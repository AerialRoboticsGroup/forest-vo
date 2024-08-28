import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...settings import DATA_PATH, ROOT_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics

# Added epipolar geometry 
from ...geometry.epipolar import T_to_E, E_to_F, generalized_epi_dist, sym_epipolar_distance_all 
from ...geometry.wrappers import Camera, Pose
import pickle

# plotting
from ...visualization.visualize_batch import make_match_figures

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                # High CUDA memory usage here
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
                # High CUDA memory usage here
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            # memory spike here
            torch.cuda.empty_cache()
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    torch.cuda.empty_cache()
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        torch.cuda.empty_cache()
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": True, ## improve backprop
        "weights": "superpoint_lightglue", #  will get from model https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
        "scrambleWeights": False,
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.useEpipolar = conf.useEpipolar
        self.debug_mode = False   
        if conf.scrambleWeights:
            self.scrambleWeights = conf.scrambleWeights
        else:
            self.scrambleWeights = False
            
        if self.useEpipolar:
            print("\n\n\n Using Lightglue epipolar error \n\n\n") 
        if self.debug_mode:
            print("lightglue in debug mode")

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim
        
        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                print(f"weights loaded from {conf.weights}")
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                print(f"weights loaded from {DATA_PATH / conf.weights}")
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:

                fname = (
                    f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                    + ".pth"
                )
                url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
                state_dict = torch.hub.load_state_dict_from_url(
                    url,
                    file_name=fname,
                )
                print(f"loaded weights from url {url}")


        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            if self.scrambleWeights:
                # Scramble the weights in the state_dict
                for key, value in state_dict.items():
                    if 'weight' in key or 'bias' in key:  # Target only weight and bias parameters
                        # Create a new random tensor with the same shape and data type as the original
                        random_tensor = torch.randn_like(value)
                        # Replace the original values with the random ones
                        state_dict[key] = random_tensor     
                print("Scrambled weights of lightglue")
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }
      
        sum_weights = 1.0
        # this goes to models,utils, losses
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i

        if self.useEpipolar:                     
            cam0, cam1 = data["view0"]["camera"], data["view1"]["camera"]
            try:
                m0, m1, mscores0, mscores1 = pred["matches0"], pred["matches1"], pred["matching_scores0"], pred["matching_scores1"]
                m0_mask = m0 > -1
                m1_mask = m1 > -1
                if self.debug_mode:
                    print("Shape of m0:", m0.shape)
                    print("Shape of m1:", m1.shape)
                    print("Shape of mscores0:", mscores0.shape)
                    print("Shape of mscores1:", mscores1.shape)
                    print("Shape of m0_mask:", m0_mask.shape)
                    print("Shape of m1_mask:", m1_mask.shape)
                    print("shape pred kps0", pred["keypoints0"].shape)
                    print("shape pred kps1", pred["keypoints1"].shape)
                    print("Total true values in m0_mask:", m0_mask.sum().item())
                    print("Total true values in m1_mask:", m1_mask.sum().item())
                # Mask out invalid keypoints (set them to zero)
                kp0 = torch.where(m0_mask.unsqueeze(-1), pred["keypoints0"], pred["keypoints0"].new_zeros(pred["keypoints0"].shape))
                kp1 = torch.where(m1_mask.unsqueeze(-1), pred["keypoints1"], pred["keypoints1"].new_zeros(pred["keypoints1"].shape))

                # Mask out invalid matching scores (set them to zero)
                mscores0 = torch.where(m0_mask, mscores0, mscores0.new_zeros(mscores0.shape))
                mscores1 = torch.where(m1_mask, mscores1, mscores1.new_zeros(mscores1.shape))
                if self.debug_mode:
                    try:
                        print("\n\n after change kps and mscores:")
                        print("Shape of filtered kp0:", kp0.shape)
                        print("Shape of filtered kp1:", kp1.shape)
                        print("Shape of cam0:", cam0.shape)
                        print("Shape of cam1:", cam1.shape)
                        print("Shape of data['T_0to1']:", data['T_0to1'].shape)
                        print("Shape of mscores0:", mscores0.shape)
                        print("Shape of mscores1:", mscores1.shape)
                        print("Shape of m0_mask:", m0_mask.shape)
                        print("Shape of m1_mask:", m1_mask.shape)
                        print("m0 shape:", m0.shape)
                        print("m1 shape:", m1.shape)                        
                    except Exception as e:
                        print("Error occurred while printing shapes:", e)
  
                epipolarError = generalized_epi_dist(
                        kp0, kp1, cam0, cam1, data["T_0to1"], all=False, essential=True
                    )
                
                mscores = []

                # Calculate the threshold based on camera intrinsics and pixel error tolerance
                # Get the focal lengths from the camera object 
                pixel_error_tolerance = 4 
                fx = cam0.f[..., 0] 
                fy = cam0.f[..., 1]

                translation = data["T_0to1"].t
                baseline = torch.norm(translation)

                # Calculate the average focal length
                f = (fx + fy) / 2
                threshold = (pixel_error_tolerance / (2 * f)) * baseline
                # print("Threshold:", threshold)
                threshold = threshold.unsqueeze(1) 
                # Set errors below the threshold to zero
                num_errors_below_threshold = (epipolarError < threshold).sum().item()

                epipolarError = torch.where(epipolarError < threshold, epipolarError.new_zeros(epipolarError.shape), epipolarError)
                # print(f"after doign {pixel_error_tolerance} pixel threshold", epipolarError)
                
                if self.debug_mode:
                    print("the m0 mask", m0_mask)
                    print("epipolar error before", epipolarError)
                    print("shape epipolar error", epipolarError.shape)
                    print("mscores0 shape", mscores0.shape)
                    print("mscores1 shape", mscores1.shape)
                confidence_scores = mscores0 
                if self.debug_mode:
                    # confidence_scores = torch.cat(mscores)
                    print("confidnec scors shape", confidence_scores.shape)
                
                # log1p ensures no log(0) issues and scales up the confidence scores in a non-linear fashion.
                confidence_log_scale = torch.log1p(confidence_scores) 
                if self.debug_mode:
                    print("confidence_log_scale.shape", confidence_log_scale.shape)
                    print("confidence_log_scale", confidence_log_scale)
                # Get the batch size dynamically from epipolarError
                batch_size = pred["keypoints0"].shape[0]

                # Reshape confidence_scores based on the dynamic batch size
                normalized_confidence = confidence_log_scale.view(batch_size, -1) 
                normalized_confidence /= normalized_confidence.max(dim=1, keepdim=True).values 

                if self.debug_mode:
                    print("epipolarError shaope", epipolarError.shape)
                    print("normalized_confidence shape", normalized_confidence.shape)
                    print("\n\n\n\n before multiply")
                    print("the normalise confience is", normalized_confidence)
                    print("the epipolar error is", epipolarError)
                
                weighted_epipolarError = epipolarError * normalized_confidence
                num_valid_matches_per_batch = m0_mask.sum(dim=1)
                zero_match_mask = num_valid_matches_per_batch == 0

                # Set epipolarErrorPerImage to zero for batches with zero matches
                epipolarErrorPerImage = torch.where(zero_match_mask, 
                                                    torch.zeros_like(num_valid_matches_per_batch), 
                                                    weighted_epipolarError.sum(dim=1) / num_valid_matches_per_batch)

                hinge_loss_threshold = 0.2 * losses["total"]  
                # clipped_hinge_loss = min(epipolarErrorPerImage, hinge_loss_threshold)
                if self.debug_mode:
                    # print("epipolar error per image shape", epipolarErrorPerImage)
                    print("epipiolar error per image", epipolarErrorPerImage)
                    print("the max loss each can be is", hinge_loss_threshold)
                clipped_hinge_loss = torch.clamp(epipolarErrorPerImage, max=hinge_loss_threshold)

                if self.debug_mode:
                    # Add the clipped hinge loss directly as the epipolar contribution to total loss
                    print("\n \n epipolar error: ")
                    print(f"losses before {losses["total"]}")
                losses["total"] += clipped_hinge_loss
                if self.debug_mode:
                    print(f"clipped loss was {clipped_hinge_loss}")
                    print(f"losses after added {losses["total"]}")
                # exit()
            except Exception as e:
                # to handle before do a forward pass
                print("\n\n\nERROR  in epipolar geometry loss: ", e)
                raise Exception


        losses["total"] /= sum_weights

        if self.debug_mode:
            try:
                print(data["name"])
                # print(data["scene"])
            except:
                pass

            import os
            import matplotlib.pyplot as plt
            scene = data["name"]
            batchsize = len(scene)
            def transform_scene_format(scenes, depth=False):
                if depth:
                    results = []
                    for scene in scenes:
                        # Split the scene into parts based on underscore
                        parts = scene.split('_')
                        
                        # Initialize a dictionary to store parts by directory
                        directory_files = {}
                        
                        for part in parts:
                            # Safely split each part into directory and filename
                            if '/' in part:
                                directory, filename = part.rsplit('/', 1)
                                filename = filename.split('.png')[0]  # Remove the .png extension
                                # Append filename to the list in the dictionary keyed by directory
                                if directory in directory_files:
                                    directory_files[directory].append(filename)
                                else:
                                    directory_files[directory] = [filename]
                        
                        # Construct new scene names for each directory
                        for directory, files in directory_files.items():
                            new_scene = f"{directory}---{'-'.join(files)}---"
                            results.append(new_scene)
                    
                    # Join all entries into a single string with underscores between them
                    name = "_".join(results)
                    return name + ".png" 
                else:
                    # Split the first element to get the directory and initial part
                    directory, initial_file = scenes[0].split('/')
                    initial_part = initial_file.split('.')[0]  # Removes the file extension

                    # Extract the remaining parts from other elements
                    remaining_parts = [scene.split('/')[1].split('.')[0] for scene in scenes[1:]]
                    
                    # Concatenate all parts with hyphens
                    final_filename = directory + '-' + '-'.join([initial_part] + remaining_parts) + '.png'

                    return final_filename

            depth = True
            if depth:
                name = "DEPTH"
            else:
                name = ""
            # print(data[""])
            img_name = transform_scene_format(scene, depth)
            # img_name = "megadepth.png"
            save_path = f"{ROOT_PATH}/LightGlueMatchPlots/{name}-{img_name}"
            # Check if the directory exists, if not create it
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # can add depth overlay if required
            figs = make_match_figures(pred, data, n_pairs=batchsize)
            # Optionally display or save the figure

            figs['matching'].savefig(save_path)
            print(f"Saved figure to {save_path}")
            plt.close(figs['matching'])  # Close the plot after saving to free up memory

        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]
            if self.debug_mode:
                for i in range(len(data['name'])):
                    image_name = data['name'][i]
                    current_loss = nll[i].item()  # Get the loss value for this image from the current nll
                    total_loss = losses["total"][i]
                    print(f"Image: {image_name}, Current Iteration Loss: {current_loss} and losses total {total_loss}")

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = LightGlue
