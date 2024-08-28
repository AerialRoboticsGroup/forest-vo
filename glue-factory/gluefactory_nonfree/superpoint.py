"""
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork

Adapted by Philipp Lindenberger (Phil26AT)
"""

import torch
from torch import nn

from gluefactory.models.base_model import BaseModel
from gluefactory.models.utils.misc import pad_and_stack

import torch.cuda.amp as amp
import torch.utils.checkpoint as checkpoint
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
import numpy as np
from .settings import ROOT_PATH, DATA_PATH
import matplotlib.pyplot as plt                
import torchvision.transforms as transforms


from gluefactory.utils.image import load_image 
from gluefactory.geometry.wrappers import Pose

def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        radius: an integer scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    indices = torch.multinomial(scores, k, replacement=False)
    return keypoints[indices], scores[indices]


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2 * radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1
    )
    ar = torch.arange(-radius, radius + 1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius
    )
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(BaseModel):
    default_conf = {
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 256,
        # Inference
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.005,
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "randomize_keypoints_training": False,
        "remove_borders": 4, 
        "legacy_sampling": False,  # True to use the old broken sampling
        "image_mode": None, # "stereo", #"RGBD", #"stereo", # 'stereo', -- set this in the confif file
        
    }
    required_data_keys = ["image"]

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"  # noqa: E501

    def _init(self, conf):
        self.relu = nn.ReLU(inplace=True) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.TESTING_STEREO = False # This has not been implemented - stereo images to only choose keypoints present in both images

        print(conf.image_mode, "\n\n\n")
        # Define channels based on image mode

        if conf.image_mode == 'RGB':
            input_channels = 3
        elif conf.image_mode == 'RGBD':
            input_channels = 4
        elif conf.image_mode == 'stereo':
            if self.TESTING_STEREO:
                input_channels = 3
            else:
                input_channels = 6
            # input_channels = 3 ## now same as RGB but same keypoints
        else:  # grayscale or unspecified
            input_channels = 1
        print("using image mode", conf.image_mode, "with input channels", input_channels)

        self.conv1a = nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        if conf.has_detector:
            self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, conf.descriptor_dim, kernel_size=1, stride=1, padding=0
            )

        state_dict = torch.hub.load_state_dict_from_url(str(self.checkpoint_url))

        # Adjust the first layer weights as necessary
        first_conv_key = 'conv1a.weight'
        if first_conv_key in state_dict:
            first_layer_weights = state_dict[first_conv_key]
            if first_layer_weights.size(1) != input_channels:  # Only adjust if there's a mismatch
                new_weights = first_layer_weights.repeat(1, input_channels // first_layer_weights.size(1), 1, 1)
                new_weights = new_weights / (input_channels / first_layer_weights.size(1))  # Average the weights
                state_dict[first_conv_key] = new_weights
                print(f"Adjusted first layer weights to have {input_channels} input channels.")

        self.load_state_dict(state_dict, strict=False)
        

    def _shared_encoder(self, image):
        def conv_relu_pool(conv, x):
            # torch.cuda.empty_cache()
            x = conv(x)
            # torch.cuda.empty_cache()
            x = self.relu(x)
            # torch.cuda.empty_cache()
            x = self.pool(x)
            # torch.cuda.empty_cache()
            return x
        x = checkpoint.checkpoint(conv_relu_pool, self.conv1a, image)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv1b, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv2a, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv2b, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv3a, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv3b, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv4a, x)
        x = checkpoint.checkpoint(conv_relu_pool, self.conv4b, x)
        return x

    
    def process_in_chunks(self, conv_layer, relu_layer, image, chunk_size=32):
        # Requires image is a 4D tensor: [batch_size, channels, height, width]

        batch_size = image.size(0)
        chunks = torch.split(image, chunk_size)
        
        # Process each chunk separately
        processed_chunks = []
        for chunk in chunks:
            chunk = conv_layer(chunk)
            chunk = relu_layer(chunk)
            processed_chunks.append(chunk)  
        
        torch.cuda.empty_cache()
        return torch.cat(processed_chunks, dim=0) 
    

    def _forward(self, data):
        # data keys (['name', 'scene', 'T_w2cam', 'depth', 'camera', 'scales', 'image_size', 'transform', 'original_image_size', 'image'])
        image = data["image"]
        
        if self.conf.image_mode in ('grayscale', None):
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        # Shared Encoder
        if self.conf.image_mode == 'RGBD':  # Check if mode is set to use RGBD
            # Clip depth values
            try:
                depth = data["depth"].unsqueeze(1).clamp(max=250)
            except:
                # For depth data in RGBD evaluations
                # Import error fix
                import numpy as np
                # ['SF_E_L_P008-000000_left.png/SF_E_L_P008-000001_left.png']
                data["name"] = data["EvaluationName"][0]  # get the string
                fileName = data["name"].split("/")[0].split('-')[-1] # 000000_left.png
                sceneName = data["name"].split("/")[0].split('-')[0] # SF_E_L_P008
                fileName = fileName.replace(".png", "_depth.npy")
                depthBaseDir = f"{DATA_PATH}/syntheticForestData/depthData/" # SF_E_L_P001/000000_left_depth.npy
                depthFinalPath = f"{depthBaseDir}{sceneName}/{fileName}"
                data["depth"] = np.load(depthFinalPath)
                depth = torch.from_numpy(data["depth"]).unsqueeze(0).unsqueeze(0).clamp(max=250)
                depth = depth.to(image.device)

            image = torch.cat([image, depth], dim=1)

        elif self.conf.image_mode == 'stereo' and self.TESTING_STEREO == False:       
            usingFinn = False
            try:
                scene = data["scene"]
                name = data["name"] 
            except:
                # This is to load the stereo pair in evaluation
                # Import error fix
                import numpy as np
                # ['SF_E_L_P008-000000_left.png/SF_E_L_P008-000001_left.png']

                data["name"] = data["EvaluationName"][0]  # get the string
                if "Hz" in data["name"]:
                    usingFinn = True
                    # 'S01_13Hz-images_cam2_sr22555667-000000.png/S01_13Hz-images_cam2_sr22555667-000001.png']
                    # images_cam2_sr22555667 , images_cam3_sr22555660
                    folderName = data["name"].split("-")[0]
                    subfolderName = data["name"].split("-")[1]
                    
                    if subfolderName == "images_cam2_sr22555667":
                        stereoSubFolder = "images_cam3_sr22555660"
                    else:
                        stereoSubFolder = "images_cam2_sr22555667"
                    imageName =  [data["name"].split("-")[2].split("/")[0]]
                    # print(folderName, subfolderName, stereoSubFolder, imageName)
                else:
                    # print(data["name"])
                    name = [data["name"].split("/")[0].split('-')[-1]] # 000000_left.png
                    scene = [data["name"].split("/")[0].split('-')[0]] # SF_E_L_P008    
                # print("new name and scene is", name, scene)                                   
            try:
                if "L" in scene:
                    pair_scene = [sc.replace("L", "R") for sc in scene]
                    image_name = [nm.replace("left", "right") for nm in name]
                else:
                    pair_scene = [sc.replace("R", "L") for sc in scene]
                    image_name = [nm.replace("right", "left") for nm in name] 

                base_path = f"{DATA_PATH}/syntheticForestData/imageData"
                pair_image_paths = [os.path.join(base_path, ps, iname) for ps, iname in zip(pair_scene, image_name)]                         
            except Exception as e:
                if usingFinn:
                    # print("using finn")
                    base_path = f"{DATA_PATH}/finnForest/"
                    normalPath = os.path.join(base_path, folderName, subfolderName, imageName[0])
                    stereoPath = os.path.join(base_path, folderName, stereoSubFolder, imageName[0])
                    pair_image_paths = sorted([stereoPath])
                else:
                    # for normal training
                    pair_image_paths = None

            if pair_image_paths:
                # print("adding stereo images")
                pair_image = [load_image(img, False) for img in pair_image_paths]
                # Move pair_image tensors to the same device as image tensors
                device = image.device
                # pair_image = [torch.tensor(img).to(device) for img in pair_image]
                pair_image = [img.clone().detach().to(device) for img in pair_image]

                # batch, channels, height, width
                pair_image = torch.stack(pair_image)
                
                # this won't work with resizing images
                # print(image.shape)
                # print(pair_image.shape)
                image = torch.cat([image, pair_image], dim=1)
                # print(image.shape)
        else:
            # If it is RGB, then we can just use the image as is
            pass
        

        def extract_keypoints_and_descriptors(image):
            def calculate_chunk_size(tensor, desired_batch_size=8):
                """
                Calculate the chunk size dynamically based on the desired batch size.
                
                Args:
                    tensor: The input tensor. -- POWER OF 2 REQUIRED
                    desired_batch_size: The desired batch size for processing.

                Returns:
                    The chunk size to use.
                """
                batch_size = tensor.size(0)
                return max(1, batch_size // desired_batch_size)


            chunking = False
            if chunking:
                # To reduce vRAM usage spike, we can process the image in chunks
                chunk_size = calculate_chunk_size(image, desired_batch_size=8)
                # print(chunk_size)

                x = self.process_in_chunks(self.conv1a, self.relu, image, chunk_size)
                x = self.process_in_chunks(self.conv1b, self.relu, x, chunk_size)
                x = self.pool(x)  

                x = self.process_in_chunks(self.conv2a, self.relu, x, chunk_size)
                x = self.process_in_chunks(self.conv2b, self.relu, x, chunk_size)
                x = self.pool(x)

                x = self.process_in_chunks(self.conv3a, self.relu, x, chunk_size)
                x = self.process_in_chunks(self.conv3b, self.relu, x, chunk_size)
                x = self.pool(x)

                x = self.process_in_chunks(self.conv4a, self.relu, x, chunk_size)
                x = self.process_in_chunks(self.conv4b, self.relu, x, chunk_size)
            else:
                x = self.relu(self.conv1a(image))
                x = self.relu(self.conv1b(x))
                x = self.pool(x)
                x = self.relu(self.conv2a(x))
                x = self.relu(self.conv2b(x))
                x = self.pool(x)
                x = self.relu(self.conv3a(x))
                x = self.relu(self.conv3b(x))
                x = self.pool(x)
                x = self.relu(self.conv4a(x))
                x = self.relu(self.conv4b(x))


            pred = {}
            if self.conf.has_detector:
                # Compute the dense keypoint scores
                cPa = self.relu(self.convPa(x))
                scores = self.convPb(cPa)
                scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
                # iadded 25-07 for memory spikes
                torch.cuda.empty_cache()
                b, c, h, w = scores.shape
                scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
                scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
                pred["keypoint_scores"] = dense_scores = scores
            if self.conf.has_descriptor:
                # Compute the dense descriptors
                cDa = self.relu(self.convDa(x))
                dense_desc = self.convDb(cDa)
                dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
                pred["descriptors"] = dense_desc

            if self.conf.sparse_outputs:
                assert self.conf.has_detector and self.conf.has_descriptor

                scores = simple_nms(scores, self.conf.nms_radius)

                # Discard keypoints near the image borders
                if self.conf.remove_borders:
                    scores[:, : self.conf.remove_borders] = -1
                    scores[:, :, : self.conf.remove_borders] = -1
                    if "image_size" in data:
                        for i in range(scores.shape[0]):
                            w, h = data["image_size"][i]
                            scores[i, int(h.item()) - self.conf.remove_borders :] = -1
                            scores[i, :, int(w.item()) - self.conf.remove_borders :] = -1
                    else:
                        scores[:, -self.conf.remove_borders :] = -1
                        scores[:, :, -self.conf.remove_borders :] = -1

                # Extract keypoints
                best_kp = torch.where(scores > self.conf.detection_threshold)
                scores = scores[best_kp]

                # Separate into batches
                keypoints = [
                    torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
                ]
                scores = [scores[best_kp[0] == i] for i in range(b)]

                # Keep the k keypoints with highest score
                max_kps = self.conf.max_num_keypoints

                # for val we allow different
                if not self.training and self.conf.max_num_keypoints_val is not None:
                    max_kps = self.conf.max_num_keypoints_val

                # Keep the k keypoints with highest score
                if max_kps > 0:
                    if self.conf.randomize_keypoints_training and self.training:
                        # instead of selecting top-k, sample k by score weights
                        keypoints, scores = list(
                            zip(
                                *[
                                    sample_k_keypoints(k, s, max_kps)
                                    for k, s in zip(keypoints, scores)
                                ]
                            )
                        )
                    else:
                        keypoints, scores = list(
                            zip(
                                *[
                                    top_k_keypoints(k, s, max_kps)
                                    for k, s in zip(keypoints, scores)
                                ]
                            )
                        )
                    keypoints, scores = list(keypoints), list(scores)

                if self.conf["refinement_radius"] > 0:
                    keypoints = soft_argmax_refinement(
                        keypoints, dense_scores, self.conf["refinement_radius"]
                    )

                # Convert (h, w) to (x, y)
                keypoints = [torch.flip(k, [1]).float() for k in keypoints]

                if self.conf.force_num_keypoints:
                    keypoints = pad_and_stack(
                        keypoints,
                        max_kps,
                        -2,
                        mode="random_c",
                        bounds=(
                            0,
                            data.get("image_size", torch.tensor(image.shape[-2:]))
                            .min()
                            .item(),
                        ),
                    )
                    scores = pad_and_stack(scores, max_kps, -1, mode="zeros")
                else:
                    keypoints = torch.stack(keypoints, 0)
                    scores = torch.stack(scores, 0)

                # Extract descriptors
                if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                    # Batch sampling of the descriptors
                    if self.conf.legacy_sampling:
                        desc = sample_descriptors(keypoints, dense_desc, 8)
                    else:
                        desc = sample_descriptors_fix_sampling(keypoints, dense_desc, 8)
                else:
                    if self.conf.legacy_sampling:
                        desc = [
                            sample_descriptors(k[None], d[None], 8)[0]
                            for k, d in zip(keypoints, dense_desc)
                        ]
                    else:
                        desc = [
                            sample_descriptors_fix_sampling(k[None], d[None], 8)[0]
                            for k, d in zip(keypoints, dense_desc)
                        ]

                pred = {
                    "keypoints": keypoints + 0.5,
                    "keypoint_scores": scores,
                    "descriptors": desc.transpose(-1, -2),
                }

                if self.conf.dense_outputs:
                    pred["dense_descriptors"] = dense_desc

            return pred
        
        pred = extract_keypoints_and_descriptors(image)
        
        if self.conf.image_mode == 'stereo' and self.TESTING_STEREO == True:
            # @TODO use stereo images to find shared keypoints in both images
            scene = data["scene"]
            name = data["name"]
            print(f"the original scene and name is {scene} and {name}")
            try:
                pair_scene = []
                image_name = []

                for i, (sc, nm) in enumerate(zip(scene, name)):
                    print("scene name!", sc, nm)
                    if "L" in sc:
                        pair_scene.append(sc.replace("L", "R"))
                        image_name.append(nm.replace("left", "right"))
                    else:
                        pair_scene.append(sc.replace("R", "L"))
                        image_name.append(nm.replace("right", "left"))


                # tartanAir has same camera matrix for all
                camera_left = camera_right = data['camera']
                # Function to generate depth image paths
                def get_depth_path(image_path):
                    """Replaces the image extension with '_depth.npy' to get the depth path."""
                    base_name, _ = os.path.splitext(image_path)  # Remove the original extension
                    depth_path = base_name + "_depth.npy"
                    depth_path = depth_path.replace("imageData", "depthData")  # Replace the folder name
                    return depth_path

                # Generate the depth image paths
                             
                base_path = f"{DATA_PATH}/syntheticForestData/imageData"
                # pair_image_path = os.path.join(base_path, pair_scene, image_name)
                pair_image_paths = [os.path.join(base_path, ps, iname) for ps, iname in zip(pair_scene, image_name)]
                pair_depth_paths = [get_depth_path(image_path) for image_path in pair_image_paths]
                # Load and stack depth data
                depth_data = [np.load(path) for path in pair_depth_paths]
                depth_tensor = torch.from_numpy(np.stack(depth_data))

            except Exception as e:
                print(e)
                exit()
                pair_image_paths = None

            if pair_image_paths:
                # pair_image = load_image(pair_image_path, False)
                print(scene, name)
                print(pair_image_paths)
                pair_image = [load_image(img, False) for img in pair_image_paths]
                print("loaded the image pairs")
                # Move pair_image tensor to the same device as image tensor
                device = image.device
                # pair_image = pair_image.clone().detach().to(device)
                pair_image = [img.clone().detach().to(device) for img in pair_image]
                # batch, channels, height, width
                pair_image = torch.stack(pair_image)
                
                # called left and right conceptually
                if "L" in scene:
                    pred_left = pred
                    print("doing pred on pair image")
                    # print("image type",type(image))
                    # print("image shape", image.shape)
                    print("pair image type",type(pair_image))
                    print("pair image shape", pair_image.shape)
                    """
                    image type <class 'torch.Tensor'>
                    image shape torch.Size([2, 3, 480, 640])
                    """
                    pred_right = extract_keypoints_and_descriptors(pair_image)
                    # Assuming:
                    # - camera_left and camera_right are tensors containing camera parameters for each image in the batch
                    # - depth_tensor is in the shape [batch_size, height, width]
                    # - pred_right is in the shape [batch_size, num_keypoints, 2] (normalized coordinates)

                    # Get focal lengths for each image in the batch (assuming fx = fy)
                    focal_lengths = camera_left.f[:, 0]

                    # Convert keypoint coordinates to pixel coordinates
                    height, width = depth_tensor.shape[1:]  # Get height and width from depth_tensor
                    pred_right_pixel = pred_right * torch.tensor([width, height])

                    # Get depth values corresponding to keypoints
                    depth_values = depth_tensor.view(depth_tensor.shape[0], -1)[
                        ..., pred_right_pixel[:, :, 1].long(), pred_right_pixel[:, :, 0].long()
                    ]

                    # Calculate disparity for each keypoint in the batch
                    disparity = (0.25 * focal_lengths.unsqueeze(-1)) / depth_values

                    # Adjust keypoint x-coordinates
                    pred_left_pixel = pred_right_pixel.clone()
                    pred_left_pixel[:, :, 0] -= disparity  # Subtract disparity from x-coordinates
                    kp_right_transformed = pred_left_pixel
                else:
                    print("RIGHT NOT IN SCENE")
                    pred_right = pred
                    pred_left = extract_keypoints_and_descriptors(pair_image)

                    pred_right_kps = pred_right["keypoints"]
                    pred_left_kps = pred_left["keypoints"] 
                    
                    # Get focal lengths for each image in the batch (assuming fx = fy)
                    focal_lengths = camera_right.f[:, 0]  # Use camera_right for focal lengths here

                    # Convert keypoint coordinates to pixel coordinates
                    height, width = depth_tensor.shape[1:] 
                    # pred_left_pixel = pred_left_kps * torch.tensor([width, height])
                    device = pred_left_kps.device  # Get the device where pred_left_kps is located
                    print("device", device)
                    pred_left_pixel = pred_left_kps * torch.tensor([width, height]).to(device)
                    # Get depth values corresponding to keypoints
                    depth_values = depth_tensor.to(device).view(depth_tensor.shape[0], -1)[  # Move depth_tensor to the same device
                        ..., pred_left_pixel[:, :, 1].long(), pred_left_pixel[:, :, 0].long()
                    ]

                    # Calculate disparity for each keypoint in the batch
                    disparity = (0.25 * focal_lengths.unsqueeze(-1)) / depth_values

                    # Adjust keypoint x-coordinates (add disparity for left to right)
                    pred_right_pixel = pred_left_pixel.clone()
                    pred_right_pixel[:, :, 0] += disparity
                    kp_left_transformed = pred_right_pixel
                    
                try:
                    print(" pred_left['keypoints']: ", pred_left['keypoints'])
                    print("kp_right_transformed: ", kp_right_transformed)
                except:
                    print("kp_left_transformed: ", kp_left_transformed)
                    print("pred_right['keypoints']: ", pred_right['keypoints'])

    
                kp0s = pred_left['keypoints'] if "L" in scene else pred_right['keypoints']
                kp1s = kp_right_transformed if "L" in scene else kp_left_transformed

                def plot_keypoints(image, kp0s, kp1s):
                    """Plots keypoints on an image.

                    Args:
                        image: A torch tensor representing the image (batch_size, channels, height, width).
                        kp0s: Keypoints from the first image (batch_size, num_keypoints, 2).
                        kp1s: Transformed keypoints from the second image (batch_size, num_keypoints, 2).
                    """

                    # Convert image tensor to numpy array and unnormalize
                    to_pil = transforms.ToPILImage()
                    img_to_plot = to_pil(image[0]) 

                    # Create plot
                    plt.figure(figsize=(10, 5))
                    plt.imshow(img_to_plot)

                    # Plot keypoints
                    plt.scatter(kp0s[0, :, 0].cpu().numpy(), kp0s[0, :, 1].cpu().numpy(), c='r', marker='o')  # Red circles

                    # Only plot kp1s if they exist (i.e., if we have a stereo pair)
                    if kp1s is not None:
                        print(kp1s)
                        plt.scatter(kp1s[0, :, 0].cpu().numpy(), kp1s[0, :, 1].cpu().numpy(), c='b', marker='x')  # Blue crosses

                    savePath = f"{ROOT_PATH}/"
                    savePath += "keypoint_visualization.png"
                    plt.title('Keypoint Visualization')
                    plt.savefig(savePath)
                    print(savePath)

                if "L" in scene:
                    testimage = image
                else:
                    testimage = pair_image
                plot_keypoints(testimage, kp0s, kp1s) 
                exit()

                # Calculate pairwise distances
                distances = torch.cdist(kp0s, kp1s)  # Shape: [batch_size, num_keypoints_kp0, num_keypoints_kp1]

                # Create mask indicating if ANY point in kp1s is within 3-pixel radius of each point in kp0s
                mask = (distances <= 30).any(dim=-1)  # Shape: [batch_size, num_keypoints_kp0]

                # Expand the mask to include the coordinate dimension
                mask = mask.unsqueeze(-1).expand_as(kp0s)

                pred = {}

                if "L" in scene:
                    pred['keypoints'] = pred_left['keypoints'][mask]
                    pred['keypoint_scores'] = pred_left['keypoint_scores'][mask]
                    pred['descriptors'] = pred_left['descriptors'][mask]
                else:
                    pred['keypoints'] = pred_right['keypoints'][mask]
                    pred['keypoint_scores'] = pred_right['keypoint_scores'][mask]
                    pred['descriptors'] = pred_right['descriptors'][mask]

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

