import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf

import pickle

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

from multiprocessing import Pool

from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

def sample_n(data, num, seed=None):
    if len(data) > num:
        print(f"in treedepth sample_n the data {len(data)} is larger than num {num}")
        # selected = np.arange(num)  # This assumes that num <= len(data)
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class TreeDepth(BaseDataset):

    default_conf = {
        # paths
        "data_dir": "syntheticForestData/",
        "depth_subpath": "depthData/",
        "image_subpath": "imageData/",
        "info_dir": "fileLists",  
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": None,
        "val_pairs": None,
        # Test
        "test_split": "test_scenes_clean.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  # only with views==3
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0, 
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
        "adjacency_range": 1,
    }

    def _init(self, conf):
        logger.info(f"Initialized TreeDepth dataset with configuration: {conf}")
        print(f"conf keys: {conf.keys()}")

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            return _TripletDataset(self.conf, split)
        else:
            return _PairDataset(self.conf, split)


def load_scene_data(base_dir, scene): 
    fileListDir = base_dir + '/fileLists'

    # Removing 'L_' or 'R_' for pose files naming
    flow_scene = scene.replace('_L', '').replace('_R', '')
    
    # Define file paths for depth, image, and pose data
    depth_file_path = os.path.join(fileListDir, f"depthData_{scene}.txt")
    image_file_path = os.path.join(fileListDir, f"imageData_{scene}.txt")

    flow_file_path  = os.path.join(fileListDir, f"flowData_{flow_scene}.txt")

    # Function to read data from a text file and convert to numpy array
    def load_data_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = np.array([line.strip() for line in file.readlines()])
                if (len(data) <= 0):
                    print(f"{file_path} data IS EMPTY loaded")
                return data
        else:
            print(f"{file_path} no data found")
            raise Exception(f"File not found: {file_path}")

    # Load data
    depth_data = load_data_from_file(depth_file_path)
    image_data = load_data_from_file(image_file_path)
    flow_data = load_data_from_file(flow_file_path)
    
    # Handle pose data based on scene naming
    poses = []

    if not partial_mode:
        # Full data loading
        if '_L_' in scene:
            pose_file_path = os.path.join(base_dir, "poseData", f"{scene}_pose_left.txt")
        elif '_R_' in scene:
            pose_file_path = os.path.join(base_dir, "poseData", f"{scene}_pose_right.txt")
        else:
            print("Neither L nor R in scene identifier. Check naming convention!")
    else: # For partial mode
        print("partial mode for poses!!")
        pose_file_path = os.path.join(base_dir, "poseDataPartial", f"{scene}_pose_left.txt")

    # return transformed_poses_array
    def load_and_transform_poses(pose_file_path):
        if os.path.exists(pose_file_path):
            poses = np.loadtxt(pose_file_path, delimiter=' ').astype(np.float64)
            # Convert NED to XYZ
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            # tx, ty, tz, qx, qy, qz, qw

            def pose_to_homogeneous(pose):
                matrix = np.eye(4)
                rotation_matrix = R.from_quat(pose[3:]).as_matrix()
                matrix[:3, :3] = rotation_matrix
                matrix[:3, 3] = pose[:3]
                return matrix

            # Convert all poses to homogeneous transformation matrice camera-to-world transformations (T_cam2w)
            poses_matrices = np.array([pose_to_homogeneous(pose) for pose in poses])
            
            # Invert the matrices to get (T_w2cam)
            poses_matrices = np.array([np.linalg.inv(pose_matrix) for pose_matrix in poses_matrices])

            return poses_matrices
    
    if os.path.exists(pose_file_path):       
        poses = load_and_transform_poses(pose_file_path)
    else:
        print("pose_file_path not exist", pose_file_path)
        raise Exception(f"File not found: {pose_file_path}")

    length = len(depth_data) 
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
    camera_intrinsics = np.array([K] * length)

    # give code to check that every image, depth, flow and pose has a corresponding value
    # Failsafe - ensure only correct files considered
    image_data = [i for i in image_data if ".txt" not in i]
    flow_data = [f for f in flow_data if "flow.npy" in f]

    image_data, depth_data  = sorted(image_data), sorted(depth_data)

    image_data = np.array(image_data)
    depth_data = np.array(depth_data)
    flow_data = np.array(flow_data)
    camera_intrinsics = np.array(camera_intrinsics)
    poses = np.array(poses)

    for i, (image, flow, camera) in enumerate(zip(image_data, flow_data, camera_intrinsics)):
        image_data[i] = np.array(image, dtype=object)
        try:
            flow_data[i] = np.array(flow, dtype=object)
        except:
            print(f"{i} out of bounds for flow data")    
        camera_intrinsics[i] = np.array(camera, dtype=object)

    return {
        "image_paths": image_data,
        "depth_paths": depth_data,
        "flow_data": flow_data,
        "intrinsics" : camera_intrinsics, 
        "poses" : poses,
    }

def load_npy_file(partial_file_path):
    base_directory = f"{DATA_PATH}/syntheticForestData"
    file_path = os.path.join(base_directory, partial_file_path)

    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.debugging = False
        self.usingNegSamples = False
        # Percentage of data that is negative samples
        self.self.negativeSamplePercentage = 0.2

        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        if thirtyOverlap:
        scene_lists_path = Path(f"{ROOT_PATH}/gluefactory/datasets/tartanSceneLists")
        print(f"scene_lists_path: {scene_lists_path}")

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):            
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}

        # load metadata
        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        
        # for every list in the sceneList file 
        for scene in scenes:
            path = self.info_dir / (scene) # + ".npz")            
            self.scenes.append(scene)                 
            
            base_directory = f"{DATA_PATH}/syntheticForestData"
            info = load_scene_data(base_directory, scene)
            
            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]   
            self.intrinsics[scene] = info["intrinsics"]
        
        if self.debugging:
            oneScene = scenes[0]
            print(f"the first image is {self.images[oneScene][0]}")
            print(f"the first depth is {self.depths[oneScene][0]}")
            print(f"the first pose is {self.poses[oneScene][0]}")
            print(f"the first intrinsics is {self.intrinsics[oneScene][0]}")

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0, "No items sampled; check configuration."


    def sample_new_items(self, seed):
        # logger.info("Sampling new %s data with seed %d.", self.split, seed)
        logger.info(f"Sampling new items for {self.split} with seed {seed}.")
   
        self.items = []
        split = self.split
        num_per_scene = self.conf[self.split + "_num_per_scene"]
        

        if isinstance(num_per_scene, Iterable):
            num_pos, num_neg = num_per_scene
        else:
            num_pos = num_per_scene
            num_neg = None
        if self.conf.views == 1:
            for scene in self.scenes:
                if scene not in self.images:
                    continue
                valid = (self.images[scene] != None) | (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                ids = np.where(valid)[0]
                if num_pos and len(ids) > num_pos:
                    ids = np.random.RandomState(seed).choice(
                        ids, num_pos, replace=False
                    )
                ids = [(scene, i) for i in ids]
                self.items.extend(ids)
                logger.info(f"Scene {scene}: {len(ids)} items added.")
        else:
            for scene in self.scenes:
                base_directory = f"{DATA_PATH}/syntheticForestData/"
                info = load_scene_data(base_directory, scene)
                                
                # have pose files in imageData folder
                self.images[scene] = [img for img in self.images[scene] if "pose" not in img] 

                self.images[scene] = np.array(self.images[scene])
                self.depths[scene] = np.array(self.depths[scene])
                                
                valid = (self.images[scene] != None) & (  # noqa: E711
                    self.depths[scene] != None  # noqa: E711
                )
                
                ind = np.where(valid)[0]       
                n = len(ind)  # Number of valid items

                adjacency_range = self.conf.adjacency_range 
                mat = np.ones((n, n)) #
                # Create sequential pairs based on adjacency_range
                
                def createSequentialPairs(n, interval=None):
                    pairs = []
                    for i in range(n):
                        if interval is not None:
                            if i + interval < n:
                                pairs.append((i, i + interval))
                        else:
                            # Generate pairs (i, j) for j = i+1 to i+adjacency_range, within bounds
                            pairs.extend([(i, j) for j in range(i + 1, min(i + 1 + adjacency_range, n))])
                    return pairs
                
                # interval allows to take staggered image pairs 
                pairs = createSequentialPairs(n) 
            
                newpairs = [(scene, ind[i], scene, ind[j], 0.9) for i, j in pairs]

                if self.usingNegSamples:
                    import random
                    if "SFW" in scene:
                        otherScenes = [s for s in self.scenes if "SFW" in s]
                    else:
                        otherScenes = [s for s in self.scenes if "SFW" not in s ]
                    temp_neg_pairs = []
                    sceneNum = scene.split("_")[-1] # get the P001, P002 etc
                    try:
                        # Ensure that the scene is not compared to itself
                        otherScenes = [s for s in otherScenes if sceneNum not in s]
                        scene_neg = np.random.choice(otherScenes)

                        idx_neg = np.random.choice(range(len(self.images[scene_neg])))
 
                        temp_neg_pairs += [(scene, ind[i], scene_neg, idx_neg, 0.0) for i , _ in pairs]

                        # percentage of negative samples to use
                        neg_sample_percentage = self.negativeSamplePercentage 
                        num_neg_samples = int(len(temp_neg_pairs) * neg_sample_percentage)
                        newpairs += random.sample(temp_neg_pairs, num_neg_samples)
                    except:
                        print("failed with the missing other neg samples",otherScenes)
                else:
                    pass
                
                pairs = newpairs
                if self.debugging:
                    print(f"for scene {scene}, the num pos is {num_pos}, the num neg is {num_neg}")
                    # Count positive and negative pairs
                    print(f"Number of pairs for scene {scene}: {len(pairs)}")
                    num_pos_pairs = sum(overlap > 0 for *_, overlap in pairs)
                    num_neg_pairs = len(pairs) - num_pos_pairs

                    # Calculate overlap distribution for positive pairs
                    overlap_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    overlap_hist, _ = np.histogram([overlap for *_, overlap  in pairs if overlap >= 0], bins=overlap_bins)

                    # Print the analysis for the current scene
                    print(f"Scene: {scene}")
                    print(f"  Positive pairs: {num_pos_pairs}")
                    print(f"  Negative pairs: {num_neg_pairs}")
                    print(f"  Overlap distribution (positive pairs):")
                    for bin_start, count in zip(overlap_bins[:-1], overlap_hist):
                        print(f"    {bin_start:.1f}-{bin_start + 0.1:.1f}: {count}")
                
                self.items.extend(pairs)
                    
        if self.conf.views == 2 and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)          
            print("Did a random shuffle")

            def shuffle_preserving_batches(items, batch_size):
                # Organize items by scene
                from collections import defaultdict
                scene_dict = defaultdict(list)
                for item in items:
                    scene_dict[item[0]].append(item)  #
                
                new_items = []
                random_state = np.random.RandomState(seed)  
                # Shuffle scene order
                scene_keys = list(scene_dict.keys())
                random_state.shuffle(scene_keys)
                # Shuffle within each scene while preserving batch order
                # for scene, scene_items in scene_dict.items():
                for scene in scene_keys:
                    scene_items = scene_dict[scene]
                    # Split into batches
                    batches = [scene_items[i:i + batch_size] for i in range(0, len(scene_items), batch_size)]
                    
                    # Shuffle batches
                    random_state.shuffle(batches)
                    
                    # Flatten shuffled batches back into the list
                    for batch in batches:
                        new_items.extend(batch)
                
                # Shuffle all batches across scenes to mix scenes
                batched_items = [new_items[i:i + batch_size] for i in range(0, len(new_items), batch_size)]
                random_state.shuffle(batched_items)
                
                # Flatten the final list
                shuffled_items = [item for batch in batched_items for item in batch]
                
                print("SHUFFLED !! self.items in sample_new_oitems treedepth!!")
                return shuffled_items

            # If want to shuffle preserving batches, uncomment the line below
            # self.items = shuffle_preserving_batches(self.items, batch_size=8)

    def _read_view(self, scene, idx):
        # print(f"Reading view from {scene}, {idx}")
        path = self.root / self.images[scene][idx]
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)         
        T = self.poses[scene][idx].astype(np.float32, copy=False)
    
        # read image
        if self.conf.read_image:
            # print(f"Read image from {self.root / self.images[scene][idx]}")
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

        # read depth
        if self.conf.read_depth:
            localRoot = Path(f"{DATA_PATH}/syntheticForestData/")
            depth_path = (
                localRoot / self.conf.depth_subpath / scene / (path.stem + "_depth.npy")
            )            
            with open(depth_path, "rb") as f:  
                depth = np.load(f)
            depth = torch.Tensor(depth)[None]           
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None

        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"

        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = np.rot90(img, k=-k, axes=(-2, -1))
                if self.conf.read_depth:
                    depth = np.rot90(depth, k=-k, axes=(-2, -1)).copy()
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)

        name = path.name

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]
        K = scale_intrinsics(K, data["scales"])

       
        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T), #T, 
            "depth": depth,
            "camera": Camera.from_calibration_matrix(K).float(),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].copy()
                x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}

        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        if self.conf.views == 2:
            if isinstance(idx, list):
                # scene, idx0, idx1, overlap = idx
                if self.usingNegSamples:
                    scene0, idx0, scene1, idx1, overlap = idx
                else:
                    scene, idx0, idx1, overlap= idx
                # scene0, idx0, scene1, idx1, overlap = idx
            else:
                if self.usingNegSamples:
                    scene0, idx0, scene1, idx1, overlap = self.items[idx]
                    # print("was one item")
                else:
                    scene, idx0, _, idx1, overlap = self.items[idx]
                # scene, idx0, idx1, overlap = self.items[idx]
            
            if self.usingNegSamples:
                data0 = self._read_view(scene0, idx0)
                data1 = self._read_view(scene1, idx1)
            else:
                data0 = self._read_view(scene, idx0)
                data1 = self._read_view(scene, idx1)
            
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()

            data["overlap_0to1"] = overlap
            
            if self.usingNegSamples:
                data["name"] = f"{scene0}/{data0['name']}_{scene1}/{data1['name']}"
            else:
                data["name"] = f"{scene}/{data0['name']}_{scene}/{data1['name']}"

        else:
            assert self.conf.views == 1
            scene, idx0 = self.items[idx]
            data = self._read_view(scene, idx0)
        if self.usingNegSamples:
            data["scene"] = [scene0, scene1] 
        else:
            data["scene"] = scene
        
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)


class _TripletDataset(_PairDataset):
    def sample_new_items(self, seed):
        logging.info("Sampling new triplets with seed %d", seed)
        self.items = []
        split = self.split
        num = self.conf[self.split + "_num_per_scene"]
        if split != "train" and self.conf[split + "_pairs"] is not None:
            if Path(self.conf[split + "_pairs"]).exists():
                pairs_path = Path(self.conf[split + "_pairs"])
            else:
                pairs_path = DATA_PATH / "configs" / self.conf[split + "_pairs"]
            for line in pairs_path.read_text().rstrip("\n").split("\n"):
                im0, im1, im2 = line.split(" ")
                assert im0[:4] == im1[:4]
                scene = im1[:4]
                idx0 = np.where(self.images[scene] == im0)
                idx1 = np.where(self.images[scene] == im1)
                idx2 = np.where(self.images[scene] == im2)
                self.items.append((scene, idx0, idx1, idx2, 1.0, 1.0, 1.0))
        else:
            for scene in self.scenes:
                path = self.info_dir / (scene + ".npz")
                assert path.exists(), path
                info = np.load(str(path), allow_pickle=True)
                if self.conf.num_overlap_bins > 1:
                    raise NotImplementedError("TODO")
                valid = (self.images[scene] != None) & (  # noqa: E711
                    self.depth[scene] != None  # noqa: E711
                ) 
                if not valid:
                    print(f"At least one condition is False for scene {scene}")              
                ind = np.where(valid)[0]
                mat = info["overlap_matrix"][valid][:, valid]
                good = (mat > self.conf.min_overlap) & (mat <= self.conf.max_overlap)
                triplets = []
                if self.conf.triplet_enforce_overlap:
                    pairs = np.stack(np.where(good), -1)
                    for i0, i1 in pairs:
                        for i2 in pairs[pairs[:, 0] == i0, 1]:
                            if good[i1, i2]:
                                triplets.append((i0, i1, i2))
                    if len(triplets) > num:
                        selected = np.random.RandomState(seed).choice(
                            len(triplets), num, replace=False
                        )
                        selected = range(num)
                        triplets = np.array(triplets)[selected]
                else:
                    # we first enforce that each row has >1 pairs
                    non_unique = good.sum(-1) > 1
                    ind_r = np.where(non_unique)[0]
                    good = good[non_unique]
                    pairs = np.stack(np.where(good), -1)
                    if len(pairs) > num:
                        selected = np.random.RandomState(seed).choice(
                            len(pairs), num, replace=False
                        )
                        pairs = pairs[selected]
                    for idx, (k, i) in enumerate(pairs):
                        # We now sample a j from row k s.t. i != j
                        possible_j = np.where(good[k])[0]
                        possible_j = possible_j[possible_j != i]
                        selected = np.random.RandomState(seed + idx).choice(
                            len(possible_j), 1, replace=False
                        )[0]
                        triplets.append((ind_r[k], i, possible_j[selected]))
                    triplets = [
                        (scene, ind[k], ind[i], ind[j], mat[k, i], mat[k, j], mat[i, j])
                        for k, i, j in triplets
                    ]
                    self.items.extend(triplets)
        # np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, idx):
        scene, idx0, idx1, idx2, overlap01, overlap02, overlap12 = self.items[idx]
        data0 = self._read_view(scene, idx0)
        data1 = self._read_view(scene, idx1)
        data2 = self._read_view(scene, idx2)
        data = {
            "view0": data0,
            "view1": data1,
            "view2": data2,
        }
        data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_0to2"] = data2["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_1to2"] = data2["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_2to0"] = data0["T_w2cam"] @ data2["T_w2cam"].inv()
        data["T_2to1"] = data1["T_w2cam"] @ data2["T_w2cam"].inv()

        data["overlap_0to1"] = overlap01
        data["overlap_0to2"] = overlap02
        data["overlap_1to2"] = overlap12
        data["scene"] = scene
        data["name"] = f"{scene}/{data0['name']}_{data1['name']}_{data2['name']}"
        ### I added
        #logger.info(f"Processed triplet: {data['name']} with overlaps {overlap01}, {overlap02}, {overlap12}")
    
        return data

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "min_overlap": 0.1,
        "max_overlap": 0.7,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = TreeDepth(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info("The dataset has elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    savePath = f"{DATA_PATH}/syntheticForestData/visualizeHeatMaps.png"
    plt.savefig(savePath)
    print(f"Saved visualization to {savePath}")


if __name__ == "__main__":
    from .. import logger  
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)

