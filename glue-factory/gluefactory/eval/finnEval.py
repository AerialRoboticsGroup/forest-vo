import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets.image_pairs import ImagePairs
from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH, ROOT_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust
import os

logger = logging.getLogger(__name__)

pairsPath = "finnForest/pairs_info_calibrated_relative_3x3.txt"

#python -m gluefactory.eval.finnEval --checkpoint 02-08-depth-pretrain-superglue-gray-32-512 --overwrite

class ForestPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",    # this is a python file
            "pairs": pairsPath , 
            # In pairs path is SF_E_R_P001/filename.jpg  SF_E_R_P001/filename.jpg intrinsic1 intrinsic2  poses: tx ty tz qx qy qz qw
            "root": "finnForest/",  
            "extra_data": "relative_pose", 
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  
            } 
        },
        "eval": {
            "estimator": "poselib", 
            "ransac_th": 1.0,
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []
    
    def _init(self, conf):

        if not (DATA_PATH / "finnForest").exists():
            logger.error("finnforest dataset not found.")
            raise FileNotFoundError("finn forest dataset directory is missing.")

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Creates a dataloader to load forest dataset images with depth information."""
        data_conf = self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        test_data = dataset.get_data_loader("test")
        return test_data
    
    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Generates predictions for each evaluation data point in the forest dataset."""
        pred_file = experiment_dir / "predictions.h5"
        def backup_existing_file(file_path):
            if file_path.exists():
                backup_path = file_path.with_suffix('.bak')
                file_path.rename(backup_path)

        pred_file = experiment_dir / "predictions.h5"
        if pred_file.exists() and overwrite:
            backup_existing_file(pred_file)

        if not pred_file.exists() or overwrite:
            if model is None:
                print(self.conf.model, self.conf.checkpoint)
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file, usingExtraStats=True):
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            for th in test_thresholds:
                pose_results_i = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

            # Store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)


        myAUCThresholds = [1, 2.5, 5, 7.5, 10, 15, 20]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=myAUCThresholds, key="rel_pose_error"
        )
    
        # I added mean and std for pose translation and rotation
        if usingExtraStats:
            
            mean_t_pose_results, best_th = eval_poses(
                pose_results, auc_ths=myAUCThresholds, key="mean_t_pose_error"
            )
            mean_r_pose_results, best_th = eval_poses(
                pose_results, auc_ths=myAUCThresholds, key="mean_r_pose_error"
            )
            summaries = {
                **summaries,
                **best_pose_results,
                **mean_t_pose_results,
                **mean_r_pose_results,
            }
        else:        
            # default pose results
            summaries = {
                **summaries,
                **best_pose_results,
            }

        results = {**results, **pose_results[best_th]}


        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results

if __name__ == "__main__":
    from .. import logger  # overwrite the logger
    
    usingExtraStats = True
    estimator_type = "poselib"

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ForestPipeline.default_conf)

    output_dir = Path(EVAL_PATH, dataset_name)

    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )    

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ForestPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    txtfolder = "Evaluations"
    imgFolder = "EvalFigures"

    if args.checkpoint is None:
        # If using pretrained conf models then save to a differrent folder
        args.checkpoint = args.conf
        txtfolder = "PretrainedEvaluations"
        imgFolder = "PretrainedEvalFigures"
    

    print("args checkpoint used is ", args.checkpoint)
    filePath = f"{str(ROOT_PATH)}{txtfolder}/{args.checkpoint}/{dataset_name}_{estimator_type}.txt" 
    os.makedirs(os.path.dirname(filePath), exist_ok=True)

    # Note that this appends to existing files
    with open(filePath, "a") as file:
        for k, v in s.items():
            file.write(f"{k}: {v}\n")
    

    if usingExtraStats:
        # This plots the pose-recall graph
        args.plot = True
        savePath = str(ROOT_DIR)
        if args.plot:
            for name, fig in f.items():
                fig.canvas.manager.set_window_title(name)
                
                finalPath = f"{savePath}/{imgFolder}/{args.checkpoint}/{name}_{dataset_name}_{estimator_type}.png"
                os.makedirs(os.path.dirname(finalPath), exist_ok=True)
                fig.savefig(finalPath)
                print(f"Saved figure to {finalPath}")
                with open(filePath, "a") as file:
                    file.write(f"\n{finalPath}")
    
    print("Saved results to", filePath)
