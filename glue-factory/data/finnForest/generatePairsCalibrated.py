import numpy as np
import os
from gluefactory.geometry.wrappers import Camera, Pose
from scipy.spatial.transform import Rotation as R
import torch
from gluefactory.settings import DATA_PATH

"""
python -m data.syntheticForestData.generatePairsCalibrated

└── forest-vo/
    └── glue-factory/
        ├── data/
        │   └── finnForest/
        │       └── generatePairsCalibrated.py
        |   └── syntheticForestData/
        │       └── generatePairsCalibrated.py
        └── gluefactory/
            ...

"""
# How many image pairs to generate for validation set
validationImageCount = 10000
outputFileName = "pairs_info_calibrated_3x3.txt" 
output_file = f'{DATA_PATH}/glue-factory/data/finnForest/{outputFileName}'


# Fixed intrinsic matrix for all images - taken from stereoParam_cam23.YAML
K = np.array([
    [1056.6287421208997, 0.0, 952.1135175072209],
    [0.0, 1056.9497022783046, 592.824593628865],
    [0.0, 0.0, 1.0]
])


imageDirs = ['S01_13Hz/images_cam2_sr22555667', 'S02_13Hz/images_cam2_sr22555667', 'W07_10Hz/images_cam2_sr22555667']
poseDirs = ['S01_13Hz/GT_S01.txt', 'S02_13Hz/GT_S02.txt', 'W07_10Hz/GT_W07.txt']
maxLoops = validationImageCount // len(imageDirs)


def load_and_transform_poses(pose_file_path):
    """
    Each line in pose file is a flattened 4x4 matrix in row-major order:
        - [R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz]
    Returns a numpy array of Pose objects representing the relative poses between image pairs
    """
    if os.path.exists(pose_file_path):
        print("Now editing poses for", pose_file_path)
        try:
            # Load the data as (N, 12) array where each row contains a flattened pose matrix
            raw_poses = np.loadtxt(pose_file_path, delimiter='\t')
            transformed_poses = []

            # Convert each raw pose into a Pose object
            for raw_pose in raw_poses:
                # Reshape from 12 elements to (3, 4) matrix
                pose_matrix = raw_pose.reshape(3, 4)
                # Convert to a homogeneous matrix (4x4)
                homogeneous_matrix = np.eye(4)
                homogeneous_matrix[:3, :] = pose_matrix
                
                # Create a Pose object from the 4x4 transformation matrix
                homogeneous_matrix = np.linalg.inv(homogeneous_matrix)
                pose = Pose.from_4x4mat(torch.tensor(homogeneous_matrix, dtype=torch.float32))
                transformed_poses.append(pose)

            poses_matrices = transformed_poses
            relative_poses = []
     
            for i in range(len(poses_matrices) - 1):
                T_0to1 = poses_matrices[i + 1] @ poses_matrices[i].inv()
                relative_poses.append(T_0to1)

            return np.array(relative_poses)

        except Exception as e:
            print(f"Error processing pose file {pose_file_path}: {e}")
            return None

first_open = True

for i, imageDir in enumerate(imageDirs):
    # Paths to the directories and files
    print(imageDirs, poseDirs[i])
    image_dir = f'{DATA_PATH}/finnForest/{imageDir}'
    pose_file_path = f'{DATA_PATH}/finnForest/{poseDirs[i]}' 

    # Read pose data from file
    try:
        poses = load_and_transform_poses(pose_file_path)
    except:
        print(f"Unable to open the pose file {pose_file_path}")
        continue

    # List all images in the directory, sorted to ensure correct pairing
    try:
        images = sorted(os.listdir(image_dir))
        if len(images) < 2:
            raise ValueError("Not enough images in directory to form pairs")
    except Exception as e:
        print(f"Error processing image directory {image_dir}: {e}")
        continue

    folder_name = os.path.basename(image_dir)
    
    file_mode = 'w' if first_open else 'a'
    first_open = False

    # Open output file to append 'a' instead of write 'w'
    with open(output_file, file_mode) as f_out:
        # Process each pair of consecutive images
        count = 0
        missingPoses = 0
        for i in range(min(len(images) - 1, maxLoops)):
            count += 1
            img1 = images[i]
            img2 = images[i+1]
            
            # Add the folder name to the image paths e.g. S01_13Hz/images_cam2_sr22555667/000000.png
            fol = imageDir.split('/')[0]
            img1 = f"{fol}/{folder_name}/{img1}"
            img2 = f"{fol}/{folder_name}/{img2}"
            
            try:
                pose_obj = poses[i]
                pose = pose_obj.as_flat_string()
                if count < 5:
                    print(f"pose: {pose}")
                row = f"{img1} {img2} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {pose}\n"
                # row = f"{folder_name}/{img1} {folder_name}/{img2} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {pose}\n"

                # Write the formatted string to the file
                f_out.write(row)
            except IndexError:
                print(f"Missing pose data for image pair {img1} and {img2}")
                missingPoses += 1
            except Exception as e:
                print(f"Error processing pose data for image {img1}: {e}")
                missingPoses += 1
            except: 
                print(f"missing the pose for {i}, image: {folder_name}/{img1}")
                missingPoses += 1
            
        print(f"total files read was {count}, and missing poses was {missingPoses}")

    # Print confirmation that the file has been written
print(f"Pairs info file has been created successfully to output_file {output_file}")

