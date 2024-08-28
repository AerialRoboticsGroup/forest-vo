import numpy as np
import os
from gluefactory.geometry.wrappers import Camera, Pose
from scipy.spatial.transform import Rotation as R
from gluefactory.settings import ROOT_PATH, DATA_PATH

# python -m data.syntheticForestData.generatePairsCalibrated

# Read the file and store each line in an array
with open(f'{ROOT_PATH}/gluefactory/datasets/tartanSceneLists/test_scenes_clean.txt', 'r') as f:
    lines = f.readlines()
imageDirs = [line.strip() for line in lines]

# usingHardPoses = False
# if usingHardPoses:
#     imageDirs = [i for i in imageDirs if "H" in i]


# Globals
first_open = True
outputFileName = "pairs_info_calibrated_relative_3x3.txt"
output_file = f'{DATA_PATH}/syntheticForestData/{outputFileName}'

# Fixed intrinsic matrix for all images
# https://github.com/castacks/tartanair_tools/blob/master/data_type.md
K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])

def load_and_transform_poses(pose_file_path):
    """
    TartanAir is camera-to-world, so we invert the matrices to get world-to-camera
    Tartan Air is in NED frame
    # NED where x=z, y=z, z=-y (points down)
    # reorder into X = right / left -- Y = up / down -- Z = forward / backward

    """
    if os.path.exists(pose_file_path):
        
        poses = np.loadtxt(pose_file_path, delimiter=' ').astype(np.float64)
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
        
        # Now are pose objects
        poses_matrices = [Pose.from_4x4mat(T) for T in poses_matrices]

        relative_poses = []
        
        # poses_matrices[i] is T_w2cam for frame i
        for i in range(len(poses_matrices) - 1):
            T_0to1 = poses_matrices[i + 1] @ poses_matrices[i].inv()
            relative_poses.append(T_0to1)

        return np.array(relative_poses)


for imageDir in imageDirs:
    poseData = imageDir + ("_pose_left.txt" if "_L_" in imageDir else "_pose_right.txt")

    # Paths to the directories and files
    image_dir = f'{DATA_DIR}/syntheticForestData/imageData/{imageDir}'
    pose_file_path = f'{DATA_DIR}/syntheticForestData/poseData/{poseData}'

    # Read pose data from file
    try:
        poses = load_and_transform_poses(pose_file_path)
    except:
        print(f"unable to open the pose file {pose_file_path}")
        # skip the for loop
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
        for i in range(len(images) - 1):
            count += 1
            # SF_E_R_P001/image.png
            img1 = images[i]
           
            img2 = images[i+1]
            try:
                pose_obj = poses[i]
                pose = pose_obj.as_flat_string()
                # Prepare the row to be written to the output file
                row = f"{folder_name}/{img1} {folder_name}/{img2} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {K[0][0]} {K[0][1]} {K[0][2]} {K[1][0]} {K[1][1]} {K[1][2]} {K[2][0]} {K[2][1]} {K[2][2]} {pose}\n"
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
