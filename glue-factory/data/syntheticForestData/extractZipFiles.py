import os
import zipfile
import shutil
from pathlib import Path
from gluefactory.settings import DATA_PATH

# Define the base directory where the zip files are located
base_dir = f"{DATA_PATH}/syntheticForestData"

# Define the destination directory for the organized data
dest_root = f"{DATA_PATH}/syntheticForestData"

def organize_data():
    """
    Unzips files in the base directory, organizes the extracted content into appropriate folders based on data type,
    scene name, and difficulty level.
    """
    renameMap = {
        "seasonsforest": "SF", 
        "seasonsforest_winter" : "SFW",
        "Easy": "E",
        "Hard": "H",
        "depth": "depthData",
        "image": "imageData",
        "flow": "flowData",
        "pose": "poseData",
        "left": "L",
        "right": "R"
    }

    # Iterate over each zip file in the base directory
    for filename in os.listdir(base_dir):
        if filename.endswith(".zip"):
            filepath = os.path.join(base_dir, filename)

            # forest-vo/data/syntheticForestData/seasonsforest_Easy_depth_left.zip
            filename = filename.split("/")[-1]
            # Extract information from the filename
            parts = filename.split("_")
            # forest-vo/data/syntheticForestData/seasonsforest_Easy_depth_left.zip
            scene_name, difficulty, data_type, camera = parts[0], parts[1], parts[2], parts[3].split(".")[0]
            scene_name = renameMap.get(scene_name)
            difficulty = renameMap.get(difficulty)
            data_type = renameMap.get(data_type)
            camera = renameMap.get(camera)

            # Map data type to destination folder
            data_type_to_folder = {
                "image": "imageData",
                "depth": "depthData",
                "flow": "flowData",
                "pose": "poseData"  # Assuming you have pose data in zip files as well
            }
            dest_folder = data_type_to_folder.get(data_type)

            if dest_folder:
                # Create a temporary directory for extraction
                temp_dir = os.path.join(base_dir, "temp_extract")
                os.makedirs(temp_dir, exist_ok=True)

                # Unzip the file to the temporary directory
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Get the trajectory directory name (PXXX) from the extracted content
                extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                if len(extracted_dirs) == 1:
                    traj_dir = extracted_dirs[0]

                    # Create the final destination directory
                    dest_dir = os.path.join(dest_root, dest_folder, f"{scene_name.upper()}_{difficulty}_{camera.upper()}_{traj_dir}")
                    os.makedirs(dest_dir, exist_ok=True)

                    # Move the extracted content to the final destination
                    for item in os.listdir(os.path.join(temp_dir, traj_dir)):
                        s = os.path.join(temp_dir, traj_dir, item)
                        d = os.path.join(dest_dir, item)
                        if os.path.isdir(s):
                            shutil.copytree(s, d)
                        else:
                            shutil.copy2(s, d)

                    # Clean up the temporary directory
                    shutil.rmtree(temp_dir)

                    print(f"Unzipped and organized: {filename}")
                else:
                    print(f"Error: Unexpected number of directories extracted from {filename}")
                    shutil.rmtree(temp_dir)
            else:
                print(f"Skipping unrecognized data type: {filename}")

if __name__ == "__main__":
    organize_data()