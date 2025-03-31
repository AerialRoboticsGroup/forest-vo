import os
import zipfile
import shutil
from pathlib import Path
from gluefactory.settings import DATA_PATH
import time

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
        print()
        if filename.endswith(".zip"):
            filepath = os.path.join(base_dir, filename)

            print(f"Found file to unzip: {filename}")
            filename = filename.split("/")[-1]
            # Extract information from the filename
            parts = filename.split("_")

            scene_name, difficulty, data_type, camera = parts[0], parts[1], parts[2], parts[3].split(".")[0]
            
            scene_name = renameMap.get(scene_name)
            orignal_difficulty = difficulty
            difficulty = renameMap.get(difficulty)
            data_type = renameMap.get(data_type)
            camera = renameMap.get(camera)
            print("scene_name, difficulty, data_type, camera:", scene_name, difficulty, data_type, camera)

            """
                inflating: seasonsforest/Easy/P011/image_left/000157_left.png  
                inflating: seasonsforest/Easy/P011/image_left/000220_left.png  
                inflating: seasonsforest/Easy/P011/image_left/000071_left.png  
                inflating: seasonsforest/Easy/P011/image_left/000108_left.png  
                inflating: seasonsforest/Easy/P011/image_left/000154_left.png  
                inflating: seasonsforest/Easy/P011/image_left/000030_left.png  
                inflating: seasonsforest/Easy/P011/pose_right.txt  
                inflating: seasonsforest/Easy/P011/pose_left.txt  
            """
            dest_folder = os.path.join(str(base_dir), data_type)         

            if dest_folder:
                # Create a temporary directory for extraction
                temp_dir = os.path.join(base_dir, "new_temp_extract")
                os.makedirs(temp_dir, exist_ok=True)
                print(f"Extracting to temporary directory: {temp_dir}")

                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        # Construct the full path to the destination file
                        destination_path = Path(temp_dir) / file
                        
                        # Check if the file already exists
                        if destination_path.exists():
                            pass
                        else:
                            # Extract the file since it does not exist
                            zip_ref.extract(file, temp_dir)

                # Get the trajectory directory name (PXXX) from the extracted content
                extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                # print("extracted_dirs", extracted_dirs)
                if len(extracted_dirs) == 1:
                    traj_dir = extracted_dirs[0]

                    try:
                        dest_dir = os.path.join(dest_root, dest_folder, f"{scene_name.upper()}_{difficulty}_{camera.upper()}")
                    except:
                        # Camera is missing for flowData etc as not L or R versions
                        dest_dir = os.path.join(dest_root, dest_folder, f"{scene_name.upper()}_{difficulty}")
                    
                    # Move the extracted content to the final destination
                    dataPath = os.path.join(temp_dir, traj_dir, orignal_difficulty)
                    print(os.listdir(dataPath))
                    for item in os.listdir(dataPath):
                        print("temp_dir", temp_dir)
                        print("traj_dir", traj_dir)
                        print("item", item)
                        s = os.path.join(temp_dir, traj_dir, orignal_difficulty, item)       
                        s = Path(s)
                        
                        # 1. Ensure s is a subdirectory of DATA_PATH
                        if not s.resolve().is_relative_to(DATA_PATH.resolve()):
                            raise ValueError(f"Path {s} is not within the base directory {DATA_PATH}")

                        # 2. Ensure s is not accidentally set to the root directory or any other sensitive directory
                        if s == Path('/') or s == Path.home():
                            raise ValueError(f"Path {s} is pointing to an unsafe directory!")              
                    
                        d = str(dest_dir) + "_" + item
                        d = os.path.join(d)
                        d = Path(d)
                        os.makedirs(d, exist_ok=True)
                        source_dir = s
                        destination_dir = d
                        
                        for item in source_dir.iterdir():
                            if item.is_dir() :
                                # Copy all files from "mode"_left/right directly to the destination
                                for file in item.iterdir():
                                    if file.is_file():  # Ensure it's a file (like .png)
                                        shutil.copy2(file, destination_dir / file.name)
                                        
                            elif item.is_file():  
                                # Copy pose files 
                                shutil.copy2(item, destination_dir / item.name)
                

                    # Clean up the temporary directory
                    temp_dir = Path(temp_dir)

                    # Sanity checks before deletion
                    if not temp_dir.resolve().is_relative_to(DATA_PATH.resolve()):
                        raise ValueError(f"Path {temp_dir} is not within the base directory {DATA_PATH}")

                    if temp_dir == Path('/') or temp_dir == Path.home():
                        raise ValueError(f"Path {temp_dir} is pointing to an unsafe directory!")

                    # Attempt to remove the directory and its contents
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"Successfully removed the temporary directory: {temp_dir}")
                    except Exception as e:
                        print(f"Failed to remove the directory: {temp_dir}. Error: {e}")

                    print(f"Unzipped and organized: {filename}")
                else:
                    print(f"Error: Unexpected number of directories extracted from {filename}")
                    # shutil.rmtree(temp_dir)
            else:
                print(f"Skipping unrecognized data type: {filename}")

def copy_pose_files(base_dir):
    base_dir = Path(base_dir)
    # Define the directories
    image_data_dir = base_dir / "imageData"
    pose_data_dir = base_dir / "poseData"
    
    # Ensure the pose_data_dir exists
    pose_data_dir.mkdir(parents=True, exist_ok=True)
    
    # List all items in the imageData directory
    for folder_name in os.listdir(image_data_dir):
        source_folder = image_data_dir / folder_name
        if source_folder.is_dir():  # Only proceed if it's a directory
            # Define source file paths
            source_pose_left = source_folder / "pose_left.txt"
            source_pose_right = source_folder / "pose_right.txt"
            
            # Define destination file paths with the new naming format
            dest_pose_left = pose_data_dir / f"{folder_name}_pose_left.txt"
            dest_pose_right = pose_data_dir / f"{folder_name}_pose_right.txt"
            
            # Copy the files if they exist in the source folder
            if source_pose_left.exists():
                shutil.copy2(source_pose_left, dest_pose_left)
                print(f"Copied {source_pose_left} to {dest_pose_left}")
            else:
                print(f"{source_pose_left} does not exist. Skipping.")
                
            if source_pose_right.exists():
                shutil.copy2(source_pose_right, dest_pose_right)
                print(f"Copied {source_pose_right} to {dest_pose_right}")
            else:
                print(f"{source_pose_right} does not exist. Skipping.")

if __name__ == "__main__":
    organize_data()
    time.sleep(10)
    copy_pose_files(base_dir)
    