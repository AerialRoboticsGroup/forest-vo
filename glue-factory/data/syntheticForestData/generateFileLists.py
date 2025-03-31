"""
Run this code if you need new train/validation/test splits
### 1) generate the file lists for the synthetic forest data
### 2) update datasets/treedepth.py to use the new file list path
### 3) update the valid pairs and scenes lists in datasets too
"""

import os
from gluefactory.settings import ROOT_PATH, DATA_PATH

folder = 'fileLists'
root_directory = f'{DATA_PATH}/syntheticForestData/'
data_types = ['imageData/', 'depthData/'] # , 'flowData/']
scenes =  os.listdir(root_directory + 'imageData/')
# or can access all scenes from f'{ROOT_PATH}/gluefactory/datasets/tartanSceneLists/allFolders.txt"

newSceneListFolder = "tartanSceneLists"

valid_file = f'{ROOT_PATH}/gluefactory/datasets/{newSceneListFolder}/valid_scenes_clean.txt'
validation = []

with open(valid_file, 'r') as file:
    for line in file:
        validation.append(line.strip())

test_file = f'{ROOT_PATH}/gluefactory/datasets/{newSceneListFolder}/test_scenes_clean.txt'
test = []

with open(test_file, 'r') as file:
    for line in file:
        test.append(line.strip())

all_file = f'{ROOT_PATH}/gluefactory/datasets/tartanSceneLists/allFolders.txt'
all_scenes = []
with open(all_file, 'r') as file:
    for line in file:
        all_scenes.append(line.strip())


train_images = list(set(all_scenes) - set(validation) - set(test))

def generate_file_lists(data_root, types, scenes=None):
    file_lists_dir = os.path.join(data_root, folder)
    os.makedirs(file_lists_dir, exist_ok=True)

    for data_type in types:
        data_type_path = os.path.join(data_root, data_type) 
      
        # specify specific scenes to generate file lists for
        if scenes is not None:
            for scene in scenes:
                if data_type == 'flowData/':
                    scene = scene.replace("_L_","_").replace("_R_","_")
                scene_path = os.path.join(data_type_path, scene)

                for subdir, _, files in os.walk(scene_path):                
                    txt_filename = os.path.join(file_lists_dir, f"{data_type.rstrip('/')}_{scene}.txt")
                    if os.path.exists(txt_filename):
                        continue
                    else:
                        print(f"Generating file: {txt_filename}")
                    if files:
                        files = sorted(files)                       
                        with open(txt_filename, 'w') as file_list:
                            for filename in files:
                                relative_path = os.path.relpath(os.path.join(subdir, filename), data_root)
                                file_list.write(relative_path + '\n')
                    else:
                        print(f"Empty directory: {subdir}")
                        continue
                    print(f"Wrote file to {txt_filename}")
        else: 
            # generating al/ files
            for subdir, _, files in os.walk(data_type_path):                
                
                txt_filename = os.path.join(file_lists_dir, f"{data_type.rstrip('/')}_{os.path.basename(subdir)}.txt")
                if os.path.exists(txt_filename):
                    print(f"Skipping existing file: {txt_filename}")
                    continue
                else:
                    print(f"Generating file: {txt_filename}")
                    # continue
                if files:
                    files = sorted(files)
                    for filename in files:
                        relative_path = os.path.relpath(os.path.join(subdir, filename), data_root)
                        print(relative_path)
                else:
                    print(f"Empty directory: {subdir}")


def check_and_regenerate_empty_file_lists(data_root):
    file_lists_dir = os.path.join(data_root, folder)
    for file_name in os.listdir(file_lists_dir):
        file_path = os.path.join(file_lists_dir, file_name)
        if os.path.getsize(file_path) == 0:
            print(f"Empty file list found: {file_path}")
            parts = file_name.split('_')
            data_type = '_'.join(parts[:-1])
            # Remove the .txt extension
            subdir_name = parts[-1].split('.')[0]  
            
            # Adjust the strings for directory name
            data_path = str(data_type).replace('Data_', 'Data/')
            # print("data_root, data_path, subdir_name", data_root, data_path, subdir_name)
            specific_subdir_path = os.path.join(data_root, data_path + "_" + subdir_name)                
            print("specific_subdir_path", specific_subdir_path)
            if os.path.exists(specific_subdir_path):
                generate_file_lists_for_subdir(data_root, specific_subdir_path, data_type, subdir_name)
            else:
                print(f"Directory not found for regeneration: {specific_subdir_path}")


def generate_file_lists_for_subdir(data_root, subdir_path, data_type, subdir_name):
    file_lists_dir = os.path.join(data_root, 'fileLists')
    txt_filename = os.path.join(file_lists_dir, f"{data_type}_{subdir_name}.txt")
    # Exclude hidden files
    files = [f for f in os.listdir(subdir_path) if not f.startswith('.')]  
    if files:
        with open(txt_filename, 'w') as file_list:
            for filename in files:
                relative_path = os.path.relpath(os.path.join(subdir_path, filename), data_root)
                file_list.write(relative_path + '\n')
        print(f"Regenerated file list for: {txt_filename}")
    else:
        print(f"No files to list in {subdir_path}, keeping file list empty.")


generate_file_lists(root_directory, data_types, scenes)

# Enable to check and regenerate empty file lists that were not done during the initial run
# check_and_regenerate_empty_file_lists(root_directory)
