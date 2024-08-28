import os

def read_file_content(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # assuming you're only interested in .txt files
            print(f"File: {filename}\n")
            with open(os.path.join(directory, filename), 'r') as file:
                for i in range(10):
                    line = file.readline()
                    if not line:
                        break
                    print(line.strip())
            print("\n")  # New line after each file content for better separation

# Specify the directory path
directory_path = '/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/datasets/tartanSceneLists'
read_file_content(directory_path)

"""
File: test_scenes_clean.txt

SF_E_L_P007
SF_H_L_P002
SF_H_R_P004
SF_E_L_P008
SFW_E_L_P003
SFW_E_L_P009
SFW_E_R_P005


File: train_scenes.txt

SFW_E_R_P006
SFW_H_L_P012
SFW_H_L_P018
SFW_H_R_P015
SF_H_L_P005
SF_H_R_P006
SFW_E_L_P005
SFW_E_R_P001
SFW_E_R_P007
SFW_H_L_P013


File: valid_pairs.txt

### THIS IS DUMMY DATA
SF_E_L_P001/example0001.jpg SF_E_L_P001/example0002.jpg
SF_E_L_P001/example0002.jpg SF_E_L_P001/example0003.jpg
SF_E_L_P001/example0003.jpg SF_E_L_P001/example0004.jpg
SFW_H_L_P017/example0001.jpg SFW_H_L_P017/example0002.jpg
SFW_H_L_P017/example0002.jpg SFW_H_L_P017/example0003.jpg
SFW_H_L_P017/example0003.jpg SFW_H_L_P017/example0004.jpg


File: valid_scenes_clean.txt

SF_E_L_P001
SFW_H_L_P017
SFW_H_R_P014
SF_H_L_P004
SF_H_R_P005
SFW_E_L_P004
SF_E_L_P009


File: train_scenes_clean.txt

SFW_E_R_P006
SFW_H_L_P012
SFW_H_L_P018
SFW_H_R_P015
SF_H_L_P005
SF_H_R_P006
SFW_E_L_P005
SFW_E_R_P001
SFW_E_R_P007
SFW_H_L_P013

File: valid_scenes.txt

SF_E_L_P001
SFW_H_L_P017
SFW_H_R_P014
SF_H_L_P004
SF_H_R_P005
SFW_E_L_P004
SF_E_L_P009
"""