

1. Download TartanAir Dataset
a. 
    i. cd glue-factory/
    ii python -m data.tartanAirDownloader.download_training --rgb --depth --flow
Check the download_training_zipfiles.txt contain the data you want to download
python download_training.py --rgb --depth --flow
# This code was edited from : https://github.com/castacks/tartanair_tools

b. The data will be downloaded in zip format like this
forest-vo/data/syntheticForestData/seasonsforest_Easy_depth_left.zip
forest-vo/data/syntheticForestData/seasonsforest_Easy_depth_right.zip
forest-vo/data/syntheticForestData/seasonsforest_Easy_flow_flow.zip
forest-vo/data/syntheticForestData/seasonsforest_Easy_flow_mask.zip
forest-vo/data/syntheticForestData/seasonsforest_Easy_image_left.zip
forest-vo/data/syntheticForestData/seasonsforest_Easy_image_right.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_depth_left.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_depth_right.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_flow_flow.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_flow_mask.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_image_left.zip
forest-vo/data/syntheticForestData/seasonsforest_Hard_image_right.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_depth_left.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_depth_right.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_flow_flow.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_flow_mask.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_image_left.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Easy_image_right.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_depth_left.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_depth_right.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_flow_flow.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_flow_mask.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_image_left.zip
forest-vo/data/syntheticForestData/seasonsforest_winter_Hard_image_right.zip

c. Extract and reformat images
    i. cd glue-factory/
    ii python -m data.syntheticForestData.extractZipFiles

The data should be in the format
├── imageData
    ├── SF_E_L_P001
        ├── 000001_left.png
    ├── SF_E_L_P002
    ...
├── depthData
    ├── SF_E_L_P001
        ├── 000001_left_depth.npy
    ├── SF_E_L_P002
├── poseData
├── flowData
...

2. (OPTIONAL) Generate NEW Test Set Image Pairs
If you want to change the existing test set image pairs then follow these steps:

a. Navigate to glue-factory
b. python -m data.syntheticForestData.generatePairsCalibrated
c. Test set file saved to:  f'{DATA_PATH}/glue-factory/data/syntheticForestData/pairs_info_calibrated_3x3.txt

Note in: forest-vo/glue-factory/gluefactory/datasets/tartanSceneLists
- there are splits for train/validation/test
If you want to update the splits, run generateImagePairs.py to get valid_pairs.txt
- SFW_E_L_P001/000000_left.png SFW_E_L_P001/000001_left.png