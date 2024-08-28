

1. Download FinnForest
a. GO to https://etsin.fairdata.fi/dataset/06926f4b-b36a-4d6e-873c-aa3e7d84ab49/data

b. Forest_dataset/RectifiedData/RectifiedImageFormat/dataset_13Hz
    - S01_13Hz_summer_seq1_shortLoop
    - S02_13Hz_summer_seq2_longLoop
    - W07_10Hz_winter_seq7_varyingIllumination

c. Download and unzip the folders into gluefactory/data/finnForest/
├── S01_13Hz
│   ├── GT_S01.txt
│   ├── images_cam2_sr22555667
│   ├── images_cam3_sr22555660
    ├── ReadMe.txt
    ├── stereoParam_cam23_0.10pxEr.mat
    ├── stereoParam_cam23.YAML
    ├── times_S01.txt
    └── timestampSecNanoSec_S01.txt
├── S02_13Hz
    ├── GT_S02.txt
    ├── images_cam2_sr22555667
    ├── images_cam3_sr22555660
    ├── ReadMe.txt
    ├── stereoParam_cam23_0.10pxEr.mat
    ├── stereoParam_cam23.YAML
    ├── times_S02.txt
    └── timestampSecNanoSec_S02.txt
...

2. Generate Test Set Image Pairs
a. Navigate to glue-factory
b. python -m data.syntheticForestData.generatePairsCalibrated
c. Validation file saved to:  f'{DATA_PATH}/glue-factory/data/finnForest/pairs_info_calibrated_3x3.txt
