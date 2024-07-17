# PerformanceRangePrediction
Code for "Conformal Performance Range Prediction for Segmentation Output Quality Control" accepted to MICCAI UNSURE 2024.

## Data
First download the official FIVES dataset from [here](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169). Then preprocess and save to H5 using the  ```FIVES_toh5.py ```  script.

## Model Training
Train all models (included in  ```src```) using the ```train.py``` script.

## Performance Range
Perform conformalized performance range prediction on the test set using ```PerformancePrediction_tta.py``` for TTA and ```PerformancePrediction.py``` for all other models.
