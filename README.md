# Conformalized Performance Range Prediction Code
Code for "Conformal Performance Range Prediction for Segmentation Output Quality Control" accepted to MICCAI UNSURE 2024.

## Virtual Environment Setup
The code is implemented in Python 3.11.2. One way of getting all the requirements is using virtualenv and the requirements.txt file.

Set up a virtual environment (e.g. conda or virtualenv) with Python 3.11.2
Install as follows:
pip install -r requirements.txt

## Data
First download the official FIVES dataset from [here](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169). Then preprocess and save to H5 using the  ```FIVES_toh5.py ```  script.

## Model Training
Train all models (included in  ```src```) using the ```train.py``` script.

## Performance Range
Perform conformalized performance range prediction on the test set using ```PerformancePrediction_tta.py``` for TTA and ```PerformancePrediction.py``` for all other models.
