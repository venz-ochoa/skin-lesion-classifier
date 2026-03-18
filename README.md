# skin-lesion-intsys
Skin Lesion Classification Model with Explainability

# Project Objectives
To develop a binary classification model (benign vs malignant) with explainability for skin lesions

# Datasets:
1. Skin Cancer MNIST: HAM10000 (License CC BY-NC 4.0)
2. ISIC 2024 - Skin Cancer Detection with 3D-TBP (License CC BY-NC 4.0)

# MVP
EfficientNet-B4 via transfer learning
Grad-CAM and Saliency
Monte Carlo dropout
ROC-AUC and Sensitivity at fixed Specificity

# Kaggle Requirements
1. Create a kaggle account, go to settings, generate legacy API, and download the kaggle.json API key
2. Place the kaggle.json in a .kaggle directory (. is for hidden directories, for security purposes) within the skin-lesion-classifier folder
3. You must go to https://www.kaggle.com/competitions/isic-2024-challenge/data and click "I understand and Accept" to authorize your account
4. Run download_data.py and select what you want to download

# Instructions
1.  To download dependencies and datasets, run download_data.py. The user is given the option to download only dependencies, only datasets, or both. Both use Kaggle API to download datasets
    and will take approx. 5-10 minutes of downloading time