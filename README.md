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

# Instructions
1. Set-up
    To download dependencies and datasets, run download_data.py
    The user is given the option to download only dependencies, only datasets, or both
    Both use Kaggle API to download datasets
    Will take approx. 5-10 minutes of downloading time