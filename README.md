# skin-lesion-intsys
Skin Lesion Classification Model with Explainability.

# Google Colab
This project was primarily made in Google Colab.

# Project Objectives
To develop a binary classification model (benign vs malignant) with explainability for skin lesions.

# Datasets:
1. Skin Cancer MNIST: HAM10000 (License CC BY-NC 4.0)
2. ISIC 2019 Challenge Dataset (Malignant-only subset for class balancing)

# MVP
- EfficientNet-B4 via transfer learning
- Augmentation (Albumentations: GridDistortion, OpticalDistortion, GaussNoise)
- Test Time Augmentation (TTA) with 6-view reformulation averaging for stability
- Grad-CAM and Saliency
- Monte Carlo dropout
- ROC-AUC and Sensitivity at fixed Specificity

# Instructions
1.  To download dependencies and datasets, run download_data.py. The user is given the option to download only dependencies, only datasets, or both. HAM10000 and ISIC 2019 uses Kaggle API to download datasets and will take approx. 7-15 minutes of downloading time.

