
1. Model Details
    - Developed by: Danielle Lenon, Seiji Liwag, Claire Ochoa, and Venice Ochoa
    - Developed on: March 2026
    - Model Type: Binary image classification (Convolutional Neural Network)
    - Architecture: EfficientNet-B4 (Pre-trained on ImageNet, Transfer Learning)

2. Model Objectives
    - Primary Use: Assist dermatologists in classifying suspicious lesions for biopsy
    - Intended Users: Medical professionals, researchers, general population
    - Out-of-scope: Self-diagnosis without medical consultation/supervision
    This model should not replace or serve as the only basis for diagnosis. It's merely for academic or research use, not actual deployment.

3. Training Data & EDA
    - Dataset: HAM10000 (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
    - Dataset Size: 10,015 images
    - Dataset: ISIC 2019 Challenge (https://www.kaggle.com/datasets/andrewmvd/isic-2019/data)
    - Dataset Size: 25,331 images
    - Preprocessing Pipelines: 
        a. resizing to 224x224 (EfficientNet standards)
        b. normalization
        c. data augmentation (Albumentations: GridDistortion, OpticalDistortion, GaussNoise)
    - Dataset Class Balance: 80/20, class imbalanced addressed by WeightedRandomSampler and integration of malignant-only samples from ISIC 2019.

4. Model Performance Evaluation
    - Benchmark Epochs (Val F1, Val Acc, Val Loss)
        a. Best F1 Score (Epoch 30): 0.9775
        b. Lowest Val Loss (Epoch 29): 0.0949
        c. Highest Accuracy (Epoch 30): 97.39%
        d. Peak Recall (Epoch 26): 98.20%
        e. Peak Precision (Epoch 13):98.20%
    
