
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
    - Preprocessing Pipelines: 
        a. resizing to 224x224 (EfficientNet standards)
        b. normalization
        c. data augmentation (rotation, flips, and color jitters)
    - Dataset Class Balance: 80/20, class imbalanced addressed by WeightedRandomSampler

4. Model Performance Evaluation
    - Benchmark Epochs (Val F1, Val Acc, Val Loss)
        a. Epoch 20 (Best F1): 0.7689, 90.51%, 0.2705
        b. Epoch 13 (Lowest Loss): 0.7406, 89.01%, 0.2590
        c. Epoch 25 (Highest Accuracy): 0.7443, 91.01%, 0.3323
    
