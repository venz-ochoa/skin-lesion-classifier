import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim

# the dataset is very imbalanced, check eda/class_distribution.png
# we use albumentations for advanced medical-grade data augmentation
# this helps the model see lesions from different angles and lighting conditions

# this is for the training data
# augmentation pipeline: advanced spatial transforms, noise, and medical distortions
train_transforms = A.Compose([
    # standard size for efficientnet
    A.Resize(224, 224),
    
    # spatial augmentations (flipping and rotating)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # advanced distortions (simulates skin stretching/camera angles)
    A.OneOf([
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
    ], p=0.3),

    # brightness, contrast, and sensor noise
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    
    # finalizing augmentations (normalization and tensor conversion)
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# this is for the val/test (no random augmentation here)
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# display a sample of the augmented data
sample_path = '/content/skin-lesion-classifier/data/model_data/train/malignant/'

# check if directory exists etc
if os.path.exists(sample_path) and len(os.listdir(sample_path)) > 0:
    
    # select the first image (malignant) to augment
    sample_file = os.listdir(sample_path)[0] 
    img_path = os.path.join(sample_path, sample_file)
    
    # albumentations works with numpy arrays (opencv)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display 5 variations
    plt.figure(figsize=(15, 5))

    for i in range(5):
        # apply albumentations (returns a dictionary)
        augmented_data = train_transforms(image=img)
        augmented_img = augmented_data["image"]
        
        # convert back for plotting
        curr_img = augmented_img.permute(1, 2, 0).numpy()
        curr_img = np.clip(curr_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(curr_img)
        plt.axis('off')
        plt.title(f"Variation {i+1}")

    plt.tight_layout()
    plt.savefig('/content/skin-lesion-classifier/EDA/augmentation_samples.png')
    print("Augmentation sample saved as 'augmentation_samples.png'")
else:
    print("Error in generating sample image of data augmentation.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")