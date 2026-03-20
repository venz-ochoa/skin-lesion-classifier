import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

#the dataset is very imbalanced, check eda/class_distribution.png
#we do data augmenting to make it fairer for the malignant class

#this is for the training data
#augmentation pipeline is rotation, flips, brightness, contrast, saturation
train_transforms = transforms.Compose([
    #this is the standard for efficient net
    transforms.Resize((224, 224)),
    #rotation and flipping
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),

    #brightness, contrast, saturation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    #finalizing augmentations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#this is for the val/test
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#display a sample of the augmented data
sample_path = '/content/skin-lesion-classifier/data/model_data/train/malignant/'

#check if directory exists etc
if os.path.exists(sample_path) and len(os.listdir(sample_path)) > 0:
    
    #select the first image (malignant) to augment
    sample_file = os.listdir(sample_path)[0] 
    img = Image.open(os.path.join(sample_path, sample_file))

    #display 5 variations
    plt.figure(figsize=(15, 5))

    for i in range(5):
        augmented_img = train_transforms(img)
        curr_img = augmented_img.permute(1, 2, 0).numpy()
        curr_img = np.clip(curr_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(curr_img)
        plt.axis('off')
        plt.title(f"Variation {i+1}")

    plt.tight_layout()
    plt.savefig('/content/skin-lesion-classifier/data/sample_data_augmentation/augmentation_samples.png')
    print("Augmentation sample saved as 'augmentation_samples.png'")
else:
    print("Error in generating sample image of data augmentation.")