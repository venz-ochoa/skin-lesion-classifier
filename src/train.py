# DO NOT RUN THIS
# VERSION 2 ALBUMENTATIONS
# if you are getting errors, run the gitclone (first code block) and only run this code block
# efficientnet-b4 (Albumentations + Weighted Sampling + AdamW)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets
import timm
import numpy as np
import cv2
import sys
import os
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# directory validation
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'source')))
current_file_path = os.path.dirname(os.path.abspath(__file__)) 
project_root_path = os.path.abspath(os.path.join(current_file_path, ".."))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

from src.data_pipeline.augment_data import train_transforms, val_transforms

# custom dataset class because Albumentations needs NumPy/OpenCV inputs
class AlbumentationsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.data.samples[index]
        # load image with opencv for albumentations compatibility
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

    def __len__(self):
        return len(self.data)

def train():
    # device configuration for better performance
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # folders containing combined HAM10000 and ISIC (Malignant-only) data
    train_dir = os.path.join(project_root_path, 'data/model_data/train')
    val_dir = os.path.join(project_root_path, 'data/model_data/val')

    # applying the custom dataset with albumentations
    train_dataset = AlbumentationsDataset(train_dir, transform=train_transforms)
    val_dataset = AlbumentationsDataset(val_dir, transform=val_transforms)

    # weightedrandomsampler for the imbalance (balances malignant/benign in batches)
    # this looks at the combined counts of both datasets
    target = np.array(train_dataset.data.targets)
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    sampler_weights = 1. / class_sample_count
    samples_weight = np.array([sampler_weights[t] for t in target])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight), len(samples_weight))

    # data loaders - using batch size 16 for efficientnet-b4 memory limits
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # efficientNet-B4 - pretrained and modified for 2 classes
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
    model = model.to(device)

    # loss weights - using square root class weights for better precision/recall balance
    loss_weights = 1. / np.sqrt(class_sample_count)
    weights_tensor = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # adamW - learning rate and weight decay optimized for stability
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)

    # learning rate scheduler to reduce LR when Val Loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # training loop
    num_epochs = 30
    print(f"Starting Training with Combined Dataset for {num_epochs} Epochs...")

    for epoch in range(num_epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # validation phase
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # evaluation metrics to track precision/recall and accuracy baseline
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Recall: {recall:.4f} | Prec: {precision:.4f}")

        # save the model state every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_dir = os.path.join(project_root_path, 'src/model/train_models')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f'efficientnet_b4_{epoch+1}.pth'))

    # final save
    save_dir = os.path.join(project_root_path, 'src/model/train_models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'efficientnet_b4_baseline_final.pth'))
    print("Training Complete. Combined models saved.")

if __name__ == '__main__':
    train()