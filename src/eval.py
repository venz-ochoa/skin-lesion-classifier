#Metrics: Val Loss, Acc, F1, Precision, Recall, AU-ROC, CM,
#Sensitivity, MC Dropout, Precision-Recall Curve

import torch
import torch.nn as nn
import timm
import numpy as np
import sys
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (f1_score, accuracy_score, recall_score,
                             precision_score, confusion_matrix,
                             roc_auc_score, roc_curve, classification_report,
                             precision_recall_curve, average_precision_score)

#directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# We still use val_transforms because the test set needs the exact same resizing/normalization
from src.data_pipeline.augment_data import val_transforms

#setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

#paths - UPDATED TO TEST
test_dir = os.path.join(project_root, 'data/model_data/test') 
model_path = os.path.join(project_root, 'src/model/efficientnetb4_v2.pth')

class AlbuImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, target

#test dataset
if not os.path.exists(test_dir):
    print(f"ERROR: Test directory not found at: {test_dir}")
    sys.exit(1)

test_dataset = AlbuImageFolder(test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

#model
print(f"Loading model from: {model_path}")
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

try:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"CRITICAL LOAD ERROR: {e}")
    print("If you see 'invalid load key', your .pth file is likely a Git LFS pointer or corrupted.")
    sys.exit(1)

criterion = nn.CrossEntropyLoss()

def run_master_audit():
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    print(f"Report on: {os.path.basename(model_path)}")
    print("─" * 60)

    #evaluation phase
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    test_loss = 0.0

    print("\033[H\033[J", end="")
    print("Standard Test Evaluation...")
    print(f"Loading model from: {model_path}\n")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #performance metrics
    avg_test_loss = test_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    ap_score = average_precision_score(all_labels, all_probs)

    performance_text = (
        f"[ CORE PERFORMANCE - TEST SET ]\n"
        f"Test Loss       : {avg_test_loss:.4f}\n"
        f"Accuracy        : {acc*100:.2f}%\n"
        f"F1 Score        : {f1:.4f}\n"
        f"Recall (Sens)   : {recall:.4f}\n"
        f"Precision       : {prec:.4f}\n"
        f"AU-ROC Score    : {auroc:.4f}\n"
        f"Avg Precision   : {ap_score:.4f}\n"
    )
    
    print("\n" + performance_text)

    #sensitivity
    target_fpr = 0.10
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fixed_sens = np.interp(target_fpr, fpr, tpr)

    print("\n[ SENSITIVITY AT FIXED CAPACITY ]")
    print(f"FPR Target     : {target_fpr*100}%")
    print(f"Sensitivity    : {fixed_sens:.4f}")

    #mc dropout
    print("\nMC Dropout Analysis (10 samples)...")
    print("This may take a moment...")
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if hasattr(model, 'classifier'):
        original_classifier = model.classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            original_classifier
        )

    mc_samples = 10
    mc_results = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pass_probs = []
            for images, _ in test_loader:
                outputs = model(images.to(device))
                pass_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            mc_results.append(pass_probs)

    if hasattr(model, 'classifier'):
        model.classifier = original_classifier
    model.eval()

    mc_results = np.array(mc_results)
    uncertainty = np.var(mc_results, axis=0)
    mean_var = np.mean(uncertainty)
    print(f"Mean Variance : {mean_var:.6f}")

    #graphs and such
    print("\nGenerating Visual Dashboards...")
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auroc:.3f}')
    ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[0].set_title('Test ROC Curve')
    ax[0].legend()

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1],
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    ax[1].set_title('Test Confusion Matrix')

    p, r, _ = precision_recall_curve(all_labels, all_probs)
    ax[2].plot(r, p, color='green', lw=2, label=f'AP = {ap_score:.3f}')
    ax[2].set_title('Test Precision-Recall Curve')
    ax[2].legend()

    plt.tight_layout()
    
    #save results - UPDATED FILENAMES
    results_path = os.path.join(project_root, 'experiments/results/metrics')
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, 'test_audit_dashboard.png'))
    
    report = classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'])
    
    with open(os.path.join(results_path, 'test_classification_report.txt'), 'w') as f:
        f.write(f"TEST DATASET AUDIT REPORT: {os.path.basename(model_path)}\n")
        f.write("="*40 + "\n")
        f.write(performance_text)
        f.write("\n" + "="*40 + "\n")
        f.write("\tDETAILED CLASSIFICATION REPORT\n")
        f.write("="*40 + "\n")
        f.write(report)
        f.write("\n" + "="*40 + "\n")
        f.write(f"MC Dropout Mean Variance: {mean_var:.6f}\n")
        f.write(f"Sensitivity @ {target_fpr*100}% FPR: {fixed_sens:.4f}\n")
    
    print(f"Saved to {results_path}")
    print("\n" + "="*40 + "\n\tDETAILED REPORT\n" + "="*40)
    print(report)
    
    plt.show()

if __name__ == "__main__":
    run_master_audit()