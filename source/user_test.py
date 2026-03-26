# ALBUMENTATIONS (final version)
# THE ONE WITHOUT ALBUMENTATION FOR USER INPUT HAS A CLOSER/HIGHER PERCENTAGE/CONFIDENCE LEVEL (RUN & TEST TO SEE) THAN THE ONE WITH ALBUMENTATION

import torch
import timm
import numpy as np
import cv2
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# change this path if needed

model_path = '/content/skin-lesion-classifier/models/efficientnetb4-v3/efficientnet_b4_v2_epoch_25.pth'

# initialize efficientnet-b4 architecture
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device).eval()
        print(f"Model loaded successfully on {device}.\n")
    except Exception as e:
        print(f"RuntimeError: {e}")
        print("Note: If 'central directory' error persists, delete and re-upload the .pth file.")
else:
    print(f"ERROR: Model not found at {model_path}")

# standard transform for normal testing
base_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# tta pipeline using albumentations (more advanced medical distortions)
# we use 8 different views to get a better average
tta_pipeline = [
    base_transform,
    A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.RandomRotate90(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.ColorJitter(brightness=0.2, contrast=0.2, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.GridDistortion(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.OpticalDistortion(distort_limit=0.1, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(224, 224), A.GaussNoise(var_limit=(10.0, 50.0), p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
]

# load image using opencv for albumentations compatibility
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# run one prediction pass
def get_prediction(img_numpy, transform_pipeline):
    augmented = transform_pipeline(image=img_numpy)["image"]
    tensor = augmented.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return probs

# main classify function with tta
def classify(image_path):
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    # load image as numpy
    img_np = preprocess_image(image_path)

    # get standard prediction
    std_probs = get_prediction(img_np, base_transform)

    # get tta predictions (average of 8 views)
    tta_results = []
    for t in tta_pipeline:
        tta_results.append(get_prediction(img_np, t))
    avg_tta_probs = np.mean(tta_results, axis=0)

    # results display
    print("\n" + "─" * 40)
    print(f"File : {os.path.basename(image_path)}")
    print("─" * 40)

    # helper for printing labels
    for label_header, probs in [("Without Albumentation", std_probs), (f"With TTA ({len(tta_pipeline)} Averaged)", avg_tta_probs)]:
        pred_idx = np.argmax(probs)
        label = "Malignant" if pred_idx == 1 else "Benign"

        print(f"[{label_header}]")
        print(f"Prediction  : {label}")
        print("\nConfidence Scores: ")
        print(f"Benign      : {probs[0]*100:.2f}%")
        print(f"Malignant   : {probs[1]*100:.2f}%")
        print("─" * 40)

    print("NOTE: For research use and testing only.")
    print("─" * 40 + "\n")

# start input loop
if __name__ == "__main__":
    if os.name == 'nt':
          # Command for Windows
          _ = os.system('cls')
    else:
          # Command for Linux and macOS
          _ = os.system('clear')
    while True:
        path = input("Enter image path (or 'q' to quit): ").strip()
        if path.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break
        classify(path)