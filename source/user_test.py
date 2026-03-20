# with image reformulation for user testing, getting mean of reformulated prediction values

# gui user testing (not test images)
# confidence level, benign or malignant classification

import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#haba wow
model_path = '/content/skin-lesion-classifier/models/efficientnetb4-v2/efficientnet_b4_v3_epoch_15.pth'
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded from '{model_path}' on {device}")

# image uploaded will be transformed for appropriate size
base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# image reformulation
tta_transforms = [ base_transforms,
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(45, 45)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(90, 90)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(135, 135)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(180, 180)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
]

# classify non-reformulated
def classify_standard(img):
    tensor = base_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return probs

# classify reformulated data
def classify_tta(img):
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(img).unsqueeze(0).to(device)
            probs  = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
            all_probs.append(probs)
    return np.mean(all_probs, axis=0)

def print_result(label_header, probs):
    pred_idx   = int(np.argmax(probs))
    label      = ["Benign", "Malignant"][pred_idx] # 0 = benign, 1 = malignant

    print(f"[{label_header}]")
    print(f"Prediction  : {label}")
    print("\nConfidence Scores: ")
    print(f"Benign      : {probs[0]*100:.2f}%")
    print(f"Malignant   : {probs[1]*100:.2f}%")

# display classification results ito
def classify(image_path):
    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    img = Image.open(image_path)
    img = img.convert("RGB")

    standard_probs = classify_standard(img)
    tta_probs      = classify_tta(img)

    print("\n" + "─" * 40)
    print(f"File : {os.path.basename(image_path)}")
    print("─" * 40)
    print_result("Without Image Reformulation", standard_probs)
    print("─" * 40)
    print_result(f"With TTA ({len(tta_transforms)} Reformulation Averaged)", tta_probs)
    print("─" * 40)
    print("NOTE: For research use and testing only.")
    print("─" * 40 + "\n")

# main na
while True:
    path = input("Enter image path (or 'q' to quit): ").strip()
    if path.lower() in ("q", "quit", "exit"):
        print("Exiting.")
        break
    classify(path)