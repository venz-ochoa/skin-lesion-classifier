import os
import sys
import subprocess
import torch
import torch.nn as nn
import timm
import joblib

def check_dependencies():
    print("--- Python Dependency Check ---")
    required = ["torch", "torchvision", "timm", "joblib", "sklearn", "pandas", "cv2", "PIL"]
    all_ok = True
    for pkg in required:
        try:
            if pkg == "cv2":
                import cv2 as _
            elif pkg == "PIL":
                from PIL import Image as _
            elif pkg == "sklearn":
                import sklearn as _
            else:
                __import__(pkg)
            print(f"{pkg.ljust(12)} Found")
        except ImportError:
            print(f"{pkg.ljust(12)} MISSING")
            all_ok = False
    return all_ok

def check_models():
    print("\n--- 🧠 Model Integrity Check ---")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = {
        "Vision (B4)": "src/model/efficientnetb4_v2.pth",
        "NLP Model": "src/model/nlp/nlp_model.joblib",
        "NLP Vector": "src/model/nlp/vectorizer.joblib",
        "RL Policy": "src/model/rl/threshold_agent.joblib"
    }
    
    all_ok = True
    for name, rel_path in models.items():
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"{name.ljust(12)} NOT FOUND at {rel_path}")
            all_ok = False
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            # Threshold for Vision model is 10MB (real one is ~71MB)
            if "Vision" in name and size_mb < 1.0:
                print(f"{name.ljust(12)} Detected LFS POINTER or INVALID FILE ({size_mb*1024:.1f} KB).")
                all_ok = False
            elif "RL Policy" in name and size_mb < 0.001:
                print(f"{name.ljust(12)} Found but EMPTY (0 bytes). Needs training.")
                all_ok = False
            else:
                print(f"{name.ljust(12)} Found ({size_mb:.2f} MB)")
    return all_ok

if __name__ == "__main__":
    print("=== SKIN LESION AI HEALTH DOCTOR ===\n")
    deps = check_dependencies()
    mods = check_models()
    
    if deps and mods:
        print("\nALL SYSTEMS NOMINAL. You can run the dashboard at 100% capacity.")
    elif not deps:
        print("\nFIX: Run 'pip install -r requirements.txt' to install missing packages.")
    elif not mods:
        print("\nFIX: The system is missing or has invalid model weight binaries.")
        print("     1. Ensure 'src/model/efficientnetb4_v2.pth' exists (~71MB).")
        print("     2. Ensure NLP models are present in 'src/model/nlp/'.")
        print("     3. For RL Policy, run 'python src/rl_threshold.py' and click 'OPTIMIZE' to generate the agent.")
