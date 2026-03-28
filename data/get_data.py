# python script to automate data downloads and execution
# use a venv for this to run on macOS
import subprocess
import sys
import os

def clear_terminal():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

def dependencies():
    print("\n--- [1] Installing Requirements ---")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_file_dir, "..", "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Requirements installed successfully.")
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to return to menu...")

def datasets():
    print("\n--- [2] Downloading Datasets ---")
    ham_path = "data/src_data/ham10000"
    isic_path = "data/src_data/isic2019"
    os.makedirs(ham_path, exist_ok=True)
    os.makedirs(isic_path, exist_ok=True)

    try:
        # HAM10000
        if not os.listdir(ham_path):
            subprocess.run(["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000", "-p", ham_path, "--unzip"], check=True)
            print("HAM10000 Downloaded.")
        else:
            print("HAM10000 already exists.")

        # ISIC 2019
        if not os.listdir(isic_path):
            subprocess.run(["kaggle", "datasets", "download", "-d", "andrewmvd/isic-2019", "-p", isic_path, "--unzip"], check=True)
            print("ISIC 2019 Downloaded.")
        else:
            print("⏩ ISIC 2019 already exists.")
    except Exception as e:
        print(f"Error: {e}")
    input("\nPress Enter to return to menu...")

def run_pipeline():
    """Combines classification and augmentation into one logical step"""
    print("\n--- [3] Sorting & Augmenting Data ---")
    pipeline_dir = os.path.join("src", "data_pipeline")
    
    scripts = [
        ("classify_data.py", "Classifying and organizing images..."),
        ("augment_data.py", "Generating balanced augmentation samples...")
    ]

    for script, message in scripts:
        script_path = os.path.join(pipeline_dir, script)
        if os.path.exists(script_path):
            print(f"\n>> {message}")
            try:
                subprocess.run([sys.executable, script_path], check=True)
            except Exception as e:
                print(f"Error in {script}: {e}")
                break
        else:
            print(f"Error: Could not find {script_path}")
            break
    
    print("\n✅ Data Pipeline processing complete.")
    input("\nPress Enter to return to menu...")

def run_script(script_name):
    script_path = os.path.join("src", script_name)
    if os.path.exists(script_path):
        print(f"\n--- Launching {script_name} ---")
        try:
            subprocess.run([sys.executable, script_path])
        except Exception as e:
            print(f"Runtime Error: {e}")
    else:
        print(f"Error: Could not find {script_path}")
    input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    while True:
        clear_terminal()
        print("========================================")
        print("      SKIN LESION PROJECT TERMINAL      ")
        print("========================================")
        print("  [1] Install Requirements")
        print("  [2] Install Datasets")
        print("  [3] Process Data (Classify & Augment)")
        print("  [4] Run Evaluation (eval.py)")
        print("  [5] Run Explainability Dashboard")
        print("  [6] Exit")
        print("========================================")
        
        try:
            select = input("\nEnter choice (1-6): ")
            
            if select == "1":
                dependencies()
            elif select == "2":
                datasets()
            elif select == "3":
                run_pipeline()
            elif select == "4":
                run_script("eval.py")
            elif select == "5":
                run_script("explainability.py")
            elif select == "6":
                print("Exiting...")
                break
            else:
                print(f"'{select}' is not a valid option.")
                input("Press Enter to try again...")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
