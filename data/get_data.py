# python script to automate data downloads and execution
# use a venv for this to run on macOS
import subprocess
import sys
import os
import time

def clear_terminal():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()

def dependencies():
    print("\n--- Installing Requirements ---")
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_file_dir, "..", "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        clear_terminal()
        print("\nRequirements installed successfully.")
        time.sleep(3)
    except Exception as e:
        print(f"Error: {e}")
    

def datasets():
    clear_terminal()
    print("\n--- Downloading Datasets ---\n")
    ham_path = "data/src_data/ham10000"
    isic_path = "data/src_data/isic2019"
    os.makedirs(ham_path, exist_ok=True)
    os.makedirs(isic_path, exist_ok=True)

    try:
        # HAM10000
        if not os.listdir(ham_path):
            subprocess.run(["kaggle", "datasets", "download", "-d", "kmader/skin-cancer-mnist-ham10000", "-p", ham_path, "--unzip"], check=True)
            print("HAM10000 Successfully downloaded.")
        else:
            print("HAM10000 already exists. Skipping download...")

        # ISIC 2019
        if not os.listdir(isic_path):
            subprocess.run(["kaggle", "datasets", "download", "-d", "andrewmvd/isic-2019", "-p", isic_path, "--unzip"], check=True)
            print("ISIC 2019 Successfully downloaded.")
        else:
            print("ISIC 2019 already exists. Skipping download...")
        time.sleep(3)
        print("ISIC 2019 already exists.")
    except Exception as e:
        print(f"Error: {e}")
    

def run_pipeline():
    """Combines classification and augmentation into one logical step"""
    clear_terminal()
    print("\n--- Sorting & Augmenting Data ---")
    pipeline_dir = os.path.join("src", "data_pipeline")
    
    scripts = [
        ("classify_data.py", "Classifying and organizing images..."),
        ("augment_data.py", "Generating balanced augmentation samples...")
    ]

    for script, message in scripts:
        script_path = os.path.join(pipeline_dir, script)
        if script == "classify_data.py" and os.path.exists('data/model_data'):
            print(f"\nSkipping {script} as model_data already exists")
            continue
        if os.path.exists(script_path):
            print(f"\n{message}")
            try:
                subprocess.run([sys.executable, script_path], check=True)
            except Exception as e:
                print(f"Error in {script}: {e}")
                break
        else:
            print(f"Error: Could not find {script_path}")
            break
    
    print("\nData Pipeline processing complete.")
    time.sleep(3)

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
    
    print("\nData Pipeline processing complete.")
    input("\nPress Enter to return to menu...")

def run_script(script_name):
    clear_terminal()
    script_path = os.path.join("src", script_name)
    if os.path.exists(script_path):
        print(f"\n--- Launching {script_name} ---")
        print("Process may take a moment to start...")
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
        print("\n===========================================")
        print("        SKIN LESION PROJECT TERMINAL      ")
        print("===========================================")
        print("  [1] Install Dependencies & Data Pipeline")
        print("  [2] Run Evaluation (eval.py)")
        print("  [3] Run Explainability Dashboard")
        print("  [4] Exit")
        print("===========================================")
        
        try:
            select = input("\nEnter choice (1-4): ")
            if select == "1":
                dependencies()
                datasets()
                run_pipeline()
            elif select == "2":
                run_script("eval.py")
            elif select == "3":
                run_script("explainability.py")
            elif select == "4":
                print("Exiting...")
                break
            else:
                print(f"'{select}' is not a valid option.")
                input("Press Enter to try again...")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
