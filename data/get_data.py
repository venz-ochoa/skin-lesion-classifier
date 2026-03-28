# python script to automate data downloads and execution
# use a venv for this to run on macOS
import subprocess
import sys
import os

def clear_terminal():
    # ASCII clear for a professional look
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

def run_script(script_name):
    # Standardizing pathing to find scripts in the src folder
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
        print("  [3] Run Evaluation (eval.py)")
        print("  [4] Run Explainability Dashboard")
        print("  [5] Exit")
        print("========================================")
        
        try:
            select = input("\nEnter choice (1-5): ")
            
            if select == "1":
                dependencies()
            elif select == "2":
                datasets()
            elif select == "3":
                run_script("eval.py")
            elif select == "4":
                run_script("explainability.py")
            elif select == "5":
                print("Exiting...")
                break
            else:
                print(f"'{select}' is not a valid option.")
                input("Press Enter to try again...")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break