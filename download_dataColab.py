#python script to automate data downloads
#use a venv for this to run on macOS
import subprocess, sys, os
from google.colab import drive
drive.mount('/content/drive')

root_drive = "/content/drive/MyDrive/skin-lesion-intsys"
ham_path = "/content/drive/MyDrive/skin-lesion-intsys/data/ham10000"

#to install dependencies
def dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Error : {e}")
    
#to install datasets (HAM10000)
def datasets():
    #create folders for dataset
    #exist_ok = True will ignore the error if the directory already exists, if not itll create a directory
    if os.path.exists(ham_path) and os.listdir(ham_path):
        print("HAM10000 is already downloaded. No downloads are necessary.")
    else:
        try:
            print("\nDownloading HAM10000 Dataset... ")
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "kmader/skin-cancer-mnist-ham10000",
                "-p", ham_path,
                "--unzip"
            ], check = True)
            print("Downloaded HAM10000 Successfully!")
        except Exception as e:
            print(f"Error : {e}")
        #check=true is for python to stop the downloads the moment theres an error
    
if __name__ == "__main__":
    #terminal. allows flexibility for both members and users
    while True:
        print("\033c", end="")
        print("Skin Lesion Project Terminal\n\t[1] Download Dependencies\n\t[2] Download Datasets\n\t[3] Download Dependencies & Datasets")
        select = int(input("\nEnter: "))
        match select:
            case 1:
                dependencies()
                exit()
            case 2:
                datasets()
                exit()
            case 3:
                dependencies()
                datasets()
                exit()
            case _:
                print(f"{select} is invalid. Select 1 or 2.")
                continue