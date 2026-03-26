#python script to automate data downloads
#use a venv for this to run on macOS
import subprocess, sys, os, zipfile

#to install dependencies
def dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Error : {e}")
    
#to install dataset
def datasets():
    os.makedirs("data/ham10000", exist_ok=True)
    os.makedirs("data/isic2019", exist_ok=True)
    
    try:
        # HAM10000
        print("\nDownloading HAM10000 Dataset...")
        if not os.listdir("data/ham10000"):
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "kmader/skin-cancer-mnist-ham10000",
                "-p", "data/ham10000",
                "--unzip"
            ], check=True)
        print("Downloaded HAM10000 Successfully!")
    except Exception as e:
        print(f"Error: {e}")
    try:
        #ISIC 2019
        print("\nDownloading ISIC 2019 Dataset...")
        if not os.listdir("data/isic2019"):
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "andrewmvd/isic-2019",
                "-p", "data/isic2019",
                "--unzip"
            ], check=True)
        print("Downloaded ISIC 2019 Successfully!")
    except Exception as e:
        print(f"Error: {e}")
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