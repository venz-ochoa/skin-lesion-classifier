#python script to automate data downloads
#use a venv for this to run on macOS
import subprocess, sys, os, zipfile

#to install dependencies
def dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Error : {e}")
    
#to install datasets
def datasets():
    #create folders for datasets
    os.makedirs("data/ham10000", exist_ok = True)
    os.makedirs("data/isic_2024", exist_ok = True)
    
    try:
        #HAM10000
        print("\nDownloading HAM10000 Dataset... ")
        if not os.listdir("data/ham10000"):
            subprocess.run([
                "kaggle", "datasets", "download",
                "-d", "kmader/skin-cancer-mnist-ham10000",
                "-p", "data/ham10000",
                "--unzip"
            ], check = True)
        print("Downloaded HAM10000 Successfully!")
        
        #ISIC 2024
        print("\nDownloading ISIC 2024 Dataset... ")
        if not os.listdir("data/isic_2024"):
            subprocess.run([
                "kaggle", "competitions", "download",
                "-c", "isic-2024-challenge",
                "-p", "data/isic_2024",
            ], check = True)
            #automatically unzips the file
            with zipfile.ZipFile("data/isic_2024/isic-2024-challenge.zip", "r") as z:
                z.extractall("data/isic_2024")
            os.remove("data/isic_2024/isic-2024-challenge.zip")
        print("Downloaded ISIC 2024 Successfully!")
        
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