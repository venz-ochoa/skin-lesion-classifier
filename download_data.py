#python script to automate data downloads
#will be added in future

#use a venv for this to run on macOS
#to install dependencies
import subprocess
import sys
def dependencies():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Error : {e}")
        
if __name__ == "__main__":
    dependencies()