import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

#HAM10000
print("\nHAM10000 Benign and Malignant class counts:")
metadata = 'data/src_data/ham10000/HAM10000_metadata.csv'
df = pd.read_csv(metadata)

#this is the malignant classes
malignant = ['mel', 'bcc', 'akiec']

#if 1 its malignant, if 0 its benign
def malignantChecker(diagnosis):
  if diagnosis in malignant:
    return 1
  else:
    return 0
df['binary_label'] = df['dx'].apply(malignantChecker)

#80% training and 20%(10 val, 10 test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['binary_label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['binary_label'])

#this is to check the class imbalance / count
print("Benign is 0, Malignant is 1")
print(df['binary_label'].value_counts())

#images
source_folder = 'data/src_data/ham10000'
base_out = 'data/model_data/'
os.makedirs(base_out, exist_ok=True)

def sort(df, split_name):
    for index, row in df.iterrows():
        img_name = row['image_id'] + '.jpg'
        if row['binary_label'] == 1:
          folder_name = 'malignant' 
        else: 
          folder_name = 'benign' 
        
        #folder path
        dest_folder = os.path.join(base_out, split_name, folder_name)
        #make if it doesnt exist
        os.makedirs(dest_folder, exist_ok=True)
        
        #check the two folders where kaggle hides the images
        part1 = os.path.join(source_folder, 'HAM10000_images_part_1', img_name)
        part2 = os.path.join(source_folder, 'HAM10000_images_part_2', img_name)
        root_dir = os.path.join(source_folder, img_name)
        
        #figure out where the file actually is, since ham has 2 parts
        if os.path.exists(part1):
            source = part1
        elif os.path.exists(part2):
            source = part2
        else:
            source = root_dir
            
        destination = os.path.join(dest_folder, img_name)
        
        #create a copy and move it to the train folder
        if os.path.exists(source):
            shutil.copy(source, destination)
        else:
            print(f"Warning: Could not find {img_name} anywhere.")

#process
print("\nSorting train data...")
sort(train_df, 'train')
print("Sorting validation data...")
sort(val_df, 'val')
print("Sorting test data...")
sort(test_df, 'test')

#check the size of each set
print(f"\nTrain size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# -----------------------------------------------

#ISIC 2019
#isic uses one-hot encoding, each class is its own column (MEL, BCC, AK are malignant)
isic_metadata = 'data/src_data/isic2019/ISIC_2019_Training_GroundTruth.csv'
isic_df = pd.read_csv(isic_metadata)

#this is the malignant classes for isic 2019
isic_malignant = ['MEL', 'BCC', 'AK', 'SCC']

#if any malignant column is 1, its malignant
isic_df['binary_label'] = isic_df[isic_malignant].max(axis=1)

#only keep malignant rows
isic_malignant_df = isic_df[isic_df['binary_label'] == 1]

#80% training and 20%(10 val, 10 test)
isic_train_df, isic_temp_df = train_test_split(isic_malignant_df, test_size=0.2, random_state=42)
isic_val_df, isic_test_df = train_test_split(isic_temp_df, test_size=0.50, random_state=42)

#this is to check the class imbalance / count
print("\nISIC 2019 Malignant class counts:")
print(isic_malignant_df[isic_malignant].sum())

#images are in a single folder unlike ham10000
isic_source_folder = 'data/src_data/isic2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

def sort_isic(df, split_name):
    for index, row in df.iterrows():
        img_name = row['image'] + '.jpg'
        
        #folder path
        dest_folder = os.path.join(base_out, split_name, 'malignant')
        #make if it doesnt exist
        os.makedirs(dest_folder, exist_ok=True)
        
        #isic images are all in one folder
        source = os.path.join(isic_source_folder, img_name)
        destination = os.path.join(dest_folder, img_name)
        
        #create a copy and move it to the train folder
        if os.path.exists(source):
            shutil.copy(source, destination)
        else:
            print(f"Warning: Could not find {img_name} anywhere.")

#process
print("\nSorting ISIC 2019 train data...")
sort_isic(isic_train_df, 'train')
print("Sorting ISIC 2019 validation data...")
sort_isic(isic_val_df, 'val')
print("Sorting ISIC 2019 test data...")
sort_isic(isic_test_df, 'test')

#check the size of each set
print(f"\nISIC Train size: {len(isic_train_df)}")
print(f"ISIC Validation size: {len(isic_val_df)}")
print(f"ISIC Test size: {len(isic_test_df)}\n")