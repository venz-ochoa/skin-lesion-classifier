import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

metadata = '/content/skin-lesion-classifier/data/ham10000/HAM10000_metadata.csv'
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
source_folder = '/content/skin-lesion-classifier/data/ham10000'
base_out = '/content/skin-lesion-classifier/data/model_data/'

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