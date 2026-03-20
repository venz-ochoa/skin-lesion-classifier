#exploratory data analysis
#distribution and information about HAM10000
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

metadata = '/content/skin-lesion-classifier/data/ham10000/HAM10000_metadata.csv'
df = pd.read_csv(metadata)

labels = {
    'nv': 'Nevus',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}
df['cell_type'] = df['dx'].map(labels)

#summary
print("---- HAM10000 Summary ----")
print(f"Total Images: {len(df)}")
print("\nClass Counts:")
print(df['cell_type'].value_counts())
print("\nTop 5 Localization Sites:")
print(df['localization'].value_counts().head(5))
print("------------------------\n")

# class distribution
plt.figure(figsize=(10,6))
sns.countplot(y='cell_type', data=df, order=df['cell_type'].value_counts().index, hue='cell_type', palette='viridis', legend=False)
plt.title('Distribution of Skin Lesion Types')
#so image doesnt clip/overlap with other elements and borders
plt.tight_layout()
plt.savefig('/content/skin-lesion-classifier/EDA/class_distribution.png')
plt.show()

# age distribution
plt.figure(figsize=(10,6))
sns.histplot(df['age'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Patients')
plt.tight_layout()
plt.savefig('/content/skin-lesion-classifier/EDA/age_distribution.png')
plt.show()

# skin lesion localization distribution
plt.figure(figsize=(10,6))
df['localization'].value_counts().plot(kind='barh', color='salmon')
plt.title('Lesion Localization on Body')
plt.tight_layout()
plt.savefig('/content/skin-lesion-classifier/EDA/localization.png')
plt.show()