import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv("D:/Project folder/anxiety_attack_dataset.csv")  

categorical_cols = ['Gender', 'Occupation', 'Medication', 'Smoking', 'Family History of Anxiety',
                    'Dizziness', 'Recent Major Life Event']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

output_dir = "preprocessed_data"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/categorical_dataset.csv", index=False)

print("Step 1 Completed: Categorical data encoded and saved to 'preprocessed_data/categorical_dataset.csv'")
