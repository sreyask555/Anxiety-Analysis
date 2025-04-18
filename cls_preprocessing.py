import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

df = pd.read_csv("preprocessed_data/smote_balanced_dataset.csv")

exclude_cols = ["Sweating Level (1-5)", "Stress Level (1-10)", "Diet Quality (1-10)", "Severity of Anxiety Attack (1-10)", "ID", "Gender", 
                "Occupation", "Smoking","Family History of Anxiety","Dizziness","Medication","Recent Major Life Event" ]

scale_cols = [col for col in df.columns if col not in exclude_cols]

X_to_scale = df[scale_cols]
X_not_scaled = df[["Sweating Level (1-5)", "Stress Level (1-10)", "Diet Quality (1-10)", "Gender",
                    "Occupation", "Smoking","Family History of Anxiety","Dizziness","Medication","Recent Major Life Event"]]
y = df["Severity of Anxiety Attack (1-10)"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_to_scale), columns=scale_cols)

final_df = pd.concat([
    X_scaled.reset_index(drop=True),
    X_not_scaled.reset_index(drop=True),
    y.reset_index(drop=True)], axis=1)

final_df.to_csv("preprocessed_data/cls_preprocessed_dataset.csv", index=False)

joblib.dump(final_df.drop(columns=["Severity of Anxiety Attack (1-10)"]).columns.tolist(), "preprocessed_data/columns.pkl")

with open("D:/Project folder/preprocessed_data/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Preprocessing completed with Gender and Occupation encoded, and class labels as 1–10.")
