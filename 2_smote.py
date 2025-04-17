import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

df = pd.read_csv("preprocessed_data/categorical_dataset.csv")

X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])
y = df["Severity of Anxiety Attack (1-10)"]

target_counts = y.value_counts()
desired_count = 3000
sampling_strategy = {label: desired_count for label in target_counts.index}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df["Severity of Anxiety Attack (1-10)"] = y_resampled

output_path = "preprocessed_data/smote_balanced_dataset.csv"
resampled_df.to_csv(output_path, index=False)

print("Step 2 Completed: SMOTE applied and data saved to 'preprocessed_data/smote_balanced_dataset.csv'")
print("New class distribution:\n", Counter(y_resampled))
