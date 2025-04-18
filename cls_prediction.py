import pandas as pd
import joblib

def predict_anxiety(user_input):
    df = pd.read_csv("preprocessed_data/cls_preprocessed_dataset.csv")
    X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])  # Feature columns only
    correct_column_order = X.columns.tolist()

    input_df = pd.DataFrame([user_input])
    input_df = input_df[correct_column_order]

    scale_cols = [
        "Age", "Sleep Hours", "Physical Activity (hrs/week)", "Caffeine Intake (mg/day)",
        "Alcohol Consumption (drinks/week)", "Heart Rate (bpm during attack)",
        "Breathing Rate (breaths/min)", "Therapy Sessions (per month)"
    ]
    
    non_scale_cols = [
        "Sweating Level (1-5)", "Stress Level (1-10)", "Diet Quality (1-10)",
        "Gender", "Occupation", "Smoking", "Family History of Anxiety", 
        "Dizziness", "Medication", "Recent Major Life Event"
    ]

    X_to_scale = input_df[scale_cols]
    X_not_scaled = input_df[non_scale_cols]

    scaler = joblib.load("preprocessed_data/scaler.pkl")
    X_scaled = pd.DataFrame(scaler.transform(X_to_scale), columns=scale_cols)

    final_input = pd.concat([X_scaled, X_not_scaled], axis=1)
    final_input = final_input[correct_column_order]  # Ensure exact order

    model = joblib.load("models/rf_classifier_model.pkl")
    prediction = model.predict(final_input)[0]

    return int(prediction)
