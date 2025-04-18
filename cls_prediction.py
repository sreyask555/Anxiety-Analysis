import pandas as pd
import joblib
from pathlib import Path

def predict_anxiety(user_input):
    try:
        # Get the base directory
        BASE_DIR = Path(__file__).parent
        
        # Define file paths
        dataset_path = BASE_DIR / "cls_preprocessed_dataset.csv"
        model_path = BASE_DIR / "cls_rf.pkl"
        
        # Check if required files exist
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        df = pd.read_csv(dataset_path)
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

        # Since we don't have the scaler file, we'll use the original values
        # This is a temporary solution until we can get the scaler file
        X_scaled = X_to_scale.copy()

        final_input = pd.concat([X_scaled, X_not_scaled], axis=1)
        final_input = final_input[correct_column_order]  # Ensure exact order

        model = joblib.load(model_path)
        prediction = model.predict(final_input)[0]

        return int(prediction)
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")
