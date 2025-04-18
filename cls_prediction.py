import pandas as pd
import joblib
from pathlib import Path
import traceback
import sys

def predict_anxiety(user_input):
    try:
        # Get the base directory
        BASE_DIR = Path(__file__).parent
        
        # Define file paths
        dataset_path = BASE_DIR / "cls_preprocessed_dataset.csv"
        model_path = BASE_DIR / "cls_rf.pkl"
        
        # Debug information
        print(f"Python path: {sys.path}")
        print(f"Base directory: {BASE_DIR}")
        print(f"Dataset path: {dataset_path}")
        print(f"Model path: {model_path}")
        print(f"Directory contents: {list(BASE_DIR.glob('*'))}")
        
        # Check if required files exist
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load and process dataset
        try:
            df = pd.read_csv(dataset_path)
            X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])  # Feature columns only
            correct_column_order = X.columns.tolist()
        except Exception as e:
            raise Exception(f"Error processing dataset: {str(e)}")

        # Prepare input data
        try:
            input_df = pd.DataFrame([user_input])
            input_df = input_df[correct_column_order]
        except Exception as e:
            raise Exception(f"Error preparing input data: {str(e)}")

        # Define columns for scaling
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

        # Process input data
        try:
            X_to_scale = input_df[scale_cols]
            X_not_scaled = input_df[non_scale_cols]
            X_scaled = X_to_scale.copy()  # Using original values since scaler is not available
            final_input = pd.concat([X_scaled, X_not_scaled], axis=1)
            final_input = final_input[correct_column_order]
        except Exception as e:
            raise Exception(f"Error processing input features: {str(e)}")

        # Make prediction
        try:
            model = joblib.load(model_path)
            prediction = model.predict(final_input)[0]
            return int(prediction)
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  # Print to console for debugging
        raise Exception(error_msg)
