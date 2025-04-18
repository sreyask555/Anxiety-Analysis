import pandas as pd
import joblib

def predict_anxiety(user_input, scaler=None, model=None):
    try:
        df = pd.read_csv("preprocessed_data/cls_preprocessed_dataset.csv")
        X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])  # Feature columns only
        correct_column_order = X.columns.tolist()
    except Exception as e:
        # If we can't read the dataset or the column doesn't exist, fallback to just using the input columns
        print(f"Warning: Error reading dataset: {str(e)}")
        correct_column_order = list(user_input.keys())

    input_df = pd.DataFrame([user_input])
    
    # Ensure all required columns exist
    for col in correct_column_order:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value
    
    # Use only columns that exist in the dataset
    existing_columns = [col for col in correct_column_order if col in input_df.columns]
    input_df = input_df[existing_columns]

    # Define known numerical columns for scaling
    scale_cols = [
        "Age", "Sleep Hours", "Physical Activity (hrs/week)", 
        "Caffeine Intake (mg/day)", "Alcohol Consumption (drinks/week)", 
        "Heart Rate (bpm during attack)", "Breathing Rate (breaths/min)", 
        "Therapy Sessions (per month)"
    ]
    
    # Only use scale_cols that actually exist in the input data
    scale_cols = [col for col in scale_cols if col in input_df.columns]
    
    # All other columns are not scaled
    non_scale_cols = [col for col in input_df.columns if col not in scale_cols]

    if scale_cols:
        X_to_scale = input_df[scale_cols]
        
        # Use provided scaler or load it if not provided
        if scaler is None:
            try:
                scaler = joblib.load("preprocessed_data/scaler.pkl")
            except Exception as e:
                # If loading fails, create a new scaler (will have default params)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
        
        try:
            X_scaled = pd.DataFrame(scaler.transform(X_to_scale), columns=scale_cols)
            
            if non_scale_cols:
                X_not_scaled = input_df[non_scale_cols]
                final_input = pd.concat([X_scaled, X_not_scaled], axis=1)
            else:
                final_input = X_scaled
        except Exception as e:
            print(f"Warning: Scaling error: {str(e)}. Using unscaled data.")
            final_input = input_df
    else:
        # If no scale columns, use unscaled input
        final_input = input_df
    
    # Ensure the columns are in the correct order for the model
    final_input = final_input[existing_columns]  # Use only existing columns

    # If model is not provided, try to load it
    if model is None:
        try:
            # Use memory mapping to efficiently load large models
            model = joblib.load("models/cls_rf.pkl", mmap_mode='r')
        except:
            # Try the older filename if the new one fails
            try:
                model = joblib.load("models/rf_classifier_model.pkl", mmap_mode='r')
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return 5  # Return a middle value as fallback
        
    try:
        prediction = model.predict(final_input)[0]
        return int(prediction)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 5  # Return a middle value as fallback
