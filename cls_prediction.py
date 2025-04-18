import pandas as pd
import joblib
import logging

# Configure basic logging if not already configured
try:
    logger = logging.getLogger(__name__)
except:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def predict_anxiety(user_input, scaler=None, model=None):
    # Log the received input
    logger.info(f"predict_anxiety called with input: {user_input}")
    logger.info(f"Using provided model: {model is not None}")
    logger.info(f"Using provided scaler: {scaler is not None}")
    
    try:
        df = pd.read_csv("preprocessed_data/cls_preprocessed_dataset.csv")
        X = df.drop(columns=["Severity of Anxiety Attack (1-10)"])  # Feature columns only
        correct_column_order = X.columns.tolist()
        logger.info(f"Dataset columns loaded: {len(correct_column_order)} columns")
    except Exception as e:
        # If we can't read the dataset or the column doesn't exist, fallback to just using the input columns
        logger.warning(f"Error reading dataset: {str(e)}")
        correct_column_order = list(user_input.keys())
        logger.info(f"Using input columns as fallback: {correct_column_order}")

    input_df = pd.DataFrame([user_input])
    
    # Ensure all required columns exist
    for col in correct_column_order:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value
            logger.info(f"Added missing column with default value: {col}")
    
    # Use only columns that exist in the dataset
    existing_columns = [col for col in correct_column_order if col in input_df.columns]
    input_df = input_df[existing_columns]
    logger.info(f"Using columns for prediction: {existing_columns}")

    # Define known numerical columns for scaling
    scale_cols = [
        "Age", "Sleep Hours", "Physical Activity (hrs/week)", 
        "Caffeine Intake (mg/day)", "Alcohol Consumption (drinks/week)", 
        "Heart Rate (bpm during attack)", "Breathing Rate (breaths/min)", 
        "Therapy Sessions (per month)"
    ]
    
    # Only use scale_cols that actually exist in the input data
    scale_cols = [col for col in scale_cols if col in input_df.columns]
    logger.info(f"Columns to scale: {scale_cols}")
    
    # All other columns are not scaled
    non_scale_cols = [col for col in input_df.columns if col not in scale_cols]
    logger.info(f"Columns not to scale: {non_scale_cols}")

    if scale_cols:
        X_to_scale = input_df[scale_cols]
        
        # Use provided scaler or load it if not provided
        if scaler is None:
            try:
                scaler = joblib.load("preprocessed_data/scaler.pkl")
                logger.info("Loaded scaler from file")
            except Exception as e:
                # If loading fails, create a new scaler (will have default params)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                logger.warning(f"Created new scaler due to error: {str(e)}")
        
        try:
            X_scaled = pd.DataFrame(scaler.transform(X_to_scale), columns=scale_cols)
            logger.info("Successfully scaled input data")
            
            if non_scale_cols:
                X_not_scaled = input_df[non_scale_cols]
                final_input = pd.concat([X_scaled, X_not_scaled], axis=1)
            else:
                final_input = X_scaled
        except Exception as e:
            logger.error(f"Scaling error: {str(e)}. Using unscaled data.")
            final_input = input_df
    else:
        # If no scale columns, use unscaled input
        logger.warning("No columns to scale, using unscaled input")
        final_input = input_df
    
    # Ensure the columns are in the correct order for the model
    final_input = final_input[existing_columns]  # Use only existing columns
    logger.info(f"Final input shape: {final_input.shape}")
    logger.info(f"First few values: {final_input.iloc[0].head(5).to_dict()}")

    # If model is not provided, try to load it
    if model is None:
        try:
            # Load without memory mapping to ensure accurate predictions
            model = joblib.load("models/cls_rf.pkl")
            logger.info("Successfully loaded model from file")
        except:
            # Try the older filename if the new one fails
            try:
                model = joblib.load("models/rf_classifier_model.pkl")
                logger.info("Successfully loaded model from alternate file")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                
                # If all else fails, create a simple prediction based on known risk factors
                logger.warning("Creating fallback prediction based on input values")
                stress = user_input.get("Stress Level (1-10)", 5)
                heart_rate = user_input.get("Heart Rate (bpm during attack)", 70)
                sleep = user_input.get("Sleep Hours", 7)
                recent_event = user_input.get("Recent Major Life Event", 0)
                
                # Higher stress, higher heart rate, less sleep and recent events increase anxiety
                base_score = (stress/2) + ((heart_rate-60)/20) + (1 if recent_event else 0) + max(0, (7-sleep)/2)
                predicted_class = max(1, min(10, round(base_score)))
                logger.info(f"Generated fallback prediction: {predicted_class}")
                return predicted_class
        
    try:
        logger.info(f"Making prediction with model type: {type(model).__name__}")
        prediction = model.predict(final_input)[0]
        predicted_class = int(prediction)
        logger.info(f"Raw prediction result: {predicted_class}")
        
        # Ensure prediction is in valid range
        if predicted_class < 1:
            logger.warning(f"Prediction {predicted_class} below valid range, adjusting to 1")
            predicted_class = 1
        elif predicted_class > 10:
            logger.warning(f"Prediction {predicted_class} above valid range, adjusting to 10")
            predicted_class = 10
            
        return predicted_class
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        
        # Create fallback prediction based on key anxiety factors
        stress = user_input.get("Stress Level (1-10)", 5)
        heart_rate = user_input.get("Heart Rate (bpm during attack)", 70)
        sleep = user_input.get("Sleep Hours", 7)
        recent_event = user_input.get("Recent Major Life Event", 0)
        
        # Higher stress, higher heart rate, less sleep and recent events increase anxiety
        base_score = (stress/2) + ((heart_rate-60)/20) + (1 if recent_event else 0) + max(0, (7-sleep)/2)
        predicted_class = max(1, min(10, round(base_score)))
        logger.info(f"Generated fallback prediction after error: {predicted_class}")
        return predicted_class
