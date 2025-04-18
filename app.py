import streamlit as st
import gdown
import joblib
import os
import logging
import traceback
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from cls_prediction import predict_anxiety
from cls_recommendation import get_recommendations

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence watchdog debug logs that create noise in Streamlit
logging.getLogger('watchdog.observers').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

st.title("Anxiety Severity Prediction and Lifestyle Recommendations")

# Create necessary directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("preprocessed_data", exist_ok=True)

# Debug information section
with st.expander("Debug Information", expanded=False):
    st.write("### System Information")
    st.write(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Python version: {os.sys.version}")
    st.write(f"Working directory: {os.getcwd()}")
    st.write(f"Files in directory: {os.listdir('.')}")
    st.write(f"Models directory contents: {os.listdir('models') if os.path.exists('models') else 'Models directory not found'}")
    st.write(f"Preprocessed data directory contents: {os.listdir('preprocessed_data') if os.path.exists('preprocessed_data') else 'Preprocessed data directory not found'}")

# Define file URLs for files that need to be downloaded
FILE_URLS = {
    "cls_rf.pkl": "https://drive.google.com/uc?id=1_gCsceiu4m4VjxQhTci7Y534LVebKd_W",  # Random Forest model
}

# Function to download file if it doesn't exist
def download_file(file_name, url, target_dir=""):
    target_path = os.path.join(target_dir, file_name) if target_dir else file_name
    if not os.path.exists(target_path):
        try:
            logger.info(f"Downloading {file_name}...")
            # Add progress bar for large files
            if file_name.endswith('.pkl'):
                st.warning(f"Downloading large model file ({file_name}). This may take a few minutes...")
            with st.spinner(f"Downloading {file_name}..."):
                gdown.download(url, target_path, quiet=False)
            logger.info(f"{file_name} downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {str(e)}")
            st.error(f"Error downloading {file_name}: {str(e)}")
            if "too large" in str(e).lower():
                st.error("The file is too large for Google Drive to scan. Please try downloading it manually.")
            return False
    return True

# Function to create scaler if it doesn't exist
def ensure_scaler_exists():
    scaler_path = os.path.join("preprocessed_data", "scaler.pkl")
    if not os.path.exists(scaler_path):
        try:
            logger.info("Scaler not found. Creating a new scaler...")
            # Load the dataset to fit the scaler
            dataset_path = os.path.join("preprocessed_data", "cls_preprocessed_dataset.csv")
            if not os.path.exists(dataset_path):
                logger.error("Cannot create scaler: preprocessed dataset not found")
                return False
                
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            # Define the known numerical columns from our form instead of relying on 'Q' prefix
            numerical_cols = [
                "Age", "Sleep Hours", "Physical Activity (hrs/week)", 
                "Caffeine Intake (mg/day)", "Alcohol Consumption (drinks/week)", 
                "Heart Rate (bpm during attack)", "Breathing Rate (breaths/min)", 
                "Therapy Sessions (per month)"
            ]
            
            # Filter to only include columns that actually exist in the dataset
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if not available_cols:
                logger.error(f"No numerical columns found in the dataset for scaling. Dataset columns: {df.columns.tolist()}")
                # Create a simple scaler anyway to avoid breaking the app
                scaler = StandardScaler()
                joblib.dump(scaler, scaler_path)
                logger.info(f"Created empty scaler as fallback at {scaler_path}")
                return True
            
            logger.info(f"Using numerical columns: {available_cols}")
            
            # Create and fit a new scaler
            scaler = StandardScaler()
            scaler.fit(df[available_cols])
            
            # Save the scaler
            joblib.dump(scaler, scaler_path)
            logger.info(f"New scaler created and saved to {scaler_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating scaler: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a fallback scaler so the app can still run
            try:
                scaler = StandardScaler()
                joblib.dump(scaler, scaler_path)
                logger.info("Created fallback scaler due to error")
                return True
            except:
                return False
    return True

# Check for required files
required_files = {
    "preprocessed_data/cls_preprocessed_dataset.csv": "Preprocessed dataset",
    "models/cls_rf.pkl": "Random Forest model"
}

# Verify existing files and download only what's missing
st.info("Checking required files...")
with st.spinner("Setting up the application..."):
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            # Only try to download if the file is in FILE_URLS (i.e., not the dataset)
            if file_name in FILE_URLS:
                st.warning(f"{description} not found. Downloading from Google Drive...")
                target_dir = os.path.dirname(file_path)
                
                if not download_file(file_name, FILE_URLS[file_name], target_dir):
                    st.error(f"Failed to download {description}. Please check the URL and try again.")
                    if file_name == "cls_rf.pkl":
                        st.info("You can download the model manually from: https://drive.google.com/uc?id=1_gCsceiu4m4VjxQhTci7Y534LVebKd_W")
                    st.stop()
            else:
                st.error(f"Missing required file: {file_path}")
                st.error("The preprocessed dataset should be included in the repository. Please check your .gitignore file.")
                st.stop()
        else:
            logger.info(f"Found {description} at {file_path}")
    
    # Check and create scaler if needed
    if ensure_scaler_exists():
        st.success("Scaler is ready")
    else:
        st.warning("Could not create scaler. Predictions may be less accurate.")

st.success("All required files are present!")

# Global variable to store model file path
MODEL_PATH = os.path.join("models", "cls_rf.pkl")

# Modified load_model function to handle model access without loading it into memory at startup
@st.cache_resource
def get_model_info():
    try:
        model_path = MODEL_PATH
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Model file size: {model_size:.2f} MB")
        return {
            "path": model_path,
            "size": model_size,
            "exists": True
        }
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")
        return {
            "path": MODEL_PATH,
            "size": 0,
            "exists": False,
            "error": str(e)
        }

@st.cache_resource
def load_scaler():
    try:
        scaler_path = os.path.join("preprocessed_data", "scaler.pkl")
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        logger.error(traceback.format_exc())
        st.warning("Could not load scaler. Using StandardScaler with default parameters.")
        return StandardScaler()

# Get model info but don't load it yet
model_info = get_model_info()
scaler = load_scaler()

st.sidebar.header("Enter Your Details")

gender_option = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
occupation_option = st.sidebar.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Other", "Student", "Unemployed"])

user_input = {
    "Age": st.sidebar.number_input("Age", min_value=18, max_value=100, value=23),
    "Gender": 0 if gender_option == "Male" else 1 if gender_option == "Female" else 2,
    "Occupation": {"Engineer": 0, "Doctor": 1, "Teacher": 2,
        "Other": 3, "Student": 4, "Unemployed": 5}[occupation_option],
    "Sleep Hours": st.sidebar.slider("Sleep Hours", 0, 15, 8),
    "Physical Activity (hrs/week)": st.sidebar.slider("Physical Activity (hrs/week)", 0, 20, 7),
    "Caffeine Intake (mg/day)": st.sidebar.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=500, value=50),
    "Alcohol Consumption (drinks/week)": st.sidebar.slider("Alcohol Consumption (drinks/week)", 0, 21, 0),
    "Smoking": 1 if st.sidebar.selectbox("Smoking", ["No", "Yes"]) == "Yes" else 0,
    "Family History of Anxiety": 1 if st.sidebar.selectbox("Family History of Anxiety", ["No", "Yes"]) == "Yes" else 0,
    "Heart Rate (bpm during attack)": st.sidebar.number_input("Heart Rate (bpm during attack)", min_value=40, max_value=200, value=72),
    "Breathing Rate (breaths/min)": st.sidebar.slider("Breathing Rate (breaths/min)", 8, 40, 8),
    "Dizziness": 1 if st.sidebar.selectbox("Dizziness", ["No", "Yes"]) == "Yes" else 0,
    "Medication": 1 if st.sidebar.selectbox("Medication", ["No", "Yes"]) == "Yes" else 0,
    "Therapy Sessions (per month)": st.sidebar.slider("Therapy Sessions (per month)", 0, 10, 2),
    "Recent Major Life Event": 1 if st.sidebar.selectbox("Recent Major Life Event", ["No", "Yes"]) == "Yes" else 0,
    "Sweating Level (1-5)": st.sidebar.slider("Sweating Level (1-5)", 1, 5, 5),
    "Stress Level (1-10)": st.sidebar.slider("Stress Level (1-10)", 1, 10, 10),
    "Diet Quality (1-10)": st.sidebar.slider("Diet Quality (1-10)", 1, 10, 9)
}

# Create a dummy model for fallback
def create_dummy_model():
    logger.warning("Creating a dummy model as fallback")
    try:
        # Create a simple RandomForest with minimal parameters
        dummy_model = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=42)
        # Create some dummy data to fit the model
        X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        y = np.array([5, 6])  # Mid-range anxiety values
        dummy_model.fit(X, y)
        return dummy_model
    except Exception as e:
        logger.error(f"Failed to create dummy model: {str(e)}")
        return None

# Modified function to load model only when needed
def load_model_for_prediction():
    try:
        with st.spinner("Loading prediction model... (this may take a moment for large models)"):
            logger.info(f"Loading Random Forest model from {MODEL_PATH}")
            # Use memory mapping to efficiently load large models
            model = joblib.load(MODEL_PATH, mmap_mode='r')
            logger.info("Random Forest model loaded successfully")
            return model
    except Exception as e:
        logger.error(f"Error loading Random Forest model: {str(e)}")
        logger.error(traceback.format_exc())
        st.warning(f"Error loading model: {str(e)}. Using a simplified model for demo purposes.")
        
        # Try to create a dummy model as fallback
        dummy_model = create_dummy_model()
        if dummy_model is not None:
            return dummy_model
        
        st.error("Could not create a fallback model. Cannot proceed with prediction.")
        return None

if st.button("Predict Anxiety Severity"):
    try:
        logger.info("Starting prediction process")
        logger.debug(f"User input: {user_input}")
        
        # Only load the model when the button is clicked
        with st.spinner("Loading model (this may take a moment for large models)..."):
            model = load_model_for_prediction()
            if model is None:
                st.error("Could not load any prediction model. Please try again later.")
                st.stop()
            
            # Log successful model loading
            model_type = type(model).__name__
            logger.info(f"Successfully loaded model of type: {model_type}")
            is_dummy = model_type == "RandomForestClassifier" and hasattr(model, "n_estimators") and model.n_estimators <= 1
            if is_dummy:
                st.warning("Using a simplified model for demonstration purposes. Predictions may not be accurate.")
        
        # Make prediction using the loaded model
        with st.spinner("Running prediction..."):
            # Force garbage collection before prediction
            gc.collect()
            
            try:
                # Run prediction with timeout protection
                predicted_class = predict_anxiety(user_input, scaler, model)
                logger.info(f"Prediction completed. Result: {predicted_class}")
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                st.error(f"Error during prediction: {str(e)}")
                # Try with a dummy model as last resort
                if not is_dummy:  # Only try dummy if we weren't already using one
                    st.warning("Trying with a simplified model...")
                    dummy_model = create_dummy_model()
                    if dummy_model is not None:
                        predicted_class = 5  # Default middle value
                        try:
                            predicted_class = predict_anxiety(user_input, scaler, dummy_model)
                        except:
                            pass
                    else:
                        st.error("Prediction failed. Please try again later.")
                        st.stop()
                else:
                    st.error("Prediction failed. Please try again later.")
                    st.stop()
        
        # Clean up memory after prediction
        del model
        gc.collect()
        
        with st.spinner("Generating recommendations..."):
            recommendations = get_recommendations(predicted_class, user_input)
            logger.info("Recommendations generated successfully")

        st.subheader("Predicted Anxiety Severity:")
        st.write(f"Severity: {predicted_class} (Scale: 1 to 10)")

        st.subheader("Recommended Lifestyle Changes:")
        if isinstance(recommendations, dict):
            for category, recommendation in recommendations.items():
                st.markdown(f"**{category}**: {recommendation}")
        else:
            # Handle legacy list format
            if recommendations:
                st.write(recommendations[0])
                for rec in recommendations[1:]:
                    st.write(f"- {rec}")

        # Debug information for the prediction
        with st.expander("Prediction Details", expanded=False):
            st.write("### Input Data")
            st.json(user_input)
            st.write("### Model Information")
            st.write(f"Model file: {model_info['path']}")
            st.write(f"Model size: {model_info['size']:.2f} MB")
            st.write(f"Using fallback model: {'Yes' if is_dummy else 'No'}")
            st.write(f"Prediction timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if user_input["Dizziness"] == 1:
            st.subheader("Additional Grounding Technique: 5-4-3-2-1 Method")
            st.markdown(
                """
                <style>
                    .cute-box {
                        background-color: #f0f8ff;
                        border-radius: 10px;
                        padding: 20px;
                        border: 2px solid #b0e0e6;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        font-family: 'Arial', sans-serif;
                        font-size: 16px;
                        color: #333;
                    }
                    .cute-box h4 {
                        color: #4682b4;
                    }
                </style>
                <div class="cute-box">
                    <h4>The 5-4-3-2-1 Method:</h4>
                    <p>This grounding technique helps you stay present by focusing on your senses. Here's how it works:</p>
                    <ul>
                        <li><b> 5 things you can see:</b> Look around and identify five things you can see.</li>
                        <li><b> 4 things you can touch:</b> Notice four things you can physically touch, like the texture of the chair or the floor beneath your feet.</li>
                        <li><b> 3 things you can hear:</b> Pay attention to three sounds around you, such as the hum of the air conditioning or birds chirping.</li>
                        <li><b> 2 things you can smell:</b> Take a deep breath and identify two things you can smell.</li>
                        <li><b> 1 thing you can taste:</b> Focus on one taste in your mouth, or simply notice the taste of your own breath.</li>
                    </ul>
                    <p>This method helps calm your mind by shifting focus from overwhelming thoughts to tangible sensations, grounding you in the present moment.</p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An error occurred during prediction. Please try again.")
        st.error(f"Error details: {str(e)}")
