import streamlit as st
import gdown
import joblib
import os
import logging
import traceback
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

st.title("Anxiety Severity Prediction and Lifestyle Recommendations")

# Debug information section
with st.expander("Debug Information", expanded=False):
    st.write("### System Information")
    st.write(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Python version: {os.sys.version}")
    st.write(f"Working directory: {os.getcwd()}")
    st.write(f"Files in directory: {os.listdir('.')}")
    st.write(f"Models directory contents: {os.listdir('models') if os.path.exists('models') else 'Models directory not found'}")
    st.write(f"Preprocessed data directory contents: {os.listdir('preprocessed_data') if os.path.exists('preprocessed_data') else 'Preprocessed data directory not found'}")

# Check for model file in both root and models directory
model_paths = [
    "cls_rf.pkl",  # Root directory
    os.path.join("models", "cls_rf.pkl")  # Models directory
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        logger.info(f"Found model at: {path}")
        break

if model_path is None:
    # If model not found, create models directory and download there
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "cls_rf.pkl")
    try:
        logger.info(f"Model file not found. Downloading to {model_path}...")
        url = "https://drive.google.com/uc?id=1_gCsceiu4m4VjxQhTci7Y534LVebKd_W"
        gdown.download(url, model_path, quiet=False)
        logger.info("Random Forest model downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading Random Forest model: {str(e)}")
        st.error(f"Error downloading Random Forest model: {str(e)}")
        st.stop()

# Check for required data files
required_files = {
    "preprocessed_data/cls_preprocessed_dataset.csv": "Preprocessed dataset",
    "preprocessed_data/scaler.pkl": "Scaler file"
}

missing_files = []
for file_path, description in required_files.items():
    if not os.path.exists(file_path):
        missing_files.append(f"{description} ({file_path})")
        logger.error(f"Missing required file: {file_path}")

if missing_files:
    st.error("The following required files are missing:")
    for file in missing_files:
        st.error(f"- {file}")
    st.error("Please ensure all required files are present in the correct locations.")
    st.stop()

@st.cache_resource
def load_model():
    try:
        logger.info(f"Loading Random Forest model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Random Forest model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Random Forest model: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error loading Random Forest model: {str(e)}")
        st.stop()

model = load_model()

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

if st.button("Predict Anxiety Severity"):
    try:
        logger.info("Starting prediction process")
        logger.debug(f"User input: {user_input}")
        
        predicted_class = predict_anxiety(user_input)
        logger.info(f"Prediction completed. Result: {predicted_class}")
        
        recommendations = get_recommendations(predicted_class, user_input)
        logger.info("Recommendations generated successfully")

        st.subheader("Predicted Anxiety Severity:")
        st.write(f"Severity: {predicted_class} (Scale: 1 to 10)")

        st.subheader("Recommended Lifestyle Changes:")
        if recommendations:
            st.write(recommendations[0])
            for rec in recommendations[1:]:
                st.write(f"- {rec}")

        # Debug information for the prediction
        with st.expander("Prediction Details", expanded=False):
            st.write("### Input Data")
            st.json(user_input)
            st.write("### Model Information")
            st.write(f"Model type: {type(model).__name__}")
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
