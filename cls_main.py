from cls_prediction import predict_anxiety
from cls_recommendation import get_recommendations

user_input = {
    "Age": 35,
    "Sleep Hours": 3,
    "Physical Activity (hrs/week)": 0,
    "Caffeine Intake (mg/day)": 160,
    "Alcohol Consumption (drinks/week)": 0,
    "Smoking": 0,
    "Family History of Anxiety": 0,
    "Heart Rate (bpm during attack)": 110,
    "Breathing Rate (breaths/min)": 25,
    "Dizziness": 1,
    "Medication": 0,
    "Therapy Sessions (per month)": 0,
    "Recent Major Life Event": 0,
    "Sweating Level (1-5)": 2,
    "Stress Level (1-10)": 10,
    "Diet Quality (1-10)": 2,
    "Gender": 0,       # 0 = Male
    "Occupation": 0    # 0 = Engineer
}

anxiety_severity = predict_anxiety(user_input)
print(f"\nPredicted Anxiety Severity: {anxiety_severity}")

recommendations = get_recommendations(anxiety_severity, user_input)

print("\n--- Personalized Recommendations ---")
for line in recommendations:
    print(line)

