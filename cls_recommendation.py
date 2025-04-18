def get_recommendations(severity, user_input):
    recommendations = []

    if severity <= 3:
        severity_msg = "Your anxiety severity is low. This suggests your lifestyle might already be relatively healthy. Keep it up!"
    elif 4 <= severity <= 6:
        severity_msg = "Your anxiety severity is moderate. Consider making positive lifestyle changes to improve your mental health."
    else:
        severity_msg = "Your anxiety severity is high. Immediate lifestyle changes are recommended for your well-being."

    recommendations.append(severity_msg)

    custom_msg = "Here are some personalized suggestions to help manage your anxiety:"

    if user_input["Alcohol Consumption (drinks/week)"] > 10:
        recommendations.append("Reduce alcohol consumption—it can increase anxiety levels.")

    if user_input["Physical Activity (hrs/week)"] == 0:
        recommendations.append("Include physical activity in your routine. Even light exercise helps reduce anxiety.")

    if user_input["Smoking"] == 1:
        recommendations.append("Try to quit smoking. Consider nicotine alternatives or professional help.")
    elif 12 < user_input["Smoking"] < 20:
        recommendations.append("Consider breathing regulation exercises like box breathing to counteract the effects of smoking.")

    if user_input["Therapy Sessions (per month)"] == 0 and severity > 3:
        recommendations.append("Consider attending therapy sessions to manage anxiety more effectively.")

    if severity > 7 and user_input["Medication"] == 0:
        recommendations.append("Since your severity is high, you may consider consulting a psychiatrist regarding medication.")

    if user_input["Diet Quality (1-10)"] < 5:
        recommendations.append("Improve your diet quality. Include more fiber-rich and protein-rich foods.")

    if user_input["Sleep Hours"] < 6:
        recommendations.append("Try to get at least 6 hours of quality sleep daily. Sleep is essential for mental wellness.")

    if user_input["Stress Level (1-10)"]>= 6:
        recommendations.append("Your stress to relativaly high.You can try meditation to reduce yout stress level and try taking a break from stressfull environment.")

    if user_input["Caffeine Intake (mg/day)"] > 400:
        recommendations.append("Too much caffeine can increase anxiety. Try reducing it to a maximum of 400 mg per day, which is considered a safe amount for daily consumption.")

    if user_input["Dizziness"]==1:
        recommendations.append("If you feel dizzy when stressed, try grounding techniques like focusing on your surroundings or using the 5-4-3-2-1 method. Staying well-hydrated also helps maintain blood pressure and reduce dizziness.")
            
    return [custom_msg] + recommendations
