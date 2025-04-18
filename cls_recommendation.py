def get_recommendations(severity, user_input=None):
    # If no user_input is provided, return general recommendations based on severity only
    if user_input is None:
        recommendations = {}
        
        if severity <= 3:
            recommendations["Severity"] = "Your anxiety severity is low. This suggests your lifestyle might already be relatively healthy."
            recommendations["Physical Activity"] = "Continue with regular physical activity, aim for at least 150 minutes per week."
            recommendations["Sleep"] = "Maintain your good sleep habits, aiming for 7-9 hours per night."
            recommendations["Diet"] = "Continue with a balanced diet rich in whole foods, fruits, and vegetables."
            recommendations["Mindfulness"] = "Consider adding mindfulness practices to further improve your well-being."
            
        elif 4 <= severity <= 6:
            recommendations["Severity"] = "Your anxiety severity is moderate. Consider making some lifestyle changes."
            recommendations["Physical Activity"] = "Increase physical activity to at least 150 minutes per week."
            recommendations["Sleep"] = "Prioritize sleep hygiene, aiming for 7-9 hours per night."
            recommendations["Diet"] = "Reduce caffeine and alcohol consumption, increase whole foods."
            recommendations["Professional Support"] = "Consider talking to a mental health professional for additional support."
            recommendations["Mindfulness"] = "Start a regular meditation or mindfulness practice, even 5-10 minutes daily can help."
            
        else:  # severity > 6
            recommendations["Severity"] = "Your anxiety severity is high. Focus on improving your mental health."
            recommendations["Professional Support"] = "Consider consulting a mental health professional as soon as possible."
            recommendations["Physical Activity"] = "Include light physical activity like walking or gentle yoga."
            recommendations["Sleep"] = "Prioritize improving sleep quality through consistent bedtime routines."
            recommendations["Diet"] = "Eliminate or significantly reduce caffeine and alcohol."
            recommendations["Mindfulness"] = "Learn and practice deep breathing exercises for immediate anxiety relief."
            recommendations["Medication"] = "Discuss medication options with a healthcare provider if appropriate."
            
        return recommendations
        
    # Original function logic for personalized recommendations when user_input is provided
    recommendations = {}

    if severity <= 3:
        recommendations["Severity"] = "Your anxiety severity is low. This suggests your lifestyle might already be relatively healthy. Keep it up!"
    elif 4 <= severity <= 6:
        recommendations["Severity"] = "Your anxiety severity is moderate. Consider making positive lifestyle changes to improve your mental health."
    else:
        recommendations["Severity"] = "Your anxiety severity is high. Immediate lifestyle changes are recommended for your well-being."

    recommendations["Overview"] = "Here are some personalized suggestions to help manage your anxiety:"

    if user_input.get("Alcohol Consumption (drinks/week)", 0) > 10:
        recommendations["Alcohol"] = "Reduce alcohol consumption—it can increase anxiety levels."

    if user_input.get("Physical Activity (hrs/week)", 0) == 0:
        recommendations["Physical Activity"] = "Include physical activity in your routine. Even light exercise helps reduce anxiety."

    if user_input.get("Smoking", 0) == 1:
        recommendations["Smoking"] = "Try to quit smoking. Consider nicotine alternatives or professional help."
    elif user_input.get("Smoking", 0) > 12 and user_input.get("Smoking", 0) < 20:
        recommendations["Breathing"] = "Consider breathing regulation exercises like box breathing to counteract the effects of smoking."

    if user_input.get("Therapy Sessions (per month)", 0) == 0 and severity > 3:
        recommendations["Therapy"] = "Consider attending therapy sessions to manage anxiety more effectively."

    if severity > 7 and user_input.get("Medication", 0) == 0:
        recommendations["Medication"] = "Since your severity is high, you may consider consulting a psychiatrist regarding medication."

    if user_input.get("Diet Quality (1-10)", 5) < 5:
        recommendations["Diet"] = "Improve your diet quality. Include more fiber-rich and protein-rich foods."

    if user_input.get("Sleep Hours", 6) < 6:
        recommendations["Sleep"] = "Try to get at least 6 hours of quality sleep daily. Sleep is essential for mental wellness."

    if user_input.get("Stress Level (1-10)", 5) >= 6:
        recommendations["Stress"] = "Your stress is relatively high. You can try meditation to reduce your stress level and take breaks from stressful environments."

    if user_input.get("Caffeine Intake (mg/day)", 200) > 400:
        recommendations["Caffeine"] = "Too much caffeine can increase anxiety. Try reducing it to a maximum of 400 mg per day, which is considered a safe amount for daily consumption."

    if user_input.get("Dizziness", 0) == 1:
        recommendations["Dizziness"] = "If you feel dizzy when stressed, try grounding techniques like focusing on your surroundings or using the 5-4-3-2-1 method. Staying well-hydrated also helps maintain blood pressure and reduce dizziness."
            
    return recommendations
