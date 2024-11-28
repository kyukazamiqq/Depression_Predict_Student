import streamlit as st
import pickle 
import numpy as np

load_model = open('model2.pkl', 'rb')
classfier = pickle.load(load_model)


def predict(Gender, Age, AcademicPressure, StudyStatisfaction, SleepDuration, 
            DietaryHabits, Suicidal, StudyHours, FinancialStress, FamilyHistory):
    # Encode inputs as shown in your code
    Gender = 1 if Gender == "Male" else 2
    SleepDuration = {"7-8 hours": 1, "More than 8 hours": 2, "5-6 hours": 3}.get(SleepDuration, 4)
    DietaryHabits = {"Healthy": 1, "Unhealthy": 3}.get(DietaryHabits, 2)
    Suicidal = 2 if Suicidal == "Yes" else 1
    FamilyHistory = 2 if FamilyHistory == "Yes" else 1

    # Prepare the feature array
    features = np.array([[Gender, Age, AcademicPressure, StudyStatisfaction, SleepDuration, 
                          DietaryHabits, Suicidal, StudyHours, FinancialStress, FamilyHistory]])

    # Make a prediction
    return classfier.predict(features)


def main():
    st.title("Depression Prediction")
    st.markdown("Depressed Prediction using Gaussian Naive Bayes with Ensemble Method (Boosting)")
    
    Gender = st.selectbox("Gender", ("Male", "Female"), key="gender")
    Age = st.number_input("Age", key="age")
    AcademicPressure = st.number_input("Scale 1-5 (Academic Pressure)", min_value=1, max_value=5, key="academic_pressure")
    StudySatisfaction = st.number_input("Scale 1-5 (Study Satisfaction)", min_value=1, max_value=5, key="study_satisfaction")
    SleepDuration = st.selectbox("Sleep Duration", ("7-8 hours", "More than 8 hours", "5-6 hours", "Less than 5 hours"), key="sleep_duration")
    DietaryHabits = st.selectbox("Dietary Habits", ("Moderate", "Healthy", "Unhealthy"), key="dietary_habits")
    Suicidal = st.selectbox("Have you ever had Suicidal Thoughts?", ("Yes", "No"), key="suicidal")
    FinancialStress = st.number_input("Scale 1-5 (Financial Stress)", min_value=1, max_value=5, key="financial_stress")
    StudyHours = st.number_input("How many hours do you study?", min_value=1, max_value=12, key="study_hours")
    FamilyHistory = st.selectbox("Family History of Mental Illness", ("Yes", "No"), key="family_history")

    if st.button("Predict"):
        result = predict(Gender, Age, AcademicPressure, StudySatisfaction, SleepDuration, 
                         DietaryHabits, Suicidal, StudyHours, FinancialStress, FamilyHistory)
        if result == "Yes":
            st.success("You are predicted to have depression, please take care of yourself")
        elif result == "No":
            st.success("Wow, You are predicted to not have depression, keep it up! :D")

if __name__ == '__main__': 
    main()


    