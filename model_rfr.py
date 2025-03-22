import streamlit as st
import numpy as np
import joblib

# Load model yang telah disimpan
model = joblib.load("model_mental_health_rfr.joblib")

# Menampilkan fitur yang digunakan model
st.write("Model ini menggunakan fitur:", model.feature_names_in_)

# Judul aplikasi
st.title("Mental Health Treatment Progress Prediction")

# Input numerik
age = st.number_input("Age", min_value=0, max_value=100, value=30)
symptom_severity = st.slider("Symptom Severity (1-10)", 1, 10, 5)
mood_score = st.slider("Mood Score (1-10)", 1, 10, 5)
sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 5)
physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0.0, max_value=50.0, value=3.0)
adherence_to_treatment = st.slider("Adherence to Treatment (%)", 0, 100, 50)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
treatment_duration = st.number_input("Treatment Duration (weeks)", min_value=0, max_value=52, value=10)  # Menambahkan fitur yang hilang

# Input kategori
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
diagnosis = st.selectbox("Diagnosis", ["Depression", "Anxiety", "Bipolar Disorder", "Schizophrenia", "Other"])
medication = st.selectbox("Medication", ["Yes", "No"])
therapy_type = st.selectbox("Therapy Type", ["Cognitive Behavioral Therapy", "Medication-Based Therapy", "Support Groups", "None"])
ai_emotional_state = st.selectbox("AI-Detected Emotional State", ["Positive", "Neutral", "Negative"])

# Encoding kategori ke numerik
gender_dict = {"Male": 0, "Female": 1, "Other": 2}
diagnosis_dict = {"Depression": 0, "Anxiety": 1, "Bipolar Disorder": 2, "Schizophrenia": 3, "Other": 4}
medication_dict = {"Yes": 1, "No": 0}
therapy_dict = {"Cognitive Behavioral Therapy": 0, "Medication-Based Therapy": 1, "Support Groups": 2, "None": 3}
ai_emotional_dict = {"Positive": 0, "Neutral": 1, "Negative": 2}

# Membentuk array input untuk model
input_data = np.array([
    age, symptom_severity, mood_score, sleep_quality, physical_activity, 
    adherence_to_treatment, stress_level, treatment_duration,  # Tambahkan fitur yang hilang
    gender_dict[gender], diagnosis_dict[diagnosis], 
    medication_dict[medication], therapy_dict[therapy_type], 
    ai_emotional_dict[ai_emotional_state]
]).reshape(1, -1)

# Tombol untuk prediksi
if st.button("Predict Treatment Progress"):
    prediction = model.predict(input_data)[0]  # Prediksi Treatment Progress
    st.success(f"Predicted Treatment Progress: {prediction:.2f}/10")
