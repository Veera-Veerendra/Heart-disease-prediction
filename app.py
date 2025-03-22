import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to predict heart disease
def predict_heart_disease(data):
    prediction = model.predict([data])
    return "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease"

# Streamlit App
st.title("Heart Disease Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=110, step=1)
sex = st.selectbox("Gender", ["0: Male", "1: Female"])
cp = st.selectbox("Chest Pain", ["0: Typical angina", "1: Atypical angina", "2: Non-anginal pain", "3: Asymptomatic"])
bp = st.number_input("Blood Pressure", min_value=0, max_value=300)
chol = st.number_input("Cholesterol", min_value=0, max_value=600)
restecg = st.selectbox("Resting ECG", ["0: Normal", "1: Abnormal", "2: Probable LVH"])
thalach = st.number_input("Heart Rate", min_value=0, max_value=300)
exang = st.selectbox("Exercise Induced", ["0: No", "1: Yes"])
oldpeak = st.number_input("Old Peak", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope", ["0: Upsloping", "1: Flat", "2: Downsloping"])
ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4)

# Feature engineering for categorical values
sex = int(sex[0])
cp = int(cp[0])
restecg = int(restecg[0])
exang = int(exang[0])
slope = int(slope[0])

input_data = np.array([age, sex, bp, chol, thalach, exang, oldpeak, slope, ca])

# Predict button
if st.button("Predict"):
    result = predict_heart_disease(input_data)
    st.success(f"Prediction: {result}")
