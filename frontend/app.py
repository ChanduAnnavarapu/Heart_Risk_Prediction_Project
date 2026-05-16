import os
import streamlit as st
from dotenv import load_dotenv
import requests
from pathlib import Path

load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Heart Disease Prediction", 
                   page_icon="🧑‍⚕️",
                   layout="wide")

st.title("🩺 Heart Disease Prediction")
st.write("Enter the following details to predict the presence of heart disease:")

st.subheader("Patient Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex (1=Male, 0=Female)", options=[0, 1])
    cp=st.number_input("Chest Pain Type", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=50, max_value=200, value=60)
    restecg = st.number_input("Resting Electrocardiographic Results", min_value=0, max_value=2, value=0)
with col3:  
    exang=st.selectbox("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment", min_value=0, max_value=2, value=1)
with col4:
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, value=2)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    ca = st.number_input("No. of Colored vessels by Fluoroscopy", min_value=0, max_value=4, value=0)
    
if st.button("Predict", use_container_width=True):
    input_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    response=requests.post(API_URL, json=input_data)
    if response.status_code==200:
        result=response.json()
        st.subheader("Prediction Result")
        prediction=result['prediction']
        if prediction==1:
            st.error("The model predicts the presence of heart disease.")
            st.write(f"Probability of heart disease: {result['probability']}%")
        else:
            st.success("The model predicts the absence of heart disease.")
            st.write(f"Probability of heart disease: {result['probability']}%")
    else:
        st.error("Error occurred while making prediction. Please try again later.")

