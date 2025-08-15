# Gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

# Load the scaler and model from the pkl files
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=100, value=10) # Corrected max_value to 100
monthly_charges = st.number_input("Enter Monthly Charges", min_value=0, max_value=130, value=10) # Corrected variable name and min/max values
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predict_button = st.button("Predict!")

if predict_button:
    gender_selected = 1 if gender == "Female" else 0
    X = [age, gender_selected, tenure, monthly_charges] # Corrected variable name
    
    X1 = np.array(X)
    
    # Corrected the error by reshaping the 1D array into a 2D array
    X_array = scaler.transform(X1.reshape(1, -1))
    
    prediction = model.predict(X_array)[0]
    
    predicted = "Yes" if prediction == 1 else "No"
    
    st.balloons()
    
    st.write(f"Predicted: {predicted}")
    
else:
    st.write("Please enter the values and use predict button")
