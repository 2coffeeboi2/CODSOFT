# app.py

import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.write("Model successfully loaded.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app layout
st.title("Titanic Survival Prediction")

st.write("""
    This app predicts Titanic survival based on passenger information.
    Please enter the details below to get the prediction.
""")

# Inputs
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, value=7.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Make prediction
try:
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    st.write("Prediction successful.")
    st.write("Prediction: **Survived**" if prediction[0] == 1 else "Prediction: **Not Survived**")
    st.write(f"Probability of survival: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of not surviving: {prediction_proba[0][0]:.2f}")
except Exception as e:
    st.error(f"Error during prediction: {e}")