import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("credit_card_model.pkl")

# Function to make predictions
def predict(transaction):
    transaction = np.array(transaction).reshape(1, -1)
    prediction = model.predict(transaction)
    return "Fraudulent Transaction" if prediction[0] == 1 else "Normal Transaction"

# Streamlit app
st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details:")

# Input features
num_features = 30  # Adjust based on the dataset
features = []
for i in range(num_features):
    feature_value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(feature_value)

# Make prediction when button is clicked
if st.button("Predict"):
    prediction = predict(features)
    st.write(f"Prediction: {prediction}")

# Example prediction
st.subheader("Example Prediction")
example_data = [
    -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443,
    -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507,
    0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348,
    -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478,
    0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705,
    -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731,
    0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215,
    0.0667120511783434, -0.20216648988976
]
if st.button("Show Example Prediction"):
    example_prediction = predict(example_data)
    st.write(f"Example Prediction: {example_prediction}")