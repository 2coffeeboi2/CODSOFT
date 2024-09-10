import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and data
model = joblib.load('sales_model.pkl')
data = pd.read_csv('sales_data.csv')

# Check the columns to verify
st.write("Columns in the data:", data.columns)

# Ensure required columns are present
required_columns = ['tv', 'radio', 'newspaper', 'sales']
if not all(col in data.columns for col in required_columns):
    st.error("Expected columns are not found in the dataset. Please check the CSV file.")
else:
    # Set up Streamlit app
    st.title('Sales Prediction App')

    # Sidebar for user input
    st.sidebar.header('User Input')
    
    # Create sliders for each advertising medium
    tv_advertising = st.sidebar.slider('TV Advertising Expenditure', min_value=int(data['tv'].min()), max_value=int(data['tv'].max()), value=int(data['tv'].mean()))
    radio_advertising = st.sidebar.slider('Radio Advertising Expenditure', min_value=int(data['radio'].min()), max_value=int(data['radio'].max()), value=int(data['radio'].mean()))
    newspaper_advertising = st.sidebar.slider('Newspaper Advertising Expenditure', min_value=int(data['newspaper'].min()), max_value=int(data['newspaper'].max()), value=int(data['newspaper'].mean()))
    
    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        'tv': [tv_advertising],
        'radio': [radio_advertising],
        'newspaper': [newspaper_advertising]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction
    st.write(f'Predicted Sales with TV: ${tv_advertising}, Radio: ${radio_advertising}, Newspaper: ${newspaper_advertising}: ${prediction:.2f}')

    # Plotting
    st.subheader('Data Visualization')

    # Scatter plot of actual data
    st.subheader('Advertising vs Sales')
    fig, ax = plt.subplots()
    ax.scatter(data['tv'], data['sales'], color='black', label='TV Advertising')
    ax.scatter(data['radio'], data['sales'], color='red', label='Radio Advertising')
    ax.scatter(data['newspaper'], data['sales'], color='blue', label='Newspaper Advertising')
    ax.set_xlabel('Advertising Expenditures')
    ax.set_ylabel('Sales')
    ax.set_title('Advertising vs Sales')
    ax.legend()
    st.pyplot(fig)