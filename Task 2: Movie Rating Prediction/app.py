import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessing info
model = joblib.load('movie_rating_model.pkl')
preprocessing_info = joblib.load('preprocessing_info.pkl')

# Streamlit app
st.title('Movie Rating Prediction')

# Input fields for movie details
year = st.number_input('Year', min_value=1900, max_value=2100, value=2020)
votes = st.number_input('Number of Votes', min_value=0, value=1000)
duration = st.number_input('Duration (in minutes)', min_value=0, value=120)
genre = st.text_input('Genre', '')
director = st.text_input('Director', '')
actor1 = st.text_input('Actor 1', '')
actor2 = st.text_input('Actor 2', '')
actor3 = st.text_input('Actor 3', '')

# Preprocess inputs
# Handle missing genres and other features by setting defaults if not found in preprocessing info
genre_mean_rating = preprocessing_info['Genre_mean_rating'].get(genre, 0)
director_encoded = preprocessing_info['Director_encoded'].get(director, 0)
actor1_encoded = preprocessing_info['Actor1_encoded'].get(actor1, 0)
actor2_encoded = preprocessing_info['Actor2_encoded'].get(actor2, 0)
actor3_encoded = preprocessing_info['Actor3_encoded'].get(actor3, 0)

input_data = pd.DataFrame([{
    'Year': year,
    'Votes': votes,
    'Duration': duration,
    'Genre_mean_rating': genre_mean_rating,
    'Director_encoded': director_encoded,
    'Actor1_encoded': actor1_encoded,
    'Actor2_encoded': actor2_encoded,
    'Actor3_encoded': actor3_encoded
}])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Rating: {prediction[0]:.2f}')