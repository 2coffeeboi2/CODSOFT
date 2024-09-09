import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load('random_forest_model.pkl')

# Load the label encoder and dataset for encoding
df = pd.read_csv('IRIS.csv')  # Make sure 'IRIS.csv' is in the same directory
le = LabelEncoder()
df_encoded = df.copy()
for column in df_encoded.columns:
    if not pd.api.types.is_numeric_dtype(df_encoded[column]):
        df_encoded[column] = le.fit_transform(df_encoded[column])

# Function to predict species
def predict_species(model, label_encoder, sepal_length, sepal_width, petal_length, petal_width):
    test_data = {'sepal_length': [sepal_length], 'sepal_width': [sepal_width], 'petal_length': [petal_length], 'petal_width': [petal_width]}
    test_df = pd.DataFrame(test_data)
    prediction = model.predict(test_df)
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit app
st.title('Iris Species Classification')

# Input fields for prediction
st.sidebar.header('Input Parameters')
sepal_length = st.sidebar.number_input('Sepal Length (cm)', min_value=0.0, value=5.2)
sepal_width = st.sidebar.number_input('Sepal Width (cm)', min_value=0.0, value=3.6)
petal_length = st.sidebar.number_input('Petal Length (cm)', min_value=0.0, value=1.4)
petal_width = st.sidebar.number_input('Petal Width (cm)', min_value=0.0, value=0.2)

if st.sidebar.button('Predict'):
    result = predict_species(model, le, sepal_length, sepal_width, petal_length, petal_width)
    st.write(f'Predicted Species: {result}')

# Visualization section
st.subheader('Dataset and Visualizations')
st.write('**Original Dataset:**')
st.write(df.head())

# Display a countplot
st.write('**Species Count Distribution**')
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(x='species', data=df_encoded, palette='Set1')
plt.title('Species Count Distribution')
st.pyplot(plt)

# Histograms of features
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
colors = ['purple', 'red', 'yellow', 'magenta']

st.write('**Histograms of Features**')
for feature, color in zip(features, colors):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_encoded[feature], color=color, kde=True)
    plt.title(f'Distribution of {feature}')
    st.pyplot(plt)

# Heatmap for feature correlations
st.write('**Feature Correlation Heatmap**')
plt.figure(figsize=(10, 8))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
st.pyplot(plt)