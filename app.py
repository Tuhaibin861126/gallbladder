import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model, scaler, and feature names
model = joblib.load('gallbladder_model.pkl')
scaler = joblib.load('gallbladder_scaler.pkl')
feature_names = joblib.load('gallbladder_features.pkl')
print("Features loaded in app.py:", feature_names)

# Set page title and layout
st.set_page_config(page_title="Gallbladder Polyp Prediction", layout="wide")
st.title('Gallbladder Polyp Prediction Model')

# Create a sidebar (optional)
st.sidebar.header('Input Features')

# Create input form
input_data = {}
for feature in feature_names:
    if feature == 'stalk':
        input_data[feature] = st.selectbox(
            f'Stalk Present:', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )
    else:
        input_data[feature] = st.number_input(f"{feature}:", min_value=0.0, format="%.2f")

# Prediction button
if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize numerical features
    numerical_features = [feature for feature in feature_names if feature != 'stalk']
    print("Numerical features in app.py:", numerical_features)
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Make prediction
    probability = model.predict_proba(input_df)[0, 1]

    # Display prediction results
    st.subheader('Prediction Results:')
    st.write(f'Probability of Neoplastic Polyp:  **{probability:.4f}**')

    if probability > 0.5:
        st.markdown("<h3 style='color:red;'>Prediction: Likely Neoplastic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>Prediction: Likely Non-Neoplastic</h3>", unsafe_allow_html=True)

# Model evaluation and information sections (as before) ...
