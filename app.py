import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model, scaler, and feature names
model = joblib.load('gallbladder_model.pkl')
scaler = joblib.load('gallbladder_scaler.pkl')
feature_names = joblib.load('gallbladder_features.pkl')
print("Features loaded in app.py:", feature_names) # 调试打印


# Set page title and layout
st.set_page_config(page_title="Gallbladder Polyp Prediction", layout="wide")
st.title('Gallbladder Polyp Prediction Model')

# Create a sidebar (optional)
st.sidebar.header('Input Features')

# Create input form
input_data = {}
col1, col2, col3 = st.columns(3)

with col1:
    input_data['diameter'] = st.number_input('Diameter (mm):', min_value=0.0, format="%.2f")
    input_data['NLR'] = st.number_input('NLR:', min_value=0.0, format="%.2f")

with col2:
    input_data['stalk'] = st.selectbox('Stalk Present:', [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    input_data['age'] = st.number_input('Age (years):', min_value=0, format="%d")

with col3:
    input_data['CA199'] = st.number_input('CA199 (U/mL):', min_value=0.0, format="%.2f")


# Prediction button
if st.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize numerical features
    numerical_features = ['diameter', 'NLR', 'age', 'CA199']
    print("Numerical features in app.py:", numerical_features) # 调试打印
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])


    # Make prediction
    probability = model.predict_proba(input_df)[0, 1]

    # Display prediction results
    st.subheader('Prediction Results:')
    st.write(f'Probability of Neoplastic Polyp:  **{probability:.4f}**')

    if probability > 0.5:
        st.markdown("<h3 style='color:red;'>Prediction: Likely Neoplastic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:green;'>Prediction: Likely Non-Neoplastic</h3>", unsafe_allow_html=True)


# Add model evaluation information (optional)
st.sidebar.header('Model Evaluation (Based on Test Set)')
st.sidebar.markdown("""
*   **Accuracy:**  [Enter your model's accuracy here]
*   **ROC AUC:**  [Enter your model's ROC AUC here]
*   **Confusion Matrix:**
    ```
    [Enter your confusion matrix here]
    ```
""")

# Add some information about the model (optional)
st.markdown("---")
st.write("""
**Model Information:**

*   This model uses Logistic Regression to predict the likelihood of a gallbladder polyp being neoplastic based on five features.
*   The model has been trained and evaluated on historical data.
*   The prediction results are for reference only and should not replace a professional medical diagnosis.  Consult with a healthcare professional for any health concerns.
""")
