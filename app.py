import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make predictions
def predict_placement(dsa_score, resume_score, communication_score, development_score, college_tier):
    try:
        features = np.array([[dsa_score, resume_score, communication_score, development_score, college_tier]])

        # Debug: print the features array
        st.write("Features array (before scaling):", features)
        st.write("Features array shape (before scaling):", features.shape)

        # Check for missing values
        if np.any(np.isnan(features)):
            st.error("Error: Some input values are missing. Please provide all required inputs.")
            return None

        # Validate input shape
        if features.shape[1] != 5:
            st.error("Error: Invalid input shape. Expected 5 features.")
            return None

        # Scale the features
        features_scaled = scaler.transform(features)

        # Debug: print the scaled features array
        st.write("Scaled features array:", features_scaled)
        st.write("Scaled features array shape:", features_scaled.shape)

        # Make prediction
        prediction = model.predict(features_scaled)
        return prediction
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app
st.set_page_config(page_title='Placement Prediction App', page_icon='🎓', layout='wide')

# HTML & CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Check Your Placement Chance</h1>
    <hr>
    """, unsafe_allow_html=True)

# Input fields
st.write("## Enter the following details:")

dsa_score = st.number_input('DSA Score', min_value=0.0, max_value=100.0, value=50.0)
resume_score = st.number_input('Resume Score', min_value=0.0, max_value=100.0, value=50.0)
communication_score = st.number_input('Communication Score', min_value=0.0, max_value=100.0, value=50.0)
development_score = st.number_input('Development Score', min_value=0.0, max_value=100.0, value=50.0)
college_tier = st.selectbox('College Tier', [1, 2, 3])

if st.button('Predict Placement'):
    result = predict_placement(dsa_score, resume_score, communication_score, development_score, college_tier)
    if result is not None:
        if result == 1:
            st.success('🎉 The student is likely to be placed.')
        else:
            st.error('😔 The student is unlikely to be placed.')
