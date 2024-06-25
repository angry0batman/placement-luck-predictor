import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app configuration
st.set_page_config(page_title='Placement Prediction App', layout='wide')

# Custom CSS for the background and navigation bar
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 10px;
        }
        .css-1avcm0n {
            background-color: #333333;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .css-qrbaxs {
            color: white;
        }
        footer {
            background-color: #1e1e1e;
            padding: 20px;
            text-align: center;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
    <div style='background-color: #333333; padding: 10px; text-align: center; border-radius: 10px;'>
        <a href='#' style='margin: 0 15px; color: white; text-decoration: none;'>Home</a>
        <a href='#' style='margin: 0 15px; color: white; text-decoration: none;'>About</a>
        <a href='#' style='margin: 0 15px; color: white; text-decoration: none;'>Contact</a>
    </div>
""", unsafe_allow_html=True)

# Streamlit app title
st.title('Check Your Placement Luck')

# Function to get user input
def get_user_input():
    dsa_score = st.number_input('DSA Score', min_value=0, max_value=100, value=50)
    resume_score = st.number_input('Resume Score', min_value=0, max_value=100, value=50)
    communication_score = st.number_input('Communication Score', min_value=0, max_value=100, value=50)
    development_score = st.number_input('Development Score', min_value=0, max_value=100, value=50)
    college_tier = st.selectbox('College Tier', options=[1, 2, 3])
    
    user_data = {
        'dsa_score': dsa_score,
        'resume_score': resume_score,
        'communication_score': communication_score,
        'development_score': development_score,
        'college_tier': college_tier
    }
    
    features = np.array([dsa_score, resume_score, communication_score, development_score, college_tier]).reshape(1, -1)
    return features

# Display HTML content
st.markdown("""
    <p style='text-align: left;'>Enter your scores below and click 'Predict Placement' to see the result.</p>
    <hr>
""", unsafe_allow_html=True)

# Get user input
user_input = get_user_input()

# Validate and scale user input
try:
    if np.isnan(user_input).any():
        st.error("Input contains missing values. Please provide complete input data.")
    elif len(user_input.shape) != 2:
        st.error("Input data shape is incorrect. Expected a 2D array.")
    else:
        scaled_input = scaler.transform(user_input)

        # Make prediction
        if st.button('Predict Placement'):
            prediction = model.predict(scaled_input)
            if prediction[0] == 1:
                st.success('The student is likely to be placed.')
            else:
                st.error('The student is not likely to be placed.')
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("""
    <footer>
        <p>&copy; 2024 Placement Prediction App. All rights reserved.</p>
    </footer>
""", unsafe_allow_html=True)
