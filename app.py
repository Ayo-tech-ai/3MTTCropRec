# Import necessary libraries
import streamlit as st
import joblib
import numpy as np

# Load the model and encoder (Ensure that the joblib files are in the correct path)
model = joblib.load('CropRec.joblib')
encoder = joblib.load('CropRec_LabelEncoder.joblib')

# Sidebar for page navigation
page = st.sidebar.selectbox("Select a page", ["Home", "About Us", "Important Information"])

# Home page for crop recommendation
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>AI-Powered Crop Recommendation Model</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    /* Main page background color */
    .stApp {
        background-color: #A8E6A1; /* Light green */
    }
    
    /* Sidebar background color */
    .css-1d391kg { 
        background-color: #BDFCC9; /* Mint green for sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.sidebar.markdown("""
## Project Description

The AI-Powered Crop Recommendation Model uses advanced machine learning techniques to provide farmers with customized crop suggestions based on soil and environmental factors. 

Inspired by the need to help farmers optimize yield and manage resources, particularly in climate-challenged areas, this project aims to boost agricultural productivity and promote sustainable practices. It emphasizes the importance of technology in connecting traditional farming with modern techniques, ultimately enhancing food security and economic stability for farming communities.
""")

    # Input fields for the seven features in a compact layout
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen Content (N)", min_value=0, max_value=171)
        temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=44.0)
        ph = st.number_input("pH Level", min_value=4.0, max_value=8.0)

    with col2:
        P = st.number_input("Phosphorous Content (P)", min_value=5, max_value=112)
        humidity = st.number_input("Humidity (%)", min_value=30.0, max_value=100.0)

    with col3:
        K = st.number_input("Potassium Content (K)", min_value=5, max_value=107)
        rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=1708.0)

    # Submit button to make predictions
    if st.button("Predict"):
        # Prepare input features for the model
        input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(input_features)
        
        # Convert the predicted label back to the crop name using the encoder
        predicted_crop = encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f"Recommended Crop: {predicted_crop[0]}")

    # Like button
    if st.button("Like ❤️"):
        st.success("Thank you for liking the model!")

    # Additional information at the bottom of the home page
    st.write("""
    ---
    **Important Information**  
    Please ensure that you provide accurate inputs for nitrogen, phosphorous, potassium, temperature, humidity, pH, and rainfall.  
    This model is designed for general recommendations of just 20 crops in Nigeria and may not capture specific local conditions.  
    Consult with local agronomists for additional guidance.
    """)

# About Us page
elif page == "About Us":
    st.title("About Us")
    st.write("""
    Welcome to the AI-Powered Crop Recommendation Model. 
    This tool is designed to help farmers make informed decisions on crop selection based on soil and climate conditions.
    Our mission is to support sustainable agriculture through advanced technology.

    ## MEET THE TEAM
    
    ### AYOOLA MUJIB AYODELE
    
    FE/23/89361170

    COHORT 2

    Learning Track : AI and ML
    
    ## Faustina Ndidiamaka Egbe

    FE/23/83253976

    COHORT 2

    Learning Track : Data analysis and Visualization

    ## Clara Okafor

    FE/23/96598382
    
    COHORT 2

    Learning Track : UI/UX
    """)

# Important Information page
elif page == "Important Information":
    st.title("Important Information")
    st.write("""
    Please ensure that you provide accurate inputs for nitrogen, phosphorous, potassium, temperature, humidity, pH, and rainfall.
    This model is designed for general recommendations and may not capture specific local conditions.
    
    Consult with local agronomists for additional guidance.
    """)
