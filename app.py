import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# Load Model & Features
@st.cache_resource
def load_assets():
    model = joblib.load('salary_model.pkl')
    features = joblib.load('salary_features.pkl')
    return model, features

try:
    model, features = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Header Section
st.title("💼 Employee Salary Estimation Engine")
st.write("Predict fair market annual compensation based on candidate profile, domain, and demographics.")
st.divider()

# Input Form Controls
st.subheader("Candidate Details")

col1, col2 = st.columns(2)

with col1:
    exp = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
    age = st.slider("Age", min_value=18, max_value=65, value=28)
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "High School"])
    job_title = st.selectbox(
        "Job Title / Role", 
        ["Software Engineer", "Data Scientist", "Data Analyst", "Product Manager", "Sales Executive", "HR Manager"]
    )

st.divider()

# Prediction Execution
if st.button("Predict Salary", type="primary", use_container_width=True):
    # Initialize zero feature vector matching model training layout
    input_data = {feat: 0 for feat in features}
    
    # Map Numerical Values
    for feat in features:
        feat_lower = feat.lower()
        if 'experience' in feat_lower or 'yearsexperience' in feat_lower:
            input_data[feat] = exp
        elif 'age' in feat_lower:
            input_data[feat] = age

    # Map One-Hot Encoded Categorical Features
    if gender == "Female":
        for feat in features:
            if 'gender_female' in feat.lower() or 'gender_f' in feat.lower():
                input_data[feat] = 1

    for feat in features:
        # Education Level matching
        if education.lower() in feat.lower():
            input_data[feat] = 1
        # Job Title matching
        if job_title.lower().replace(" ", "") in feat.lower().replace(" ", ""):
            input_data[feat] = 1

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict Salary
    predicted_salary = model.predict(input_df)[0]
    
    # Render Result Card
    st.success(f"### Estimated Annual Salary: ₹ {max(300000, round(predicted_salary, 2)):,}")
    
    st.info("""
    **Model Valuation Summary:**
    - **Demographic & Role Factors:** Compensation predictions adjust based on years of experience, formal education tier, and job function demand.
    - **Engine Logic:** Model employs Random Forest ensemble regression trained on processed enterprise salary datasets.
    """)
    