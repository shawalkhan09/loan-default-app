import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Loan Default Predictor", layout="centered")

st.title("Loan Default Prediction")
st.write("Fill in the applicant details below and click Predict.")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    revolving_utilization = st.number_input(
        "Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0,
        help="Total balance on credit cards divided by credit limit"
    )
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )
    times_30_59 = st.number_input(
        "Times 30-59 Days Late", min_value=0, max_value=20, value=0
    )
    debt_ratio = st.number_input(
        "Debt Ratio", min_value=0.0, max_value=100.0, value=20.0,
        help="Monthly debt payments divided by monthly income"
    )
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0.0, value=5000.0
    )

with col2:
    open_credit_lines = st.number_input(
        "Open Credit Lines", min_value=0, max_value=50, value=5
    )
    times_90_days_late = st.number_input(
        "Times 90+ Days Late", min_value=0, max_value=20, value=0
    )
    real_estate_loans = st.number_input(
        "Real Estate Loans", min_value=0, max_value=20, value=1
    )
    times_60_89 = st.number_input(
        "Times 60-89 Days Late", min_value=0, max_value=20, value=0
    )
    dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=20, value=0
    )

st.divider()

# Predict button
if st.button("Predict", use_container_width=True):

    input_data = np.array([[
        revolving_utilization,
        age,
        times_30_59,
        debt_ratio,
        monthly_income,
        open_credit_lines,
        times_90_days_late,
        real_estate_loans,
        times_60_89,
        dependents
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"High Risk: This applicant is likely to default.")
    else:
        st.success(f"Low Risk: This applicant is unlikely to default.")

    st.metric(label="Default Probability", value=f"{probability * 100:.1f}%")