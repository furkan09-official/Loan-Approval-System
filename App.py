import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model = RandomForestClassifier()
model = pickle.load(open('model.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_loan_approval(credit_score, annual_income, debt_to_income_ratio, employment_status, loan_duration, 
                          home_ownership_status, marital_status, number_of_open_credit_lines, 
                          previous_loan_defaults, risk_score):
    # Label encoding for categorical values (this should match the encoding during training)
    employment_mapping = {'Employed': 2, 'Unemployed': 1, 'Self-employed': 0}
    home_ownership_mapping = {'Own': 1, 'Rent': 0}
    marital_status_mapping = {'Married': 1, 'Single': 0}
    previous_loan_mapping = {'Yes': 1, 'No': 0}

    employment_status_num = employment_mapping[employment_status]
    home_ownership_status_num = home_ownership_mapping[home_ownership_status]
    marital_status_num = marital_status_mapping[marital_status]
    previous_loan_defaults_num = previous_loan_mapping[previous_loan_defaults]

    # Create a data frame for the input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'AnnualIncome': [annual_income],
        'DebtToIncomeRatio': [debt_to_income_ratio],
        'EmploymentStatus': [employment_status_num],
        'LoanDuration': [loan_duration],
        'HomeOwnershipStatus': [home_ownership_status_num],
        'MaritalStatus': [marital_status_num],
        'NumberOfOpenCreditLines': [number_of_open_credit_lines],
        'PreviousLoanDefaults': [previous_loan_defaults_num],
        'RiskScore': [risk_score]
    })

    # Scale the input data 
    input_data_scaled = scaler.transform(input_data)

    # Predict using the loaded model
    prediction = model.predict(input_data_scaled)

    return prediction[0]

# Streamlit UI
st.title("Loan Approval Prediction")

st.markdown("""
    Enter the following details to predict if your loan application is likely to be approved or not.
""")

# User inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)
annual_income = st.number_input("Annual Income (INR)", min_value=1000, step=1000)
debt_to_income_ratio = st.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed', 'Self-employed'])
loan_duration = st.slider("Loan Duration (Months)", min_value=1, max_value=120, value=36)
home_ownership_status = st.selectbox("Home Ownership Status", ['Own', 'Rent'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single'])
number_of_open_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, max_value=30, value=5)
previous_loan_defaults = st.selectbox("Previous Loan Defaults", ['Yes', 'No'])
risk_score = st.number_input("Risk Score", min_value=0, max_value=1000, value=500, step=1)

# Predict button
if st.button("Predict Loan Approval"):
    result = predict_loan_approval(
        credit_score, annual_income, debt_to_income_ratio, employment_status, loan_duration, 
        home_ownership_status, marital_status, number_of_open_credit_lines, 
        previous_loan_defaults, risk_score
    )
    
    # Display the result
    if result == 1:
        st.success("Congratulations! Your loan is likely to be approved.")
    else:
        st.error("Sorry, your loan is likely to be rejected.")

# Note to the user
st.markdown("""
            """
   )
