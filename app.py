#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb # Import xgboost to create DMatrix
import numpy as np # Import numpy for potential dummy values

# Load your saved model
try:
    model = joblib.load("xgb_fraud.pkl")
except FileNotFoundError:
    st.error("Error: Model file 'xgb_fraud.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Smart Fraud Detector", page_icon="💳")

st.title("💳 Smart Fraud Detector")
st.markdown("Enter the transaction details below to get a fraud probability prediction.")

# Collect user inputs
st.header("Transaction Details")
transaction_amt = st.number_input("Transaction Amount ($)", value=50.0, min_value=0.01, format="%.2f")
card_type = st.selectbox("Card Type", ["Visa", "MasterCard", "Discover", "American Express", "Other"], index=0)
issuer_bank = st.number_input("Card Issuer Code (e.g., first few digits, if available)", value=1500, min_value=1)
device_type = st.selectbox("Device Type", ["desktop", "mobile"], index=0)
browser = st.selectbox("Browser Used", ["chrome", "safari", "firefox", "edge", "other"], index=0)
days_since_use = st.slider("Days Since Last Use", 0, 365, value=1)
email_match = st.selectbox("Billing & Recipient Email Match", ["Yes", "No"], index=1)

# --- Feature Engineering and Mapping ---
card4_map = {"Visa": 1, "MasterCard": 2, "Discover": 3, "American Express": 4, "Other": 0}
device_map = {"desktop": 0, "mobile": 1}
browser_map = {"chrome": 1, "safari": 2, "firefox": 3, "edge": 4, "other": 0}
email_match_flag = 1 if email_match == "Yes" else 0

# --- IMPORTANT: Map your collected features to the 'feature_X' names ---
# You NEED to get the correct mapping from your training script.
# This is an EXAMPLE mapping. Adjust it based on your actual X_train.columns.tolist()
# You also need to include ALL 20 features ('feature_0' to 'feature_19')
# with appropriate default/dummy values if they are not collected from the user.

# Create a dictionary to hold all 20 features, initialized to 0 or a reasonable default
all_features_data = {f'feature_{i}': [0.0] for i in range(20)} # Initialize with float type for consistency

# Populate the features that you collect from the user
# This mapping needs to be accurate to your training data's column order/names
# Example (replace feature_X with the correct index based on your training data):
# If 'TransactionAmt' was feature_0
all_features_data['feature_0'] = [transaction_amt]
# If 'card1' was feature_1
all_features_data['feature_1'] = [float(issuer_bank)] # Ensure type matches training, if it was float
# If 'card4' was feature_2
all_features_data['feature_2'] = [float(card4_map[card_type])]
# If 'DeviceType' was feature_3
all_features_data['feature_3'] = [float(device_map[device_type])]
# If 'id_31' was feature_4
all_features_data['feature_4'] = [float(browser_map[browser])]
# If 'D1' was feature_5
all_features_data['feature_5'] = [float(days_since_use)]
# If 'email_match' was feature_6
all_features_data['feature_6'] = [float(email_match_flag)]

# You MUST fill in the remaining 'feature_X' fields (e.g., feature_7 to feature_19)
# with default/dummy values that make sense for your model.
# These values should ideally represent a "neutral" state or the mean/median of that feature
# if it's not provided by the user. For this example, they remain 0.0 as initialized above.
# If these were categorical and are One-Hot Encoded, ensure correct 0/1 values for those.

df_input = pd.DataFrame(all_features_data)

# Ensure the columns are in the exact order the model expects
# You can get this order from model.feature_names_in_ if it's an XGBClassifier,
# or from the exact order of columns in X_train during training for Booster.
# Let's assume for a Booster, the order is 'feature_0', 'feature_1', ..., 'feature_19'
expected_feature_order = [f'feature_{i}' for i in range(20)]
df_input = df_input[expected_feature_order]


# --- Prediction Logic ---
if st.button("Detect Fraud"):
    # Convert the Pandas DataFrame to a DMatrix for the Booster model
    dinput = xgb.DMatrix(df_input)

    # Make prediction using model.predict()
    if hasattr(model, 'best_iteration'):
        predictions = model.predict(dinput, iteration_range=(0, model.best_iteration + 1))
    else:
        predictions = model.predict(dinput)

    prob = predictions[0]

    st.write(f"📊 Fraud Probability: **{prob:.4f}**")

    if prob > 0.5:
        st.error("🚨 Likely Fraudulent!")
        st.markdown("**Action Recommended:** Review transaction, potentially flag for manual check or decline.")
    else:
        st.success("✅ Legitimate Transaction")
        st.markdown("**Action Recommended:** Proceed with transaction.")

st.markdown("---")
st.markdown("Developed for Advanced Programming 2025")