import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('drug_classifier_rf.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Drug Classification Predictor")
st.write("Enter patient details to predict the appropriate drug type.")

# --- User Inputs ---
age = st.number_input("Age", min_value=0, max_value=120, value=25)

sex = st.selectbox("Sex", ["F", "M"])
bp = st.selectbox("Blood Pressure", ["HIGH", "LOW", "NORMAL"])
cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
na_to_k = st.number_input("Sodium to Potassium Ratio (Na_to_K)", value=10.0, step=0.1)

# Create a dataframe for the input
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'BP': [bp],
    'Cholesterol': [cholesterol],
    'Na_to_K': [na_to_k]
})

# --- Preprocessing: One-Hot Encoding ---
input_df_encoded = pd.get_dummies(input_df, drop_first=True)

# Align columns with training set
# Load X_train columns (saved during training)
# For simplicity, we assume these are the columns after one-hot encoding in training
training_columns = ['Age', 'Na_to_K', 'Sex_M', 'BP_LOW', 'BP_NORMAL', 'Cholesterol_NORMAL']

for col in training_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# Ensure column order is same as training
input_df_encoded = input_df_encoded[training_columns]

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_df_encoded)
    st.success(f"The predicted drug type is: {prediction[0]}")
