import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

model = tf.keras.models.load_model("model.h5")

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

##streamlit app
st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", label_encoder_gender.classes_)
geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
age = st.number_input("Age", 18, 92)
tenure = st.number_input("Tenure", 0, 10)
balance = st.number_input("Balance", 0, 1000000)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0, 1000000)
credit_score = st.number_input("Credit Score", 0, 1000)

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
    "Geography": [geography],
})

#onehot encode 'Geography'
geo_encoded = one_hot_encoder_geo.transform(input_data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]))

#Combine one-hot encoded 'Geography' with the rest of the input data
input_data = pd.concat(
    [input_data.drop(columns=["Geography"]).reset_index(drop=True), geo_encoded_df],
    axis=1,
)

# Align inference columns with the scaler fit order.
input_data = input_data[scaler.feature_names_in_]

#Scale the input data
input_data = scaler.transform(input_data)

#Make prediction
prediction = model.predict(input_data)
prediction_proba = prediction[0][0]
st.write(f"Prediction: {prediction[0]}")

if prediction_proba > 0.5:
    st.write(f"Prediction Probability: {prediction_proba:.2f} (Churn)")
else:
    st.write(f"Prediction Probability: {prediction_proba:.2f} (Not Churn)")
