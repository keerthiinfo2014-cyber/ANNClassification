import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')
# Load the encoders and scaler
with open('onehotencoder_geography.pkl', 'rb') as f:
    onehotencoder_geography = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the Streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', onehotencoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score')
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data in the correct order
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode Geography
geo_encoded = onehotencoder_geography.transform(
    input_data[['Geography']]).toarray()
geo_df = pd.DataFrame(
    geo_encoded, columns=onehotencoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data, geo_df], axis=1)

# Label encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Drop original Geography column
input_data.drop('Geography', axis=1, inplace=True)

# Reorder columns to match training data (if needed)
# Example: input_data = input_data[model_input_columns]

# Scaling the input data
input_data = scaler.transform(input_data)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction = (prediction > 0.5).astype(int)
    if prediction[0][0] == 1:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is unlikely to churn.')
