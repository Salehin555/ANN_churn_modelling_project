import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import tensorflow as tf
import pickle

#load the trained model
from tensorflow.keras.models import load_model
model=load_model('model.h5')
#load the encoder and scaler
with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender=pickle.load(f)
with open('onehotencoder_geography.pkl','rb') as f:
    onehotencoder=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

# Define the Streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")
# Input fields
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender= st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])


#Prepare the input data
input_data=pd.DataFrame({
    
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

# One hot encode the geography
geography_df=pd.DataFrame([geography],columns=['Geography'])
geography_encoded=onehotencoder.transform(geography_df).toarray()
geography_encoded_df=pd.DataFrame(geography_encoded,columns=onehotencoder.get_feature_names_out(['Geography']))
# Concatenate the encoded geography with the input data
input_df=pd.concat([input_data.reset_index(drop=True),geography_encoded_df],axis=1)

# Scaling the input data
input_df=scaler.transform(input_df)
# Predict churn
prediction=model.predict(input_df)
churn_probability=prediction[0][0]
st.write(f"Churn Probability: {churn_probability:.2f}")
if churn_probability>0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
