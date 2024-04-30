import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('XGB_churn.pkl')
scaler = joblib.load('StandardScaler.pkl')


def main():
    st.title('🏦 Churn Model Deployment')
    st.write("Created by Andrew Jonathan")
    
    st.header('Enter Customer Information')
    
    name = st.text_input("Name: ")
    gender = st.radio("👫 Gender:", ["Male", "Female"])
    geography = st.selectbox("🗺️ Geography:", ["Germany", "France", "Spain"])
    creditscore = st.number_input("💳 Credit Score:", 300, 900)
    age = st.number_input("Age:", 0, 100)
    tenure = st.number_input("⏳ Tenure", 0, 100)
    balance = st.number_input("💰 Balance:", 0, 999999)
    numofproducts = st.number_input("📦 Number of Products:", 0, 10)
    hascrcard = st.radio("💳 Has credit card?", ["Yes", "No"])
    isactivemember = st.radio("Active member?", ["Yes", "No"])
    esitmatedsalary= st.number_input("💵 Salary:", 0, 1000000000)
    
    
    data = {'Age': int(age), 'Gender': gender, 'Geography': geography, 'CreditScore':creditscore,'Tenure':int(tenure), 'Balance':int(balance),
            'NumOfProducts':int(numofproducts), 'HasCrCard': hascrcard,
            'isActiveMember': isactivemember, 'EstimatedSalary': int(esitmatedsalary)}
    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore','Geography','Gender', 'Age','Tenure', 'Balance',
                                                    'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary'])

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['IsActiveMember'] = df['IsActiveMember'].map({'Yes': 1, 'No': 0})
    df['HasCrCard'] = df['HasCrCard'].map({'Yes': 1, 'No': 0})
    df['Geography'] = df['Geography'].map({'Spain': 0, 'France': 1, 'Germany': 2})
    
    df_scale = scaler.transform(df)
    df = pd.DataFrame(df_scale,columns=df.columns)
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'{name} is predicted: {result}')
    

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Not Churned" if prediction[1] == 0 else "Churned"

if __name__ == '__main__':
    main()
