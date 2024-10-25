
import pandas as pd
import numpy as np
# import seaborn as sns
#import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# %pip install h5py graphviz pydot
import numpy as np


data = pd.read_csv('Loan_Data.csv')
ds = data.head(6)
data.head()
df = data.copy()

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('Loan.h5')

st.sidebar.image('cheerful-african-guy-with-narrow-dark-eyes-fluffy-hair-dressed-elegant-white-shirt.jpg', width = 200)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Loan Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>Loan Approval application helps predict the approval of a loan based on several key factors related to the applicant's background and financial situation. This project uses a machine learning model trained on historical loan data to assist users in determining whether a loan will likely be approved.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('portrait-woman-surrounded-by-money.jpg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)


# sel = ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome',
#        'Dependents', 'Property_Area',
#        'Credit_History', 'Loan_Amount_Term', 'Education', 'Married', 'Self_Employed']
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>ApplicantIncome</h3>", unsafe_allow_html=True)
    st.markdown("<p>The applicant’s monthly income. Higher incomes generally support better repayment capacity, increasing the chances of loan approval.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>LoanAmount</h3>", unsafe_allow_html=True)
    st.markdown("<p>The number of months for loan repayment. Longer terms mean smaller monthly payments, which might make loan approval easier for lower-income applicants.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>CoapplicantIncome</h3>", unsafe_allow_html=True)
    st.markdown("<p>The income of any co-applicant on the loan. A higher combined income (applicant + coapplicant) can boost the overall loan eligibility by strengthening the repayment capacityp>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'> Dependents </h3>", unsafe_allow_html=True)
    st.markdown("<p>Shows the number of dependents. More dependents often imply higher expenses, potentially reducing the loan repayment capacity.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Property_Area</h3>", unsafe_allow_html=True)
    st.markdown("<p>The area where the applicant's property is located (Urban, Rural, or Semiurban). Certain areas might show lower or higher default risks, which can influence loan decisions.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Credit_History</h3>", unsafe_allow_html=True)
    st.markdown("<p>Shows whether the applicant has a credit history (1 for yes, 0 for no). A positive credit history (1) indicates responsible financial behavior and significantly improves loan eligibility.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Loan_Amount_Term</h3>", unsafe_allow_html=True)
    st.markdown("<p>The number of months for loan repayment. Longer terms mean smaller monthly payments, which might make loan approval easier for lower-income applicants.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Education</h3>", unsafe_allow_html=True)
    st.markdown("<p>If the applicant is a graduate or not. Higher education often correlates with higher-paying jobs, which can increase the likelihood of loan approval.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Married</h3>", unsafe_allow_html=True)
    st.markdown("<p>If the applicant is married. Being married might suggest dual income sources, which can positively influence the loan eligibility assessment.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Self_Employed</h3>", unsafe_allow_html=True)
    st.markdown("<p>If the applicant is self-employed. Employment stability (often associated with non-self-employed individuals) can affect loan eligibility as self-employment income may fluctuate.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


  
    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Modelled with ❤️ by Oluwaseunfunmi </p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: CENTER; color: #2B2A4C;'>Modelling Section </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(ds[['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome',      'Dependents', 'Property_Area', 'Credit_History', 'Loan_Amount_Term', 'Education', 'Married', 'Self_Employed']])

# 'CLIENTNUM', 'ApplicantIncome', 'Dependent_count',
#        'CoapplicantIncome', 'Dependents', 'Property_Area',
#        'Credit_History', 'Loan_Amount_Term', 'Education',
#        'Married', 'Self_Employed', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'

if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    st.sidebar.markdown("Add your input here")
    ApplicantIncome = st.sidebar.number_input("ApplicantIncome",0,100000)
    LoanAmount = st.sidebar.number_input("LoanAmount",0,1000)
    CoapplicantIncome = st.sidebar.number_input("CoapplicantIncome",0,10000)
    Dependents = st.sidebar.selectbox("Dependents", [0,1,2,3])
    Property_Area = st.sidebar.selectbox("Property_Area", ["Urban", "Rural", "Semiurban"])
    Credit_History = st.sidebar.selectbox("Credit_History", [0,1])
    Loan_Amount_Term = st.sidebar.number_input("Loan_Amount_Term",0,1000)
    Education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
    Married = st.sidebar.selectbox("Married", ['Yes', 'No'])
    Self_Employed = st.sidebar.selectbox("Self_Employed", ['Yes', 'No'])


    input_variables = pd.DataFrame([{
        'ApplicantIncome': ApplicantIncome,
        'LoanAmount':LoanAmount,
        'CoapplicantIncome': CoapplicantIncome, 
        'Dependents': Dependents,
        'Property_Area': Property_Area,
        'Credit_History': Credit_History,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Education': Education,
        'Married': Married,
        'Self_Employed': Self_Employed
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)

    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Input Your Name</h2>", unsafe_allow_html=True)
    customer_name = st.text_input("")

    if customer_name:
            if st.button('Press To Predict'):
                st.markdown("<h4 style='color: #2B2A4C; text-align: left; font-family: montserrat;'>Model Report</h4>", unsafe_allow_html=True)
                print(input_variables)
                input_variables['Property_Area'] = input_variables['Property_Area'].map({'Urban': 0, 'Rural': 1, 'Semiurban': 2})
                input_variables['Education'] = input_variables['Education'].map({'Graduate': 0, 'Not Graduate': 1})
                input_variables['Married'] = input_variables['Married'].map({'Yes': 0, 'No': 1})
                input_variables['Self_Employed'] = input_variables['Self_Employed'].map({'Yes': 0, 'No': 1})
                input_variables = input_variables.astype('float32')
                predicted = model.predict(input_variables)
                st.toast('Predicted Successfully')
                st.image('Check.jpeg', width=100)
                if predicted >= 0.5:
                    st.error(f"OOpppss!! {customer_name} you are not Eligible for the Loan")
                else:
                    st.success(f"Congratulation!! {customer_name} you are eligible for the Loan")
    else:
            st.warning("Please enter the customer's name.")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>Loan Prediction MODEL BUILT BY Oluwaseunfunmi</h8>",unsafe_allow_html=True)

