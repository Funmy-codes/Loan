
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# %pip install h5py graphviz pydot


data = pd.read_csv('BankChurners.csv')
ds = data.head(6)
data.head()
df = data.copy()

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('BankChurners.h5')

st.sidebar.image('C.png', width = 200)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Bank Churn Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>Customer churn is a significant challenge for banks, as acquiring new customers is often more expensive than retaining existing ones. This project focuses on creating a machine learning model that analyzes customer behavior and predicts the likelihood of churn. The model will use various features such as customer demographics, account balance, transaction history, credit score, tenure with the bank, product usage, and other relevant financial indicators to make these predictions.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('woman-interacting-with-money.jpg',  width = 700)
    st.markdown('<br>', unsafe_allow_html= True)



    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)


    # 'CLIENTNUM', 'Customer_Age', 'Dependent_count',
#        'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Months_on_book',
#        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Customer_Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>Younger customers (e.g., those in their 20s or 30s) might have different banking needs compared to older customers (e.g., those in their 50s or 60s). For instance, younger customers may be more focused on loans or credit-building products, while older customers may prioritize savings, retirement plans, or high credit limits.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Dependent_count</h3>", unsafe_allow_html=True)
    st.markdown("<p>The number of dependents may influence a customer’s likelihood to churn. Customers with many dependents might be more loyal to their bank if they have favorable terms, or they might churn if they find better financial services that help them manage their responsibilities more effectively.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Relationship_Count</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers with a lower Total_Relationship_Count (i.e., those with only one or two products) may be more at risk of attrition, as they have fewer dependencies on the bank and could more easily switch to a competitor. On the other hand, customers with more products are likely to stay, given the complexity of moving all their financial relationships elsewhere.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'> Months_Inactive_12_mon </h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers who have been inactive for several months are at a higher risk of attrition. This can serve as an early warning sign for the bank, allowing them to proactively reach out to these customers with personalized offers, incentives, or engagement strategies to prevent churn.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Months_on_book</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers who have been with the bank for many months or years are generally less likely to churn compared to newer customers. Conversely, newer customers might have a higher risk of attrition, especially if they have not yet developed a strong relationship with the bank.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Contacts_Count_12_mon</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers who have had fewer contacts may be at a higher risk of attrition. Lack of engagement could signify dissatisfaction, confusion about services, or simply a lack of need for banking products. Monitoring contact frequency can help identify potential churn risks early.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Credit_Limit</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers with high credit limits may exhibit different behaviors compared to those with lower limits. For instance, a significant reduction in a customer's credit limit could lead to increased dissatisfaction and a higher risk of churn. Conversely, customers with limited credit may feel constrained and more inclined to seek better options elsewhere.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Revolving_Bal</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers with high revolving balances may be at greater risk of attrition if they face financial challenges or feel overwhelmed by debt. Conversely, customers with lower balances relative to their limits may be more financially stable and engaged with the bank.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Avg_Open_To_Buy</h3>", unsafe_allow_html=True)
    st.markdown("<p>Customers with low open-to-buy amounts may feel financially constrained and could be more likely to switch to another bank or credit provider offering better terms or higher credit limits. Monitoring this metric can help banks identify customers who might be at risk of attrition.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Amt_Chng_Q4_Q1</h3>", unsafe_allow_html=True)
    st.markdown("<p>Significant fluctuations in spending may correlate with customer satisfaction and engagement. For instance, a sharp decline in spending could indicate that a customer is dissatisfied with the bank's services or that they have found alternatives, increasing the likelihood of churn..</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Trans_Amt</h3>", unsafe_allow_html=True)
    st.markdown("<p>Changes in total transaction amounts can serve as indicators of customer satisfaction and potential churn risk. A significant decline in transaction amounts may suggest disengagement, dissatisfaction, or financial difficulties, prompting the bank to take proactive measures to retain the customer..</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)


    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Trans_Ct</h3>", unsafe_allow_html=True)
    st.markdown("<p>A significant decline in the total number of transactions can signal a potential risk of churn. If a customer is not engaging with their accounts regularly, it may indicate dissatisfaction or that they are considering switching to another bank..</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Total_Ct_Chng_Q4_Q1</h3>", unsafe_allow_html=True)
    st.markdown("<p>Significant declines in transaction counts can serve as early warning signals for potential churn. If a customer who previously engaged actively shows a notable drop in transactions, it may suggest dissatisfaction or disengagement, prompting the bank to take proactive measures.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)

    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Avg_Utilization_Ratio</h3>", unsafe_allow_html=True)
    st.markdown("<p>High utilization ratios may correlate with financial distress or dissatisfaction with banking services, increasing the risk of customer churn. Monitoring changes in utilization ratios can help banks identify customers who may be struggling financially and might benefit from targeted outreach or financial counseling.</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
  
    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by Oluwaseunfunmi </p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: CENTER; color: #2B2A4C;'>Modelling Section </h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(ds[['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 
                    'Months_on_book', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 
                    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']])

# 'CLIENTNUM', 'Customer_Age', 'Dependent_count',
#        'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Months_on_book',
#        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
#        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
#        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'

if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    st.sidebar.markdown("Add your input here")
    Customer_Age = st.sidebar.number_input("Customer_Age",0,1000)
    Dependent_count = st.sidebar.selectbox("Dependent_count", df['Dependent_count'].unique())
    Total_Relationship_Count = st.sidebar.selectbox("Total_Relationship_Count", df['Total_Relationship_Count'].unique())
    Months_Inactive_12_mon = st.sidebar.number_input("Months_Inactive_12_mon",0,1000)
    Months_on_book = st.sidebar.number_input("Months_on_book",0,1000)
    Contacts_Count_12_mon = st.sidebar.number_input("Contacts_Count_12_mon",0,1000)
    Credit_Limit = st.sidebar.number_input("Credit_Limit",0,100000000)
    Total_Revolving_Bal = st.sidebar.number_input("Total_Revolving_Bal",0,100000)
    Avg_Open_To_Buy = st.sidebar.number_input("Avg_Open_To_Buy",0,1000000)
    Total_Amt_Chng_Q4_Q1 = st.sidebar.number_input("Total_Amt_Chng_Q4_Q1", 0.0, 100.0,format="%.1f")
    Total_Trans_Amt = st.sidebar.number_input("Total_Trans_Amt",0,100000)
    Total_Trans_Ct = st.sidebar.number_input("Total_Trans_Ct",0,1000)
    Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input("Total_Ct_Chng_Q4_Q1", 0.0, 100.0,format="%.1f")
    Avg_Utilization_Ratio = st.sidebar.number_input("Avg_Utilization_Ratio", 0.0, 100.0,format="%.1f")
    
   

    input_variables = pd.DataFrame([{
        'Customer_Age': Customer_Age,
        'Dependent_count':Dependent_count,
        'Total_Relationship_Count': Total_Relationship_Count, 
        'Months_Inactive_12_mon': Months_Inactive_12_mon,
        'Months_on_book': Months_on_book,
        'Contacts_Count_12_mon': Contacts_Count_12_mon,
        'Credit_Limit': Credit_Limit,
        'Total_Revolving_Bal': Total_Revolving_Bal,
        'Avg_Open_To_Buy': Avg_Open_To_Buy,
        'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt': Total_Trans_Amt,
        'Total_Trans_Ct': Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio': Avg_Utilization_Ratio
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)

    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Input Patient Name</h2>", unsafe_allow_html=True)
    customer_name = st.text_input("")

    if customer_name:
            if st.button('Press To Predict'):
                st.markdown("<h4 style='color: #2B2A4C; text-align: left; font-family: montserrat;'>Model Report</h4>", unsafe_allow_html=True)
                predicted = model.predict(input_variables)
                st.toast('Predicted Successfully')
                st.image('Check.jpeg', width=100)
                if predicted >= 0.5:
                    st.error(f"It is likely {customer_name} leaves")
                else:
                    st.success(f"It is likely {customer_name} stays")
    else:
            st.warning("Please enter the customer's name.")
    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>BankChurners Prediction MODEL BUILT BY Oluwaseunfunmi</h8>",unsafe_allow_html=True)