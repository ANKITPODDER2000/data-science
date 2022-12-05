import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from sklearn.model_selection import train_test_split
sns.set_style('darkgrid')

def is_need_to_build_model(site_param):
    if st.session_state['param'] != site_param:
        st.session_state['model_created'] = False

data = pd.read_csv('./Salary_Data.csv')
df_max = data.abs().max()
df = data / df_max

st.title("Experience vs Salary")

navigation_opt = st.sidebar.radio("Navigation", ["View Dataset", "Create Own model"], index = 0)

if navigation_opt == "View Dataset":
    st.subheader("Data Frame")
    st.dataframe(data.head(), width = 800)

    st.subheader("Scatter plot of Data")
    st.pyplot(sns.jointplot(x = 'YearsExperience', y = 'Salary', data = data))

elif navigation_opt == "Create Own model":
    split = st.slider("Test data size", min_value = 0.0, max_value = 1.0, value = 0.25, step = 0.05)
    lr = st.slider("Learning Rate", min_value = 0.0, max_value = 2.0, value = 0.01, step = 0.001)
    epoch = st.slider("No of Epochs", min_value = 100, max_value = 10000, value = 1000, step = 100)
    
    site_param = {
        'split' : split,
        'lr' : lr,
        'epoch' : epoch
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['YearsExperience'], 
        df['Salary'], 
        test_size= split,
        random_state=42
    )
    
    if 'model_created' not in st.session_state:
        st.session_state['model_created'] = False
        st.session_state['param'] = None
        
    is_need_to_build_model(site_param)
    
    if st.session_state['model_created'] or st.button('Build the Model'):
        training_msg = st.text("Model training started")
        history = None
        if st.session_state['model_created'] == False:
            st.session_state['model_created'] = True
            st.session_state['param'] = {
                'split' : split,
                'lr' : lr,
                'epoch' : epoch
            }
            placeholder2 = st.empty()
            my_bar = placeholder2.progress(0)
            history = train_model(0, 0, lr, epoch, X_train, y_train, X_test, y_test, my_bar)
            st.session_state['model'] = history
            placeholder2.empty()
        
        opt = st.selectbox("Select the option", ["View training history", "Predict Salary"])
        
        
        training_msg.text('Model building is completed')
        
        if opt == 'View training history':
            plot_df = pd.DataFrame()
            plot_df['train_loss'] = history['train_loss']
            plot_df['test_loss'] = history['test_loss']

            st.line_chart(plot_df)
        elif opt == 'Predict Salary':
            exp = st.number_input('Insert experience in Year', min_value = 0.5, max_value = 50.0, step = 0.5)
            pre_price = hypothesis(
                st.session_state['model']['theta0'],
                st.session_state['model']['theta0'],
                exp / df_max['YearsExperience']
            ) * df_max['Salary']
            st.write(f"Predicted Salary : **{pre_price}**")
            
        
        
        
        