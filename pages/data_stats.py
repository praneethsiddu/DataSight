# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:19:25 2021

@author: Partapu Praneeth
"""

import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
def app():
    #path = os.getcwd()
    #st.write(path)
    if 'main_data.csv' not in os.listdir("data"):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        
        df = pd.read_csv('data/main_data.csv')
        st.subheader("Exploratory Data Analysis")
        with st.expander("Check Data"):
            select_ = st.radio("HEAD OR TAIL",('All','HEAD','TAIL'))
            if select_ == 'All':
                st.dataframe(df)
            elif select_ == 'HEAD':
                st.dataframe(df.head())
            elif select_ == 'TAIL':
	            st.dataframe(df.tail())
        with st.expander("Check Columns"):
            
            select_ = st.radio("Select Columns",('All Columns','Specific Column'))
            if select_ == "All Columns":
                st.write(df.columns)
            if select_ == "Specific Column":
                col_spe = st.multiselect("Select Columns To Show",df.columns)
                st.write(df[col_spe])
        with st.expander("Check Dimentions"):
            select_ = st.radio('Select Dimension',('All','Row','Column'))
            if select_ == "All":
                st.write(df.shape)
            elif select_ == "Row":
                st.write(df.shape[0])
            elif select_ == "Column":
                st.write(df.shape[1])
        with st.expander("Check Summary"):
            st.write(df.describe())
        with st.expander("Value Counts"):
            select_ = st.multiselect("Select values",df.columns.tolist())
            st.write(df[select_].count())
        with st.expander("Check Data Types"):
            
            df1=pd.read_csv('data/column_type_desc.csv')
            st.dataframe(df1)
            
                
    	
    
    
