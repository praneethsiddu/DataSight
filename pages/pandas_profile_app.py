# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:33:41 2021

@author: Partapu Praneeth
"""
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import pandas as pd
def app():
    st.write("Pandas Profiling Report")
    if st.button("click to generate"):
        #path=os.getcwd()
        # Load the data 
        if 'main_data.csv' not in os.listdir('data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            df = pd.read_csv('data/main_data.csv')
            st.header('**Input DataFrame**')
            st.write(df)
            pr = ProfileReport(df, explorative=True)
            st_profile_report(pr)
    
        
