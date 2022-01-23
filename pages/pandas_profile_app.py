# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:33:41 2021

@author: Partapu Praneeth
"""
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from pages import utils
import base64
import pandas as pd
def filedownload(path1):
    
    href = f'<a href="data:file/html" download={path1}>Download {path1} File</a>'
    return href
def app():
    st.write("Pandas Profiling Report")
    if st.button("click to generate"):
        path=os.getcwd()
        # Load the data 
        if 'main_data.csv' not in os.listdir(path+'/data'):
            st.markdown("Please upload data through `Upload Data` page!")
        else:
            df = pd.read_csv(path+'/data/main_data.csv')
            with st.spinner('Cooking Report'):
                report = ProfileReport(df)
                path1=path+'/data/output.html'
                report.to_file(output_file = path+'/data/output.html')
            st.success('Done!')
            st.write("You can view the report by downloading output.html file")
            #if st.button("Explore Columns"):
             #   x = st.selectbox('Select desired variable',df.columns)
              #  report.description_set['variables'][x] 
    
        
