# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:42:15 2022

@author: Partapu Praneeth
"""

import json
from joblib import load

import pandas as pd
import streamlit as st

import numpy as np

# Custom classes 
from .utils import isNumerical
import os
def app():
    
    f = open('data/metadata/model_params.json')
    data = json.load(f)
    st.write(data['y'])
    l=len(data['X'])
    if 'user_x' not in st.session_state:
        st.session_state.user_x=[0]*l
    il=data['X']
    col=st.selectbox('Select The column',data['X'])
    z=st.number_input('Enter vaule Of {col}')
    index=data['X'].index(col)
    if st.button("submit"):
        st.session_state.user_x[index]=z
    st.write(pd.DataFrame(st.session_state.user_x,index=data['X']))
    rr=load('{pred_type}.joblib')
    if st.button("Predict"):
        data = np.array([st.session_state.user_x])
        st.write(rr.predict(data))
