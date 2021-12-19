# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:38:01 2021

@author: Partapu Praneeth
"""

import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import os
def app():
    path = os.getcwd()
    st.write(path)
    if 'main_data.csv' not in os.listdir(path+"\data"):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv(path+'\data\main_data.csv')
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
        gridOptions = gb.build()
        AgGrid(data,gridOptions=gridOptions,enable_enterprise_modules=True,allow_unsafe_jscode=True)