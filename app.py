# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:02:13 2021

@author: Partapu Praneeth
"""

import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload,AandG,data_stats,data_visualization,AutoML,pandas_profile_app,welcome
st.set_page_config(
        page_title="DataSight",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

# Create an instance of the app 
app = MultiPage()


# Add all your application here
app.add_page("Get Started", welcome.app)
app.add_page("Upload Data", data_upload.app)
app.add_page("Data Stats", data_stats.app)
app.add_page("Aggregation and flitering", AandG.app)
app.add_page("Data Visualization",data_visualization.app)
app.add_page("pandas profile report", pandas_profile_app.app)
app.add_page("AutoMl", AutoML.app)


# The main app
app.run()
