# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 01:53:08 2021

@author: Partapu Praneeth
"""

import streamlit as st
text= """
A **big heartfelt Thanks** for your time in trying our tiny-simple-data visualization tool.<br>
<p>DataSight is an open-source web app designed via streamlit.<b>Its incredibly Simple - Just drag & drop any csv and explore the data visually.</b></span> By default the application will figure out better visualization based on user selections. \
Feel free to customize as needed. </p>
<b>Advantages</b>
<br>
- Simple and quick for any Exploratory Data Analysis.
<br>
- Create interactive Dashboard web app within minutes from any csv and share the findings
<br>
- Download your results with just single click
<br>
- Highly Customizable
<br>
- No more static reports<br>
"""  
def app():
    
    
    st.markdown("# Welcome.")
    
    st.markdown(text, unsafe_allow_html=True)
    #st.markdown("Please provide your valuable suggestions, feature requests, notifying issues in [github](https://github.com/Vinothsuku/vizdxp)", unsafe_allow_html=True)
    st.sidebar.markdown("")
    # st.markdown(slide_link, unsafe_allow_html=True)
    st.sidebar.markdown("")
    st.sidebar.markdown("<br><br><br> <br><br><br><br> <br><br><br><br><br> <br><br><br><br><br> ", unsafe_allow_html=True)
    st.sidebar.markdown("")
