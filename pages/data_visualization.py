# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:45:52 2021

@author: Partapu Praneeth
"""

import streamlit as st
import base64
import io
#import plotly_express as px
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from pages import utils
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href
def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href
def app():
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    #path=os.getcwd()
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df1 = pd.read_csv('data/main_data.csv')
        # df_visual = pd.DataFrame(df_analysis)
        df_visual = df1.copy()
        with st.expander("Categorial Data Plots"):
            select_ = st.radio("Select Type for Categorial Analysis",('None','count plot','box plot','violin plot','bar chart','Histogram'))
            if select_ == "count plot":
                
                s = st.selectbox('select the column',df1.columns,key=1)
                ax = sns.countplot(df1[s])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
                st.write(sns.countplot(df1[s]))
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'count_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("**Text area for marking observations**",placeholder="Notes",value="None",key=1)
                if str1 != "None":
                    st.session_state.insights.append(str1)
                    
                
            if select_=="bar chart":
               # st.write(df1.dtypes)
                s = st.multiselect("Select Columns To Show",df1.columns)
                st.bar_chart(df1[s])
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=2)
                if str1 != "None":
                    st.session_state.insights.append(str1)
            if select_ == 'box plot':
                #t.write(df1.dtypes)
                x = st.selectbox('Select X Column',df1.columns,key=2)
                y = st.selectbox('Select Y Column',df1.columns,key=3)
                st.write(x,y)
                plt.figure(figsize=(10, 7))
                #sns.set_theme(style="whitegrid")
                ax3 = sns.boxplot(x,y,data=df1,palette='rainbow')
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'box_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=4)
                if str1 != "None":
                    st.session_state.insights.append(str1)
            if select_ == 'violin plot':
                #t.write(df1.dtypes)
                x = st.selectbox('Select X Column',df1.columns,key=4)
                y = st.selectbox('Select Y Column',df1.columns,key=5)
                st.write(x,y)
                plt.figure(figsize=(10, 7))
                #sns.set_theme(style="whitegrid")
                ax3 = sns.violinplot(x,y,data=df1,palette='rainbow')
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'violin.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=42)
                if str1 != "None":
                    st.session_state.insights.append(str1)
                
            if select_ == "Histogram":
                #st.write(df1.dtypes)
                x = st.selectbox('Select Numerical Variables',df1.columns,key=66)
                plt.figure(figsize=(12, 8))
                #sns.set_theme(style="whitegrid")
                ax3 =sns.distplot(df1[x])
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'Histrogram.pdf'), unsafe_allow_html=True)
                
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=3)
                st.write(str1)
                if str1 != "None":
                    st.write(str1)
                    st.session_state.insights.append(str1)
                
        with st.expander("Regression Plots"):
            select_ = st.radio("Select Type for Regression Analysis",('None','area chart','line chart','lmplot','scatter plot'))
            if select_ == 'scatter plot':
                #t.write(df1.dtypes)
                x = st.selectbox('Select X Column',df1.columns,key=79)
                y = st.selectbox('Select Y Column',df1.columns,key=89)
                st.write(x,y)
                plt.figure(figsize=(10, 7))
                #sns.set_theme(style="whitegrid")
                ax3 = sns.scatterplot(x,y,data=df1)
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'scatter_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=41)
                if str1 != "None":
                    st.session_state.insights.append(str1)
            if select_ == 'lmplot':
                #t.write(df1.dtypes)
                x = st.selectbox('Select X Column',df1.columns,key=9)
                y = st.selectbox('Select Y Column',df1.columns,key=10)
                cols = pd.read_csv('data/column_type_desc.csv')
                
                Categorical,Numerical,Object = utils.getColumnTypes(cols)
                hue = st.selectbox('Select hue parameter',Categorical+Object)
                st.write(x,y)
                plt.figure(figsize=(10, 7))
                #sns.set_theme(style="whitegrid")
                ax3 = sns.lmplot(x,y,data=df1,palette='rainbow',hue=hue)
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'lmplot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=43)
                if str1 != "None":
                    st.session_state.insights.append(str1)
                
            if select_=="area chart":
                #st.write(df1.dtypes)
                s = st.multiselect("Select Columns To Show",df1.columns,key=11)
                st.area_chart(df1[s])
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=51)
                if str1 != "None":
                    st.session_state.insights.append(str1)
                
            if select_ == "line chart":
                
                s = st.multiselect("Select Columns To Show",df1.columns,key=12)
                st.line_chart(df1[s])
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=6)
                if str1 != "None":
                    st.session_state.insights.append(str1)
                
        with st.expander("Matrix Plots"):
            select_ = st.radio("Select Type for Matrix Analysis",('None','Correlation Heatmap','Yet to Impelement'))
            if select_=='Correlation Heatmap':
                plt.figure(figsize=(12, 8))
                #sns.set_theme(style="whitegrid")
                ax3 =sns.heatmap(df1.corr())
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'correlation_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=7)
                if str1 != "None":
                    st.session_state.insights.append(str1)
            '''if select_=='clustermap':
                plt.figure(figsize=(12, 8))
                #sns.set_theme(style="whitegrid")
                ax3 =sns.clustermap(df1)
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'clustermap.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=7)
                if str1 != "None":
                    st.session_state.insights.append(str1)'''
                
        with st.expander("Distrubtion plots"):
            select_ = st.radio("Select Type for Matrix Analysis",('None','Rug Plot','Dist Plot','Pair Plot','Joint Plot','yet to implement'))
            if select_=="Pair Plot":
                  plt.figure(figsize=(12, 8))
                  #sns.set_theme(style="whitegrid")
                  ax3 =sns.pairplot(df1)
                  plt.xticks(rotation=90)
                  st.pyplot(plt)
                  st.markdown(imagedownload(plt,'pair_plot.pdf'), unsafe_allow_html=True)
                  str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=8)
                  
                  if str1 != "None":
                      st.session_state.insights.append(str1)
            if select_=="Dist Plot":
                s = st.multiselect("Select Columns To Show",df1.columns,key=13)
                plt.figure(figsize=(12, 8))
                #sns.set_theme(style="whitegrid")
                ax3 =sns.distplot(df1[s])
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'dist_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=81)
                
                if str1 != "None":
                    st.session_state.insights.append(str1)
            if select_ == 'Joint Plot':
                #t.write(df1.dtypes)
                x = st.selectbox('Select X Column',df1.columns,key=14)
                y = st.selectbox('Select Y Column',df1.columns,key=15)
                st.write(x,y)
                kind = st.selectbox('Select Kind parameter to compare',["scatter","reg","resid","kde","hex"])
                plt.figure(figsize=(10, 7))
                #sns.set_theme(style="whitegrid")
                ax3 = sns.jointplot(x,y,data=df1,kind=kind)
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'joint_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=47)
                if str1 != "None":
                    st.session_state.insights.append(str1)
            if select_=="Rug Plot":
                s = st.multiselect("Select Columns To Show",df1.columns)
                plt.figure(figsize=(12, 8))
                #sns.set_theme(style="whitegrid")
                ax3 =sns.rugplot(data=df1,x=s[0])
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'rug_plot.pdf'), unsafe_allow_html=True)
                str1 = st.text_area("Text area for marking observations",placeholder="Notes",value="None",key=87)
                
                if str1 != "None":
                    st.session_state.insights.append(str1)
                
        if st.button("Your Insights"):
            st.write(st.session_state.insights)
    
            
            
        ''' cols = pd.read_csv('column_type_desc.csv')
        Categorical,Numerical,Object = utils.getColumnTypes(cols)
        cat_groups = {}
        unique_Category_val={}

        for i in range(len(Categorical)):
                unique_Category_val = {Categorical[i]: utils.mapunique(df1, Categorical[i])}
                cat_groups = {Categorical[i]: df_visual.groupby(Categorical[i])}
                '''
        
       # category = st.selectbox("Select Category ", Categorical + Object)

        #sizes = (df_visual[category].value_counts()/df_visual[category].count())

        #labels = sizes.keys()

        #maxIndex = np.argmax(np.array(sizes))
        #explode = [0]*len(labels)
        #explode[int(maxIndex)] = 0.1
        #explode = tuple(explode)
        
        #fig1, ax1 = plt.subplots()
        #ax1.pie(sizes,explode = explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=0)
        #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ##ax1.set_title('Distribution for Categorical Column - ' + (str)(category))
        #st.pyplot(fig1)
        
       # corr = df1.corr(method='pearson')
        
        #fig2, ax2 = plt.subplots()
        #mask = np.zeros_like(corr, dtype=np.bool)
        #mask[np.triu_indices_from(mask)] = True
        # Colors
        #cmap = sns.diverging_palette(240, 10, as_cmap=True)
        #sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
        #ax2.set_title("Correlation Matrix")
        #st.pyplot(fig2)
        
        
        #categoryObject=st.selectbox("Select " + (str)(category),unique_Category_val[category])
        #st.write(cat_groups[category].get_group(categoryObject).describe())
        #colName = st.selectbox("Select Column ",Numerical)

        #st.bar_chart(cat_groups[category].get_group(categoryObject)[colName])
        
