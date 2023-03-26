import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.title("Regression Analysis using Streamlit")

st.subheader("AN ILLUSTRATIVE WEBPAGE FOR CURIOUS PEOPLES :)")

st.write("It was created with the intention of helping non-programmers in performing Regression Analysis with the help of drag and drop facilities")

uploaded_file = st.file_uploader('Select Your data file (CSV)')



if uploaded_file is not None:
	data_file = pd.read_csv(uploaded_file)

	if st.checkbox("Profiling",help="Click here to view full profiling of your dataset"):
		profile = ProfileReport(data_file, explorative=True)
		st_profile_report(profile)

	if st.checkbox("Summary",help = 'Click here to view quick summary of your data'):
		st.write("Summary: ")
		st.write(data_file.describe(include='all'))
	

	if st.checkbox("Visualization"):

		col1, col2 = st.columns(2)
		with col1:
			if st.checkbox("Single Variable",help = 'Click here to view histogram of individual variable'):
				var = st.selectbox('Select Variable',data_file.columns)
				fig, ax = plt.subplots()
				plt.hist(data_file[var])
				plt.xlabel(var)	
				plt.ylabel("frequency")
				st.pyplot(fig)
		with col2:
			if st.checkbox("Bi-Variables",help = 'Click here to view scatterplot between variables'):
				selected_x_var = st.selectbox('What do want the x variable to be?',data_file.columns)
				selected_y_var = st.selectbox('What about the y?',data_file.columns)
				fig, ax = plt.subplots()
				ax = sns.scatterplot(x = data_file[selected_x_var],y = data_file[selected_y_var])
				plt.xlabel(selected_x_var)
				plt.ylabel(selected_y_var)
				plt.title('Scatterplot')
				st.pyplot(fig)

	if st.checkbox("Fit a Linear Regression"):
		col3 , col4 = st.columns(2)
		with col3:
			if st.checkbox("Simple linear regression"):
				y_var = st.selectbox('Select dependent variable (Y)',data_file.columns)
				x_var = st.selectbox('Select independent variable (X)',data_file.columns)
				y = data_file[y_var]	
				X = data_file[x_var]
				x=sm.add_constant(X)
				results = sm.OLS(y,x).fit()
				st.write(results.summary())
		
		with col4:
			if st.checkbox("Multiple Linear Regression"):
				

				x_columns = st.multiselect("Select the independent variable(s) (X):", data_file.columns.tolist())
				
	
	

	
	

					
				



        

