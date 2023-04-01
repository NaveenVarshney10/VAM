import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import ydata_profiling
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from statsmodels.stats.diagnostic import het_white
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


st.title("Regression Analysis using Streamlit")

st.subheader("AN ILLUSTRATIVE WEBPAGE FOR CURIOUS PEOPLES :)")

st.write("It was created with the intention of helping non-programmers in performing Regression Analysis with the help of drag and drop facilities")

uploaded_file = st.file_uploader('Select Your data file (CSV)')



if uploaded_file is not None:
	data_file = pd.read_csv(uploaded_file)
	st.write("It was assumed that your data was pre-processed and ready to use. If that is not the case, please preprocess your data first.")

	if st.checkbox("Profiling",help="Click here to view full profiling of your dataset"):
		profile = ProfileReport(data_file, explorative=True)
		st_profile_report(profile)

	if st.checkbox("Head",help="Click here to view top 10 rows of your data"):
		st.write(data_file.head(10))	

	if st.checkbox("Summary",help = 'Click here to view quick summary of your data'):
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
				selected_y_var = st.selectbox('What about the y?',data_file.columns)
				fil = data_file.drop(selected_y_var,axis=1)
				selected_x_var = st.selectbox('What do want the x variable to be?',fil.columns)
				fig, ax = plt.subplots()
				ax = sns.scatterplot(x = data_file[selected_x_var],y = data_file[selected_y_var])
				plt.xlabel(selected_x_var)
				plt.ylabel(selected_y_var)
				plt.title('Scatterplot')
				st.pyplot(fig)

	if st.checkbox("Fit a Linear Regression"):
		y_var = st.selectbox('Select dependent variable (Y)',data_file.columns)
		col3 , col4 = st.columns(2)
		with col3:
			if st.checkbox("Simple linear regression"):
				
				fil = data_file.drop(y_var,axis=1)
				x_var = st.selectbox('Select independent variable (X)',fil.columns)
				y = data_file[y_var]	
				X = data_file[x_var]
				x=sm.add_constant(X)
				results = sm.OLS(y,x).fit()

				st.write("**"+"VALIDATING ASSUMPTIONS:"+"**")

				#LINEARITY
				st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Linearity", unsafe_allow_html=True)

				#INDEPENDENCE
				st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Independence of observation", unsafe_allow_html=True)
				
				#MULTICOLINEARITY
				st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Multicollinearity", unsafe_allow_html=True)

				count = 3

				residual = results.resid
					
				white_test = het_white(residual, results.model.exog)

				#st.write('White test statistic: ', white_test[0])
				#st.write('White test p-value: ', white_test[1])

				#HOMOSCEDASTICITY
				if white_test[1] > 0.05:
					count = count + 1
					st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Homoscedasticity", unsafe_allow_html=True)
				else:
					st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Homoscedasticity", unsafe_allow_html=True)

				#NORMALITY
				stat, p = shapiro(residual)

				#st.write('Shapiro test statistic: ', stat)
				#st.write('Shapiro test p-value: ', p)

				if p > 0.05:
					count = count + 1
					st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Normality", unsafe_allow_html=True)
				else:	
					st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Normality", unsafe_allow_html=True)
	
				#AUTOCORRELATION  
					
				dw = sm.stats.stattools.durbin_watson(results.resid)

				#st.write(dw)

				if dw <=3 and dw >= 1:
					count = count + 1
					st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Autocorrelation", unsafe_allow_html=True)
				else:
					st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Autocorrelation", unsafe_allow_html=True)
							
				
				st.write(results.summary())
				fig = sns.lmplot(x=x_var,y=y_var,data=data_file)
				st.subheader("Visual Representation of the fitted line")
				st.pyplot(fig)

				st.write(count)

				if (6-count) != 0 and (6-count) != 1:
					st.write("**"+"Word of Caution :"+"**")
					st.write("As "+ str(6-count) +" regression model assumptions are not statisfied. Therefore the provided results can lead to biased and unreliable estimates of the regression coefficients, and incorrect or misleading inferences about the relationship between the predictor variables and the response variable")
					st.write("**"+"Piece Of Advice :"+"**")
					st.write("To address these issues, it may be necessary to use alternative regression methods, such as \
							generalized linear models, robust regression methods, or machine learning algorithms that \
							can handle nonlinear relationships, heteroscedasticity, autocorrelation, multicollinearity, and outliers more effectively.\
							Alternatively, data transformations, outlier detection and removal, and other data cleaning techniques can be used to preprocess the data before fitting the regression model.") 
					
				elif (6-count) == 1:
					st.write("**"+"Word of Caution :"+"**")
					st.write("As "+str(6-count)+" regression model assumption is not statisfied. Therefore the provided results can lead to biased and unreliable estimates of the regression coefficients, and incorrect or misleading inferences about the relationship between the predictor variables and the response variable")
					st.write("**"+"Piece Of Advice :"+"**")
					st.write("To address that issues, it may be necessary to use alternative regression methods, such as \
							generalized linear models, robust regression methods, or machine learning algorithms that \
							can handle nonlinear relationships, heteroscedasticity, autocorrelation, multicollinearity, and outliers more effectively.\
							Alternatively, data transformations, outlier detection and removal, and other data cleaning techniques can be used to preprocess the data before fitting the regression model.") 
					

				else:
					st.write("BINGO :)")
	
					st.write("All assumptions are statisfied, therefore the provided results are relaible.")	
							

		with col4:
			if st.checkbox("Multiple Linear Regression"):
				fil = data_file.drop(y_var,axis=1)
				x_columns = st.multiselect("Select the independent variable(s) (X):",fil.columns.tolist())
				y =  data_file[y_var]
				X = data_file[x_columns]
				x= sm.add_constant(X)
				results = sm.OLS(y,x).fit()

				if len(x_columns) >= 2: 
					st.write("**"+"VALIDATING ASSUMPTIONS:"+"**")

					#LINEARITY
					st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Linearity", unsafe_allow_html=True)

					#INDEPENDENCE
					st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Independence of observation", unsafe_allow_html=True)
					
					residual = results.resid
					
					white_test = het_white(residual, results.model.exog)

					#st.write('White test statistic: ', white_test[0])
					#st.write('White test p-value: ', white_test[1])

					count = 2
					#HOMOSCEDASTICITY
					if white_test[1] > 0.05:
						count = count + 1
						st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Homoscedasticity", unsafe_allow_html=True)
					else:
						st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Homoscedasticity", unsafe_allow_html=True)

					#NORMALITY
					stat, p = shapiro(residual)

					#st.write('Shapiro test statistic: ', stat)
					#st.write('Shapiro test p-value: ', p)

					if p > 0.05:
						count = count + 1
						st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Normality", unsafe_allow_html=True)
					else:	
						st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Normality", unsafe_allow_html=True)

					#MULTICOLINEARITY
					VIF = pd.DataFrame()
					VIF['VIF'] = [vif(X.values,i) for i in range(X.shape[1])]
					VIF['FEATURES'] = X.columns	
					vif_values = VIF['VIF'].values
					#st.write(vif_values)
					bvif_values = pd.Series(list(vif_values < 5))


					if 	bvif_values.all():
						count = count + 1
						st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Multicollinearity", unsafe_allow_html=True)
					else: 
						st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Multicollinearity", unsafe_allow_html=True)
	
					#AUTOCORRELATION  
					
					dw = sm.stats.stattools.durbin_watson(results.resid)

					#st.write(dw)

					if dw <=3 and dw >= 1:
						count = count + 1 
						st.markdown("<span style='color:green;font-size:20px'>✓</span> " + "Autocorrelation", unsafe_allow_html=True)
					else:
						st.markdown("<span style='color:red;font-size:20px'>X</span> " + "Autocorrelation", unsafe_allow_html=True)
							
					
					st.write(results.summary())	

					st.write("\n")	
		

					if (6-count) != 0 and (6-count) != 1:
						st.write("**"+"Word of Caution :"+"**")
						st.write("As "+ str(6-count) +" regression model assumptions are not statisfied. Therefore the provided results can lead to biased and unreliable estimates of the regression coefficients, and incorrect or misleading inferences about the relationship between the predictor variables and the response variable")
						st.write("**"+"Piece Of Advice :"+"**")
						st.write("To address these issues, it may be necessary to use alternative regression methods, such as \
								generalized linear models, robust regression methods, or machine learning algorithms that \
								can handle nonlinear relationships, heteroscedasticity, autocorrelation, multicollinearity, and outliers more effectively.\
								Alternatively, data transformations, outlier detection and removal, and other data cleaning techniques can be used to preprocess the data before fitting the regression model.") 
					
					elif (6-count) == 1:
						st.write("**"+"Word of Caution :"+"**")
						st.write("As "+str(6-count)+" regression model assumption is not statisfied. Therefore the provided results can lead to biased and unreliable estimates of the regression coefficients, and incorrect or misleading inferences about the relationship between the predictor variables and the response variable")
						st.write("**"+"Piece Of Advice :"+"**")
						st.write("To address that issues, it may be necessary to use alternative regression methods, such as \
								generalized linear models, robust regression methods, or machine learning algorithms that \
								can handle nonlinear relationships, heteroscedasticity, autocorrelation, multicollinearity, and outliers more effectively.\
								Alternatively, data transformations, outlier detection and removal, and other data cleaning techniques can be used to preprocess the data before fitting the regression model.") 
					

					else:
						st.write("BINGO :)")
						st.write("\n")
						st.write("All assumptions are statisfied, therefore the provided results are relaible.")	




	
	

	
	

					
				



        

