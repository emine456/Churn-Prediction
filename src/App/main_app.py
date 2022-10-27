import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras as keras
import base64



# Title of the application
st.title('Application that predict customer churn')

# Loading our machine learning model

with open('../../models/RandomForestClassifier2_Model_Pickle','rb') as f:

	model = pickle.load(f)

# Loading deep learning model
model_deep_learning = keras.models.load_model('../../models/Deep_Learning_model')


# Create a slidebar
model_selectbox = st.sidebar.selectbox(
"Choose your model",
("Machine Learning model", "Deep Learing model"))

# Machine Learning model

if model_selectbox == "Machine Learning model":

	# Function that makes the prediction
	def predict(model,input_df):
		prediction_proba = model.predict_proba(input_df)
		prediction = 0
		for pred in prediction_proba[:,1]:
			if pred >= 0.52:
				prediction=1
			elif pred < 0.52:
				prediction=0
		return prediction

	# Create a slidebar
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict customer churn')

	if add_selectbox == 'Online':
		
		# Customer caracteristics
		credit_score = st.number_input('Credit Score', min_value=1, max_value=3000, value=82, step=1)
		age = st.number_input('Age', min_value=1, max_value=120, value=30, step=1)
		products_number = st.selectbox("Products_number",(1,2,3))
		active_member = st.selectbox("Active_member",(0,1))
		balance = st.number_input("Balance")
		estimated_salary = st.number_input("Estimated_salary")
		country = st.selectbox("Country",('France', 'Spain', 'Germany'))
		gender = st.selectbox('Gender',('Female','Male'))
		tenure = st.selectbox('Tenure',('Old_client', 'Medium_client', 'New_client'))
		credit_card = st.selectbox('Credit_card',(0,1))
		output=""
		output1=""

		# Create a user define dataframe
		user_df_data = [[credit_score,age,products_number,active_member,balance,estimated_salary,country,gender,tenure,credit_card]]
		user_df_colnames = ["credit_score","age","products_number","active_member","balance","estimated_salary","country","gender","tenure","credit_card"]
		input_df = pd.DataFrame(user_df_data,columns=user_df_colnames)
		
		# Standardize continious feature
		encoder = StandardScaler()
		input_df["credit_score"] = encoder.fit_transform(input_df["credit_score"].to_numpy().reshape(-1,1))
		input_df["balance"] = encoder.fit_transform(input_df["balance"].to_numpy().reshape(-1,1))
		input_df["estimated_salary"] = encoder.fit_transform(input_df["estimated_salary"].to_numpy().reshape(-1,1))

		# LabelEncoder for categorical features
		label_encoder = LabelEncoder()
		input_df["country"] = label_encoder.fit_transform(input_df["country"].to_numpy().reshape(-1,1))
		input_df["gender"] = label_encoder.fit_transform(input_df["gender"].to_numpy().reshape(-1,1))
		input_df["tenure"] = label_encoder.fit_transform(input_df["tenure"].to_numpy().reshape(-1,1))

		if st.button("Predict"):
			output = predict(model=model, input_df=input_df)
			output_dict = {1 : 'Churn', 0 : 'No Churn'}
			final_label = ""
			final_label = np.where(output == 1, 'Churn',np.where(output ==0,"No Churn","???????"))
			st.success(f'The Client will be {final_label}')

	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			
			# Drop Unnamed column
			data.drop(["Unnamed: 0"],axis=1,inplace=True)
			
			# Standardize continious feature
			encoder = StandardScaler()
			data["credit_score"] = encoder.fit_transform(data["credit_score"].to_numpy().reshape(-1,1))
			data["balance"] = encoder.fit_transform(data["balance"].to_numpy().reshape(-1,1))
			data["estimated_salary"] = encoder.fit_transform(data["estimated_salary"].to_numpy().reshape(-1,1))

			# LabelEncoder for categorical features
			label_encoder = LabelEncoder()
			data["country"] = label_encoder.fit_transform(data["country"].to_numpy().reshape(-1,1))
			data["gender"] = label_encoder.fit_transform(data["gender"].to_numpy().reshape(-1,1))
			data["tenure"] = label_encoder.fit_transform(data["tenure"].to_numpy().reshape(-1,1))
			
			predictions = []
			predictions_proba = model.predict_proba(data)

			for pred in predictions_proba[:,1]:
				if pred >= 0.52:
					predictions.append("Churn") # Predict churn
				else:
					predictions.append("No churn") # Predict no churn

			data["Prediction"] = predictions
			
			st.write(data)



# Deep Learning model

else:
	# Function that makes the prediction
	def predict(model,input_df):
		prediction_proba = model.predict(input_df)
		prediction = 0
		for element in prediction_proba:
			if element > 0.56:
				prediction=1
			else:
				prediction=0
		return prediction

	# Create a slidebar
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict customer churn')

	if add_selectbox == 'Online':
		
		# Customer caracteristics
		credit_score = st.number_input('Credit Score', min_value=1, max_value=3000, value=82, step=1)
		age = st.number_input('Age', min_value=1, max_value=120, value=30, step=1)
		products_number = st.selectbox("Products_number",(1,2,3))
		active_member = st.selectbox("Active_member",(0,1))
		balance = st.number_input("Balance")
		estimated_salary = st.number_input("Estimated_salary")
		country = st.selectbox("Country",('France', 'Spain', 'Germany'))
		gender = st.selectbox('Gender',('Female','Male'))
		tenure = st.selectbox('Tenure',('Old_client', 'Medium_client', 'New_client'))
		credit_card = st.selectbox('Credit_card',(0,1))
		output=""
		output1=""

		# Create a user define dataframe
		user_df_data = [[credit_score,age,products_number,active_member,balance,estimated_salary,country,gender,tenure,credit_card]]
		user_df_colnames = ["credit_score","age","products_number","active_member","balance","estimated_salary","country","gender","tenure","credit_card"]
		input_df = pd.DataFrame(user_df_data,columns=user_df_colnames)
		
		# Standardize continious feature
		encoder = StandardScaler()
		input_df["credit_score"] = encoder.fit_transform(input_df["credit_score"].to_numpy().reshape(-1,1))
		input_df["balance"] = encoder.fit_transform(input_df["balance"].to_numpy().reshape(-1,1))
		input_df["estimated_salary"] = encoder.fit_transform(input_df["estimated_salary"].to_numpy().reshape(-1,1))

		# LabelEncoder for categorical features
		label_encoder = LabelEncoder()
		input_df["country"] = label_encoder.fit_transform(input_df["country"].to_numpy().reshape(-1,1))
		input_df["gender"] = label_encoder.fit_transform(input_df["gender"].to_numpy().reshape(-1,1))
		input_df["tenure"] = label_encoder.fit_transform(input_df["tenure"].to_numpy().reshape(-1,1))
		

		if st.button("Predict"):
			output = predict(model=model_deep_learning , input_df=input_df)
			output_dict = {1 : 'Churn', 0 : 'No Churn'}
			final_label = ""
			final_label = np.where(output == 1, 'Churn',np.where(output ==         0,"No Churn","???????"))
			st.success(f'The Client will be {final_label}')

	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			
			# Drop Unnamed column
			data.drop(["Unnamed: 0"],axis=1,inplace=True)

			# Standardize continious feature
			encoder = StandardScaler()
			data["credit_score"] = encoder.fit_transform(data["credit_score"].to_numpy().reshape(-1,1))
			data["balance"] = encoder.fit_transform(data["balance"].to_numpy().reshape(-1,1))
			data["estimated_salary"] = encoder.fit_transform(data["estimated_salary"].to_numpy().reshape(-1,1))

			# LabelEncoder for categorical features
			label_encoder = LabelEncoder()
			data["country"] = label_encoder.fit_transform(data["country"].to_numpy().reshape(-1,1))
			data["gender"] = label_encoder.fit_transform(data["gender"].to_numpy().reshape(-1,1))
			data["tenure"] = label_encoder.fit_transform(data["tenure"].to_numpy().reshape(-1,1))

		
			
			predictions = []
			predictions_proba = model_deep_learning .predict(data.to_numpy())

			for element in predictions_proba:
				if element > 0.82:
					predictions.append("Churn") # Predict churn
				else:
					predictions.append("No churn") # Predict no churn

			data["Prediction"] = predictions
			
			st.write(data)





				