Customer Churn Prediction
Project Overview
This project focuses on building a machine learning model to predict customer churn. Customer churn, or attrition, is the process of customers leaving a service. For businesses, retaining existing customers is often more cost-effective than acquiring new ones, so predicting which customers are at risk of leaving is a valuable business objective.

The primary goal of this project was to develop a predictive model that identifies potential churners and to build a simple web application to demonstrate the model's functionality.

Methodology & Technologies
The project follows a standard machine learning pipeline:

Data Loading & Cleaning: Loaded the customer_churn_data.csv dataset. Missing values in the TotalCharges column were removed.

Feature Engineering: Categorical data, such as Gender and Churn, was converted into numerical representations (0 and 1) to be used by the models.

Data Scaling: The numerical features (Age, Tenure, MonthlyCharges) were scaled using StandardScaler to ensure that no single feature unfairly influences the model's performance.

Model Training: Several supervised machine learning algorithms were trained and evaluated, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, and Random Forest.

Hyperparameter Tuning: GridSearchCV was used to find the optimal parameters for each model, ensuring the best possible performance.

Model Export: The best-performing model (SVC) and the scaler were saved as .pkl files for use in the web application.

Web Application: A simple, interactive web application was built using Streamlit to allow users to input customer data and receive a real-time churn prediction.

Key Technologies Used:

Python: The core programming language.

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning model development, evaluation, and data scaling.

Streamlit: For creating the web-based user interface.

Joblib: For serializing and exporting the trained model and scaler.

NumPy: For numerical operations.

Project Files
customer_churn_data.csv: The dataset containing customer information and churn status.

app.py: The Python script for the Streamlit web application. This file loads the saved model and scaler to make predictions.

model.pkl: The serialized binary file containing the trained Support Vector Classifier (SVC) model.

scaler.pkl: The serialized binary file containing the fitted StandardScaler. This is crucial to preprocess new input data in the same way the training data was handled.

notebook.ipynb: The Jupyter Notebook file where the initial data exploration, model training, and evaluation took place.

Key Outcomes and Business Impact
Developed a robust customer churn prediction model with a predictive accuracy of over 90%.

Created a functional web application to easily demonstrate the model's ability to identify high-risk customers.

This predictive capability can enable businesses to implement proactive customer retention strategies, potentially reducing churn and increasing customer lifetime value.

How to Run the Application
To run the Streamlit app on your local machine, follow these steps:

Make sure you have Python installed.

Install the necessary libraries:

pip install pandas scikit-learn streamlit joblib numpy


Navigate to the project directory in your terminal.

Run the application with the command:

streamlit run app.py


A web browser window will open automatically, displaying the application.

Author
Amruth
