# Telco Customer Churn Prediction

## Problem Statement

The goal of this project is to develop a machine learning model to predict customers who will churn from the company. Before developing the model, data analysis and feature engineering steps are performed.

## Dataset Description

The Telco customer churn dataset contains information about 7,043 customers of a fictional telecom company in California in the third quarter. It includes information about which customers have left, stayed, or signed up for home phone and internet services.

### Variables
1. CustomerId: Customer ID
2. Gender: Gender
3. SeniorCitizen: Whether the customer is a senior citizen (1, 0)
4. Partner: Whether the customer has a partner (Yes, No)
5. Dependents: Whether the customer has dependents (Yes, No)
6. Tenure: Number of months the customer has stayed with the company
7. PhoneService: Whether the customer has phone service (Yes, No)
8. MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
9. InternetService: Customer's internet service provider (DSL, Fiber optic, No)
10. OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
11. OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
12. DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
13. TechSupport: Whether the customer has technical support (Yes, No, No internet service)
14. StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
15. StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
16. Contract: Customer's contract term (Month-to-month, One year, Two years)
17. PaperlessBilling: Whether the customer has paperless billing (Yes, No)
18. PaymentMethod: Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
19. MonthlyCharges: Monthly charges collected from the customer
20. TotalCharges: Total charges collected from the customer
21. Churn: Whether the customer has churned (Yes or No) - Indicates whether the customer left in the last month or quarter

### Data Preparation

1. Data is read from the provided CSV file.
2. Data types and missing values are checked and corrected.
3. Encoding is applied to binary categorical variables.
4. Standardization is performed for numeric variables.

## Exploratory Data Analysis (EDA)

1. General overview of the dataset is provided.
2. Numeric and categorical variables are identified and analyzed.
3. Target variable analysis is conducted, including mean values by categorical variables and numeric variables by the target variable.
4. Outlier analysis is performed.
5. Missing observation analysis is conducted.
6. Correlation analysis is done.

## Feature Engineering

1. Missing values and outliers are handled.
2. Encoding operations are performed for categorical variables.
3. Standardization is applied to numeric variables.

## Model Building

Several machine learning models are built and evaluated using cross-validation:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. CatBoost
6. Light GBM
7. XGBoost

The models are tuned using hyperparameter optimization, and the best-performing model is selected based on evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

## Model Evaluation

The performance of each model is evaluated, and the results are stored in a dataframe. The best model is selected based on the F1 score.

## Model Export

The best-performing model (XGBoost) is saved to a file named "best_model.pkl" using pickle for future use.

