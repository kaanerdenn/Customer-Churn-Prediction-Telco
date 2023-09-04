##############################
# Telco Customer Churn Prediction
##############################

# Problem: Developing a machine learning model to predict customers who will churn from the company.
# Before developing the model, data analysis and feature engineering steps are expected to be performed.

# Telco customer churn data contains information about 7043 customers of a fictional telecom company in California in the third quarter.
# It includes information about which customers have left, stayed, or signed up for home phone and internet services.

# 21 Variables and 7043 Observations

# CustomerId: Customer ID
# Gender: Gender
# SeniorCitizen: Whether the customer is a senior citizen (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has phone service (Yes, No)
# MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
# InternetService: Customer's internet service provider (DSL, Fiber optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
# TechSupport: Whether the customer has technical support (Yes, No, No internet service)
# StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
# StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
# Contract: Customer's contract term (Month-to-month, One year, Two year)
# PaperlessBilling: Whether the customer has paperless billing (Yes, No)
# PaymentMethod: Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges: Monthly charges collected from the customer
# TotalCharges: Total charges collected from the customer
# Churn: Whether the customer has churned (Yes or No) - Indicates whether the customer left in the last month or quarter

# Each row represents a unique customer.
# Variables contain information about customer services, accounts, and demographic data.
# Services customers have signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information - how long they have been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# Demographic information about customers - gender, age range, and whether they have partners and dependents


# TASK 1: EXPLORATORY DATA ANALYSIS
           # Step 1: Examine the general overview.
           # Step 2: Capture numeric and categorical variables.
           # Step 3: Analyze numeric and categorical variables.
           # Step 4: Analyze the target variable. (Mean of the target variable by categorical variables, mean of numeric variables by the target variable)
           # Step 5: Perform outlier analysis.
           # Step 6: Perform missing observation analysis.
           # Step 7: Perform correlation analysis.

# TASK 2: FEATURE ENGINEERING
           # Step 1: Perform necessary operations for missing and outlier values.
           # You can apply operations.
           # Step 2: Create new variables.
           # Step 3: Perform encoding operations.
           # Step 4: Standardize numeric variables.
           # Step 5: Build the model.


# Required Libraries and Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# TotalCharges should be a numerical variable
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

df.head()

##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

##################################
# GENERAL OVERVIEW
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df["Churn"].value_counts().plot.bar()
plt.show()

# Categorical Variables: Gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
# InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
# PaperlessBilling, PaymentMethod
# Numerical Variables: tenure, MonthlyCharges, TotalCharges

cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
num_cols = [col for col in df.columns if df[col].dtypes != "O"]

print(len(cat_cols))
print(len(num_cols))

##################################
# ANALYSIS OF NUMERIC VARIABLES
##################################

df[num_cols].describe([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).T

##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

df[cat_cols].describe()

for col in cat_cols:
    print(col, ":", len(df[col].unique()), "categories")

# Step 4: Analyze the target variable. (Mean of the target variable by categorical variables, mean of numeric variables by the target variable)
df.groupby("Churn").agg(np.mean)

# Step 5: Perform outlier analysis.
# There are no outliers in the data.
# Step 6: Perform missing observation analysis.
df.isnull().sum()

# First, it is checked whether TotalCharges is missing in the TotalCharges variable, where there are few observations with missing TotalCharges variables.
# Observations with missing TotalCharges variable are deleted from the data set since they are few.
df[df["TotalCharges"].isnull()]

df.dropna(inplace=True)

# Outlier values for numeric variables are checked, but there is no data that requires removal of observations with outlier values.
##################################
# CORRELATION ANALYSIS
##################################

df.corr()

# There is a high positive correlation between MonthlyCharges and TotalCharges. This is normal because TotalCharges is the product of MonthlyCharges and Tenure.
# There is also a positive correlation between Age and Churn.
# Churn variable has 0.4 positive correlation with MonthlyCharges and 0.2 negative correlation with Tenure.

##################################
# TASK 2: FEATURE ENGINEERING
##################################

##################################
# Handling Outliers and Missing Values
##################################

# Outliers are not removed because there is no data with outliers in the data set.

# First, it is checked whether TotalCharges is missing in the TotalCharges variable, where there are few observations with missing TotalCharges variables.
# Observations with missing TotalCharges variable are deleted from the data set since they are few.
df[df["TotalCharges"].isnull()]

df.dropna(inplace=True)

##################################
# Encoding Categorical Variables
##################################

# Yes-No variables are converted to 0-1.
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and col not in "Churn"]
binary_cols

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Multiple classes are one hot encoded for variables with more than 2 categories.
# We can also use the "get_dummies" function to apply one-hot encoding.
# df = pd.get_dummies(df, columns=["Contract"])

##################################
# Scaling Numerical Variables
##################################

# Standardization is performed for numeric variables.
# Standardization is performed for better results for algorithms that use distance as a criterion.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[num_cols])
df[num_cols] = scaler.transform(df[num_cols])

##################################
# Building Machine Learning Models
##################################

# The data set is divided into 2 as train and test.

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Models
# Logistic Regression
log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_

# Logistic Regression - Model Tuning
log_model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
y_pred = log_model.predict(X_test)

cross_val_score(log_model, X_test, y_test, cv=10).mean()

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

cross_val_score(knn_model, X_test, y_test, cv=10).mean()

knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

# K-Nearest Neighbors (KNN) - Model Tuning
knn_tuned = KNeighborsClassifier(n_neighbors=47)
knn_tuned.fit(X_train, y_train)

y_pred = knn_tuned.predict(X_test)

cross_val_score(knn_tuned, X_test, y_test, cv=10).mean()

# Decision Tree
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
cart_model

y_pred = cart_model.predict(X_test)
cross_val_score(cart_model, X_test, y_test, cv=10).mean()

# Decision Tree - Model Tuning
cart_grid = {"max_depth": range(1,10),
             "min_samples_split": list(range(2,50))}

cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv=10, n_jobs=-1, verbose=2)
cart_cv_model = cart_cv.fit(X_train, y_train)

print("Best Parameters: " + str(cart_cv_model.best_params_))

# Decision Tree - Model Tuning (Best Parameters)
cart_tuned = DecisionTreeClassifier(max_depth=6, min_samples_split=27)
cart_tuned_model = cart_tuned.fit(X_train, y_train)

y_pred = cart_tuned_model.predict(X_test)

cross_val_score(cart_tuned_model, X_test, y_test, cv=10).mean()

# Random Forest
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
cross_val_score(rf_model, X_test, y_test, cv=10).mean()

rf_params = {"max_depth": [2,5,8,10],
             "max_features": [2,5,8],
             "n_estimators": [10,50,100],
             "min_samples_split": [5,10]}

rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2)
rf_cv_model.fit(X_train, y_train)

print("Best Parameters: " + str(rf_cv_model.best_params_))

# Random Forest - Model Tuning (Best Parameters)
rf_tuned = RandomForestClassifier(max_depth=10, max_features=5, min_samples_split=5, n_estimators=100)
rf_tuned.fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)

cross_val_score(rf_tuned, X_test, y_test, cv=10).mean()

# Catboost
catb = CatBoostClassifier()
catb_model = catb.fit(X_train, y_train)

y_pred = catb_model.predict(X_test)

cross_val_score(catb_model, X_test, y_test, cv=10).mean()

catb_params = {"iterations": [200,500],
              "learning_rate": [0.01,0.05],
              "depth": [3,5]}

catb = CatBoostClassifier()
catb_cv = GridSearchCV(catb, catb_params, cv=10, n_jobs=-1, verbose=2)
catb_cv.fit(X_train, y_train)

print("Best Parameters: " + str(catb_cv.best_params_))

# Catboost - Model Tuning (Best Parameters)
catb_tuned = CatBoostClassifier(depth=3, iterations=200, learning_rate=0.05)
catb_tuned.fit(X_train, y_train)

y_pred = catb_tuned.predict(X_test)

cross_val_score(catb_tuned, X_test, y_test, cv=10).mean()

# Light GBM
lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test)

cross_val_score(lgbm_model, X_test, y_test, cv=10).mean()

lgbm_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
               "n_estimators": [20, 40, 100, 200]}

lgbm = LGBMClassifier()
lgbm_cv = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=2)
lgbm_cv.fit(X_train, y_train)

print("Best Parameters: " + str(lgbm_cv.best_params_))

# Light GBM - Model Tuning (Best Parameters)
lgbm_tuned = LGBMClassifier(learning_rate=0.01, n_estimators=100)
lgbm_tuned.fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

cross_val_score(lgbm_tuned, X_test, y_test, cv=10).mean()

# XGBoost
xgb = XGBClassifier()
xgb_model = xgb.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

cross_val_score(xgb_model, X_test, y_test, cv=10).mean()

xgb_params = {"n_estimators": [100, 500],
              "subsample": [0.1, 0.5, 1.0],
              "max_depth": [3, 4, 5],
              "learning_rate": [0.01, 0.02, 0.1]}

xgb = XGBClassifier()
xgb_cv = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=2)
xgb_cv.fit(X_train, y_train)

print("Best Parameters: " + str(xgb_cv.best_params_))

# XGBoost - Model Tuning (Best Parameters)
xgb_tuned = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=500, subsample=1.0)
xgb_tuned.fit(X_train, y_train)

y_pred = xgb_tuned.predict(X_test)

cross_val_score(xgb_tuned, X_test, y_test, cv=10).mean()

# Creating a Dataframe for Model Results
models = [log_model, knn_tuned, cart_tuned, rf_tuned, catb_tuned, lgbm_tuned, xgb_tuned]
results = []

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    result = {"Model Name": names,
              "Accuracy": acc,
              "Precision": precision,
              "Recall": recall,
              "F1 Score": f1,
              "Confusion Matrix": cm}
    results.append(result)

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="F1 Score", ascending=False)
result_df

# Exporting the Best Model
import pickle

# Saving the best model to a file
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_tuned, model_file)
