#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:19:48 2024

@author: heiyo
"""

# A large software company's HR department has hired you to better understand 
# why some employees are more likely to leave the firm than others. 
# For this purpose, they give you a data set of 1,470 current and 
# former employees with information on whether or not they have left the company, 
# their tenure, gender, education, and several other variables 
# (a description of discrete data can be found below). 
# The company wants you to provide two classifiers

# 1. a neural network that can predict employee attrition (variable name is attrition)
# 2. a boosted ensemble of trees that can predict employee attrition 
# but also provide a ranking of feature importance 
# (i.e. the features that have the largest influence on the decision to quit their jobs)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
path = '/Users/heiyo/Documents/Pitt/MQE学习/MQE/ECON2824_machine/Finalweek'
data = pd.read_csv(path + '/Simulate_HRemployee_attrition.csv')

# Data cleaning and preprocessing
data = data.drop(['Over18', 'EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis=1)
data['Attrition'] = data['Attrition'].replace({'Yes': 1, 'No': 0})

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Normalize numerical features
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols] / data[numerical_cols].max()

# Define target and predictors
target_column = 'Attrition'
predictors = data.columns.drop(target_column)

# Split data into training and test sets
X = data[predictors].values
y = data[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Neural network model
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1200, random_state=0)
mlp.fit(X_train, y_train)

# Predictions and evaluation
y_pred = mlp.predict(X_test)
print("Neural Network Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# Confusion matrix plot
plt.figure(figsize=(6, 4))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Regularized neural network
mlp_regularized = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=1200, random_state=0, alpha=0.001)
mlp_regularized.fit(X_train, y_train)
y_pred_regularized = mlp_regularized.predict(X_test)
print("Regularized Neural Network Accuracy:", round(accuracy_score(y_test, y_pred_regularized), 3))

# Cross-validation
cv_scores = cross_val_score(mlp_regularized, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy (Regularized NN):", round(cv_scores.mean(), 3))

# Boosting with RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
rf_regressor.fit(X_train, y_train)
rf_y_pred = rf_regressor.predict(X_test)

print('Random Forest RMSE:', np.sqrt(mean_squared_error(y_test, rf_y_pred)))

# Feature importance visualization
feature_importances = rf_regressor.feature_importances_
importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, random_state=0)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)

print('Gradient Boosting RMSE:', np.sqrt(mean_squared_error(y_test, gb_y_pred)))

