#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:30:48 2024
@author: Tiffany
"""

# This script analyzes a wine quality dataset to predict 
# whether a wine is classified as high quality (rating ≥ 7) or not. 
# It performs data preprocessing by identifying and removing outliers, 
# cleaning missing values, and engineering a binary target variable. 
# The data is then standardized and used to train two machine learning models: 
# K-Nearest Neighbors (KNN) and Bernoulli Naive Bayes (BernoulliNB). 
# Model performance is evaluated using accuracy metrics and a confusion matrix, 
# providing insights into the effectiveness of each approach for classifying wine quality.

import pandas as pd
# Import data
wine_data = pd.read_csv('/Users/XXX/winequality.csv')

# Summary/Drop Outliers and missing values 
summary_stats = wine_data.describe()
print(summary_stats)

# Delect outliers before standardization
import matplotlib.pyplot as plt
import seaborn as sns

# Create box plots for each column to see the specific outliers
numerical_cols = wine_data.select_dtypes(include=['float64']).columns
plt.figure(figsize=(12, 8))
for col in numerical_cols:
    sns.boxplot(x=wine_data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Replace the specified value (Outliers by looking at the graph and dataset) with NaN in the some columns 
wine_data = wine_data.replace({'fixed acidity': {-999: pd.NA}})
wine_data = wine_data.replace({'chlorides': {11.2: pd.NA}})
wine_data = wine_data.replace({'total sulfur dioxide': {289: pd.NA}})
wine_data = wine_data.replace({'total sulfur dioxide': {278: pd.NA}})
wine_data = wine_data.replace({'density': {-9: pd.NA}})
wine_data = wine_data.replace({'pH': {31.3: pd.NA}})

# Drop All the NaN Values in the dataset
wine_data_clean = wine_data.dropna()

### Creat a new col for wine quality, which set the high quality(above 7) as 1, and the rest as 0
wine_data_clean['high quality'] = wine_data_clean['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Assuming 'df' is your DataFrame and 'column_to_drop' is the name of the column you want to remove
column_to_drop = 'quality'
wine_data_clean.drop(column_to_drop, axis=1, inplace=True)

#### Split Data ####
# Charateristics are the features, predictors
# "high quality" is the label,class
feature = wine_data_clean.drop(['high quality'], axis=1)
label = wine_data_clean['high quality']

#### First method - K-nearest Neighbor ####
#### Standlized the feature columns for K-nearest Neighbor ####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feature = scaler.fit_transform(feature)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

from sklearn.model_selection import cross_val_score
for k in range(1,11):
    model.n_neighbors = k
    accuracy = cross_val_score(model, feature, label, cv = 10)
    print(f"Mean accuracy score for k ={k}: {accuracy.mean()}")
    
#### Second method - BernoulliNM ####
from sklearn.naive_bayes import BernoulliNB
label_NB = wine_data_clean['high quality']
model_NB = BernoulliNB()
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_NB_train, label_NB_test = train_test_split(feature, label_NB, test_size=0.2, random_state=21)
model_NB.fit(feature_train, label_NB_train)

label_NB_pred = model_NB.predict(feature_test)
# print(label_NB_pred)
#### Check and evaluate how the model works ####
accuracy_NB = model_NB.score(feature_test, label_NB_test)
print("Accuracy:", accuracy_NB)

# Plot Confusion Matrix
import numpy as np
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(label_NB_pred, label_NB_test)
names = np.unique(label_NB_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')     

