# Machine Learning
This repository showcases three distinct machine learning and data analysis projects, each addressing a specific real-world problem using various modeling techniques. From predicting wine quality based on chemical properties to analyzing employee attrition and forecasting office supply sales, the scripts illustrate a range of approaches in classification, regression, and time series forecasting. The projects emphasize data preprocessing, model implementation, and evaluation, offering practical examples of how machine learning can generate actionable insights across diverse domains.


# 1. Wine Quality Prediction

A machine learning script for predicting the quality of wine samples based on chemical properties. 
The script demonstrates the use of two classification techniques: **K-Nearest Neighbors (KNN)** and **Bernoulli Naive Bayes (BNB)**.

## Overview

The `Knearest_BernoulliNM.py` script processes a wine quality dataset and performs classification to determine whether a wine sample is of high quality (quality score >= 7). The key steps include data cleaning, outlier removal, feature engineering, and model evaluation.

## Features

- **Data Preprocessing:**
  - Removal of outliers and missing values using visual inspection and boxplots.
  - Standardization of numerical features for KNN compatibility.
  - Creation of a binary target column (`high quality`), categorizing wines as high quality (`1`) or not (`0`).

- **Models Implemented:**
  1. **K-Nearest Neighbors (KNN):**
     - Hyperparameter tuning with cross-validation for optimal `k`.
     - Standardized feature input for better performance.
  2. **Bernoulli Naive Bayes (BNB):**
     - Splits data into training and testing sets.
     - Generates predictions and evaluates accuracy.
     - Confusion matrix visualization for performance insights.

- **Visualization:**
  - Boxplots to identify outliers in numerical features.
  - Heatmap of confusion matrix for Naive Bayes model evaluation.

## Dataset

The script uses a wine quality dataset (CSV file), which should be available locally. Update the file path in the script (`winequality.csv`) to point to your dataset.

### Expected Columns:
- Numerical features such as acidity, chlorides, sulfur dioxide, etc.
- Target variable: `quality` (used to create the binary `high quality` column).

## Requirements

The following Python libraries are required to run the script:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

2. Run the script:

```bash
python Knearest_BernoulliNM.py
```

3. Update the dataset path (winequality.csv) in the script as needed.

## Output
- KNN Results: Mean accuracy scores for k ranging from 1 to 10 using cross-validation.
- Naive Bayes Results: Model accuracy and confusion matrix heatmap.



# 2. Employee Attrition Analysis

This script provides machine learning models to analyze employee attrition data and identify key factors that influence employees' decisions to leave a company. It implements two predictive models:

1. A **Neural Network** for predicting employee attrition.
2. A **Boosted Ensemble of Trees** to predict attrition and analyze feature importance.

## Overview

The `employee_attrition_analysis.py` script processes a dataset of 1,470 current and former employees. The dataset includes variables such as tenure, gender, education, and other factors that influence attrition. The script applies data preprocessing techniques and builds classification models to analyze the data.

---

## Features

### Data Preprocessing
- Drops irrelevant columns (e.g., `Over18`, `EmployeeCount`).
- Encodes categorical variables using `LabelEncoder`.
- Normalizes numerical features for consistent model performance.

### Predictive Models
1. **Neural Network (MLPClassifier):**
   - Implements a multi-layer perceptron with regularization.
   - Evaluates performance using accuracy and confusion matrix.
   - Includes cross-validation to validate the model.

2. **Boosted Ensemble of Trees:**
   - Random Forest for feature importance analysis.
   - Gradient Boosting Regressor for enhanced prediction accuracy.
   - Visualizes feature importance for interpretability.

---

## Dataset

The dataset, `Simulate_HRemployee_attrition.csv`, is used to train and test the models. Ensure the file path is correctly set in the script.

### Key Columns:
- **Attrition (Target):** Binary indicator (`1` for Yes, `0` for No).
- **Predictors:** Variables such as `tenure`, `gender`, `education`, etc.

---

## Requirements

The following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

2. Run the script:

```bash
python employee_attrition_analysis.py
```

3. Update the dataset path (Simulate_HRemployee_attrition.csv) in the script if necessary.

## Output
1. Neural Network:
  - Accuracy score.
  - Confusion matrix visualization.
  - Cross-validation results.

2. Boosted Ensemble of Trees:
  - RMSE for predictions.
  - Feature importance bar chart.




# 3. Office Supplies Sales Forecasting

This script analyzes historical sales data for office supplies, models the time series using the **Seasonal ARIMA (SARIMA)** framework, and generates forecasts for future sales. The goal is to provide actionable insights into sales trends and patterns.

---

## Overview

The `Forecasting_Supplies.py` script processes time series data for office supplies sales, focusing on the following tasks:
1. **Data Cleaning and Preprocessing:** Prepare the sales data for time series analysis.
2. **Time Series Analysis:** Check stationarity and decompose the series into trend, seasonal, and residual components.
3. **SARIMA Modeling:** Optimize the SARIMA model parameters using grid search and fit the best model to the data.
4. **Forecasting:** Generate one-step-ahead forecasts and evaluate their accuracy.

---

## Features

### Data Preprocessing
- Filters sales data for the "Office Supplies" category.
- Removes unnecessary columns and sets the `Order Date` column as a time series index.
- Resamples the sales data to calculate monthly averages for analysis.

### Time Series Analysis
- Performs the **Augmented Dickey-Fuller (ADF)** test to assess stationarity.
- Decomposes the time series into **trend**, **seasonal**, and **residual** components.

### SARIMA Modeling
- Conducts a **grid search** to find optimal SARIMA parameters using the AIC score.
- Fits the best SARIMA model to the data and evaluates its performance.

### Forecasting
- Generates short-term forecasts with confidence intervals.
- Computes **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to assess forecast accuracy.
- Produces long-term forecasts and visualizes them alongside historical data.

---

## Dataset

The script uses a dataset from an Excel file (`Superstore.xls`) containing historical sales data. Update the file path in the script to point to your local dataset.

### Key Columns:
- **Category:** Filtered to include only "Office Supplies."
- **Sales:** Target variable for time series modeling.

---

## Requirements

The following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib statsmodels
```

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

2. Run the script:

```bash
python Forecasting_Supplies.py
```

3. Update the dataset path (Superstore.xls) in the script if necessary.

## Output
1. Time Series Analysis:
- ADF test results (stationarity check).
- Decomposition plots (trend, seasonal, residual).

2. SARIMA Modeling:
- AIC scores for model selection.
- Summary of the best SARIMA model.

3. Forecasting:
- Visualization of observed vs. forecasted values.
- Forecast error metrics (MSE and RMSE).
- Long-term forecast with confidence intervals.

