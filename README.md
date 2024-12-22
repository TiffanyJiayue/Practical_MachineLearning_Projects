# Machine Learning



# Wine Quality Prediction

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
KNN Results: Mean accuracy scores for k ranging from 1 to 10 using cross-validation.
Naive Bayes Results: Model accuracy and confusion matrix heatmap.
