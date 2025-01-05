# Bank Marketing Classification Documentation

![GitHub License](https://img.shields.io/github/license/Sambonic/bank-marketing-classification)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

This project uses machine learning to classify bank marketing campaign outcomes, aiming to improve prediction accuracy.

#### Last Updated: January 5th, 2025

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)

<a name="installation"></a>
## Installation

Make sure you have [python](https://www.python.org/downloads/) downloaded if you haven't already.
Follow these steps to set up the environment and run the application:

1. Clone the Repository:
   
```bash
git clone https://github.com/Sambonic/bank-marketing-classification
```

```bash
cd bank-marketing-classification
```

2. Create a Python Virtual Environment:
```bash
python -m venv env
```

3. Activate the Virtual Environment:
- On Windows:
  ```
  env\Scripts\activate
  ```

- On macOS and Linux:
  ```
  source env/bin/activate
  ```
4. Ensure Pip is Up-to-Date:
  ```
  python.exe -m pip install --upgrade pip
  ```
5. Install Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Import Bank Marketing Classification as shown below.


<a name="usage"></a>
## Usage
## Running the Bank Marketing Classification Project

To use this project, follow these steps:

1. **Execute the notebook:** Open `main-repo/src/bank-marketing-classification-ml.ipynb` in a Jupyter Notebook environment.  Run all code cells sequentially.

2. **Data Exploration:** The notebook begins by loading the dataset, displaying initial rows, and providing summary statistics (data types, unique values, descriptive statistics, and missing values). Data visualizations (histograms, pie charts, count plots, and correlation heatmaps) are generated to understand the data distribution and relationships between features.

3. **Data Preprocessing:** The notebook then performs data preprocessing steps, including:
    - Renaming columns for better readability.
    - Encoding categorical features using label encoding.
    - Handling missing values.  Multiple imputation methods (KNN, KMeans, median imputation for numerical features; mode, KModes imputation for categorical features) are evaluated using Random Forest classification accuracy. The best performing method is then applied.  The notebook also demonstrates dropping null values as a comparison.
    - Discretizing numerical features ('Age' and 'Campaign_Contacts') into categorical features using binning and applying outlier handling techniques. 
    - Removing duplicate rows.
    - Converting data types to be appropriate for models.

4. **Feature Selection:** Two feature selection methods are applied:
    - Chi-squared test for categorical features.
    - Random Forest feature importance for all features.
    The notebook then identifies common features selected by both methods.

5. **Class Imbalance Handling:** The notebook addresses the class imbalance by employing both undersampling and oversampling. It demonstrates the undersampling method by downsampling the majority class to match the size of the minority class. It then shows using SMOTENC for oversampling minority class instances, accounting for categorical features.

6. **Model Training and Evaluation:** Several classification models (KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, LGBMClassifier, and XGBClassifier) are trained and evaluated using the preprocessed data.  Each model is evaluated using different combinations of feature selection and hyperparameter optimization strategies. The notebook evaluates models with and without feature selection. It provides a learning curve for each model showing training accuracy and validation accuracy at various training sizes to show whether a model is overfitting or underfitting. A confusion matrix and ROC curve with AUC values are generated to evaluate model performance.  Metrics like accuracy, precision, recall, and F1-score are calculated and displayed for each model.  Comparative analysis is provided via bar charts.

7. **Model Comparison:** The notebook presents a comparison of all evaluated models, showcasing their performance metrics (accuracy, precision, recall, F1-score) across different scenarios (with/without feature selection and hyperparameter tuning), to determine the best-performing model for the given dataset and task.  Plots will show ROC curves comparing various models.

<a name="features"></a>
## Features
- **Bank Marketing Campaign Classification:** Predicts customer subscription based on various features.
- **Data Preprocessing:** Handles missing values using KNN, K-means, median, mode, and k-modes imputation; evaluates different imputation methods based on RandomForestClassifier accuracy.  Drops duplicates.
- **Feature Engineering:** Discretizes 'Age' and 'Campaign_Contacts' features into meaningful categories, and normalizes 'Campaign_Contacts'.
- **Feature Selection:** Employs Chi-squared test and Random Forest feature importance for selecting relevant features. Combines results from both methods to choose a final feature subset.
- **Class Imbalance Handling:** Addresses class imbalance using undersampling of the majority class and SMOTENC for oversampling the minority class.  Also explores class weights as an alternative.
- **Model Training and Evaluation:** Trains and evaluates multiple classification models (KNN, Decision Tree, Logistic Regression, Random Forest, LightGBM, XGBoost) with and without feature selection and hyperparameter tuning, reporting accuracy, precision, recall, and F1-score. Uses learning curves to assess model complexity and overfitting. ROC curves compare model performance.
- **Hyperparameter Tuning:** Uses `RandomizedSearchCV` for hyperparameter optimization of all models to maximize the F1 score.
- **Comprehensive Visualization:** Uses various plots (histograms, pie charts, bar charts, heatmaps, confusion matrices, ROC curves, learning curves) to visualize data, results, and model performance.


