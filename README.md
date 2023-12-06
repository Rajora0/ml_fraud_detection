# Predictive Models for Fraud Detection

Fraud detection is a critical aspect of data analysis, especially in financial transactions. This notebook focuses on building predictive models for fraud detection using machine learning techniques. The code is available in a Jupyter Notebook, which can be accessed [here](https://colab.research.google.com/github/Rajora0/ml_fraud_detection/blob/main/Modelos_preditivos_em_dados_detec%C3%A7%C3%A3o_de_fraude.ipynb).

## Loading and Preparing Data

The initial steps involve mounting Google Drive to access the dataset and performing ETL (Extract, Transform, Load) operations. The dataset used is an example from Kaggle, which contains information about financial transactions.

## Project Pipeline

The notebook follows a structured pipeline for the project:

1. **Discovering the Problem:** Clearly defining the problem at the outset is crucial to guide subsequent phases.

2. **Data Collection:** Gathering data, whether internal or external, structured or unstructured, that is aligned with the identified problem.

3. **Data Cleaning/Processing:** Addressing missing values, inconsistent records, and outliers to ensure the data is suitable for analysis.

4. **Exploratory Data Analysis (EDA):** Identifying patterns and gaining insights from the cleaned data.

5. **Data Modeling:** Utilizing machine learning models to extract patterns and address the identified business problem.

6. **Interpreting Results:** Subjectively evaluating and interpreting the results, making them understandable to other teams.

7. **Applying Improvements:** Iteratively revising the model or analysis as needed, considering changes in data over time.

## Loading Data and Exploratory Analysis

The dataset includes various features such as transaction type, amount, originator, recipient, and indicators for fraud. An exploratory analysis is performed to understand the characteristics of the data.

## Data Encoding

Categorical variables are encoded using one-hot encoding to prepare the data for machine learning models.

## Logistic Regression Model

A logistic regression model is trained on the data to predict fraud. The model's performance metrics, such as accuracy, recall, precision, and F1 score, are evaluated.

## Handling Imbalanced Data

To address imbalanced data, the Synthetic Minority Over-sampling Technique (SMOTE) is applied, creating a balanced dataset for better model training.

## Decision Tree and Random Forest Models

Decision tree and random forest models are implemented and evaluated. The models' performance is assessed using metrics such as accuracy, precision, recall, and F1 score.

## Model Comparison

Logistic regression, decision tree, and random forest models are compared based on their performance metrics to identify the most suitable model for fraud detection.

## Hyperparameter Tuning with RandomizedSearchCV

RandomizedSearchCV is used to fine-tune the hyperparameters of the random forest model. This helps optimize the model for better performance.

## Conclusion

This notebook provides a comprehensive overview of building predictive models for fraud detection in financial transactions. The process includes data loading, cleaning, exploratory analysis, model training, and performance evaluation. The comparison of different models and hyperparameter tuning further enhances the effectiveness of fraud detection. This pipeline serves as a foundation for developing robust fraud detection systems in real-world scenarios.
