# Machine Learning Model Evaluation Using Apache Spark

## Overview

This project implements machine learning models to predict mental health treatment outcomes based on a survey dataset using Apache Spark. The dataset contains information on mental health conditions, demographics, work-life balance, and access to treatment options. The objective is to preprocess the data, train various classification models, and evaluate their performance in predicting whether an individual sought treatment for mental health issues.

## Dataset

The dataset used for this project is a mental health survey dataset containing demographic information, mental health conditions, and treatment history. It is available as survey.csv. The dataset includes columns such as age, gender, country, self-employment status, family history, and others that can help predict the likelihood of treatment for mental health issues.

## Objectives

- Preprocess and clean the survey dataset for machine learning analysis.
- Train and evaluate multiple machine learning models including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, and Neural Networks.
- Evaluate model performance using accuracy, precision, recall, AUC, and ROC curves.
- Visualize model evaluation metrics and ROC curves for comparison.

## Tools & Technologies

- **Apache Spark** (PySpark) for big data processing and scalable machine learning.
- **Python** for scripting and model implementation.
- **Pandas** for data manipulation.
- **Scikit-learn** for machine learning algorithms.
- **Matplotlib** for data visualization.

## Prerequisites

Ensure you have the following installed:

- Python 3.6+
- Apache Spark 3.0+
- Jupyter Notebook (optional, for interactive analysis)

Install required Python libraries:

'''pip install pyspark pandas scikit-learn matplotlib'''

##Project Workflow

**1. Model Training and Evaluation**
The model_training.py script trains and evaluates multiple machine learning models on the preprocessed dataset:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP) Classifier
- Deep Neural Network (DNN) Classifier
It evaluates the models based on accuracy, precision, recall, AUC score, and generates ROC curves.

'''python src/model_training.py'''

**3. Visualizations**
The script generates the following visualizations to evaluate model performance:

Bar chart showing accuracy, precision, and recall for each model.
ROC curves for each model comparing true positive rate vs false positive rate.
'''python src/visualization.py'''
## Results

After running the model training and evaluation, the following visualizations will be produced:

Model Performance Metrics (Accuracy, Precision, Recall)


ROC Curves
