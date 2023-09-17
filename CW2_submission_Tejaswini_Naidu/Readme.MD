# Financial Consumer Complaints Classification with Machine Learning

## Overview

This project aims to classify consumer complaints into various financial product categories. Utilizing Natural Language Processing (NLP) techniques and machine learning algorithms, the project not only categorizes complaints but also offers actionable insights into areas of improvement for financial institutions.

## Table of Contents

1. [Installation](#installation)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Unit Testing](#unit-testing)
8. [Future Work](#future-work)
9. [Appendix](#appendix)

## Installation

1. Clone this repository.
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Data Exploration

- The dataset is sourced from the Consumer Financial Protection Bureau (CFPB).
- In-depth exploratory data analysis is performed to understand the data distribution, imbalance, and potential features.

## Data Preprocessing

- The preprocessing steps include text cleaning, handling class imbalance, and encoding categorical variables.
- Specialized preprocessing functions are developed for easy reusability.

## Feature Engineering

- Text features are engineered using TF-IDF Vectorization.
- Dimensionality reduction is performed using Truncated SVD.

## Model Training

- Multiple machine learning algorithms are evaluated, including Naive Bayes, Decision Trees, Random Forests, Neural Networks, and Generalized Linear Models (GLMs).
- Hyperparameter tuning is performed where applicable.

## Model Evaluation

- Models are rigorously evaluated using metrics like accuracy, precision, and recall.
- Detailed performance metrics for each financial product category are generated.

## Unit Testing

- Unit testing is a critical aspect of the project to ensure the correctness and robustness of various components. We have implemented a suite of unit tests to verify the functionality of key components related to data processing and model evaluation. These tests help catch potential issues early in development and ensure that the code functions as expected.

### Test Coverage

Our unit tests cover the following key functionalities:

1. **Data Loading**: We verify that the data loading function correctly loads the dataset and checks its integrity.

2. **Data Preprocessing**: This includes tests to ensure that text cleaning, handling class imbalance, and encoding of categorical variables are performed accurately.

3. **Feature Engineering**: Unit tests validate the TF-IDF vectorization process and dimensionality reduction using Truncated SVD.

4. **Model Evaluation**: We assess the correctness of model evaluation by checking accuracy, precision, and recall metrics. These tests ensure that our models are being evaluated correctly.

### Running Unit Tests
Need to run unittesting.py which automatically triggers CustomerComplaint_GLM_Refactored.py file which has code for Model training and prediction 

## Future Work

- Integration into a real-time consumer interaction system.
- Implementation of additional machine learning algorithms for improved accuracy.
  
## Appendix

- Details of all the Python libraries used and their documentation links.
