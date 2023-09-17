"""
Complaints Classification Program

This program has been recreated from the original .ipynb notebook to facilitate the
execution of unit test cases. It performs the following tasks:
- Loads a balanced dataset from a CSV file.
- Encodes the 'Product' column using LabelEncoder.
- Performs TF-IDF vectorization on the 'Transformed_Complaints' column.
- Standardizes the TF-IDF matrix.
- Applies Truncated SVD for dimensionality reduction.
- Splits the data into training and testing sets.
- Trains a logistic regression model.
- Evaluates and prints the accuracy of the logistic regression model on the test data.

Usage:
Run this script to execute the entire workflow for complaints classification.

Original Author: Tejaswini Naidu

Date: 12-09-2023

"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class ComplaintsClassification:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.balanced_df = None
        self.tfidf_matrix = None
        self.tfidf_matrix_standardized = None
        self.tfidf_matrix_reduced = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.glm_model = None

    def load_balanced_dataset(self):
        """
        Loads the balanced dataset from the specified path.
        """
        self.balanced_df = pd.read_csv(self.dataset_path)
        return self.balanced_df

    def encode_product_column(self):
        """
        Encodes the 'Product' column using LabelEncoder.
        """
        label_encoder = LabelEncoder()
        self.balanced_df['Product'] = label_encoder.fit_transform(self.balanced_df['Product'])
        return self.balanced_df

    def tfidf_vectorization(self, max_features=10000):
        """
        Performs TF-IDF vectorization on the 'Transformed_Complaints' column.
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.balanced_df['Transformed_Complaints'])
        return self.tfidf_matrix

    def standardize_tfidf_matrix(self):
        """
        Standardizes the TF-IDF matrix.
        """
        scaler = StandardScaler(with_mean=False)
        self.tfidf_matrix_standardized = scaler.fit_transform(self.tfidf_matrix)
        return self.tfidf_matrix_standardized

    def apply_svd(self, num_components=300):
        """
        Applies Truncated SVD for dimensionality reduction.
        """
        svd = TruncatedSVD(n_components=num_components)
        self.tfidf_matrix_reduced = svd.fit_transform(self.tfidf_matrix_standardized)
        return self.tfidf_matrix_reduced

    def split_data(self, test_size=0.25, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.tfidf_matrix_reduced, self.balanced_df['Product'],
            test_size=test_size, random_state=random_state)

    def train_logistic_regression(self, random_state=42):
        """
        Trains a logistic regression model.
        """
        self.glm_model = LogisticRegression(random_state=random_state)
        self.glm_model.fit(self.X_train, self.y_train)
        return self.glm_model

    def evaluate_model(self):
        """
        Evaluates the logistic regression model and returns the accuracy.
        """
        glm_predictions = self.glm_model.predict(self.X_test)
        glm_accuracy = accuracy_score(self.y_test, glm_predictions)
        test_accuracy_glm = round(glm_accuracy * 100, 2)
        return test_accuracy_glm


def main():
    # Instantiate the ComplaintsClassification class with the dataset path
    dataset_path = 'input_data/transformed_complaints.csv'
    complaints_classifier = ComplaintsClassification(dataset_path)

    # Load and preprocess the data
    complaints_classifier.load_balanced_dataset()
    complaints_classifier.encode_product_column()
    complaints_classifier.tfidf_vectorization()
    complaints_classifier.standardize_tfidf_matrix()
    complaints_classifier.apply_svd()

    # Split the data into training and testing sets
    complaints_classifier.split_data()

    # Train the logistic regression model
    complaints_classifier.train_logistic_regression()

    # Evaluate and print the model's accuracy
    test_accuracy = complaints_classifier.evaluate_model()
    print(f"Test Accuracy (Logistic Regression): {test_accuracy}%")


if __name__ == "__main__":
    main()
