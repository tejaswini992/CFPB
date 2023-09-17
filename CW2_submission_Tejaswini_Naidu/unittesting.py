import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression

# Import the ComplaintsClassification class from your code
from CustomerComplaint_GLM_Refactored import ComplaintsClassification


class CustomTestResult(unittest.TestResult):
    def __init__(self, stream, verbosity=1):
        super().__init__(stream, verbosity)
        self.results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append((test, 'Pass'))

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append((test, 'Error'))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append((test, 'Failure'))

    def save_results(self, file_path):
        with open(file_path, 'w') as f:
            for test, result in self.results:
                f.write(f"{test}: {result}\n")


class TestComplaintsClassification(unittest.TestCase):
    @classmethod
    def setUp(self):
        dataset_path = 'input_data/transformed_complaints.csv'
        self.sample_data = pd.read_csv(dataset_path)
        # Create an instance of ComplaintsClassification for each test with the sample data
        self.complaints_classifier = ComplaintsClassification(dataset_path)

    def tearDown(self):
        # Clean up after each test
        self.complaints_classifier = None

    def test_encode_product_column(self):
        """
        Test the encoding of the 'Product' column using LabelEncoder.
        """
        self.complaints_classifier.load_balanced_dataset()
        df_encoded = self.complaints_classifier.encode_product_column()
        self.assertTrue('Product' in df_encoded.columns)
        self.assertTrue(all(isinstance(x, int) for x in df_encoded['Product']))

    def test_tfidf_vectorization(self):
        """
        Test TF-IDF vectorization on the 'Transformed_Complaints' column.
        """
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        tfidf_matrix = self.complaints_classifier.tfidf_vectorization()

        # Check if it's a sparse matrix using issparse from scipy.sparse
        self.assertTrue(issparse(tfidf_matrix))
        self.assertEqual(tfidf_matrix.shape[0], len(self.sample_data))

    def test_standardize_tfidf_matrix(self):
        """
        Test standardization of the TF-IDF matrix.
        """
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        self.complaints_classifier.tfidf_vectorization()
        tfidf_matrix_standardized = self.complaints_classifier.standardize_tfidf_matrix()

        # Check if it's a sparse matrix using issparse from scipy.sparse
        self.assertTrue(issparse(tfidf_matrix_standardized))
        self.assertEqual(tfidf_matrix_standardized.shape, self.complaints_classifier.tfidf_matrix.shape)

    def test_apply_svd(self):
        """
        Test the application of Truncated SVD for dimensionality reduction.
        """
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        self.complaints_classifier.tfidf_vectorization()
        self.complaints_classifier.standardize_tfidf_matrix()
        tfidf_matrix_reduced = self.complaints_classifier.apply_svd()
        self.assertIsInstance(tfidf_matrix_reduced, np.ndarray)
        self.assertTrue(tfidf_matrix_reduced.shape[0] == len(self.sample_data))

    def test_split_data(self):
        """
        Test the splitting of data into training and testing sets.
        """
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        self.complaints_classifier.tfidf_vectorization()
        self.complaints_classifier.standardize_tfidf_matrix()
        self.complaints_classifier.apply_svd()
        self.complaints_classifier.split_data(test_size=0.2)
        self.assertEqual(len(self.complaints_classifier.X_train), int(0.8 * len(self.sample_data)))

    @patch('sklearn.linear_model.LogisticRegression.fit')
    def test_train_logistic_regression(self, mock_fit):
        """
        Test training of the logistic regression model.
        """
        mock_fit.return_value = None
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        self.complaints_classifier.tfidf_vectorization()
        self.complaints_classifier.standardize_tfidf_matrix()
        self.complaints_classifier.apply_svd()
        self.complaints_classifier.split_data()
        model = self.complaints_classifier.train_logistic_regression()
        self.assertIsInstance(model, LogisticRegression)

    def test_evaluate_model(self):
        """
        Test the evaluation of the logistic regression model.
        """
        self.complaints_classifier.load_balanced_dataset()
        self.complaints_classifier.encode_product_column()
        self.complaints_classifier.tfidf_vectorization()
        self.complaints_classifier.standardize_tfidf_matrix()
        self.complaints_classifier.apply_svd()
        self.complaints_classifier.split_data()
        self.complaints_classifier.train_logistic_regression()
        accuracy = self.complaints_classifier.evaluate_model()
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 100)

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == '__main__':
    # Create a custom result object
    custom_result = CustomTestResult(open('unit_test_results.txt', 'w'))

    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComplaintsClassification)

    # Run the tests with the custom result object
    suite.run(result=custom_result)

    # Save the test results
    custom_result.save_results('unit_test_results.txt')
