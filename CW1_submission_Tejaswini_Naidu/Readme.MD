# Coursework 1: Consumer Complaint Database
- Author: Tejaswini Naidu - tn152
- Current date: 16 July 2023 
- Module: Data programming in Python (DSM020-2023-APR)

# Table of Contents

1. [Consumer Complaint Database](#consumer-complaint-database)
2. [Author](#author)
3. [Overview](#overview)
4. [Motivation](#motivation)
5. [Dataset Information](#dataset-information)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [References](#references)
9. [Step 1: Import Libraries](#step-1-import-libraries)
    - [Libraries Used](#libraries-used)
10. [Step 2: Data Preprocessing](#step-2-data-preprocessing)
11. [Step 3: Explore Structured Columns](#step-3-explore-structured-columns)
12. [Step 4: Explore Unstructured Columns](#step-4-explore-unstructured-columns)
13. [Conclusion](#conclusion)
14. [Citations](#citations)
15. [Steps to Execute Code](#steps-to-execute-code)

## Overview
The Consumer Complaint Database is a rich dataset featuring customer complaints across various banking domain products and services. It acts as an essential resource for identifying patterns, understanding, and addressing consumer complaints, and ultimately enhancing banking customer satisfaction. 

This database provides insights into areas such as:
- Customer Complaints
- Product and Service Analysis
- Trends and Patterns
- Consumer Satisfaction

This Jupyter Notebook documents and explores the Consumer Complaint Database. We aim to uncover valuable insights through data analysis and visualizations, offering actionable recommendations to businesses to enhance their customer satisfaction and address consumer issues effectively.

## Motivation
As an individual interested in the banking domain, this exploration of the complaints dataset serves to provide insights into the common issues faced by consumers within the Banking or FinTech industries. This analysis aims to understand the types of complaints and identify frequently mentioned companies.

Key motivations for this exploration are:
- Improving Customer Experience
- Enhancing Products and Services
- Identifying Consumer Pain Points
- Benchmarking and Best Practices

Common issues faced by consumers include:
- Incorrect information on credit reports
- Loan servicing, payments, and escrow account problems
- Dealing with lenders or servicers
- Managing an account
- Problems with credit reporting companies' investigations
- Attempts to collect debts not owed
- Disclosure verification of debt
- Communication tactics
- and more...

Through this exploration, we aim to improve the banking industry by providing actionable insights and recommendations based on the complaints dataset analysis.

## Dataset Location
- Full Dataset: input_data/customer_complaints_full.csv
  This dataset is too large to be uploaded to portal. It can be downloaded from the following link: https://www.consumerfinance.gov/data-research/consumer-complaints/
- sample dataset: input_data/customer_complaints_sample.csv
  The sample dataset is a subset of the full dataset, containing first few rows. It is used for testing and debugging purposes.

## Dataset Information
The Consumer Complaint Database consists of the following columns:
- Date received
- Product
- Sub-product
- Issue
- Sub-issue
- Consumer complaint narrative
- Company public response
- Company
- State
- ZIP code
- Tags
- Consumer consent provided?
- Submitted via
- Date sent to company
- Company response to consumer
- Timely response?
- Consumer disputed?
- Complaint ID

## Usage
This Jupyter Notebook is meant to explore and analyze the Consumer Complaint Database. It offers various data exploration and visualization techniques to understand customer complaints. Tasks that can be accomplished include:

- Data Exploration:
  - Examining distribution of complaints across different product categories
  - Analyzing the frequency of complaints over time
  - Exploring content of customer complaint narratives

- Visualizations:
  - Creating word clouds to visualize most frequent terms in complaint narratives
  - Generating bar charts or pie charts to illustrate distribution of complaints by product category
  - Plotting line charts to show the trend of complaint submissions over time

## Dependencies
This notebook relies on the following Python libraries:
- pandas
- plotly
- wordcloud
- mlflow

Ensure these dependencies are installed before running the notebook.

## References
- Consumer Complaint Database
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [plotly Documentation](https://plotly.com/python/)
- [wordcloud Documentation](https://amueller.github.io/word_cloud/)
- [mlflow Documentation](https://mlflow.org/docs/latest/index.html)

## Step 1: Import Libraries
In this step, we import the necessary libraries that will be used throughout the notebook. These libraries provide functionality for data manipulation, visualization, and experiment tracking.

### Libraries Used
The following libraries are imported in this step:
- pandas
- plotly
- wordcloud
- mlflow

## Step 2: Data Preprocessing
This step focuses on preprocessing the data to ensure its suitability for further analysis. The preprocessing steps include:

- **Dropping Rows with Missing Consumer Complaint Narrative**: We exclude rows where the narrative is NaN or null, ensuring complete complaint narratives for the analysis.
- **Dropping Duplicate Consumer Complaint Narratives**: Any duplicate rows based on the consumer complaint narrative column are removed to prevent redundancy.
- **Changing Date Formats**: The format of the "Date received" and "Date sent to company" columns are converted to a standardized date format using the pd.to_datetime function for easier manipulation and analysis.
- **Dropping Rows with Inconsistent Dates**: Any rows where the "Date received" is later than the "Date sent to company" are removed to maintain logical consistency in the date information.
- **Dropping Irrelevant Column**: The column "Consumer disputed?" is dropped as it does not provide any relevant information for our analysis, since it contains only 0 values indicating no disputes.

These preprocessing steps ensure that the dataset is clean and ready for subsequent analysis, visualization, and modeling tasks. The preprocessed dataset, represented as a pandas DataFrame, is the output of this step.

Data preprocessing is crucial in assuring the quality and reliability of the data. It addresses issues such as missing values and duplicates, enhancing the data's accuracy and integrity.

## Step 3: Explore Structured Columns
This step involves the exploration of structured columns in the dataset. The purpose is to derive insights from various aspects of the consumer complaints data. Below are the explorations:

- **Product**: The `explore_product` function analyzes the distribution and frequency of different products mentioned in consumer complaints. It generates visualizations such as bar plots or pie charts.

- **Issue**: The `explore_issue` function investigates the types of issues reported by consumers and their frequency. It generates visualizations or summaries highlighting common issues faced by consumers.

- **Date received by the CFPB**: The `explore_date_received` function examines the timeline of consumer complaints received by the Consumer Financial Protection Bureau (CFPB), aiding in identifying any temporal patterns or trends.

- **Company response columns**: The `explore_company_response` function focuses on different responses provided by companies to consumer complaints and their distribution. It may generate visualizations, such as bar plots, showcasing these distributions.

- **Count by State**: The `explore_state` function determines the number of consumer complaints reported from each state, potentially producing visualizations like bar plots or choropleth maps for geographical distribution.

To execute these explorations, run the respective function calls providing the preprocessed DataFrame `processDF` as an input.

Upon completion of this step, we will have a more comprehensive understanding of the structured columns in the dataset, thereby gaining valuable insights into the consumer complaints data.

## Step 4: Explore Unstructured Columns
In this step, we delve into the 'Consumer Complaint Narrative' column. This particular column contains detailed narratives or descriptions provided by consumers about their complaints. These narratives can encompass a wide range of information, from specific issues consumers faced to relevant experiences they wish to share. This unstructured text data offers a wealth of insights that can aid in understanding the context of the complaints and consumers' experiences in more depth.

## Conclusion
This Jupyter Notebook, together with this README.md guide, offers a robust and systematic approach to delve into the Consumer Complaint Database. The exploratory and analytical procedures delineated here, with emphasis on consumer complaints related to the banking domain, open avenues for remarkable discoveries.

By utilizing this guide, users are equipped to unveil critical insights encompassing the scope of consumer complaints, the implicated products and services, frequencies of these complaints, and discernible trends over time. Such profound understanding proves invaluable for organizations who aspire to augment their customer experience, refine their product and service offerings, and address consumer grievances proactively and effectively.

The overarching aim of this exploration is to bring about tangible improvements in the banking industry, by offering actionable insights and practical recommendations. In a rapidly evolving landscape of the banking sector, marked by constant technological innovations and shifting consumer expectations, the continual exploration and examination of complaint data becomes increasingly imperative. 

A deep understanding of consumer pain points and grievances is essential in tailoring effective solutions and meeting ever-changing demands. Armed with the insights garnered from this data exploration, businesses can better respond to challenges, devise strategies to improve, and thus enhance their service offerings. It enables them to turn these complaints into opportunities for development and improvement, thereby contributing to a more customer-centric banking environment.

Ultimately, we strive to create an improved banking ecosystem that not only resolves consumer complaints effectively but also anticipates potential issues and addresses them proactively. By providing this framework for analysis, we hope to empower businesses to leverage data, understand their customers better, and create a more robust and responsive banking industry for the future.

## Citations
1. Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Python Official Documentation: https://www.python.org/doc/
3. Pandas Official Documentation: https://pandas.pydata.org/pandas-docs/stable/
4. Plotly Official Documentation: https://plotly.com/python/
5. Wordcloud Documentation: https://amueller.github.io/word_cloud/
6. MLflow Documentation: https://mlflow.org/docs/latest/index.html

## Steps to Execute Code

To replicate this analysis and utilize the Jupyter Notebook, follow the below steps:

1. **Clone the Repository:** Start by copying this repository to your local machine.
2. **Install Necessary Libraries from requirements.txt:** Make sure to install all the required Python libraries mentioned in the "requirements.txt". You can do so using pip installation.
3. **Download the Sample Data:** Download the sample data provided, which represents 0.01% of the last two years of the Consumer Complaint Database.
4. **Move the Sample Data to the Local Repository:** Move the downloaded data into the same directory where you've cloned this repository.
5. **Open the Jupyter Notebook:** Open the Jupyter Notebook in the local repository.
6. **Run Each Cell:** Navigate through each cell in the notebook, running them one at a time. Make sure to read the comments and markdown cells for context.
7. **Review the Results:** After you have run all cells, review the output of the cells to gain insights from the data analysis.
8. **MLflow Usage:** MLflow is used in this notebook for experiment tracking. Once the notebook is executed, you can view the MLflow UI to see the tracked experiments. Run the MLflow UI in a new terminal using the command: _mlflow ui_.
The MLflow UI runs at `localhost:5000` by default. Open it in your web browser to view the logged runs.
9. **Review the Results:** After you have run all cells and checked the MLflow UI, review the output of the cells and the logged experiments to gain insights from the data analysis.

Please note, it's important to run all the cells in the order they are in to ensure that all variables and functions are declared before they are used.



