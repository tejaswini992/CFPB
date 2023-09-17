#!/usr/bin/env python
# coding: utf-8

# # Data programming in Python (DSM020-2023-APR)
# ## Coursework 1 Submission
# 
# ### _Consumer Complaint Database_
# 
# **Author**: Tejaswini Naidu - tn152
# 
# ### Overview
# The Consumer Complaint Database is a valuable dataset that contains customer complaints related to various products and services in banking domain. It serves as a comprehensive resource for analyzing and addressing consumer concerns, identifying patterns, and improving overall customer satisfaction from banking.
# 
# The database offers insights into the following areas:
# 
# 1. **Customer Complaints**: It provides a wide range of customer complaints, covering multiple industries such as banking, credit reporting, mortgage, debt collection, etc. These complaints shed light on the issues faced by consumers and the areas that need attention.
# 
# 2. **Product and Service Analysis**: By examining the complaints, we can gain a deeper understanding of the strengths and weaknesses of different products and services. This analysis can help companies identify areas for improvement and make informed business decisions.
# 
# 3. **Trends and Patterns**: The database enables the identification of trends and patterns in consumer complaints over time. This information is vital for companies to proactively address recurring issues and implement effective solutions.
# 
# 4. **Consumer Satisfaction**: By studying the complaints, companies can assess consumer satisfaction levels and take appropriate measures to enhance the customer experience. This includes improving customer service, refining product offerings, and streamlining processes.
# 
# This Jupyter Notebook serves as a documentation and exploration of the Consumer Complaint Database. Through data analysis and visualizations, we aim to uncover valuable insights and provide actionable recommendations for businesses and organizations to improve their customer satisfaction and address consumer concerns effectively.
# 
# ### Motivation
# As someone with a keen interest in the banking domain, my motivation behind exploring the complaints dataset is to gain insights into the common issues faced by consumers within the FinTech or Banking industries. By analyzing this dataset, I aim to understand the types of complaints submitted by consumers and identify the companies that are frequently mentioned in these complaints.
# 
# The primary motivations for exploring the dataset are as follows:
# 
# 1. **Improving Customer Experience**: By identifying the common issues faced by consumers, organizations can take proactive steps to address these concerns and enhance the overall customer experience. This includes streamlining processes, improving communication, and providing better support and resolution to customer complaints.
# 
# 2. **Enhancing Products and Services**: Understanding the types of complaints submitted by consumers helps in identifying areas where improvements can be made to the products and services offered by the company. This analysis can guide companies in refining their offerings, introducing new features, and ensuring customer satisfaction.
# 
# 3. **Identifying Consumer Pain Points**: By exploring the dataset, we can uncover specific pain points faced by consumers, such as incorrect information on credit reports, loan servicing issues, communication problems, and more. This knowledge allows organizations to prioritize their efforts and allocate resources to address the most critical areas affecting consumer satisfaction.
# 
# 4. **Benchmarking and Best Practices**: Analyzing the complaints dataset not only provides insights into the challenges faced by consumers but also allows for benchmarking against industry standards and best practices. This benchmarking helps companies identify areas where they may be falling behind competitors and adopt strategies to improve their performance.
# 
# Examples of common issues faced by consumers include:
# 
# - Incorrect information on credit reports
# - Loan servicing, payments, and escrow account problems
# - Dealing with lenders or servicers
# - Managing an account
# - Problems with credit reporting companies' investigations
# - Attempts to collect debts not owed
# - Disclosure verification of debt
# - Communication tactics
# - and more...
# 
# Through this exploration, we aim to contribute to the betterment of the banking industry by providing actionable insights and recommendations based on the analysis of the complaints dataset.
# 
# 
# ### Dataset Information
# The Consumer Complaint Database consists of the following columns:
# 
# 
# 0. `Date received`: Date when the complaint was received.
# 1. `Product`: Product associated with the complaint.
# 2. `Sub-product`: Sub-product associated with the complaint.
# 3. `Issue`: Issue raised in the complaint.
# 4. `Sub-issue`: Sub-issue associated with the complaint.
# 5. `Consumer complaint narrative`: Detailed narrative provided by the consumer for the complaint.
# 6. `Company public response`: Public response provided by the company for the complaint.
# 7. `Company`: Company against which the complaint was filed.
# 8. `State`: State associated with the complaint.
# 9. `ZIP code`: ZIP code associated with the complaint.
# 10. `Tags`: Tags associated with the complaint.
# 11. `Consumer consent provided?`: Indicates whether consumer consent was provided.
# 12. `Submitted via`: Method through which the complaint was submitted.
# 13. `Date sent to company`: Date when the complaint was sent to the company.
# 14. `Company response to consumer`: Response provided by the company to the consumer.
# 15. `Timely response?`: Indicates whether the response was provided in a timely manner.
# 16. `Consumer disputed?`: Indicates whether the consumer disputed the complaint.
# 17. `Complaint ID`: Unique identifier for each complaint.
# 
# ### Usage
# This Jupyter Notebook aims to explore and analyze the Consumer Complaint Database. It provides various data exploration and visualization techniques to gain insights into the customer complaints. Some of the tasks that can be performed using this notebook include:
# 
# 1. Data Exploration:
#    - Examining the distribution of complaints across different product categories.
#    - Analyzing the frequency of complaints over time.
#    - Exploring the content of customer complaint narratives.
# 
# 2. Visualizations:
#    - Creating word clouds to visualize the most frequent terms in complaint narratives.
#    - Generating bar charts or pie charts to illustrate the distribution of complaints by product category.
#    - Plotting line charts to show the trend of complaint submissions over time.
# 
# ### Dependencies
# This notebook relies on the following Python libraries:
# - pandas: for data manipulation and analysis.
# - plotly: for interactive and visually appealing visualizations.
# - wordcloud: for generating word clouds based on complaint narratives.
# - mlflow: for experiment tracking and managing machine learning workflows.
# 
# Make sure to install these dependencies before running the notebook.
# 
# ### References
# - [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
# - [pandas Documentation](https://pandas.pydata.org/docs/)
# - [plotly Documentation](https://plotly.com/python/)
# - [wordcloud Documentation](https://amueller.github.io/word_cloud/)
# - [mlflow Documentation](https://www.mlflow.org/docs/latest/index.html)
# 

# ---

# ## Step 1: Import Libraries
# 
# ### **Overview**
# In this step, we import the necessary libraries that will be used throughout the notebook. These libraries provide functionality for data manipulation, visualization, and experiment tracking.
# 
# ### **Libraries Used**
# The following libraries are imported in this step:
# - `pandas`: A powerful library for data manipulation and analysis.
# - `plotly`: An interactive and visually appealing library for creating visualizations.
# - `wordcloud`: A library for generating word clouds based on text data.
# - `mlflow`: A library for experiment tracking and managing machine learning workflows.
# 
# ### **Usage**
# By importing these libraries, we gain access to a wide range of functions and tools that enable us to perform data exploration, visualization, and experiment tracking in the subsequent steps of this notebook. These libraries provide a solid foundation for our analysis and help streamline our workflow.
# 
# ### Step 1.1: System Libraries

# In[1]:


# Importing the libraries
import os
import numpy as np
import time  # For time-related operations
import mlflow  # For experiment tracking and management
import pandas as pd  # For data manipulation and analysis
import plotly.express as px  # For creating interactive visualizations
import plotly.io as pio  # For configuring plot settings
import plotly.graph_objects as go  # For creating custom visualizations
from plotly.subplots import make_subplots  # For creating subplots

from wordcloud import WordCloud, STOPWORDS  # For generating word clouds and handling stopwords


# ### Step 1.2: User-defined Libraries
# In this sub-step, we import any additional user-defined libraries or custom functions that are specific to this notebook or the analysis being performed. This may include libraries or functions located in the `pyFuncs` folder. The following user-defined libraries are imported:
# - `mlflow_setup`: A module for setting up and managing MLflow experiments.
# - `plotly_config`: A module for configuring and customizing Plotly visualizations.
# 

# In[2]:


from pyFuncs import mlflow_setup as ml
from pyFuncs import plotly_config as pc


# ### Step 1.3: Pandas Configuration
# In this sub-step, we configure the display options for Pandas to ensure that column widths are widened to accommodate larger content. This allows for better readability when working with wide columns in the DataFrame.

# In[3]:

pd.set_option('display.max_columns', None)


# ### Step 1.4: Plotly Configuration
# In this sub-step, we configure Plotly using the `plotly_config` module to customize various aspects of the visualizations. The `plotly_config` module contains functions for setting global plot configurations, such as default color schemes, font styles, and layout options.

# In[4]:


pc.configure_plotly()


# ### Step 1.5: MLflow Configuration Object
# In this sub-step, we return the MLflow configuration object. The MLflow configuration object allows us to set and retrieve various MLflow configurations, such as the tracking server URL and experiment settings.

# In[5]:


use_mlflow = False

def return_mlflow_config(exp_name):
    """
    Return the mlflow config object
    """
    mlflow_config = ml.MLflowConfig(experiment_name=exp_name)
    # if experiment does not exist, create it
    if mlflow.get_experiment_by_name(mlflow_config.experiment_name) is None:
        mlflow_config.configure_mlflow()
    # else:
    #     mlflow_config.create_experiment()
    mlflow_config.get_experiment_id()
    return mlflow_config

if use_mlflow:
    # 0. Configure MLflow
    mlflow_config = return_mlflow_config("Consumer_Complaints")
    # start the run for this experiment
    mlflow.start_run(experiment_id=mlflow_config.experiment_id)
    # Save run id in a variable
    run_id = mlflow.active_run().info.run_id


# ---

# ## Step 2: Import Dataset
# 
# ### Overview
# In this step, we import the Consumer Complaint Database as our dataset. The dataset contains customer complaints and relevant information that we will be analyzing and visualizing in subsequent steps.
# 
# ### Dataset Description
# The Consumer Complaint Database is a structured dataset that includes the following columns:
# 
# 0. `Date received`: Date when the complaint was received.
# 1. `Product`: Product associated with the complaint.
# 2. `Sub-product`: Sub-product associated with the complaint.
# 3. `Issue`: Issue raised in the complaint.
# 4. `Sub-issue`: Sub-issue associated with the complaint.
# 5. `Consumer complaint narrative`: Detailed narrative provided by the consumer for the complaint.
# 6. `Company public response`: Public response provided by the company for the complaint.
# 7. `Company`: Company against which the complaint was filed.
# 8. `State`: State associated with the complaint.
# 9. `ZIP code`: ZIP code associated with the complaint.
# 10. `Tags`: Tags associated with the complaint.
# 11. `Consumer consent provided?`: Indicates whether consumer consent was provided.
# 12. `Submitted via`: Method through which the complaint was submitted.
# 13. `Date sent to company`: Date when the complaint was sent to the company.
# 14. `Company response to consumer`: Response provided by the company to the consumer.
# 15. `Timely response?`: Indicates whether the response was provided in a timely manner.
# 16. `Consumer disputed?`: Indicates whether the consumer disputed the complaint.
# 17. `Complaint ID`: Unique identifier for each complaint.
# 
# Below code snippet handles the Customer Complaint Database obtained from the Consumer Financial Protection Bureau (CFPB) website. The data source can be accessed through the following link: [Customer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
# 
# The code performs the following steps:
# 
# 1. Specify the URL of the Customer Complaint Database and define the file paths for the full dataset and the sample dataset.
# 
# 2. Check if the full dataset file and the sample dataset file already exist. If they exist, load them into separate DataFrames (`inDF` and `sample_dataset_df`).
# 
# 3. If the files don't exist, read the Customer Complaint Database from the URL, convert the 'Date Received' column to datetime type, and filter the DataFrame to include only the data from the last two years.
# 
# 4. Create the 'input_data' folder (if it doesn't exist) and save the filtered DataFrame as a CSV file (`full_dataset_file_path`) representing the full dataset.
# 
# 5. Additionally, save a sample of the filtered DataFrame (`sample_df`) as a separate CSV file (`sample_dataset_file_path`) for further analysis.
# 
# 6. Assign the filtered DataFrame to `inDF`, representing the full filtered dataset, and store the sample dataset in `sample_dataset_df`.
# 
# This code ensures that the Customer Complaint Database from the CFPB website is available, saves the full dataset and a sample of it for subsequent use, and avoids redundant downloading and processing if the files already exist.
# 

# In[6]:

#
# # Specify the URL of the Customer Complaint Database
# database_url = 'https://files.consumerfinance.gov/ccdb/complaints.csv.zip'
#
# # Define the file paths
# full_dataset_file_path = 'input_data/customer_complaints_full.csv'
# sample_dataset_file_path = 'input_data/customer_complaints_sample.csv'
#
# if os.path.exists(full_dataset_file_path) and os.path.exists(sample_dataset_file_path):
#     # If both files already exist, load them into DataFrames
#     inDF = pd.read_csv(full_dataset_file_path)
#     sample_dataset_df = pd.read_csv(sample_dataset_file_path)
# else:
#     # Read the Customer Complaint Database from the URL
#     complaints_df = pd.read_csv(database_url, compression='zip')
#
#     # This dataset is really huge, so will limit it to last 2 years data.
#
#     # Convert the 'Date Received' column to datetime type
#     complaints_df['Date received'] = pd.to_datetime(complaints_df['Date received'])
#
#     # Get the current year
#     current_year = pd.Timestamp.now().year
#
#     # Calculate the start and end dates for the last two years
#     start_date = pd.Timestamp(year=current_year - 2, month=1, day=1)
#     end_date = pd.Timestamp(year=current_year, month=12, day=31)
#
#     # Filter the DataFrame to include only the last two years' data
#     filtered_df = complaints_df[(complaints_df['Date received'] >= start_date) &
#                                 (complaints_df['Date received'] <= end_date)]
#
#     # Create the 'input_data' folder if it does not exist
#     os.makedirs('input_data', exist_ok=True)
#
#     # Save the filtered DataFrame as a CSV file (full dataset)
#     filtered_df.to_csv(full_dataset_file_path, index=False)
#
#     # Save a sample of the filtered DataFrame as a CSV file
#
#     sample_size = int(len(filtered_df) * 0.01)  # Calculate 0.01% of the dataframe's length
#     sample_df = filtered_df.head(sample_size)  # Get the first 0.01% of the dataframe
#     sample_df.to_csv(sample_dataset_file_path, index=False)  # Save the dataframe to a csv file
#
#     # Assign the filtered DataFrame to the full dataset DataFrame
#     inDF = filtered_df.copy()
#
# full_dataset_df contains the full filtered dataset
# sample_dataset_df contains a sample of the filtered dataset


# ---

# ## Step 3: Data Preprocessing
# 
# In this step, we preprocess the data to ensure its quality and suitability for further analysis. The preprocessing steps applied to the dataset are as follows:
# 
# 1. **Dropping Rows with Missing Consumer Complaint Narrative**: In order to focus on the consumer complaint narratives, we drop any rows where the narrative is NaN or null. This ensures that we have complete complaint narratives for analysis.
# 
# 2. **Dropping Duplicate Consumer Complaint Narratives**: We remove any duplicate rows based on the consumer complaint narrative column. This helps to eliminate redundancy in the dataset and ensures that each complaint is represented only once.
# 
# 3. **Changing Date Formats**: We convert the format of the "Date received" (which was already converted before) and "Date sent to company" columns to a standardized date format using the `pd.to_datetime` function. This allows for easier manipulation and analysis of the date information.
# 
# 4. **Dropping Rows with Inconsistent Dates**: We drop any rows where the "Date received" is greater than the "Date sent to company". This step ensures that the date information is logically consistent, as the complaint should have been received before it was sent to the company.
# 
# 5. **Dropping Irrelevant Column**: The column "Consumer disputed?" contains only 0 values, indicating that there are no disputes. Therefore, we drop this column as it does not provide any useful information for our analysis.
# 
# By performing these preprocessing steps, we ensure that the dataset is cleaned and ready for subsequent analysis, visualization, and modeling tasks. The preprocessed dataset, represented as a pandas DataFrame, is returned as the output.
# 
# It is important to preprocess the data to ensure its quality and reliability. Preprocessing helps to address missing values, eliminate duplicates
# 

# In[7]:


def preprocess_data(inDF):
    """
    Preprocess the data. Below are the steps:
    1. Drop the Rows if Consumer Complaint Narrative is NaN or Null
    2. Drop the Duplicates on Consumer Complaint Narrative
    3. Change format of Date received and Date sent to company
    4. Drop rows if Date received is greater than Date sent to company
    5. The columns `Date received` and `Date sent to company` were used to create a new column `Days to resolve`
    6. Column "Consumer disputed?" has only 0 values. Hence, dropping it.
    :param inDF: pandas dataframe representing the input data
    :return: pandas dataframe representing the preprocessed data (out_df)
    """
    start_time = time.time()
    
    # 1. Drop Rows if Consumer Complaint Narrative is NaN or Null
    # Going forward, we will be using only Consumer Complaint Narratives
    preprocess_df = inDF.dropna(subset=['Consumer complaint narrative'])

    # 2. Drop Duplicates on Consumer Complaint Narrative
    preprocess_df = preprocess_df.drop_duplicates(subset=['Consumer complaint narrative'])

    # 3. Change format of Date received and Date sent to company
    preprocess_df['Date received'] = pd.to_datetime(preprocess_df['Date received'])
    preprocess_df['Date sent to company'] = pd.to_datetime(preprocess_df['Date sent to company'])

    # 4. Drop rows if Date received is greater than Date sent to company
    # Reason: Date received of consumer complaint should be less than Date sent to company.
    # If not, it is an error
    preprocess_df = preprocess_df[preprocess_df['Date received'] <= preprocess_df['Date sent to company']]

    # 5. The columns `Date received` and `Date sent to company` were used to create a new column `Days to resolve`
    # which is the number of days taken by the company to resolve the complaint.
    preprocess_df['Days to resolve'] = (preprocess_df['Date sent to company'] -
                                        preprocess_df['Date received']).dt.days + 1

    # 6. Column "Consumer disputed?" has only 0 values. Hence, dropping it.
    out_df = preprocess_df.drop(columns=['Consumer disputed?'])
    
    out_df.sort_values(by=['Date received'], inplace=True)

    
    end_time = time.time()
    time_taken = end_time - start_time
    print("Preprocessing completed in {:.2f} seconds.".format(time_taken))

    return out_df

# processDF = preprocess_data(inDF)


# In[16]:


# processDF.to_csv('input_data/preprocessed_data.csv')


# ---

# ## Step 4: Explore Structured Columns
# 
# In this step, we explore all the structured columns in the dataset. The objective is to gain insights into various aspects of the consumer complaints data. Below are the details of each exploration:
# 
# 1. **Product**: Analyze the distribution and frequency of different products mentioned in consumer complaints.
#    - The `explore_product` function analyzes the "Product" column and generates visualizations such as bar plots or pie charts to show the distribution of different products mentioned in consumer complaints.
# 
# 2. **Issue**: Investigate the types of issues reported by consumers and their frequency.
#    - The `explore_issue` function examines the "Issue" column and generates visualizations or summaries to highlight the types of issues reported by consumers and their frequency. This helps in understanding the common issues faced by consumers.
# 
# 3. **Date received by the CFPB**: Examine the timeline of consumer complaints received by the Consumer Financial Protection Bureau (CFPB).
#    - The `explore_date_received` function analyzes the "Date received" column and presents the timeline of consumer complaints received by the CFPB. This can help identify any temporal patterns or trends in consumer complaints.
# 
# 4. **Company response columns**: Understand the different responses provided by companies to consumer complaints and their distribution.
#    - The `explore_company_response` function focuses on the company response columns and investigates the different responses provided by companies to consumer complaints. It may generate visualizations, such as bar plots, to showcase the distribution of company responses.
# 
# 5. **Count by State**: Determine the number of consumer complaints reported from each state.
#    - The `explore_state` function counts the number of consumer complaints reported from each state. It may produce visualizations, such as bar plots or choropleth maps, to visualize the distribution of complaints across different states.
# 
# To perform the exploration, please execute the corresponding function calls mentioned above, providing the preprocessed DataFrame `processDF` as the input.
# 
# After executing this step, we will have a better understanding of the structured columns in the dataset and gain valuable insights into the consumer complaints data.
# 

# ---
# ### Step 4.1: Explore Product Column
# 
# In this sub-step, we focus on exploring the "Product" column of the consumer complaints dataset. The goal is to analyze the distribution and frequency of different products mentioned in consumer complaints. Below is an explanation of the exploration process:
# 
# 1. Retrieve the "Product" column from the preprocessed DataFrame `processDF`.
# 2. Calculate the count of each unique product using the `value_counts()` function.
# 3. Sort the product counts in descending order for better visualization.
# 4. Utilize the Plotly library to create a bar graph representing the product distribution.
#    - The x-axis of the bar graph represents the different product categories.
#    - The y-axis indicates the count of consumer complaints for each product.
#    - Each bar is colored based on the respective product category.
#    - The graph is titled "Product By Highest Complaint Volume".
#    - The actual count of complaints is displayed on top of each bar as an integer.
#    - The total count of complaints across all products is displayed at the top of the plot.
# 
# Please execute the following code to explore the "Product" column:

# In[9]:


def custom_tick_labels(label):
    if label == "Credit reporting, credit repair services, or other personal consumer reports":
        return "Credit reporting, repair or reports"
    if label == "Credit card or prepaid card":
        return "Credit or prepaid card"
    return label


def explore_product(processDF):
    """
    Explore the Product column and generate a bar graph representing the distribution of products
    mentioned in consumer complaints.

    Parameters:
    processDF (pandas.DataFrame): Preprocessed DataFrame containing the consumer complaints data.

    Returns:
    None
    """
    data = processDF['Product'].value_counts()
    data = data.sort_values(ascending=False)

    fig = px.bar(data, x=data.values, y=data.index, color=data.index,
                 labels={'x': 'Count', 'y': 'Product', 'color': 'Product'},
                 title='Product By Highest Complaint Volume')

    fig.update_traces(texttemplate='%{x}', textposition='outside',
                      textfont_size=8)

    fig.add_annotation(x=0.5, y=1.1,
                       text="Total Complaints: " + str(data.sum()),
                       showarrow=False,
                       font=dict(size=12, color="black"),
                       xref="paper",
                       yref="paper")

    # Rotate the y-axis tick names for better readability
    fig.update_layout(yaxis_tickangle=-45)

    # Adjust the margin to fit the rotated plot
    fig.update_layout(margin=dict(l=100, r=20, b=100, t=100))

    # Remove the legend
    fig.update_layout(showlegend=False)

    # Remove the x-axis name
    fig.update_yaxes(title_text='', tickfont=dict(size=7))

    # Reduce the size of x-axis tick names
    fig.update_xaxes(tickfont=dict(size=8))

    # Customize the y-axis tick labels
    fig.update_yaxes(tickvals=data.index, ticktext=[custom_tick_labels(label) for label in data.index])

    pio.show(fig)
    # fig.write_html("plots/1.1_product_histogram.html")
    # mlflow.log_artifact("plots/1.1_product_histogram.html")

    return None




# explore_product(processDF)


# After exploring the Product type in the dataset, it appears that the category **Credit reporting, credit repair services, or other personal consumer reports** has a higher frequency of complaints compared to other product types. This suggests that issues related to **credit reporting, credit repair services, and personal consumer reports** are more prevalent among the complaints recorded in the database.

# ---
# ### Step 4.2: Explore Issue Column
# 
# 
# The `explore_issue` function analyzes the "Issue" column of the preprocessed DataFrame and generates a bar graph to visualize the distribution of issues mentioned in the consumer complaints.
# 
# 1. Count the occurrences of each issue in the "Issue" column.
# 2. Sort the issues in descending order based on their frequency.
# 3. Calculate the total number of different issues.
# 4. Select the top 10 most common issues.
# 5. Create a bar graph using Plotly, where the x-axis represents the issues and the y-axis represents the count of complaints for each issue.
# 6. Add labels to the graph to indicate the axis names and color legend.
# 7. Include the total number of complaints in the graph as an annotation.
# 8. Hide the x-axis labels on the ticks.
# 9. Save the generated graph as an HTML file for further reference.
# 10. Log the generated graph as an artifact in MLflow for tracking and reproducibility.
# 
# This function provides insights into the distribution of issues in the consumer complaints data, highlighting the top 10 issues with the highest complaint volume.

# In[10]:


def explore_issue(processDF):
    """
    Explore the "Issue" column of the preprocessed DataFrame and generate a bar graph.

    Parameters:
    processDF (pandas.DataFrame): Preprocessed DataFrame containing the consumer complaints data.

    Returns:
    None
    """
    data = processDF['Issue'].value_counts()
    data = data.sort_values(ascending=False)

    total_issues = len(data)
    top10_issues = data.index[:10]

    fig = px.bar(data[top10_issues], x=data[top10_issues].index,
                 y=data[top10_issues].values,
                 color=data[top10_issues].index,
                 labels={'x': 'Issue', 'y': 'Count', 'color': 'Issue'},
                 title='Showing Top 10 (out of {}) Issues by Highest Complaint Volume'.format(total_issues))

    fig.update_traces(texttemplate='%{y}', textposition='outside',
                      textfont_size=8)

    fig.add_annotation(x=0.5, y=1.1,
                       text="Total Complaints: " + str(data.sum()),
                       showarrow=False,
                       font=dict(size=12, color="black"),
                       xref="paper",
                       yref="paper")

    fig.update_xaxes(showticklabels=False)
    # replace 'index' with 'Issues' on x-axis
    fig.update_layout(xaxis_title="Issues")
    pio.show(fig)
    # fig.write_html("plots/2.1_issue_histogram.html")
    # mlflow.log_artifact("plots/2.1_issue_histogram.html")

# explore_issue(processDF)


# After exploring the Issue type in the Customer Complaint Database, it is observed that certain **issues stand out** and have a **higher frequency** (like _Incorrect information on your report_) of occurrence compared to others. The **Issue type** provides specific details about the nature of the **complaints** filed by consumers.
# 
# By examining the **Issue type**, we can gain insights into the most common **problems** or concerns raised by consumers. This information is valuable for identifying **patterns**, understanding customer **pain points**, and making improvements to **products**, **services**, and customer **support**.

# ---
# ### Step 4.3: Explore Date Features
# 
# The `explore_date_received` function analyzes the "Date received" and "Date sent to company" columns of the preprocessed DataFrame and generates line charts to visualize the distribution of complaints over time.
# 
# 1. Group the complaints by date received and calculate the count of complaints for each date.
# 2. Convert the "Date received" column to datetime format if it's not already in that format.
# 3. Sort the DataFrame by "Date received" in ascending order.
# 4. Repeat steps 1-3 for the "Date sent to company" column.
# 5. Create a Plotly figure and add two line charts: one for "Date received" and one for "Date sent to company".
# 6. Set the x-axis as the respective dates and the y-axis as the count of complaints.
# 7. Add markers to the line charts to indicate each data point.
# 8. Update the layout of the figure, including the title and axis labels.
# 9. Save the generated line chart as an HTML file for further reference.
# 10. Log the generated line chart as an artifact in MLflow for tracking and reproducibility.
# 
# Additionally, the function groups the complaints by product and date received to generate line charts showing the count of complaints for each product over time. The process is similar to steps 1-10, but applied to each product separately.
# 
# These line charts provide insights into the volume of complaints received and sent by the CFPB over time, as well as the distribution of complaints across different products.

# In[11]:


def explore_date_received(processDF):
    """
    Explore the Date received column using line chart
    :param processDF: DataFrame containing the preprocessed complaints data
    :return:
    """
    # Group the complaints by date received and calculate the count
    data_receive = processDF.groupby(pd.Grouper(key='Date received', freq='7D')).size().reset_index(name='Count')

    # Convert the 'Date received' column to datetime if it's not already in datetime format
    data_receive['Date received'] = pd.to_datetime(data_receive['Date received'])

    # Sort the DataFrame by 'Date received' in ascending order
    data_receive = data_receive.sort_values('Date received')

    # Now add the date the CFPB sent the complaint to the company
    # Group the complaints by date sent to company and calculate the count
    data_sent = processDF.groupby(pd.Grouper(key='Date sent to company', freq='7D')).size().reset_index(name='Count')
    data_sent['Date sent to company'] = pd.to_datetime(data_sent['Date sent to company'])
    data_sent = data_sent.sort_values('Date sent to company')

    # Using plotly to plot the graph with dotted lines for date received and date sent
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_receive['Date received'], y=data_receive['Count'],
                             mode='lines+markers',
                             name='Date received by the CFPB'))
    fig.add_trace(go.Scatter(x=data_sent['Date sent to company'], y=data_sent['Count'],
                             mode='lines+markers',
                             name='Date sent to company by the CFPB'))

    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    fig.update_layout(title_text='Total Complaints by Date Received and Date Sent to Company',
                      title_x=0.5,
                     legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.01
    ))
    pio.show(fig)

    # fig.write_html("plots/3.1_dates_line_chart.html")
    # mlflow.log_artifact("plots/3.1_dates_line_chart.html")

    # Group the complaints by product and date received
    complaints_by_product_date = processDF.groupby(['Product', pd.Grouper(key='Date received', freq='7D')]).size().reset_index(name='Count')

    complaints_by_product_date['Date received'] = pd.to_datetime(complaints_by_product_date['Date received'])

    # Sort the DataFrame by product and date received
    complaints_by_product_date = complaints_by_product_date.sort_values(['Product', 'Date received'])

    # Plot the count of complaints for each product using go.Scatter
    fig = go.Figure()

    for product in complaints_by_product_date['Product'].unique():
        product_data = complaints_by_product_date[complaints_by_product_date['Product'] == product]
        fig.add_trace(go.Scatter(
            x=product_data['Date received'],
            y=product_data['Count'],
            mode='lines',
            name=product
        ))

    # Set the x-axis and y-axis labels
    fig.update_layout(
        xaxis_title='Date Received (7-Day Interval)',
        yaxis_title='Count of Complaints',
        title_text='Total Complaints by Date Received and Product',
        title_x=0.5,
        legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.01
    )
    )
    pio.show(fig)

    # fig.write_html("plots/3.2_dates_line_chart_product.html")
    # mlflow.log_artifact("plots/3.2_dates_line_chart_product.html")


# explore_date_received(processDF)


# After analyzing the complaint data, it appears that there has been a **significant drop** in complaint volume in **May 2022** and in **recent months**. This observation suggests a decrease in the number of complaints received or sent to companies during these periods.
# 
# The reasons for this decline could be attributed to various factors, such as **improved customer experiences**, enhanced **complaint resolution processes**, or changes in **consumer behavior**. It is also important to consider external factors, industry trends, or regulatory changes that may have influenced the complaint volume.
# 
# By closely monitoring and analyzing complaint volume trends, organizations can proactively address any potential issues, enhance **customer satisfaction**, and continuously improve their products, services, and **complaint handling processes**.
# 
# When analyzing the complaint data and observing significant drops in complaint volume, it may be beneficial to consider the inclusion of exogenous variables. Exogenous variables are external factors that could potentially explain or indicate such drops in complaint volume.
# 
# Possible exogenous variables to consider include:
# 
# 1. **Seasonal Factors**: Some industries may experience fluctuations in complaint volume due to seasonal variations in consumer behavior or market conditions. For example, there might be decreased complaint activity during holiday seasons or specific times of the year.
# 
# 2. **Economic Factors**: Economic indicators such as economic downturns, changes in unemployment rates, or fluctuations in interest rates could impact consumer behaviors and subsequently influence complaint volumes.
# 
# 3. **Regulatory Changes**: Changes in regulations or industry standards can have an impact on consumer behavior and the number of complaints filed. Monitoring regulatory changes relevant to the industry can help identify any potential correlation between these changes and complaint volume.
# 
# 4. **Marketing Campaigns**: The launch of new marketing campaigns or promotional activities can lead to changes in consumer perceptions or behaviors, potentially affecting complaint volumes. It is important to track the timing of such campaigns and evaluate their impact on complaint trends.
# 
# By including relevant exogenous variables in the analysis, it becomes possible to uncover additional insights into the factors contributing to drops in complaint volume. This allows organizations to gain a more comprehensive understanding of the overall landscape and make informed decisions to improve customer satisfaction and complaint management processes.
# 
# To ensure that the decline in complaint volume is not indicative of underlying issues going unnoticed or unresolved, it is crucial for organizations to further investigate the underlying causes. Additionally, **tracking complaint trends** over time will help identify any potential patterns or emerging concerns that may require attention.
# 
# 

# ---
# ### Step 4.4: Explore Days to Resolve
# 
# The `explore_days_to_resolve` function explores the distribution of the "Days to resolve" column in the preprocessed DataFrame.
# 
# 1. Grouping the complaints by the "Days to resolve" column and calculating the count for each unique value, generating a DataFrame called `data`.
# 2. Sorting the DataFrame `data` by the "Days to resolve" column in ascending order.
# 3. Creating intervals of 25 days by calculating the minimum and maximum values of the "Days to resolve" column.
# 4. Using pandas' `cut` function to categorize the "Days to resolve" column into these intervals, assigning corresponding labels to each interval.
# 5. Plotting a bar chart using plotly (`px.bar`) with the categorized "Days to resolve" on the x-axis, complaint count on the y-axis, and colors based on the categories.
# 6. Setting the x-axis label as "Days to Resolve," y-axis label as "Count of Complaints," and title as "Total Complaints by Days to Resolve."
# 
# This function provides a visual representation of the distribution of complaint resolution times, allowing the identification of common resolution periods and potential outliers. The histogram helps understand the frequency of different intervals and their corresponding complaint counts, enabling insights into the overall resolution efficiency.
# 

# In[12]:


def explore_days_to_resolve(preprocessDF):
    """
    Explore the Days to resolve column using a histogram
    """
    # Group the complaints by days to resolve and calculate the count
    data = preprocessDF.groupby(['Days to resolve']).size().reset_index(name='Count')

    # Sort the DataFrame by days to resolve in ascending order
    data = data.sort_values('Days to resolve')

    # Lets create days interval of 25 days by calculating the min and max days to resolve
    min_days = data['Days to resolve'].min()
    max_days = data['Days to resolve'].max()

    data['Days to resolve'] = pd.cut(data['Days to resolve'], bins=[min_days, 25, 50, 75, 100, max_days],
                                        labels=['1-25', '25-50', '50-75', '75-100', '100 and beyond'])

    # Plot the histogram using plotly
    fig = px.bar(data, x='Days to resolve', y='Count', color='Days to resolve',
                    title='Total Complaints by Days to Resolve')

    # Set the x-axis and y-axis labels
    fig.update_layout(
        xaxis_title='Days to Resolve',
        yaxis_title='Count of Complaints',
        title_text='Total Complaints by Days to Resolve',
        title_x=0.5,
        showlegend=False
    )
    pio.show(fig)

    # fig.write_html("plots/3.3_days_to_resolve_histogram.html")
    #
    # # Add the plot to MLflow
    # mlflow.log_artifact("plots/3.3_days_to_resolve_histogram.html")
    
# explore_days_to_resolve(processDF)


# Based on the results of the provided code, it can be concluded that a significant number of complaints are resolved within the first **25 days**. This finding is based on the histogram analysis of the **Days to resolve** column above. This also suggests the presence of efficient complaint resolution processes during the initial stages.
# 
# **Hence**, efficient complaint handling within the early stages is crucial for ensuring customer satisfaction and prompt issue resolution. By focusing on resolving complaints within the first 25 days, organizations can strive to maintain high customer satisfaction levels and address issues in a timely manner.
# 
# These insights from the histogram analysis of **Days to resolve** provide valuable information for organizations to optimize their complaint management processes, improve response times, and enhance overall customer experience.

# ---
# ### Step 4.5: Explore Company Response
# 
# This function explores the "Companies" and "Company response" columns in the complaints data. It performs the following steps:
# 
# 1. Group the complaints by company and company response to calculate the frequency of each combination.
# 2. Sort the data by company and the count of responses in descending order.
# 3. Select the top 15 companies based on complaint volume.
# 4. Create a pie chart to visualize the distribution of complaint volume among the top 15 companies and their timely response status.
# 5. Save the pie chart as an HTML file and log it as an artifact in MLflow.
# 6. Create a bar chart to visualize the complaint volume of the top 15 companies and their timely response status.
# 7. Add the actual count on top of each bar and display the total complaint count at the top of the plot.
# 8. Save the bar chart as an HTML file and log it as an artifact in MLflow.
# 9. Group the complaints by company response to the consumer and count the number of complaints for each response type.
# 10. Create a bar chart to visualize the distribution of company responses to consumer complaints.
# 11. Add the actual count on top of each bar and display the total complaint count at the top of the plot.
# 12. Save the bar chart as an HTML file and log it as an artifact in MLflow.
# 
# These visualizations provide insights into the complaint volume and response patterns of different companies. The pie chart shows the distribution of complaint volume among the top 15 companies and their timely response status. The bar charts display the complaint volume for each company and their response types. These visualizations can help identify companies with high complaint volumes and understand their response patterns.
# 

# In[13]:


def explore_company_response(processDF):
    """
    Explore the Companies and company response columns
    :param processDF: DataFrame containing the preprocessed complaints data
    """
    # Group the complaints by company and company response
    company_response_df = processDF.groupby(['Company', 'Timely response?']).size().reset_index(name='Frequency')

    # Sort the DataFrame by company and count of responses
    company_response_df = company_response_df.sort_values(['Company', 'Frequency'], ascending=False)

    # Get the top 15 companies
    top15_companies = company_response_df['Company'].unique()[:15]

    # Get the data for only the top 15 companies
    company_response_df = company_response_df[company_response_df['Company'].isin(top15_companies)]

    # Using plotly to plot the graph with stacked bars
    fig = px.pie(company_response_df, values='Frequency', names='Company',
                 title="Top 15 Companies by Complaint Volume and Timely Response - Pie Chart")

    pio.show(fig)
    fig.write_html("plots/4.1_company_timely_response_pie_chart.html")
    mlflow.log_artifact("plots/4.1_company_timely_response_pie_chart.html")

    # Using plotly to plot the graph with stacked bars
    fig = px.bar(company_response_df, x="Company", y="Frequency", color="Timely response?",
                 title="Top 15 Companies by Complaint Volume and Timely Response - Bar Chart")

    # Add actual count on top of the bar as an int
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont_size=8)

    # Add all together total count at top of the plot
    fig.add_annotation(x=0.5, y=1.1, text="Total Complaints: " + str(company_response_df['Frequency'].sum()),
                       showarrow=False, font=dict(size=14, color="black"), xref="paper", yref="paper")

    # Set the order of categories on the x-axis
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})

    # Position the legend on the left inside the figure
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.01
    ))

    pio.show(fig)
    # fig.write_html("plots/4.2_company_timely_response_bar_chart.html")
    # mlflow.log_artifact("plots/4.2_company_timely_response_bar_chart.html")

    # Explore how companies respond to complaints using "Company response to consumer" only
    # Group the complaints by company response and count the number of complaints
    company_response_df = processDF.groupby('Company response to consumer').size().reset_index(name='Count')

    # Using plotly to plot the graph with stacked bars
    fig = px.bar(company_response_df, x="Company response to consumer", y="Count",
                 title="Company Response to Consumer")

    # Add actual count on top of the bar as an int
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont_size=8)

    # Add all together total count at top of the plot
    fig.add_annotation(x=0.5, y=1.1, text="Total Complaints: " + str(company_response_df['Count'].sum()),
                       showarrow=False, font=dict(size=14, color="black"), xref="paper", yref="paper")

    # Set the order of categories on the x-axis
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})

    # Position the legend on the left inside the figure
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="left",
        x=0.01
    ))

    pio.show(fig)
    # fig.write_html("plots/4.3_company_response_bar_chart.html")
    # mlflow.log_artifact("plots/4.3_company_response_bar_chart.html")

# explore_company_response(processDF)


# Based on the analysis of the figures from **Figure 4.5.1** and **Figure 4.5.2**, it can be observed that **Zions Debt Holdings** and **eToro USA LLC** hold the highest number of complaint volumes among the **top 15 companies** from dataset. Also, this insight highlights the need for special attention to be given to these companies in terms of addressing customer concerns, improving services, and ensuring timely responses.
# 
# Identifying companies with high complaint volumes allows organizations to prioritize their efforts in investigating the root causes of the complaints, enhancing their complaint resolution processes, and taking necessary actions to improve customer satisfaction and experience.
# 
# It is crucial for Zions Debt Holdings and eToro USA LLC to address the concerns raised by customers, improve their services, and establish effective communication channels to provide timely and satisfactory resolutions to complaints.
# 
# Analyzing the bar chart in **Figure 4.5.3**, it is evident that the majority of consumer complaints were resolved with the **"Closed with explanation"** status. This indicates that companies provided explanations or resolutions to address the concerns raised by consumers.
# 
# The high frequency of "Closed with explanation" responses suggests that companies are actively engaging with consumer complaints and making efforts to provide satisfactory explanations or resolutions. This response category demonstrates a commitment to addressing consumer concerns and ensuring transparency in the complaint resolution process.
# 
# However, it is important for companies to continuously monitor and improve their complaint resolution practices. By analyzing the distribution of different response categories, companies can identify areas for improvement and strive to provide even better resolutions to consumer complaints.
# 
# It is noteworthy that the "Company Response to Consumer" attribute plays a crucial role in shaping consumer perceptions and overall satisfaction. Companies should prioritize effective communication, empathy, and swift resolution of consumer complaints to maintain positive relationships with their customers.

# ---
# 
# ### Step 4.6: Explore response by state
# 
# The `explore_state` function explores the distribution of complaints across different states in the United States using a choropleth map.
# 
# 1. Grouping the complaints by the "State" column and calculating the count of complaints for each state, generating a DataFrame called `state_df`.
# 2. Using plotly's `Choropleth` to create a choropleth map, where each state is represented by a color based on the count of complaints.
# 3. Providing the spatial coordinates (locations) of each state, the count of complaints as the data to be color-coded (z), and specifying the location mode as 'USA-states'.
# 4. Setting the colorscale to 'Reds' to represent the variation in complaint counts.
# 5. Adding a color bar with the title "Count" to the plot to provide a visual reference for the color scale.
# 6. Updating the layout of the plot, including the title as "Total Complaints by State," aligning the title to the center (title_x=0.5), and limiting the map scope to the United States (geo_scope='usa').
# 7. Setting the template to "plotly_white" for a clean and visually appealing plot.
# 8. Adding a color bar title as "Map shading: Count of Complaints" to provide an explanation of the color coding.
# 
# This function provides a visual representation of the distribution of complaints across different states in the United States. The choropleth map helps identify states with a higher volume of complaints and allows for easy comparison between states. It provides valuable insights into the geographic distribution of complaints, enabling further analysis and investigation based on location.

# In[14]:


def explore_state(preprocessDF):
    """
    Explore the State column. We will use the layout of the United States to show the count.
    """
    # Group the complaints by state and count the number of complaints
    state_df = preprocessDF.groupby('State').size().reset_index(name='Count')

    # Using plotly to plot the graph with Choropleth and use the best template
    fig = go.Figure(data=go.Choropleth(
        locations=state_df['State'],  # Spatial coordinates
        z=state_df['Count'].astype(float),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Blues',
        colorbar_title="Count"
    ))

    fig.update_layout(
        title_text='Total Complaints by State',
        title_x=0.5,
        geo_scope='usa',  # limit map scope to USA
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Map shading: <br>Count of Complaints",
        )
    )
    pio.show(fig)

    # fig.write_html("plots/4.4_usa_map_state.html")
    # # Add the plot to MLflow
    # mlflow.log_artifact("plots/4.4_usa_map_state.html")

    
# explore_state(processDF)


# The choropleth map in above Figure represents the total number of complaints by state in the United States. The varying shades of blue on the map indicate the count of complaints for each state, with darker shades representing higher complaint volumes.
# 
# Analyzing the map provides insights into the geographical distribution of consumer complaints across the United States. It is evident that certain states have a higher concentration of complaints compared to others. These states might require closer attention from companies and regulatory bodies to address consumer concerns effectively. The colorbar on the right side of the map provides a scale to interpret the shading of the map. The darker the shade, the higher the count of complaints in that particular state.
# 
# This visualization enables a quick visual assessment of complaint hotspots and can assist in identifying geographic patterns in consumer dissatisfaction. It is essential for companies to leverage this information to understand regional variations in consumer experiences and tailor their customer service strategies accordingly.
# 
# Understanding the distribution of complaints by state can help companies identify potential areas for improvement, prioritize resources, and implement targeted measures to address consumer concerns. By addressing regional disparities and focusing on improving customer experiences across different states, companies can enhance overall customer satisfaction and loyalty.

# --- 
# 
# ## Step 5: Explore Unstructured Columns
# 
# Here, The `Consumer Complaint Narrative` column refers to a specific column in a dataset that contains narratives or descriptions provided by consumers regarding their complaints. It is a textual column where consumers can provide detailed information about their complaints, including the issues they faced, any relevant experiences, or specific details they want to share. Since dataset is large, We will use only first 10000 samples

# In[15]:


def explore_consumer_complaint_narrative(preprocessDF):
    """
    Consumer complaint narrative is a free form text field.
    We will use wordcloud to explore this field with the help of the wordcloud library and Plotly
    """
    # Get the text from consumer complaint narrative removing stop words
    text = " ".join(review for review in preprocessDF['Consumer complaint narrative'].dropna())
    stopwords = set(STOPWORDS)

    # Also remove any XXXX kind of words
    stopwords.update(["XXXX", "XX", "xx", "xxxx", "xxxxxx", "xxxxxxxx", "xxxxxxxxx", "xxxxxxxxxx",
                      "XXXXXX", "XXXXXXXX"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

    # Display the generated image with a nice plotly layout
    fig = px.imshow(wordcloud, template="plotly_white")
    fig.update_layout(
        title_text='Consumer Complaint Narrative Word Cloud',
        title_x=0.5
    )
    pio.show(fig)
    
    # fig.write_html("plots/5.1_consumer_complaint_narrative_wordcloud.html")
    # # Add the plot to MLflow
    # mlflow.log_artifact("plots/5.1_consumer_complaint_narrative_wordcloud.html")

    
# explore_consumer_complaint_narrative(processDF.head(10000))


# The `explore_consumer_complaint_narrative` function performs an analysis on the "Consumer complaint narrative" column in the dataset. This column contains free-form text provided by consumers to describe their complaints.
# 
# The function starts by extracting the text from the "Consumer complaint narrative" column, removing any missing values. It then preprocesses the text by removing common stop words and specific words like "XXXX" or variations of it, which are often used as placeholders for sensitive information.
# 
# Next, the function generates a word cloud visualization using the `wordcloud` library. A word cloud is a visual representation of the most frequently occurring words in a text corpus, where the size of each word represents its frequency. This visualization provides a quick overview of the most common terms used in consumer complaints.

# # Conclusion
# 
# Throughout this notebook, we've taken a deep dive into analyzing the dataset of consumer complaints. We've loaded and preprocessed the data, carried out exploratory data analysis, created visualizations to understand the distribution of data, and identified patterns and insights. 
# 
# We have made extensive use of several Python libraries such as pandas, matplotlib, seaborn, plotly, and WordCloud for data handling, visualization, and exploration. Moreover, we've ensured our reproducibility of results and traceability by leveraging MLflow for experiment tracking and artifact logging.
# 
# Through our analysis, we have gained substantial insights into the nature of consumer complaints, their classification, the frequency of different complaint types, and key words in the consumer narratives. These insights can be extremely valuable for financial institutions to improve their products and customer service, as well as for regulatory bodies to identify and mitigate systemic issues in the financial market. 
# 
# While we have already gathered a considerable amount of information, there's always more to be explored. In Coursework 2, we could employ natural language processing (NLP) techniques to further analyze the consumer complaint narratives and gain deeper insights. In addition, machine learning models could be trained to predict the product or issue based on the narrative, which could facilitate more efficient handling of incoming complaints.
# 
# Finally, please note that the `requirements.txt` file contains all the dependencies needed to run this notebook. The sample data, MLflow tracking data, and other resources used in this notebook have also been included. The `README.md` file provides a detailed guide about the content of this notebook and how to use it. Please follow the instructions in the `README.md` file to navigate and reproduce the results in this notebook.
# 

# In[ ]:




