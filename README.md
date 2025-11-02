# Phishing Detection through URL and Content Analysis

## Project Overview
This project aims to develop a machine learning model to distinguish between phishing and legitimate URLs using feature extraction techniques. The objective is to create an effective and accurate classification system to help protect users from phishing attacks.

## Table of Contents
- [Installation and Running](#installation)
- [Features](#features)
- [Data](#data)
- [Models](#models)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)


## Installation
To set up this project on your local machine, follow these steps:

First, ensure that Python and pip are installed on your system. Then, install the required packages using pip:

1. Download all necessary libraries:
   ```bash
   pip scrapy install pandas scikit-learn imbalanced-learn 
   ```


## Running the Web Scraping Legitimete URLs
Code not included in repo

1. Navigate to the project directory:
   ```bash
   cd urlcrawler
   ```

2. Run the following command to extract legitimate URLs using Scrapy Web Scraper using Spider: 
   ```bash
   scrapy crawl legit_urls
   ```
   optional to output the list of URLs:
   ```bash
   scrapy crawl legit_urls -o urls.json
   ```

This script will:

- This spider starts with a list of trusted and known legitimate websites (start_urls) and extracts all links found on those pages until it hits its set limit. 



## Running the Scrpts for Feature Extraction
1. Navigate to the project directory:
   

2. Run the following command to extract features from URLs and save them:
   ```bash
   python extract_features.py
   ```

This script will:

- Parse URLs to extract features like path length, use of HTTPS, and presence of IP addresses.
- Check for common phishing indicators within the URL.
- Output a CSV file urls_train.csv containing the features and labels.

## Running the Scrpts for Model Training and Evaluation
1. Navigate to the project directory:
   

2. Run the following command to extract features from URLs and save them:
   ```bash
   python main.py
   ```

This script will:

- Load the dataset from urls_train2.csv.
- Split the data into training and testing sets.
- Train multiple models such as Random Forest, Decision Tree, and K-Nearest Neighbors.
- Evaluate the models using metrics like accuracy, ROC-AUC, and provide a confusion matrix for each model.
- Perform cross-validation to assess model reliability.



### Notes
- Ensure that the paths to the CSV files and other dependencies are correctly set up in the scripts.
- Modify the Python and pip commands according to your operating system and Python environment (e.g., using `pip3` or `python3` if necessary).


## Features
The feature extraction process isolates key URL characteristics crucial for distinguishing phishing from legitimate URLs. Extracted features include:
- **Domain Characteristics**: Number of subdomains, presence of an IP address.
- **Path Analysis**: Path length, suspicious keyword presence.
- **Security Features**: Use of HTTPS protocol.
- **Deception Techniques**: URL shortening, '@' symbol presence, hyphen count.
- **Other Features**: URL length, query count.

## Data
- **Datasets**: 
  - **Phishing Database by PyFuncable** (for phishing URLs)
  - **Scrapy Web Scraper** (for legitimate URLs)
- **Preprocessing**: 
  - Labels: "phishing" and "legitimate"
  - Data Aggregation: Combines URLs from both sources and applies feature extraction.

## Models
The following classification models are used:
- **Random Forest Classifier**: Combines multiple decision trees for better accuracy.
- **Decision Tree Classifier**: Splits data into subsets for classification.
- **K-Nearest Neighbors (KNN)**: Classifies based on the nearest neighbors.

## Training and Evaluation
- **Training Pipelines**: Each model is wrapped in a pipeline for consistent preprocessing.
- **Cross-Validation**: Uses 5-fold cross-validation to ensure robustness.
- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of the model.
  - **Precision and Recall**: Important for phishing detection.
  - **F1-Score**: Combines precision and recall.
  - **ROC-AUC Score**: Measures discrimination ability.

## Results
**Average Results for 5 Extracted Features**:
| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Random Forest      | 0.81     | 0.83      | 0.81   | 0.81     | 0.94    |
| Decision Tree      | 0.82     | 0.85      | 0.83   | 0.82     | 0.82    |
| K-Nearest Neighbors| 0.82     | 0.85      | 0.83   | 0.82     | 0.82    |

**Average Results for 10 Extracted Features**:
| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Random Forest      | 0.86     | 0.87      | 0.86   | 0.86     | 0.95    |
| Decision Tree      | 0.81     | 0.83      | 0.81   | 0.81     | 0.81    |
| K-Nearest Neighbors| 0.86     | 0.87      | 0.86   | 0.86     | 0.86    |



