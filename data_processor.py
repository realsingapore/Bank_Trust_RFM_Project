import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

# Configure logging for background execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# csv_file_path = "C:\\Users\\elsingy\\Documents\\AMDARI DS\\Internship\\BankTrust\\Data\\BankTrustDataset.csv"


def fetch_data(csv_file_path):
    """Fetch data from CSV file"""
    
    # Check if file exists
    # good practise to log errors so it can be reviewed later in background tasks
    if not os.path.exists(csv_file_path):
        logging.error("CSV file not found: {}".format(csv_file_path))
        return None

    # Read CSV file
    try:
        data = pd.read_csv(csv_file_path)
        print("Successfully loaded {} transactions from {}".format(len(data), csv_file_path))
        return data

    except Exception as e:
        logging.error("Error loading data from CSV: {}".format(e))
        return None

def preprocess_data(data):
    """Preprocess the transaction data with consistency checks."""
    if data is None:
        print("No data available. Please fetch data first.")
        return None

    print("Starting Data Preprocessing...")
    df = data.copy()

    # Check for duplicates (Exactly like you have in Jupyter notebook)
    logging.info("Checking for duplicates in the dataset.")
    total_duplicates = df.duplicated().sum()
    id_duplicates = df['TransactionID'].duplicated().sum()
    logging.info(f"The total duplicates is : {total_duplicates}")
    logging.info(f"The total transaction duplicates is : {id_duplicates}")

    # Drop duplicated transaction IDs
    df = df.drop_duplicates(subset=['TransactionID'])

    # Data consistency checks
    logging.info("Performing data consistency checks...")
    # Count unique customers
    unique_customers = df['CustomerID'].nunique()
    logging.info(f"Unique customers: {unique_customers}")

    # Check for invalid dates
    logging.info("Checking for invalid transaction dates...")
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')

    # Drop missing values
    df = df.dropna()

    # Ensure proper datetime format
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')

    return df



def calculate_rfm_metrics(data):
    """Calculate RFM metrics and merge customer demographics."""
    if data is None:
        print("No data available. Please fetch and preprocess data first.")
        return None

    # Reference date is one day after the latest transaction
    reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)

    # RFM calculation
    rfm_df = data.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
        'TransactionID': 'count',                                      # Frequency
        'TransactionAmount': 'sum'                                     # Monetary
    }).reset_index()

    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Customer demographics (first known values)
    customer_demographics = data.groupby('CustomerID').agg({
        'CustomerDOB': 'first',
        'CustGender': 'first',
        'CustLocation': 'first',
        'CustAccountBalance': 'last'
    }).reset_index()

    # Merge RFM with demographics
    rfm_df = rfm_df.merge(customer_demographics, on='CustomerID', how='left')

    return rfm_df
