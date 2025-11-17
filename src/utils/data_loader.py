import pandas as pd

def load_data(file_path):
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Performs basic data preprocessing.
    """
    # Add your preprocessing steps here
    return df
