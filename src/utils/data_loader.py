import pandas as pd
from pathlib import Path

def load_data(file_path, **kwargs):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path: The absolute or relative path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
        The loaded pandas DataFrame.
    """
    try:
        # Use Path for robust file handling
        data_path = Path(file_path)
        print(f"Loading data from {data_path}")
        return pd.read_csv(data_path, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: No data to parse from file {file_path}")
        raise

# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Performs basic data preprocessing (e.g., cleaning, feature engineering).

#     Args:
#         df: The input pandas DataFrame.

#     Returns:
#         The processed pandas DataFrame.
#     """
#     # 1. Drop duplicates
#     df = df.drop_duplicates()

#     # 2. Handle missing values (Example)
#     # df['reviews_rating'].fillna(df['reviews_rating'].median(), inplace=True)

#     # 3. Type conversion (Example)
#     # df['reviews_date'] = pd.to_datetime(df['reviews_date'])

#     return df.copy() # Return a copy to avoid SettingWithCopyWarning
