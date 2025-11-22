import pandas as pd

def get_column_value_counts(df: pd.DataFrame, column_name: str, dropna: bool = False) -> None:
    """
    This function prints the value counts of a column in a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.
        dropna (bool, optional): Whether to drop NaN values. Defaults to False.
    
    Returns:
        pd.Series: The value counts of the column.
    """
    distribution: pd.Series = df[column_name].value_counts(dropna=dropna)
    unique_count = distribution.shape[0]

    print(f"Found {unique_count} unique values (including NaN if present).")
    print("Value Counts Distribution")
    
    if unique_count < 10:
        display(distribution)
    else:
        print(distribution.to_markdown(numalign="left", stralign="left"))
    
    return distribution

def get_missing_columns(df_raw: pd.DataFrame, threshold: float = 90) -> pd.DataFrame:
    """
    The function returns the columns with missing values above the threshold.
    
    Args:
        df_raw (pd.DataFrame): The DataFrame to analyze.
        threshold (float, optional): The threshold for missing values. Defaults to 90.
    
    Returns:
        pd.DataFrame: The missing values and percentages of the DataFrame.
    """
    missing_data = df_raw.isnull().sum()
    missing_percentage = (df_raw.isnull().sum() / len(df_raw)) * 100

    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage (%)': missing_percentage
    })

    # sort them
    missing_info = missing_info.sort_values(by='Missing Count', ascending=False)

    # Add a new column for unique values
    missing_info['Unique Values List'] = [df_raw[col].unique().tolist() for col in missing_info.index]

    missing_info['Unique Values Count'] = [len(df_raw[col].unique()) for col in missing_info.index]

    print("📊 Missing Values and Percentages:")
    if missing_info.empty:
        print("No missing values found in the DataFrame.")
    else:
        display(missing_info)
    missing_columns = missing_info[missing_info["Missing Percentage (%)"] > threshold].index
    print(f"Found {missing_columns.shape[0]} columns with missing values above {threshold}%.")
    return missing_columns.tolist()
