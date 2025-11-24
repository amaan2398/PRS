import re
import random
from typing import List, Optional, Union, Dict, Any

import pandas as pd

class DataAnalyzer:
    """
    Provides static utility methods for exploratory data analysis (EDA), 
    reporting, and transformation of Pandas DataFrames, adhering to 
    Data Science and Google Python Style Guide best practices.
    """

    @staticmethod
    def analyze_value_distribution(
        df: pd.DataFrame, 
        column_name: str, 
        dropna: bool = False, 
        return_distribution: bool = False,
        limit: int = -1
    ) -> Union[pd.DataFrame, None]:
        """
        Calculates and displays the count and percentage distribution of unique values in a column.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            column_name (str): The name of the column to analyze.
            dropna (bool, optional): Whether to exclude NaN values from the distribution. Defaults to False.
            return_distribution (bool, optional): If True, returns the distribution DataFrame; otherwise, prints it.
            
        Returns:
            Union[pd.DataFrame, None]: The distribution DataFrame if requested, otherwise None.
        
        Raises:
            KeyError: If the specified column name is not found in the DataFrame.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")

        # Vectorized calculation of counts and percentages (O(n) optimized)
        counts: pd.Series = df[column_name].value_counts(dropna=dropna)
        total_rows: int = len(df)
        
        # Calculate percentage using vectorized operation
        percentage: pd.Series = (counts / total_rows) * 100
        
        # Combine results into a structured DataFrame
        distribution_df: pd.DataFrame = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentage.round(2)
        })

        unique_count: int = distribution_df.shape[0]
        
        print(f"🔍 Distribution Analysis for '{column_name}'")
        print(f"Found {unique_count} unique values (including NaN if dropna is False).")
        
        if return_distribution:
            return distribution_df
        else:
            print("Value Counts Distribution:")
            if limit == -1:
                print(distribution_df.to_markdown(numalign="left", stralign="left"))
            else:
                print(distribution_df.head(limit).to_markdown(numalign="left", stralign="left"))

    @staticmethod
    def analyze_missing_data(
        df: pd.DataFrame, 
        reporting_threshold_percent: float = 0, 
        drop_list_threshold_percent: float = 90
    ) -> List[str]:
        """
        Calculates and reports missing data statistics and identifies columns suitable for dropping.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            reporting_threshold_percent (float, optional): Columns must have at least this percentage 
                of missing data to be displayed in the report. Defaults to 0.
            drop_list_threshold_percent (float, optional): Columns with missing data above this 
                percentage are returned as a list recommended for dropping. Defaults to 90.
                
        Returns:
            List[str]: A list of column names exceeding the drop threshold.
        """
        missing_count: pd.Series = df.isnull().sum()
        
        if missing_count.sum() == 0:
            print("No missing values found in the DataFrame.")
            return []
            
        # Vectorized calculation for percentage
        missing_percentage: pd.Series = (missing_count / len(df)) * 100

        missing_info: pd.DataFrame = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage (%)': missing_percentage.round(2)
        })

        # Add unique count information (Vectorized Pandas)
        missing_info['Unique Values Count'] = df.nunique()
        
        # Filter out rows with threshold missing % and sort
        missing_info = missing_info[missing_info['Missing Percentage (%)'] >= reporting_threshold_percent]
        missing_info = missing_info.sort_values(by='Missing Count', ascending=False)

        print("📊 Missing Values Report")
        
        # Report based on the display threshold
        report_df = missing_info[missing_info["Missing Percentage (%)"] >= reporting_threshold_percent]
        
        if report_df.empty:
            print(f"No columns found with missing values above {reporting_threshold_percent}%.")
        else:
            print("Missing Values and Percentages (Filtered by reporting threshold):")
            print(report_df.to_markdown())

        # Identify columns for dropping based on the drop threshold
        columns_to_drop: List[str] = missing_info[
            missing_info["Missing Percentage (%)"] > drop_list_threshold_percent
        ].index.tolist()
        
        print(f"\nFound {len(columns_to_drop)} columns recommended for dropping (>{drop_list_threshold_percent}% missing).")
        return columns_to_drop

    @staticmethod
    def sample_random_data(df: pd.DataFrame, columns: List[str], n_samples: int = 10, return_sampled_data: bool = False):
        """
        Samples a random subset of rows from the DataFrame based on the specified columns and number of samples.
        
        Args:
            df (pd.DataFrame): The DataFrame to sample from.
            columns (List[str]): List of column names to include in the sample.
            n_samples (int, optional): Number of random samples to select. Defaults to 10.
            
        Returns:
            pd.DataFrame: A DataFrame containing the sampled rows and specified columns.
        """
        # Get 10 random indices
        random_indices = random.sample(range(len(df)), n_samples)

        # Select the random rows and specified columns
        sampled_data = df.iloc[random_indices][columns]

        # Display the sampled data
        print(sampled_data.to_markdown())
        if return_sampled_data:
            return sampled_data
