import pandas as pd
from typing import List, Set, Union, Optional

class DataCleaner:
    """
    Provides static methods for robust, defensivve data cleaning and validation
    operations on Pandas DataFrames.
    """

    @staticmethod
    def drop_specified_columns(
        df: pd.DataFrame, 
        missing_value_columns: Optional[List[str]] = None, 
        invalid_value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Safely drops a set of columns identified for removal.
        
        Args:
            df (pd.DataFrame): The DataFrame to process.
            missing_value_columns (Optional[List[str]]): Columns with excessive missing values.
            invalid_value_columns (Optional[List[str]]): Columns containing invalid data types or values.
            
        Returns:
            pd.DataFrame: A new DataFrame with specified columns dropped.
            
        Raises:
            TypeError: If the input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df).__name__}.")

        cols_missing = set(missing_value_columns or [])
        cols_invalid = set(invalid_value_columns or [])

        # Combine all target columns using set union
        target_cols_set: Set[str] = cols_missing | cols_invalid
        existing_cols: Set[str] = set(df.columns)

        # Find intersection: Columns that exist AND need dropping
        columns_to_drop_verified: List[str] = list(target_cols_set.intersection(existing_cols))
        
        # Find difference: Columns requested but not found in DF
        non_existent_columns: Set[str] = target_cols_set - existing_cols

        print("🗑️ Column Dropping Utility")

        if non_existent_columns:
            print(f"Warning: The following columns were not found and will be ignored: {non_existent_columns}")

        if not columns_to_drop_verified:
            print("No matching columns found to drop. Returning a copy of the original DataFrame.")
            return df.copy()

        # Execution
        try:
            df_cleaned = df.drop(columns=columns_to_drop_verified)
            
            print(f"Dropped columns: {columns_to_drop_verified}")
            print(f"Initial DataFrame shape: {df.shape}")
            print(f"Final DataFrame shape: {df_cleaned.shape}")
            
            print("\nCleaned Data Preview:")
            print(df_cleaned.head().to_markdown())
            
            return df_cleaned
            
        except Exception as e:
            print(f"Error: An unexpected error occurred during column dropping: {e}")
            raise
    
    @staticmethod
    def filter_by_date_validity(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Filters rows, keeping only those where the specified column contains a valid date.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            date_col (str): Name of the column containing date strings.
            
        Returns:
            pd.DataFrame: DataFrame with only valid date rows, index reset.
            
        Raises:
            KeyError: If the specified column is not found.
        """
        if date_col not in df.columns:
            raise KeyError(f"Column '{date_col}' not found in DataFrame.")

        # Create a temp column for processing and validation
        parsed_col_name = f'{date_col}_parsed_temp'
        df_processed = df.copy()

        print("📅 Date Validation Filter")
        print(f"Original shape: {df_processed.shape}")

        # CRITICAL OPTIMIZATION: pd.to_datetime is vectorized (C-speed)
        # It handles mixed formats and errors='coerce' turns junk/invalid dates into NaT (Not a Time)
        # We avoid the slow df[col].apply(lambda...) call from the original code
        df_processed[parsed_col_name] = pd.to_datetime(
            df_processed[date_col], 
            errors='coerce'
        )

        # Filter Logic: Keep rows where the parsed date is NOT null (i.e., valid)
        df_valid = df_processed.dropna(subset=[parsed_col_name])

        dropped_count = len(df_processed) - len(df_valid)
        print(f"Removed {dropped_count} invalid date rows.")
        
        # Clean up helper column and reset index
        df_valid = df_valid.drop(columns=[parsed_col_name])
        df_valid = df_valid.reset_index(drop=True)

        print(f"Final shape: {df_valid.shape}")
        return df_valid
