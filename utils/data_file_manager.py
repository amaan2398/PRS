import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union

class DataFileManager:
    """
    Provides static methods for robust, defensible data loading operations.
    """
    @staticmethod
    def load_csv_data(file_path: Union[str, Path], **kwargs: Dict[str, Any]):
        """
        Loads data from a CSV file into a pandas DataFrame.

        Args:
            file_path: The absolute or relative path to the CSV file.
            **kwargs: Additional keyword arguments to pass to pandas.read_csv.

        Returns:
            The loaded pandas DataFrame.
        """
        try:
            # Use Path for robust file handling and strict type checking
            data_path = Path(file_path)

            if not data_path.exists():
                 raise FileNotFoundError(f"File not found at: {data_path}")
            
            print(f"Loading data from {data_path.resolve()}")
            data_frame = pd.read_csv(data_path, **kwargs)
            print(f"Data successfully loaded. Shape: {data_frame.shape}")
            # Basic cleaning: ensure required columns exist and convert rating to numeric
            if "reviews_rating" in data_frame.columns:
                data_frame["reviews_rating"] = pd.to_numeric(data_frame["reviews_rating"], errors="coerce")
            return data_frame
        except FileNotFoundError as e:
            # Re-raise FileNotFoundError with the original error
            raise e
            
        except pd.errors.EmptyDataError:
            # Handle empty file
            print(f"Error: No data to parse from file: {file_path}")
            raise
            
        except Exception as e:
            # Catch other potential I/O or parsing errors (e.g., permission denied)
            print(f"An unexpected error occurred during file loading: {e}")
            raise IOError("Failed to read CSV due to an unexpected IO error.")

    @staticmethod
    def save_csv_data(data_frame: pd.DataFrame, file_path: Union[str, Path], **kwargs: Dict[str, Any]):
        """
        Saves a pandas DataFrame to a CSV file.

        Args:
            data_frame: The pandas DataFrame to save.
            file_path: The absolute or relative path to the CSV file.
            **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_csv.

        Returns:
            None
        """
        try:
            # Use Path for robust file handling and strict type checking
            data_path = Path(file_path)

            print(f"Saving data to {data_path.resolve()}")
            data_frame.to_csv(data_path, **kwargs)
            print(f"Data successfully saved. Shape: {data_frame.shape}")
        except Exception as e:
            # Catch other potential I/O or saving errors (e.g., permission denied)
            print(f"An unexpected error occurred during file saving: {e}")
            raise IOError("Failed to save CSV due to an unexpected IO error.")

    @staticmethod
    def exists(file_path: Union[str, Path]) -> bool:
        """
        Checks if a file exists at the specified path.

        Args:
            file_path: The absolute or relative path to the file.

        Returns:
            True if the file exists, False otherwise.
        """
        return Path(file_path).exists()