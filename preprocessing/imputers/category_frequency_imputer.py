
# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
"""
Imputes the most frequent category for each row in a DataFrame.
"""
import collections
from typing import Dict, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.manager import ConfigManager
from preprocessing.nlp.text_processing import TextProcessor

class CategoryFrequencyImputer(BaseEstimator, TransformerMixin):
    """
    Analyzes category frequencies within a dataset and imputes the single most
    globally frequent category for each row.
    """
    _config: ConfigManager

    def __init__(self) -> None:
        """Initializes the imputer, setting up the empty frequency map."""
        self._logger_name = self.__class__.__name__
        if not hasattr(self, '_config'):
            self._config = ConfigManager()

        self._category_frequency_map: Dict[str, int] = collections.Counter()
        # Load config
        self._delimiter = self._config.get_config('category_imputation')['delimiter']
        self._default_category = self._config.get_config('category_imputation')['default_category']
        self._replacements = self._config.get_config('category_imputation')['replacements']

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoryFrequencyImputer':
        """
        Fits the imputer to the data, calculating the global frequency map.
        
        Args:
            X (pd.DataFrame): The input DataFrame containing the category column.
            y (pd.Series, optional): The target column (not used).
        
        Returns:
            CategoryFrequencyImputer: The fitted imputer.
        """
        return self.fit_frequency_map(X[y])

    def fit_frequency_map(self, categories_series: pd.Series) -> None:
        """
        Calculates and stores the global frequency map of all categories.

        Args:
            categories_series (pd.Series): The Pandas Series containing raw category strings.
        """
        # Ensure NaN values are treated as empty strings before processing
        series_filled = categories_series.fillna(self._default_category)
        print(f"[{self._logger_name}] Starting frequency calculation on {len(series_filled)} rows...")

        # Apply the cleaning and splitting function
        # This is O(N * S) where N=rows, S=avg string length, unavoidable with custom parsing
        all_categories_list = series_filled.apply(lambda ctg: TextProcessor.clean_and_split_str(ctg, delimiter=self._delimiter, replacements=self._replacements))

        # Flatten the list of lists
        # O(N*C) where C=avg categories per row. Flattening with generator expression is fast.
        flattened_categories = [
            ctg for sublist in all_categories_list 
            for ctg in sublist
        ]
        
        # Calculate frequencies
        self._category_frequency_map = collections.Counter(flattened_categories)
        return self

    def __str__(self) -> str:
        # Display Top 5 Categories (Replaces original snippet's display logic)
        return f"[{self._logger_name}] Found {len(self._category_frequency_map)} unique categories."

    def __repr__(self) -> str:
        return self.__str__()

    def display_top_n_categories(self, top_n: int = 5) -> None:
        """
        Displays the top N most frequent categories.
        
        Args:
            top_n (int, optional): Number of top categories to display. Defaults to 5.
        """
        # Display Top 5 Categories (Replaces original snippet's display logic)
        print("\n📊 Top Global Categories:")
        print(self._category_frequency_map.most_common(top_n))

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transforms the input DataFrame by imputing the most frequent category for each row.
        
        Args:
            X (pd.DataFrame): The input DataFrame containing the category column.
            y (pd.Series, optional): The target column (not used).
        
        Returns:
            pd.DataFrame: The transformed DataFrame with imputed categories.
        """
        return X.apply(lambda row: self._impute_category(row[y]), axis=1)

    def _impute_category(self, categories_str: Optional[str]) -> str:
        """
        Imputes the most frequent category for a given row.
        
        Args:
            categories_str (Optional[str]): The raw category string from a DataFrame row.
        
        Returns:
            str: The imputed category.
        """
        if not isinstance(categories_str, str) or not categories_str.strip():
            return self._default_category

        # Clean and split the categories
        cleaned_list =  TextProcessor.clean_and_split_str(categories_str, delimiter=self._delimiter, replacements=self._replacements)

        # If no valid categories, return default
        if not cleaned_list:
            return self._default_category

        max_count = -1
        final_category = self._default_category
        for ctg in cleaned_list:
            if (ctg in self._category_frequency_map) and (self._category_frequency_map[ctg] > max_count):
                max_count = self._category_frequency_map[ctg]
                final_category = ctg
        return final_category
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the imputer to the data and transforms the input DataFrame.
        
        Args:
            X (pd.DataFrame): The input DataFrame containing the category column.
            y (pd.Series, optional): The target column (not used).
        
        Returns:
            pd.DataFrame: The transformed DataFrame with imputed categories.
        """
        return self.fit(X, y).transform(X, y)