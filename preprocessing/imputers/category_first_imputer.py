import collections
from typing import List, Dict, Optional
import re
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.manager import ConfigManager
from preprocessing.nlp.text_processing import TextProcessor


class CategoryFirstImputer(BaseEstimator, TransformerMixin):
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
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoryFirstImputer':
        """
        Fits the imputer to the data, calculating the global frequency map.
        
        Args:
            X (pd.DataFrame): The input DataFrame containing the category column.
            y (pd.Series, optional): The target column (not used).
        
        Returns:
            CategoryFirstImputer: The fitted imputer.
        """
        if y is None:
            raise ValueError("y is not used in this imputer.")
        return self
    
    def __str__(self) -> str:
        # Display Top 5 Categories (Replaces original snippet's display logic)
        return  f"[{self._logger_name}] Imputing the 1st category for each row."
    
    def __repr__(self) -> str:
        return self.__str__()

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

        return cleaned_list[0]
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)
