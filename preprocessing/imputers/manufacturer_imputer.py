# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
import pandas as pd
from typing import List, Optional, Dict, Union

class ManufacturerImputer:
    """
    Implements a fast, hierarchical imputation strategy for missing manufacturer values.

    The imputation follows a strict hierarchy of specificity: Product > Brand > 
    Brand + Category > Category. Uses vectorized operations (.map) for speed.
    """

    # Dictionary to store the pre-calculated imputation maps
    _imputation_maps: Dict[str, pd.Series]

    def __init__(self) -> None:
        """Initializes the imputer instance."""
        self._imputation_maps = {}
        self._required_cols: List[str] = ['manufacturer', 'product_name', 'brand', 'categories']
        print(f"[{self.__class__.__name__}] Initialized. Ready to fit hierarchy.")

    def _calculate_mode_map(self, df: pd.DataFrame, group_cols: Union[str, List[str]]) -> pd.Series:
        """Helper to calculate the mode (most frequent) manufacturer for a given grouping."""
        
        # Use .dropna() only on the grouping column and the target column
        # .agg(lambda x: x.mode().iloc[0]) is the standard way to get the single mode.
        return (
            df.dropna(subset=['manufacturer'])
              .groupby(group_cols)['manufacturer']
              .agg(lambda x: x.mode().iloc[0])
        )

    def fit(self, df: pd.DataFrame) -> 'ManufacturerImputer':
        """
        Calculates the four hierarchical imputation maps based on the input data.

        Args:
            df (pd.DataFrame): The DataFrame used to calculate the frequency maps.

        Returns:
            ManufacturerImputer: The fitted imputer instance.
        
        Raises:
            KeyError: If required columns are missing.
        """
        if not all(col in df.columns for col in self._required_cols):
            missing = [col for col in self._required_cols if col not in df.columns]
            raise KeyError(f"Required columns missing for fitting: {missing}")

        print("\n[Fit] Calculating Imputation Maps...")

        # 1. Product-level mapping (Most Specific)
        self._imputation_maps['product'] = self._calculate_mode_map(df, 'product_name')

        # 2. Brand-level mapping
        self._imputation_maps['brand'] = self._calculate_mode_map(df, 'brand')

        # 3. Brand + Category mapping (Requires MultiIndex)
        self._imputation_maps['brand_cat'] = self._calculate_mode_map(df, ['brand', 'categories'])

        # 4. Category-level mode (Least Specific)
        self._imputation_maps['category'] = self._calculate_mode_map(df, 'categories')
        
        print("[Fit] Maps calculated successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the hierarchical imputation using vectorized Pandas operations (.map) 
        to maximize speed.

        Args:
            df (pd.DataFrame): The DataFrame to impute missing manufacturers in.

        Returns:
            pd.DataFrame: DataFrame with the new 'manufacturer_imputed' column.
        
        Raises:
            RuntimeError: If the imputer has not been fitted.
        """
        if not self._imputation_maps:
            raise RuntimeError("Imputer must be fitted using .fit() before calling .transform().")

        # Create a working copy and initialize the imputed column with the existing manufacturer
        df_imputed = df.copy() #.rename(columns={'manufacturer': 'manufacturer_original'})
        df_imputed['manufacturer_imputed'] = df_imputed['manufacturer']

        mask_missing = df_imputed['manufacturer_imputed'].isna()
        initial_missing_count = mask_missing.sum()
        print(f"\n[Transform] Starting imputation on {initial_missing_count} missing values.")

        # --- Vectorized Imputation Hierarchy ---

        # 1. Product-level mapping (Product_name -> Manufacturer)
        # Apply the map only where manufacturer is currently missing (mask_missing)
        df_imputed.loc[mask_missing, 'manufacturer_imputed'] = df_imputed.loc[mask_missing, 'product_name'].map(
            self._imputation_maps['product']
        )
        print("[Transform] Applied Product-level map.")

        # 2. Brand-level mapping
        mask_missing = df_imputed['manufacturer_imputed'].isna()
        df_imputed.loc[mask_missing, 'manufacturer_imputed'] = df_imputed.loc[mask_missing, 'brand'].map(
            self._imputation_maps['brand']
        )
        print("[Transform] Applied Brand-level map.")

        # 3. Brand + Category mapping (Requires MultiIndex/Merge)
        # Using .merge() is the vectorized way to map MultiIndex results
        mask_missing = df_imputed['manufacturer_imputed'].isna()
        
        # Prepare the subset for merging
        df_subset = df_imputed[mask_missing].copy()
        
        # Merge the subset with the brand_cat map on the join keys
        # We perform a left join to align manufacturer from the map onto the missing rows
        df_subset = df_subset.merge(
            self._imputation_maps['brand_cat'].rename('manufacturer_map'),
            left_on=['brand', 'categories'],
            right_index=True,
            how='left'
        )
        
        # Update the imputed column in the original DF copy based on the merge result
        df_imputed.loc[mask_missing, 'manufacturer_imputed'] = df_imputed.loc[mask_missing, 'manufacturer_imputed'].fillna(
            df_subset['manufacturer_map']
        )
        print("[Transform] Applied Brand+Category map (via Merge).")
        
        # 4. Category-level mode
        mask_missing = df_imputed['manufacturer_imputed'].isna()
        df_imputed.loc[mask_missing, 'manufacturer_imputed'] = df_imputed.loc[mask_missing, 'categories'].map(
            self._imputation_maps['category']
        )
        print("[Transform] Applied Category-level map.")
        
        final_missing_count = df_imputed['manufacturer_imputed'].isna().sum()
        imputed_count = initial_missing_count - final_missing_count
        print(f"[Transform] Imputation complete. Successfully imputed {imputed_count} values.")

        return df_imputed