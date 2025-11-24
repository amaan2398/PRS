# data_loader.py
"""
Helpers to load the CSV dataset. If no local CSV exists, app accepts an uploaded file.
"""
import pandas as pd
from typing import Optional


def load_data(csv_path: str = "data/processed/df_cleaned.csv", uploaded_file: Optional[object] = None) -> Optional[pd.DataFrame]:
    """Load data from an uploaded file (Streamlit) or a local path.

    Args:
        csv_path: default local csv relative path
        uploaded_file: file-like (from Streamlit file_uploader)

    Returns:
        pandas.DataFrame or None if not found
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Basic cleaning: ensure required columns exist and convert rating to numeric
    if "reviews_rating" in df.columns:
        df["reviews_rating"] = pd.to_numeric(df["reviews_rating"], errors="coerce")
    return df