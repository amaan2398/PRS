# sentiment_analyzer.py
"""
Sentiment Analyzer using pre-trained Logistic Regression model.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from utils.data_file_manager import DataFileManager
from sklearn.preprocessing import FunctionTransformer

# Import transformers to ensure pickle can load them
# Note: The notebook imported them from `transformers` package.
# We assume the directory structure allows this import.
from transformers import TemporalFeatures, TargetEncoder, TextStats

# Helper functions that might be needed for unpickling FunctionTransformers
# These must match the names used in the notebook
# Helper transformers
def bool_to_int_func(X):
    return X.astype(int)

def fill_na_text(X):
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]          # ✅ force 1D
    return X.fillna("").astype(str)  # ✅ return Series of strings

bool_to_int = FunctionTransformer(bool_to_int_func, validate=False)
text_fill_na = FunctionTransformer(fill_na_text, validate=False)


class SentimentAnalyzer:
    """
    Sentiment Analyzer using pre-trained Logistic Regression model.
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize the analyzer by loading necessary models.
        
        Args:
            models_dir (Path): Path to the directory containing model files.
        """
        self.models_dir = models_dir
        self.model = None
        self.feature_pipeline = None
        self.feature_selection_mask = None
        self.label_encoder = None
        
        self._load_models()

    def _load_models(self):
        """Load the pickle files."""
        try:
            # Load feature pipeline
            self.feature_pipeline = DataFileManager.load_pickle_file(
                self.models_dir / "feature_pipeline.pkl"
            )
            
            # Load feature selection mask
            self.feature_selection_mask = DataFileManager.load_pickle_file(
                self.models_dir / "feature_selection_mask.pkl"
            )
            
            # Load best model (Logistic Regression)
            self.model = DataFileManager.load_pickle_file(
                self.models_dir / "best_model.pkl"
            )
            
            # Load label encoder (if available, otherwise assume 0=Negative, 1=Positive)
            # Notebook saved it as 'label_encoder.pkl'
            if (self.models_dir / "label_encoder.pkl").exists():
                self.label_encoder = DataFileManager.load_pickle_file(
                    self.models_dir / "label_encoder.pkl"
                )
            
        except Exception as e:
            st.error(f"Error loading sentiment models: {e}")
            raise e

    def predict_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict sentiment for a DataFrame of reviews.
        
        Args:
            df (pd.DataFrame): DataFrame containing review data.
                               Must contain columns expected by the pipeline.
        
        Returns:
            pd.DataFrame: Original DataFrame with 'predicted_sentiment' and 'sentiment_score' columns.
        """
        if df.empty:
            return df
        
        try:
            # 1. Transform features
            X_transformed = self.feature_pipeline.transform(df)
            
            # 2. Apply feature selection
            if self.feature_selection_mask is not None:
                # Handle sparse matrix or dense array
                if hasattr(X_transformed, "toarray"):
                    # If it's sparse, we might need to be careful. 
                    # But usually slicing works on csr_matrix.
                    X_selected = X_transformed[:, self.feature_selection_mask]
                else:
                    X_selected = X_transformed[:, self.feature_selection_mask]
            else:
                X_selected = X_transformed
            
            # 3. Predict
            predictions = self.model.predict(X_selected)
            probabilities = self.model.predict_proba(X_selected)
            
            # Map predictions to labels
            if self.label_encoder:
                predicted_labels = self.label_encoder.inverse_transform(predictions)
            else:
                predicted_labels = ["Positive" if p == 1 else "Negative" for p in predictions]
            
            # Get positive probability
            # Assuming class 1 is Positive (standard for binary classification)
            # If label_encoder is present, check classes_
            pos_class_idx = 1
            if self.label_encoder:
                # Find index of 'Positive'
                classes = list(self.label_encoder.classes_)
                if "Positive" in classes:
                    pos_class_idx = classes.index("Positive")
            
            pos_probs = probabilities[:, pos_class_idx]
            
            # Add to DataFrame
            result_df = df.copy()
            result_df['predicted_sentiment'] = predicted_labels
            result_df['sentiment_score'] = pos_probs
            
            return result_df
            
        except Exception as e:
            st.error(f"Error during sentiment prediction: {e}")
            print("Error during sentiment prediction:", e)
            return df
