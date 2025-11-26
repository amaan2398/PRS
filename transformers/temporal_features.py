# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# -------- Temporal Feature Generator --------
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="reviews_date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        self.min_date_ = pd.to_datetime(X[self.date_col]).min()
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return pd.DataFrame({
            "review_month": df[self.date_col].dt.month,
            "review_day_of_week": df[self.date_col].dt.dayofweek,
            "review_year": df[self.date_col].dt.year,
            "review_recency_days": (df[self.date_col] - self.min_date_).dt.days
        })
    
    # Add these methods for pickling
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)