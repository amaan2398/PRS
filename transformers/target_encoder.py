# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encodes categorical variables with mean target value."""
    def __init__(self, col):
        self.col = col
        self.target_map_ = None
    
    def fit(self, X, y):
        df = X.copy()
        df["target"] = y
        self.target_map_ = df.groupby(self.col)["target"].mean().to_dict()
        return self
    
    def transform(self, X):
        return X[self.col].map(self.target_map_).fillna(np.mean(list(self.target_map_.values()))).values.reshape(-1,1)

    # Add these methods for pickling
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
