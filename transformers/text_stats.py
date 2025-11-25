
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TextStats(BaseEstimator, TransformerMixin):
    """Extract text metrics such as length, word count, punctuation count."""
    def __init__(self, text_col):
        self.text_col = text_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.text_col] = df[self.text_col].fillna("")

        return np.vstack([
            df[self.text_col].str.len(),
            df[self.text_col].str.split().str.len(),
            df[self.text_col].str.count("!"),
            df[self.text_col].str.count(r"\?"),
            df[self.text_col].apply(lambda t: sum(1 for w in t.split() if w.isupper()))
        ]).T
        
    # Add these methods for pickling
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
