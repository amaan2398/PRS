# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
# recommender_user.py
"""
User-based collaborative filtering recommender using adjusted cosine similarity.
"""
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from utils.data_file_manager import DataFileManager

class UserBasedRecommender:
    """
    User-based Collaborative Filtering Recommender.
    Uses pre-computed similarity matrices and user-item matrices.
    """
    
    def __init__(self, models_dir: Path):
        """
        Initialize the recommender by loading necessary models.
        
        Args:
            models_dir (Path): Path to the directory containing model files.
        """
        self.models_dir = models_dir
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_means = None
        self.user_list = []
        
        self._load_models()

    def _load_models(self):
        """Load the pickle files."""
        try:
            # Load user-item matrix
            self.user_item_matrix = DataFileManager.load_pickle_file(
                self.models_dir / "recommendation" / "user_item_matrix_full.pkl"
            )
            self.user_list = list(self.user_item_matrix.index)
            
            # Load user similarity matrix
            self.user_similarity_matrix = DataFileManager.load_pickle_file(
                self.models_dir / "recommendation" / "user_similarity_full.pkl"
            )
            
            # Load user means
            self.user_means = DataFileManager.load_pickle_file(
                self.models_dir / "recommendation" / "user_means_full.pkl"
            )
            
        except Exception as e:
            st.error(f"Error loading recommendation models: {e}")
            print(f"Error loading recommendation models: {e}")
            raise e

    def predict_rating(self, user: str, product: str, k: int = 10) -> float:
        """
        Predict rating for a user-product pair using User-User CF (Adjusted Cosine).
        
        Args:
            user (str): Username.
            product (str): Product name.
            k (int): Number of similar users to consider.
            
        Returns:
            float: Predicted rating (1-5).
        """
        # Handle unknown user/product
        if user not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if product not in self.user_item_matrix.columns:
            return self.user_means.get(user, self.user_item_matrix.mean().mean())

        # Get similar users who rated this product
        # Note: user_similarity_matrix is a DataFrame with users as index and columns
        if user not in self.user_similarity_matrix.index:
             return self.user_means.get(user, self.user_item_matrix.mean().mean())

        similar_users = self.user_similarity_matrix[user].sort_values(ascending=False)
        similar_users = similar_users.drop(user, errors='ignore')
        
        # Filter for users who have rated the target product
        # user_item_matrix has users as rows, products as columns
        rated_mask = self.user_item_matrix[product].notna()
        # We need to intersect similar_users index with rated_mask index
        # similar_users is a Series (index=users, value=similarity)
        # rated_mask is a Series (index=users, value=bool)
        
        # Align indices
        common_users = similar_users.index.intersection(rated_mask[rated_mask].index)
        similar_users_rated = similar_users.loc[common_users]
        
        top_k_users = similar_users_rated.head(k)
        
        if len(top_k_users) == 0:
            return self.user_means.get(user, self.user_item_matrix.mean().mean())
        
        # Weighted average with user mean adjustment
        user_mean = self.user_means[user]
        
        numerator = 0
        denominator = 0
        
        for similar_user, similarity_score in top_k_users.items():
            similar_user_rating = self.user_item_matrix.loc[similar_user, product]
            similar_user_mean = self.user_means[similar_user]
            
            # Add (rating - mean) * similarity
            numerator += similarity_score * (similar_user_rating - similar_user_mean)
            denominator += abs(similarity_score)
        
        if denominator == 0:
            return user_mean
        
        # Predicted rating = user_mean + weighted_average_of_deviations
        predicted_rating = user_mean + (numerator / denominator)
        
        # Clip to valid rating range
        return float(np.clip(predicted_rating, 1, 5))

    def get_recommendations(self, user: str, k: int = 20) -> pd.DataFrame:
        """
        Get top-k recommended products for a user.
        
        Args:
            user (str): Username.
            k (int): Number of recommendations to generate.
            
        Returns:
            pd.DataFrame: DataFrame with columns ['product_name', 'predicted_rating'].
        """
        if user not in self.user_item_matrix.index:
            # Fallback for new users: return popular products
            product_avg_ratings = self.user_item_matrix.mean().sort_values(ascending=False)
            return pd.DataFrame({
                'product_name': product_avg_ratings.head(k).index,
                'predicted_rating': product_avg_ratings.head(k).values
            })

        # Get unrated products
        user_ratings = self.user_item_matrix.loc[user]
        unrated_products = user_ratings[user_ratings.isna()].index
        
        predictions = []
        # Optimization: If unrated products list is huge, we might want to limit candidates.
        # But for this dataset (268 products), it's fine to iterate.
        
        for product in unrated_products:
            pred_rating = self.predict_rating(user, product, k=10) # k=10 neighbors as per notebook default
            predictions.append({'product_name': product, 'predicted_rating': pred_rating})
            
        # Sort by predicted rating
        recommendations = pd.DataFrame(predictions).sort_values('predicted_rating', ascending=False)
        
        return recommendations.head(k).reset_index(drop=True)
