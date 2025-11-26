# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
# recommender.py
"""
Item-based collaborative filtering recommender using adjusted cosine similarity.
Only uses classic item-item CF.
"""
import numpy as np
import pandas as pd
import streamlit as st
from typing import List


class ItemBasedRecommender:
    """OOP wrapper for building and querying an item-based CF recommender.

    Attributes:
        user_item (pd.DataFrame): pivot table (index: users, columns: items) of ratings
        item_index (dict): mapping item -> column index in matrices
        index_item (dict): reverse mapping
        sim_matrix (np.ndarray): item-item similarity matrix
        items_meta (pd.DataFrame): product metadata (name, brand, categories)
    """

    def __init__(self):
        self.user_item = None
        self.item_index = None
        self.index_item = None
        self.sim_matrix = None
        self.items_meta = None
        self.user_list = []

    def fit(self, df: pd.DataFrame, user_col: str = "reviews_username", item_col: str = "name", rating_col: str = "reviews_rating"):
        """Build pivot and compute similarity matrix.
        This method caches the similarity matrix computation using Streamlit's cache.
        """
        # Build pivot table: rows users, columns items
        pivot = df.pivot_table(index=user_col, columns=item_col, values=rating_col, aggfunc="mean")
        self.user_item = pivot
        self.user_list = list(pivot.index)

        # Items metadata: keep first observed brand/categories
        meta = df[[item_col, "brand", "categories"]].drop_duplicates(subset=[item_col]).set_index(item_col)
        self.items_meta = meta

        # Map items to indices
        items = list(pivot.columns)
        self.item_index = {item: idx for idx, item in enumerate(items)}
        self.index_item = {idx: item for item, idx in self.item_index.items()}

        # Compute similarity matrix (cached)
        self.sim_matrix = compute_adjusted_cosine_similarity(pivot)

    def recommend_for_user(self, username: str, top_k: int = 10) -> pd.DataFrame:
        """Return top_k recommended items for a given user.

        Formula (item-based CF):
            pred(u, j) = sum_i sim(j, i) * r(u, i) / sum_i |sim(j, i)|
        where i iterates over items rated by user u.
        and j iterates over all items (not rated by user u)
        """
        if username not in self.user_item.index:
            return pd.DataFrame()

        user_ratings = self.user_item.loc[username]
        rated_mask = user_ratings.notna()
        rated_items = user_ratings.index[rated_mask]
        if len(rated_items) == 0:
            return pd.DataFrame()

        rated_idx = [self.item_index[it] for it in rated_items if it in self.item_index]
        ratings_vector = user_ratings.fillna(0).values  # length = n_items (aligns with item order)

        # Compute scores for all items: vectorized
        sim_sub = self.sim_matrix[:, rated_idx]  # shape (n_items, n_rated)
        numerators = sim_sub.dot(ratings_vector[rated_idx])
        denominators = np.sum(np.abs(sim_sub), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.where(denominators > 0, numerators / denominators, 0.0)

        # Exclude already rated items
        for it in rated_idx:
            scores[it] = -np.inf

        # Top-k
        top_indices = np.argpartition(-scores, range(min(top_k, len(scores))))[:top_k]
        top_sorted = top_indices[np.argsort(-scores[top_indices])]

        result_rows = []
        for rank, idx in enumerate(top_sorted, start=1):
            item_name = self.index_item[idx]
            meta = self.items_meta.loc[item_name] if item_name in self.items_meta.index else {"brand": None, "categories": None}
            result_rows.append({
                "rank": rank,
                "product_name": item_name,
                "categories": meta.get("categories") if isinstance(meta, dict) else meta.get("categories"),
                "brand": meta.get("brand") if isinstance(meta, dict) else meta.get("brand"),
                "pred_score": float(scores[idx])
            })

        rec_df = pd.DataFrame(result_rows)
        return rec_df

    def get_user_rated_items(self, username: str) -> pd.DataFrame:
        """Return items the user has rated (name, rating)."""
        if username not in self.user_item.index:
            return pd.DataFrame()
        s = self.user_item.loc[username].dropna().reset_index()
        s.columns = ["product_name", "rating"]
        return s


@st.cache_data(show_spinner=False)
def compute_adjusted_cosine_similarity(pivot_table: pd.DataFrame) -> np.ndarray:
    """Compute item-item similarity matrix using adjusted cosine similarity.

    Steps:
    - pivot_table: users x items, values are ratings (NaN if not rated)
    - subtract each user's mean from their rated entries
    - fill missing entries with 0 (they don't contribute)
    - similarity = centered_matrix.T @ centered_matrix / (norms product)

    Returns:
        sim_matrix (n_items x n_items) numpy array with values in [-1,1]
    """
    # pivot_table: index = users, columns = items
    # compute user means (only over rated items)
    user_means = pivot_table.mean(axis=1)
    # subtract user means (broadcast across columns)
    centered = pivot_table.sub(user_means, axis=0).fillna(0)
    R = centered.values  # shape (n_users, n_items)

    # dot product between items
    item_dot = R.T.dot(R)  # shape (n_items, n_items)

    # item norms
    item_norms = np.sqrt(np.sum(R ** 2, axis=0))  # shape (n_items,)
    denom = np.outer(item_norms, item_norms)  # shape (n_items, n_items)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.divide(item_dot, denom)
        sim[denom == 0] = 0.0

    # set diagonal to 1
    np.fill_diagonal(sim, 1.0)
    return sim


# -----------------
# Commented: User-User CF snippet (DO NOT run; provided for reference only)
# -----------------
# def compute_user_user_similarity(pivot_table):
#     """User-based CF: compute cosine similarity between users (center by item mean if desired).
#     This is commented out intentionally (requirement: only item-item CF active).
#     """
#     from sklearn.metrics.pairwise import cosine_similarity
#     user_item = pivot_table.filln