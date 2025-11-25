# app.py
"""
Streamlit app entrypoint for Ebuss Item-Based CF recommender.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from utils.data_file_manager import DataFileManager
from recommender import ItemBasedRecommender

# --- Page config & title
st.set_page_config(page_title="Ebuss — Item CF Recommender", layout="wide")
st.title("🛍️ Ebuss — Item-Based Collaborative Filtering Recommender")
st.markdown(
    """
    Select a user and get product recommendations powered by classic **item-item collaborative filtering** (adjusted cosine similarity).
    - Pure item-based CF only (user-user code provided commented out)
    - Works with a local CSV (see `data/processed/df_cleaned.csv`) or upload your own file
    """
)

# --- Sidebar: load data
# with st.sidebar:
#     st.header("Data")
#     st.caption("The app expects a CSV with columns including 'name' and 'reviews_username' and 'reviews_rating'.")
csv_path = "data/processed/df_cleaned.csv"  # default local path
df = DataFileManager.load_csv_data(csv_path=csv_path)

if df is None or df.empty:
    st.warning("No data available. Please upload a CSV or place 'data/processed/df_cleaned.csv' in the project folder.")
    st.stop()

# --- Basic data check
required_cols = {"name", "reviews_username", "reviews_rating", "categories", "brand"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"Input CSV missing columns. Required: {required_cols}")
    st.stop()

# --- Instantiate recommender (will compute similarity once and cache)
with st.spinner("Preparing recommender — computing item similarities (cached)..."):
    recommender = ItemBasedRecommender()
    recommender.fit(df)

# --- UI controls
users = recommender.user_list
col1, col2, col3 = st.columns([3, 1, 2])
with col1:
    selected_user = st.selectbox("Select username", users)
with col2:
    n_rec = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
with col3:
    get_btn = st.button("Get Recommendations")

# --- Get recommendations
if get_btn:
    if selected_user not in users:
        st.error("Selected user not in dataset")
    else:
        with st.spinner("Computing recommendations..."):
            recs = recommender.recommend_for_user(selected_user, top_k=n_rec)

        if recs.empty:
            st.info("No recommendations could be produced for this user.")
        else:
            st.success(f"Top {len(recs)} recommendations for '{selected_user}'")

            # Display nice table
            st.dataframe(recs.style.format({"pred_score": "{:.3f}"}), height=500)

            # Optional: show rated items by user
            with st.expander("Show items this user has already rated"):
                rated = recommender.get_user_rated_items(selected_user)
                st.table(rated)

# Footer
st.markdown("---")
st.caption("Implemented: Item-based CF with adjusted cosine similarity. No SVD, no deep learning.")