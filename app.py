"""
Streamlit app for Ebuss Recommendation System.
Combines User-User Collaborative Filtering with Sentiment Analysis.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from utils.data_file_manager import DataFileManager
from recommenders.user_based import UserBasedRecommender
from sentiment.analyzer import SentimentAnalyzer
from sklearn.preprocessing import FunctionTransformer

# --- Page config & title
st.set_page_config(page_title="Ebuss — Hybrid Recommender", layout="wide")
st.title("🛍️ Ebuss — Sentiment-Enhanced Recommendation System")
st.markdown(
    """
    Get personalized product recommendations powered by **User-User Collaborative Filtering**, 
    refined by **Sentiment Analysis** of product reviews.
    """
)

# Helper transformers
def bool_to_int_func(X):
    return X.astype(int)

def fill_na_text(X):
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]          # ✅ force 1D
    return X.fillna("").astype(str)  # ✅ return Series of strings

bool_to_int = FunctionTransformer(bool_to_int_func, validate=False)
text_fill_na = FunctionTransformer(fill_na_text, validate=False)


# --- Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# --- Caching functions
@st.cache_resource
def load_recommender():
    """Load and cache the User-Based Recommender."""
    return UserBasedRecommender(MODELS_DIR)

@st.cache_resource
def load_sentiment_analyzer():
    """Load and cache the Sentiment Analyzer."""
    return SentimentAnalyzer(MODELS_DIR)

@st.cache_data
def load_data():
    """Load and cache the final dataset."""
    csv_path = DATA_DIR / "df_final.csv"
    if not csv_path.exists():
        st.error(f"Data file not found: {csv_path}")
        print(f"Data file not found: {csv_path}")
        return pd.DataFrame()
    return DataFileManager.load_csv_data(csv_path)

# --- Load Resources
with st.spinner("Loading models and data..."):
    try:
        recommender = load_recommender()
        sentiment_analyzer = load_sentiment_analyzer()
        df_final = load_data()
    except Exception as e:
        st.error(f"Critical error loading resources: {e}")
        print(f"Critical error loading resources: {e}")
        st.stop()

if df_final.empty:
    st.stop()

# --- Sidebar / Input Section
with st.sidebar:
    st.header("Configuration")
    
    # User Selection
    # Filter users to those present in the CF model
    valid_users = sorted(recommender.user_list)
    selected_user = st.selectbox("Select Username", valid_users)
    
    st.markdown("---")
    
    # Sliders
    k_candidates = st.slider(
        "Number of products from CF (k)", 
        min_value=5, 
        max_value=50, 
        value=20,
        help="How many products to retrieve from the Collaborative Filtering model."
    )
    
    n_recommendations = st.slider(
        "Final recommendations (n)", 
        min_value=3, 
        max_value=k_candidates, 
        value=max(k_candidates//2, 3),
        help="How many products to show after sentiment filtering."
    )
    
    get_btn = st.button("Get Recommendations", type="primary")

# --- Recommendation Logic
if get_btn:
    st.divider()
    st.subheader(f"Top {n_recommendations} Recommendations for '{selected_user}'")
    
    # Step 1: Collaborative Filtering
    with st.status("Running Recommendation Pipeline...", expanded=True) as status:
        st.write("🔍 **Step 1:** Generating candidate products using User-User CF...")
        
        # Get top-k candidates
        # We ask for k_candidates
        candidates_df = recommender.get_recommendations(selected_user, k=k_candidates)
        
        if candidates_df.empty:
            status.update(label="No recommendations found.", state="error")
            st.warning("No recommendations could be generated for this user.")
            st.stop()
            
        candidate_products = candidates_df['product_name'].tolist()
        st.write(f"✅ Found {len(candidate_products)} candidate products.")
        
        # Step 2: Filter Data
        st.write("📂 **Step 2:** Retrieving reviews for candidate products...")
        # Filter df_final for these products
        df_candidates = df_final[df_final['product_name'].isin(candidate_products)].copy()
        
        if df_candidates.empty:
            status.update(label="No reviews found for candidates.", state="error")
            st.warning("No reviews found in dataset for the recommended products.")
            st.stop()
            
        # Step 3: Sentiment Analysis
        st.write("🧠 **Step 3:** Analyzing sentiment of reviews...")
        # Predict sentiment
        df_analyzed = sentiment_analyzer.predict_sentiment(df_candidates)
        # Step 4: Calculate Stats
        st.write("📊 **Step 4:** Calculating positive sentiment percentage...")
        
        product_stats = []
        for product in candidate_products:
            # Get reviews for this product
            prod_reviews = df_analyzed[df_analyzed['product_name'] == product]
            
            if prod_reviews.empty:
                continue
                
            total_reviews = len(prod_reviews)
            # Count positive sentiments
            # Assuming 'predicted_sentiment' column has "Positive"/"Negative" strings
            positive_count = (prod_reviews['predicted_sentiment'] == 'Positive').sum()
            positive_pct = (positive_count / total_reviews) * 100
            
            # Get metadata (brand, category) from the first review
            brand = prod_reviews['brand'].iloc[0] if 'brand' in prod_reviews.columns else "N/A"
            category = prod_reviews['categories'].iloc[0] if 'categories' in prod_reviews.columns else "N/A"
            avg_rating = prod_reviews['reviews_rating'].mean() if 'reviews_rating' in prod_reviews.columns else 0.0
            
            # Get CF predicted rating
            cf_rating = candidates_df[candidates_df['product_name'] == product]['predicted_rating'].values[0]
            
            product_stats.append({
                'Product Name': product,
                'Brand': brand,
                'Category': category,
                'CF Predicted Rating': cf_rating,
                'Avg Actual Rating': avg_rating,
                'Total Reviews': total_reviews,
                'Positive Sentiment %': positive_pct
            })
            
        # Create DataFrame
        final_df = pd.DataFrame(product_stats)
        
        # Step 5: Sort and Filter
        st.write("🔝 **Step 5:** Ranking by sentiment...")
        final_df = final_df.sort_values('Positive Sentiment %', ascending=False)
        final_recs = final_df.head(n_recommendations)
        
        status.update(label="Recommendation Pipeline Completed!", state="complete")

    # --- Display Results
    
    # Display as a nice table with progress bar for sentiment
    st.dataframe(
        final_recs,
        column_config={
            "Positive Sentiment %": st.column_config.ProgressColumn(
                "Positive Sentiment",
                help="Percentage of positive reviews",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "CF Predicted Rating": st.column_config.NumberColumn(
                "CF Score",
                format="%.2f"
            ),
            "Avg Actual Rating": st.column_config.NumberColumn(
                "Avg Rating",
                format="%.2f"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Optional: Detailed view
    with st.expander("View Detailed Analysis"):
        for _, row in final_recs.iterrows():
            st.markdown(f"### {row['Product Name']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Brand", row['Brand'])
            col2.metric("Positive Sentiment", f"{row['Positive Sentiment %']:.1f}%")
            col3.metric("Total Reviews", row['Total Reviews'])
            st.divider()

# Footer
st.markdown("---")
st.caption("Powered by: User-User Collaborative Filtering (Adjusted Cosine) & Logistic Regression Sentiment Model")