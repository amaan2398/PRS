# Ebuss Sentiment-Enhanced Recommendation System

## Overview

Ebuss is a hybrid product recommendation system designed to improve user experience by combining **User-User Collaborative Filtering** with **Sentiment Analysis**.

The system recommends products based on:

1.  **Collaborative Filtering**: Identifying products liked by users with similar rating patterns (Adjusted Cosine Similarity).
2.  **Sentiment Analysis**: Filtering and ranking these recommendations based on the sentiment of their reviews (Positive/Negative).

## Project Structure

```
d:\Projects\PRS\
├── app.py                      # Main Streamlit application entry point
├── config.json                 # Configuration file
├── recommenders/               # Recommendation logic
│   ├── user_based.py           # User-User CF implementation (Active)
│   └── item_based.py           # Item-Item CF implementation (Legacy)
├── sentiment/                  # Sentiment analysis logic
│   └── analyzer.py             # Sentiment Analyzer using Logistic Regression
├── utils/                      # Utility functions
│   └── data_file_manager.py    # Data loading and saving
├── transformers/               # Custom Scikit-learn transformers
├── models/                     # Pre-trained ML models (Pickle files)
├── data/                       # Dataset directory
│   └── processed/
│       └── df_final.csv        # Final processed dataset
└── notebooks/                  # Jupyter notebooks for training and analysis
```

## Key Features

- **Hybrid Pipeline**: Combines the breadth of Collaborative Filtering with the quality assurance of Sentiment Analysis.
- **User-User CF**: Uses Adjusted Cosine Similarity to account for user rating bias.
- **Sentiment Filtering**: Re-ranks products to ensure highly-rated items also have positive textual reviews.
- **Interactive UI**: Built with Streamlit, allowing dynamic configuration of recommendation parameters.

## Setup and Usage

1.  **Install Dependencies**:
    Ensure you have the required Python packages installed (pandas, numpy, streamlit, scikit-learn, xgboost, lightgbm).

2.  **Run the Application**:

    ```bash
    streamlit run app.py
    ```

3.  **Using the App**:
    - Select a **Username** from the sidebar.
    - Adjust the **Number of products from CF (k)** slider to control the candidate pool size.
    - Adjust the **Final recommendations (n)** slider to control how many top products to display.
    - Click **Get Recommendations**.

## Models

The system uses the following pre-trained models located in `models/`:

- `recommendation/user_similarity_full.pkl`: User-User similarity matrix.
- `recommendation/user_item_matrix_full.pkl`: User-Item rating matrix.
- `best_model.pkl`: Logistic Regression model for sentiment classification.
- `feature_pipeline.pkl`: TF-IDF and feature engineering pipeline.

## Notebooks

The `notebooks/` directory contains the research and training code:

- `03_feature_extraction_and_sentiment_model.ipynb`: Sentiment model training and evaluation.
- `04_recommendation_system.ipynb`: Collaborative filtering model development and comparison.
