# Ebuss Sentiment-Enhanced Recommendation System

- 🚀 **Deployed App:** [https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/](https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/)
- 📂 **GitHub Repo:** [https://github.com/amaan2398/PRS](https://github.com/amaan2398/PRS)

## Overview

Ebuss is a hybrid product recommendation system designed to improve user experience by combining **User-User Collaborative Filtering** with **Sentiment Analysis**.

The system recommends products based on:

1.  **Collaborative Filtering**: Identifying products liked by users with similar rating patterns (Adjusted Cosine Similarity).
2.  **Sentiment Analysis**: Filtering and ranking these recommendations based on the sentiment of their reviews (Positive/Negative).

## Project Structure

```
├── app.py                                                  # Main Streamlit application entry point
├── config/                                                 # Configuration files
│   ├── __init__.py                                         # Initialization file
│   └── manager.py                                          # Manager class for handling app state
├── config.json                                             # Configuration file
├── recommenders/                                           # Recommendation logic
│   ├── __init__.py                                         # Initialization file
│   ├── user_based.py                                       # User-User CF implementation (Active)
│   └── item_based.py                                       # Item-Item CF implementation (Legacy)
├── sentiment/                                              # Sentiment analysis logic
│   ├── __init__.py                                         # Initialization file
│   └── analyzer.py                                         # Sentiment Analyzer using Logistic Regression
├── preprocessing/                                          # Data preprocessing logic
│   ├── imputers/                                           # Imputer logic
│   │   ├── __init__.py                                     # Initialization file
│   │   ├── manufacturer_imputer.py                         # Manufacturer imputer
│   │   ├── category_frequency_imputer.py                   # Category frequency imputer
│   │   └── category_first_imputer.py                       # Category first imputer
│   ├── nlp/                                                # NLP logic
│   │   ├── __init__.py                                     # Initialization file
│   │   ├── nlp_engine.py                                   # NLP engine
│   │   └── text_processing.py                              # Text processing
│   └── __init__.py                                         # Initialization file
├── utils/                                                  # Utility functions
│   ├── data_analyzer.py                                    # Data analyzer
│   ├── data_cleaner.py                                     # Data cleaner
│   └── data_file_manager.py                                # Data loading and saving
├── transformers/                                           # Custom Scikit-learn transformers
│   ├── target_encoder.py                                   # Target encoder
│   ├── temporal_features.py                                # Temporal features
│   └── text_stats.py                                       # Text stats
├── models/                                                 # Pre-trained ML models (Pickle files)
├── data/                                                   # Dataset directory
│   ├── processed/
│   │   ├── df_cleaned.csv                                  # Cleaned dataset
│   │   └── df_final.csv                                    # Final processed dataset
│   └── raw/
│       ├── Data+Attribute+Description.csv                  # Raw dataset
│       └── dataset.csv                                     # Raw dataset
└── notebooks/                                              # Jupyter notebooks for training and analysis
    ├── notebook_setup.py                                   # Notebook setup script
    ├── 01_data_cleaning_and_preprocessing.ipynb            # Data preprocessing notebook
    ├── 02_exploratory_data_analysis.ipynb                  # Exploratory data analysis notebook
    ├── 03_feature_extraction_and_sentiment_model.ipynb     # Feature extraction and sentiment model
    └── 04_recommendation_system.ipynb                      # Recommendation system notebook
```

## Key Features

- **Hybrid Pipeline**: Combines the breadth of Collaborative Filtering with the quality assurance of Sentiment Analysis.
- **User-User CF**: Uses Adjusted Cosine Similarity to account for user rating bias.
- **Sentiment Filtering**: Re-ranks products to ensure highly-rated items also have positive textual reviews.
- **Interactive UI**: Built with Streamlit, allowing dynamic configuration of recommendation parameters.

## Setup and Usage

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Before you begin

Run following command for the first time:

```bash
venv\Scripts\activate
python download_models.py
```

1.  **Install Dependencies**:
    Ensure you have the required Python packages installed (pandas, numpy, streamlit, scikit-learn, xgboost, lightgbm).

2.  **Run the Application**:

    ```bash
    venv\Scripts\activate
    streamlit run app.py
    ```

3.  **Using the App**:
    - Select a **Username** from the sidebar.
    - Adjust the **Number of products from CF (k)** slider to control the candidate pool size.
    - Adjust the **Final recommendations (n)** slider to control how many top products to display.
    - Click **Get Recommendations**.
