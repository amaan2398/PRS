# Ebuss - A Sentiment-Based Product Recommendation System

## Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

Build a model that will improve the recommendations given to the users given their past reviews and ratings.

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── dataset.csv
│   │   └── Data+Attribute+Description.csv
│   └── processed/
│       ├── df_cleaned.csv
│       └── df_final.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_sentiment_model_building.ipynb
│   ├── 05_recommendation_model_building.ipynb
│   └── notebook_setup.py # Setup notebook environment for running notebooks
├── app.py
├── models/
│   ├── __init__.py
│   └── recommendation_model.py
├── config/
│   ├── __init__.py
│   └── manager.py
├── utils/
│   ├── __init__.py
│   └── data_file_manager.py
├── preprocessing/
│   ├── imputers/
│   │   ├── __init__.py
│   │   ├── category_first_imputer.py
│   │   └── category_frequency_imputer.py
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── nlp_engine.py
│   │   └── text_processing.py
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.12 or later
- [uv](https://github.com/astral-sh/uv) installed

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your-username/PRS.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd PRS
    ```
3.  Install dependencies using `uv`. This project uses separate environments for notebooks and the application.

    **For the Notebook Environment (Research & Experimentation):**

    ```sh
    uv sync --group notebook
    ```

    **For the Application Environment (Frontend & Backend):**

    ```sh
    uv sync --group app
    ```

## Running Notebooks

To start the Jupyter Notebook server, make sure you have synced the `notebook` group, then use:

```sh
uv run --group notebook jupyter notebook
```

## Running the application

To run the FastAPI application, make sure you have synced the `app` group, then use:

```sh
uv run uvicorn src.main:app --reload
```

<!-- ## Running Tests

To run the tests, use the following command:

```sh
uv run python -m unittest discover tests
``` -->
