"""
Data loading and sampling utilities for H&M Kaggle dataset.
Designed for practical Streamlit deployment (sampling + caching).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

# Expected file names after user places data in data/ folder
ARTICLES_FILE = "articles.csv"
CUSTOMERS_FILE = "customers.csv"
TRANSACTIONS_FILE = "transactions_train.csv"

DATA_DIR = Path(__file__).parent.parent / "data"


def _build_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a small deterministic dataset for running without Kaggle files."""
    rng = np.random.default_rng(42)
    article_count = 120
    customer_count = 800
    article_ids = [f"{100000000 + i:010d}" for i in range(article_count)]
    customer_ids = [f"{i:016d}" for i in range(customer_count)]

    product_types = ["T-shirt", "Sweater", "Dress", "Trousers", "Jacket", "Skirt"]
    product_groups = ["Garment Upper body", "Garment Lower body", "Garment Full body"]
    colours = ["Black", "White", "Blue", "Red", "Beige", "Green"]
    garments = ["Jersey Basic", "Knitwear", "Dresses Ladies", "Trousers", "Outdoor"]

    articles = pd.DataFrame(
        {
            "article_id": article_ids,
            "prod_name": [f"Demo {product_types[i % len(product_types)]} {i + 1}" for i in range(article_count)],
            "product_type_name": [product_types[i % len(product_types)] for i in range(article_count)],
            "product_group_name": [product_groups[i % len(product_groups)] for i in range(article_count)],
            "colour_group_name": [colours[i % len(colours)] for i in range(article_count)],
            "garment_group_name": [garments[i % len(garments)] for i in range(article_count)],
            "detail_desc": [f"Demo {colours[i % len(colours)].lower()} everyday fashion item" for i in range(article_count)],
        }
    )
    customers = pd.DataFrame({"customer_id": customer_ids})

    transaction_count = 24000
    transactions = pd.DataFrame(
        {
            "t_dat": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 365, transaction_count), unit="D"),
            "customer_id": rng.choice(customer_ids, transaction_count),
            "article_id": rng.choice(article_ids, transaction_count),
            "price": rng.uniform(0.01, 0.20, transaction_count).round(4),
            "sales_channel_id": rng.choice([1, 2], transaction_count, p=[0.35, 0.65]),
        }
    )
    transactions = transactions.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
    return transactions, articles, customers


@st.cache_data(show_spinner="Loading articles...")
def load_articles(sample_frac: float = 1.0) -> pd.DataFrame:
    path = DATA_DIR / ARTICLES_FILE
    if not path.exists():
        st.error(f"Missing {ARTICLES_FILE}. Place it in the `data/` folder.")
        st.stop()
    df = pd.read_csv(path, dtype={"article_id": str})
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


@st.cache_data(show_spinner="Loading customers...")
def load_customers(sample_frac: float = 1.0) -> pd.DataFrame:
    path = DATA_DIR / CUSTOMERS_FILE
    if not path.exists():
        st.error(f"Missing {CUSTOMERS_FILE}. Place it in the `data/` folder.")
        st.stop()
    df = pd.read_csv(path, dtype={"customer_id": str})
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


@st.cache_data(show_spinner="Loading & sampling transactions...")
def load_transactions(
    sample_customers: int = 50000,
    min_date: str = "2019-09-01",
    max_date: str = None,
) -> pd.DataFrame:
    """
    Load transactions and sample a manageable number of customers.
    Full dataset is ~32M rows – sampling is essential for Streamlit.
    """
    path = DATA_DIR / TRANSACTIONS_FILE
    if not path.exists():
        st.error(f"Missing {TRANSACTIONS_FILE}. Place it in the `data/` folder.")
        st.stop()

    # Read only necessary columns and filter early
    usecols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    df = pd.read_csv(
        path,
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
        usecols=usecols,
    )

    if min_date:
        df = df[df["t_dat"] >= pd.to_datetime(min_date)]
    if max_date:
        df = df[df["t_dat"] <= pd.to_datetime(max_date)]

    # Sample customers to keep memory reasonable
    unique_customers = df["customer_id"].unique()
    if len(unique_customers) > sample_customers:
        rng = np.random.default_rng(42)
        chosen = rng.choice(unique_customers, size=sample_customers, replace=False)
        df = df[df["customer_id"].isin(chosen)]

    df = df.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
    return df


@st.cache_data
def get_merged_data(
    sample_customers: int = 30000,
    min_date: str = "2019-09-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, articles, customers) ready for analysis."""
    required_files = [DATA_DIR / ARTICLES_FILE, DATA_DIR / CUSTOMERS_FILE, DATA_DIR / TRANSACTIONS_FILE]
    if not all(path.exists() for path in required_files):
        return _build_demo_data()

    articles = load_articles()
    customers = load_customers()
    transactions = load_transactions(sample_customers=sample_customers, min_date=min_date)

    # Keep only customers and articles that appear in the sampled transactions
    transactions = transactions[
        transactions["customer_id"].isin(customers["customer_id"])
        & transactions["article_id"].isin(articles["article_id"])
    ]
    customers = customers[customers["customer_id"].isin(transactions["customer_id"])]
    articles = articles[articles["article_id"].isin(transactions["article_id"])]

    return transactions, articles, customers


def get_data_summary(transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame) -> dict:
    return {
        "n_transactions": len(transactions),
        "n_customers": transactions["customer_id"].nunique(),
        "n_articles": transactions["article_id"].nunique(),
        "date_range": (
            transactions["t_dat"].min().strftime("%Y-%m-%d"),
            transactions["t_dat"].max().strftime("%Y-%m-%d"),
        ),
        "total_revenue": transactions["price"].sum(),
        "avg_price": transactions["price"].mean(),
        "online_share": (transactions["sales_channel_id"] == 2).mean(),  # usually 2 = online
    }