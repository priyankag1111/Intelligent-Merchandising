"""
Product-level analytics, affinity, and simple embeddings for H&M.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


@st.cache_data(show_spinner="Computing product performance...")
def product_performance(transactions: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Units, revenue, unique buyers, avg price per article."""
    perf = (
        transactions.groupby("article_id")
        .agg(
            units=("customer_id", "count"),
            revenue=("price", "sum"),
            unique_buyers=("customer_id", "nunique"),
            avg_price=("price", "mean"),
            first_sale=("t_dat", "min"),
            last_sale=("t_dat", "max"),
        )
        .reset_index()
    )
    perf = perf.merge(
        articles[
            [
                "article_id",
                "prod_name",
                "product_type_name",
                "product_group_name",
                "colour_group_name",
                "garment_group_name",
                "detail_desc",
            ]
        ],
        on="article_id",
        how="left",
    )
    perf["weeks_active"] = ((perf["last_sale"] - perf["first_sale"]).dt.days / 7).clip(lower=1)
    perf["units_per_week"] = perf["units"] / perf["weeks_active"]
    return perf.sort_values("units", ascending=False)


@st.cache_data(show_spinner="Building TF-IDF product embeddings...")
def build_text_embeddings(articles: pd.DataFrame, max_features: int = 5000):
    """
    Simple content-based embeddings from product name + description.
    Returns (vectorizer, matrix, article_ids)
    """
    text = (
        articles["prod_name"].fillna("")
        + " "
        + articles["product_type_name"].fillna("")
        + " "
        + articles["colour_group_name"].fillna("")
        + " "
        + articles["detail_desc"].fillna("")
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(text)
    return vectorizer, matrix, articles["article_id"].values


def find_similar_products(
    article_id: str,
    article_ids: np.ndarray,
    matrix,
    top_k: int = 10,
) -> list:
    """Return top-k similar article_ids by TF-IDF cosine similarity."""
    try:
        idx = np.where(article_ids == article_id)[0][0]
    except IndexError:
        return []

    sims = cosine_similarity(matrix[idx], matrix).flatten()
    top_idx = sims.argsort()[::-1][1 : top_k + 1]  # skip self
    return list(zip(article_ids[top_idx], sims[top_idx]))


@st.cache_data(show_spinner="Computing association rules (sample)...")
def simple_market_basket(transactions: pd.DataFrame, min_support: int = 20, top_n: int = 30):
    """
    Lightweight co-purchase pairs (not full Apriori – memory friendly).
    """
    baskets = (
        transactions.groupby("customer_id")["article_id"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    pair_counts = {}
    item_counts = {}

    for basket in baskets:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1
        for i, a in enumerate(basket):
            for b in basket[i + 1 :]:
                pair = tuple(sorted([a, b]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Filter and score
    results = []
    for (a, b), cnt in pair_counts.items():
        if cnt < min_support:
            continue
        support = cnt
        conf_a = cnt / item_counts[a]
        conf_b = cnt / item_counts[b]
        results.append(
            {
                "item_a": a,
                "item_b": b,
                "co_purchases": cnt,
                "confidence_a_to_b": conf_a,
                "confidence_b_to_a": conf_b,
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df
    return df.sort_values("co_purchases", ascending=False).head(top_n)