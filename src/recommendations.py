"""
Simple but effective recommendation engines for H&M data.
Includes popularity, recent-purchase, and a lightweight item-item CF.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st


@st.cache_data(show_spinner="Building popularity baseline...")
def popularity_recommender(transactions: pd.DataFrame, top_k: int = 12) -> list:
    """Global top sellers in the last N days of data."""
    max_date = transactions["t_dat"].max()
    recent = transactions[transactions["t_dat"] >= max_date - pd.Timedelta(days=28)]
    top = recent["article_id"].value_counts().head(top_k).index.tolist()
    return top


@st.cache_data(show_spinner="Building customer purchase history...")
def get_customer_history(transactions: pd.DataFrame) -> dict:
    """customer_id -> list of article_ids ordered by time (most recent last)."""
    history = (
        transactions.sort_values(["customer_id", "t_dat"])
        .groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )
    return history


def recommend_from_history(
    customer_id: str,
    history: dict,
    top_k: int = 12,
    exclude_purchased: bool = True,
) -> list:
    """Simple: recommend the customer's most frequent past purchases (repurchase bias)."""
    if customer_id not in history:
        return []
    counts = Counter(history[customer_id])
    items = [item for item, _ in counts.most_common(top_k * 2)]
    if exclude_purchased:
        # In a real system we would still allow repurchase; here we just return top past
        pass
    return items[:top_k]


@st.cache_data(show_spinner="Computing item co-occurrence...")
def build_item_item_cf(
    transactions: pd.DataFrame,
    min_support: int = 5,
) -> dict:
    """
    Item-item collaborative filtering via co-occurrence.
    Returns dict: article_id -> list of (similar_article, score)
    """
    # Build baskets per customer (simplified: all purchases of a customer)
    baskets = (
        transactions.groupby("customer_id")["article_id"]
        .apply(set)
        .tolist()
    )

    co_counts = defaultdict(lambda: defaultdict(int))
    item_counts = defaultdict(int)

    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
        for i in basket:
            for j in basket:
                if i != j:
                    co_counts[i][j] += 1

    # Jaccard-like similarity
    similarities = {}
    for item, related in co_counts.items():
        sims = []
        for other, cnt in related.items():
            if item_counts[other] < min_support:
                continue
            # cosine approximation
            sim = cnt / np.sqrt(item_counts[item] * item_counts[other])
            sims.append((other, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        similarities[item] = sims[:30]  # keep top 30 neighbours

    return similarities


def recommend_item_cf(
    customer_id: str,
    history: dict,
    similarities: dict,
    top_k: int = 12,
) -> list:
    """Recommend items similar to what the customer has bought."""
    if customer_id not in history:
        return []

    scores = defaultdict(float)
    purchased = set(history[customer_id])

    for item in purchased:
        if item not in similarities:
            continue
        for neighbour, sim in similarities[item]:
            if neighbour in purchased:
                continue
            scores[neighbour] += sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_k]]


def hybrid_recommend(
    customer_id: str,
    history: dict,
    similarities: dict,
    popular: list,
    top_k: int = 12,
) -> list:
    """Blend personal CF + popularity for cold-start robustness."""
    personal = recommend_item_cf(customer_id, history, similarities, top_k=top_k)
    if len(personal) >= top_k:
        return personal[:top_k]

    # Fill remaining with popularity
    seen = set(personal)
    for item in popular:
        if item not in seen:
            personal.append(item)
        if len(personal) >= top_k:
            break
    return personal[:top_k]


def evaluate_map_at_k(
    recommendations: dict,
    ground_truth: dict,
    k: int = 12,
) -> float:
    """Simple MAP@k for offline evaluation."""
    aps = []
    for cust, recs in recommendations.items():
        if cust not in ground_truth:
            continue
        truth = set(ground_truth[cust])
        if not truth:
            continue
        hits = 0
        sum_prec = 0.0
        for i, item in enumerate(recs[:k], 1):
            if item in truth:
                hits += 1
                sum_prec += hits / i
        aps.append(sum_prec / min(len(truth), k))
    return float(np.mean(aps)) if aps else 0.0