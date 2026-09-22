"""
RFM analysis and customer segmentation for H&M data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st


@st.cache_data(show_spinner="Computing RFM features...")
def compute_rfm(transactions: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Classic RFM:
    - Recency: days since last purchase
    - Frequency: number of purchases
    - Monetary: total spend
    """
    if reference_date is None:
        reference_date = transactions["t_dat"].max() + pd.Timedelta(days=1)

    rfm = (
        transactions.groupby("customer_id")
        .agg(
            recency=("t_dat", lambda x: (reference_date - x.max()).days),
            frequency=("article_id", "count"),
            monetary=("price", "sum"),
            unique_articles=("article_id", "nunique"),
            first_purchase=("t_dat", "min"),
            last_purchase=("t_dat", "max"),
        )
        .reset_index()
    )

    # Quantile-based RFM scores (1–5)
    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    return rfm


def assign_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Business-friendly segment labels based on RFM scores."""
    def label(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Promising / New"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost / Hibernating"
        elif m >= 4:
            return "Big Spenders"
        else:
            return "Need Attention"

    rfm = rfm.copy()
    rfm["segment"] = rfm.apply(label, axis=1)
    return rfm


@st.cache_data(show_spinner="Running K-Means clustering...")
def kmeans_segmentation(rfm: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Unsupervised clustering on scaled RFM features."""
    features = ["recency", "frequency", "monetary", "unique_articles"]
    X = rfm[features].copy()

    # Log-transform skewed monetary & frequency
    X["monetary"] = np.log1p(X["monetary"])
    X["frequency"] = np.log1p(X["frequency"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    # Order clusters by monetary value for nicer labels
    cluster_order = (
        rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index.tolist()
    )
    mapping = {old: new for new, old in enumerate(cluster_order)}
    rfm["cluster"] = rfm["cluster"].map(mapping)

    return rfm, kmeans, scaler


def get_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round(1)
        .sort_values("total_revenue", ascending=False)
    )
    summary["pct_customers"] = (summary["customers"] / summary["customers"].sum() * 100).round(1)
    summary["pct_revenue"] = (summary["total_revenue"] / summary["total_revenue"].sum() * 100).round(1)
    return summary