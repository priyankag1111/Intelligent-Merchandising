"""
H&M Intelligent Assortment & Merchandising – Streamlit Prototype
Works purely on the official Kaggle H&M dataset.

Deploy:
  1. Place articles.csv, customers.csv, transactions_train.csv in ./data/
  2. pip install -r requirements.txt
  3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Make src importable
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import get_merged_data, get_data_summary
from segmentation import compute_rfm, assign_rfm_segments, kmeans_segmentation, get_segment_summary
from demand_forecast import build_weekly_demand, forecast_item, get_top_items_by_volume, simple_trend_velocity
from recommendations import (
    popularity_recommender,
    get_customer_history,
    build_item_item_cf,
    hybrid_recommend,
)
from product_analytics import product_performance, build_text_embeddings, find_similar_products, simple_market_basket

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="H&M Intelligent Assortment POC",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("H&M Intelligent Assortment & Merchandising")
st.caption("POC Prototype · Kaggle-compatible analytics · Free-tool Stack")

# ---------------------------------------------------------------------------
# Sidebar – data controls
# ---------------------------------------------------------------------------
st.sidebar.header("Data Controls")
sample_customers = st.sidebar.slider("Sample customers", 5_000, 80_000, 25_000, 5_000)
min_date = st.sidebar.selectbox(
    "Transactions from",
    ["2019-01-01", "2019-06-01", "2019-09-01", "2020-01-01"],
    index=2,
)

with st.sidebar.expander("About this POC"):
    st.markdown(
        """
        **Data source**: Official H&M Personalized Fashion Recommendations (Kaggle)  
        **Analyses included**:
        - RFM + Customer Segmentation
        - Hierarchical Demand Forecasting
        - Trend / Velocity Detection
        - Personalized Recommendations
        - Product Performance & Similarity
        - Market Basket (co-purchase)
        
        All models run on sampled data for interactive performance.
        """
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_all(sample_customers, min_date):
    return get_merged_data(sample_customers=sample_customers, min_date=min_date)

with st.spinner("Loading and sampling H&M data..."):
    transactions, articles, customers = load_all(sample_customers, min_date)

if not all((Path("data") / filename).exists() for filename in ["articles.csv", "customers.csv", "transactions_train.csv"]):
    st.info("Running in demo mode with generated sample data. Add the three Kaggle CSVs to data/ to use real H&M data.")

summary = get_data_summary(transactions, articles, customers)

# KPI strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{summary['n_transactions']:,}")
c2.metric("Customers", f"{summary['n_customers']:,}")
c3.metric("Articles", f"{summary['n_articles']:,}")
c4.metric("Date Range", f"{summary['date_range'][0]} → {summary['date_range'][1]}")
c5.metric("Total Revenue (sample)", f"${summary['total_revenue']:,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1. Customer Segmentation",
        "2. Demand Forecasting",
        "3. Trend Detection",
        "4. Recommendations",
        "5. Product Analytics",
        "6. How to Deploy",
    ]
)

# ===================== TAB 1: SEGMENTATION =====================
with tab1:
    st.header("Customer Segmentation (RFM + K-Means)")
    st.markdown(
        "Classic RFM scoring plus unsupervised clustering. Segments map directly to the "
        "**Segmentation** component of the original opportunity."
    )

    rfm = compute_rfm(transactions)
    rfm = assign_rfm_segments(rfm)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("RFM Segment Distribution")
        seg_summary = get_segment_summary(rfm)
        st.dataframe(seg_summary, use_container_width=True)

        fig = px.bar(
            seg_summary.reset_index(),
            x="segment",
            y="pct_revenue",
            color="pct_customers",
            title="% of Revenue vs % of Customers by Segment",
            labels={"pct_revenue": "% Revenue", "pct_customers": "% Customers"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("K-Means Clusters (optional)")
        n_clusters = st.slider("Number of clusters", 3, 8, 5)
        rfm_km, model, scaler = kmeans_segmentation(rfm, n_clusters=n_clusters)

        cluster_sum = (
            rfm_km.groupby("cluster")
            .agg(
                customers=("customer_id", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
            )
            .round(1)
        )
        st.dataframe(cluster_sum, use_container_width=True)

        fig2 = px.scatter(
            rfm_km.sample(min(5000, len(rfm_km))),
            x="recency",
            y="monetary",
            color="cluster",
            size="frequency",
            title="Recency vs Monetary (size = Frequency)",
            log_y=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "**Business use**: Champions & Loyal → protect & upsell. "
        "At Risk / Lost → win-back campaigns. "
        "Promising → nurture with personalised ranking."
    )

# ===================== TAB 2: DEMAND FORECASTING =====================
with tab2:
    st.header("Hierarchical Demand Forecasting")
    st.markdown(
        "Weekly unit forecasts at product-type or garment-group level using Prophet. "
        "Supports the **Demand Prediction** box of the original architecture."
    )

    level = st.selectbox(
        "Aggregation level",
        ["product_type_name", "garment_group_name", "product_group_name", "colour_group_name"],
        index=0,
    )
    weekly = build_weekly_demand(transactions, articles, level=level)
    top_items = get_top_items_by_volume(weekly, top_n=25)

    selected_item = st.selectbox("Select item to forecast", top_items)
    periods = st.slider("Forecast horizon (weeks)", 4, 16, 8)

    if st.button("Run Forecast", key="forecast_btn"):
        with st.spinner("Fitting Prophet..."):
            forecast, history = forecast_item(weekly, selected_item, periods=periods)

        if forecast is None:
            st.warning("Not enough history for a reliable forecast.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=history["ds"], y=history["y"], name="Actual", mode="lines+markers")
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    name="Forecast",
                    mode="lines",
                    line=dict(dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    name="Upper",
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    name="Lower",
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,200,0.15)",
                    showlegend=False,
                )
            )
            fig.update_layout(
                title=f"Weekly Demand Forecast – {selected_item}",
                xaxis_title="Week",
                yaxis_title="Units",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            future = forecast[forecast["ds"] > history["ds"].max()][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].round(1)
            st.subheader("Forecast Table")
            st.dataframe(future, use_container_width=True)

# ===================== TAB 3: TREND DETECTION =====================
with tab3:
    st.header("Trend / Velocity Detection")
    st.markdown(
        "Compares recent 4-week average sales vs the previous 4 weeks. "
        "Flags **Accelerating**, **Stable**, or **Declining** styles."
    )

    level_t = st.selectbox(
        "Level for trend analysis",
        ["product_type_name", "garment_group_name", "colour_group_name"],
        key="trend_level",
    )
    weekly_t = build_weekly_demand(transactions, articles, level=level_t)
    velocity_df = simple_trend_velocity(weekly_t, recent_weeks=4)

    st.subheader("Top Accelerating Items")
    st.dataframe(
        velocity_df[velocity_df["trend_flag"] == "Accelerating"].head(15),
        use_container_width=True,
    )

    st.subheader("Top Declining Items")
    st.dataframe(
        velocity_df[velocity_df["trend_flag"] == "Declining"].head(15),
        use_container_width=True,
    )

    fig = px.scatter(
        velocity_df.head(80),
        x="total_units",
        y="velocity",
        color="trend_flag",
        hover_name="item",
        title="Velocity vs Total Volume (size of opportunity)",
        log_x=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 4: RECOMMENDATIONS =====================
with tab4:
    st.header("Personalized Recommendations")
    st.markdown(
        "Hybrid recommender: item-item collaborative filtering + popularity fallback. "
        "Maps to the **Ranking / Visibility** commercial decision."
    )

    with st.spinner("Building recommendation models (first run may take 1–2 min)..."):
        popular = popularity_recommender(transactions, top_k=20)
        history = get_customer_history(transactions)
        similarities = build_item_item_cf(transactions, min_support=5)

    # Pick a sample customer who has history
    sample_customers_list = list(history.keys())[:500]
    selected_cust = st.selectbox("Select a customer_id", sample_customers_list)

    if selected_cust:
        past = history.get(selected_cust, [])
        st.write(f"**Past purchases (last 10):** {past[-10:]}")

        recs = hybrid_recommend(selected_cust, history, similarities, popular, top_k=12)

        # Enrich with article names
        rec_df = articles[articles["article_id"].isin(recs)][
            ["article_id", "prod_name", "product_type_name", "colour_group_name"]
        ].drop_duplicates("article_id")
        # Preserve order
        rec_df["rank"] = rec_df["article_id"].apply(lambda x: recs.index(x) if x in recs else 999)
        rec_df = rec_df.sort_values("rank").drop(columns="rank")

        st.subheader("Top 12 Recommendations")
        st.dataframe(rec_df, use_container_width=True)

        st.success(
            "In production these scores would feed the ranking engine on category pages, "
            "search results and homepage carousels."
        )

# ===================== TAB 5: PRODUCT ANALYTICS =====================
with tab5:
    st.header("Product Performance & Similarity")
    st.markdown("Supports **Assortment** decisions – which styles to keep, promote or drop.")

    perf = product_performance(transactions, articles)

    st.subheader("Top 20 Articles by Units Sold")
    st.dataframe(
        perf[
            [
                "article_id",
                "prod_name",
                "product_type_name",
                "colour_group_name",
                "units",
                "revenue",
                "unique_buyers",
                "units_per_week",
            ]
        ].head(20),
        use_container_width=True,
    )

    st.subheader("Content-Based Similar Products")
    with st.spinner("Building TF-IDF embeddings..."):
        vectorizer, matrix, article_ids = build_text_embeddings(articles)

    # Let user pick a popular article
    top_articles = perf["article_id"].head(100).tolist()
    chosen = st.selectbox("Choose an article to find similar items", top_articles)

    if chosen:
        similar = find_similar_products(chosen, article_ids, matrix, top_k=8)
        if similar:
            sim_ids = [s[0] for s in similar]
            sim_scores = [s[1] for s in similar]
            sim_df = articles[articles["article_id"].isin(sim_ids)][
                ["article_id", "prod_name", "product_type_name", "colour_group_name"]
            ].copy()
            sim_df["similarity"] = sim_df["article_id"].map(dict(zip(sim_ids, sim_scores)))
            st.dataframe(sim_df.sort_values("similarity", ascending=False), use_container_width=True)

    st.subheader("Frequent Co-Purchase Pairs (Market Basket)")
    basket = simple_market_basket(transactions, min_support=15, top_n=20)
    if not basket.empty:
        # Map to names
        id_to_name = articles.set_index("article_id")["prod_name"].to_dict()
        basket["item_a_name"] = basket["item_a"].map(id_to_name)
        basket["item_b_name"] = basket["item_b"].map(id_to_name)
        st.dataframe(
            basket[["item_a_name", "item_b_name", "co_purchases", "confidence_a_to_b"]],
            use_container_width=True,
        )
    else:
        st.info("Not enough co-purchase volume in the current sample. Try increasing sample size.")

# ===================== TAB 6: DEPLOY =====================
with tab6:
    st.header("Deploy this app on Streamlit via GitHub")
    st.markdown(
        """
### 1. Repository structure
```
hm_streamlit_app/
├── app.py
├── requirements.txt
├── data/                  ← put the 3 CSV files here (or use Git LFS / external storage)
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
├── src/
│   ├── data_loader.py
│   ├── segmentation.py
│   ├── demand_forecast.py
│   ├── recommendations.py
│   └── product_analytics.py
└── README.md
```

### 2. Local run
```bash
cd hm_streamlit_app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy to Streamlit Community Cloud (free)
1. Push the repo to GitHub (do **not** commit the large CSVs – add `data/*.csv` to `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Connect the GitHub repo, set main file to `app.py`.
4. For data: either  
   - host the CSVs on a public Google Drive / S3 and download in `data_loader.py`, or  
   - use Streamlit Secrets + a private file host, or  
   - keep a heavily sampled version of the CSVs in the repo for the demo.

### 4. Recommended sampling for public demo
In `data_loader.py` the default already samples ~25 k customers.  
For a public GitHub demo you can further reduce to 5–10 k customers so the app stays fast.

### 5. Next upgrades (after CEO demo)
- Add real-time identity resolution & event streams (replace generated signals).
- Move models to a feature store + scheduled retraining.
- Connect the ranking scores to an actual e-commerce ranking API.
        """
    )

st.divider()
st.caption("")