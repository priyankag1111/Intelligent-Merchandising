"""
Hierarchical demand forecasting on H&M transaction data.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
from typing import Optional


@st.cache_data(show_spinner="Building weekly demand series...")
def build_weekly_demand(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    level: str = "product_type_name",
) -> pd.DataFrame:
    """
    Aggregate to weekly demand at a chosen hierarchy level.
    level can be: 'article_id', 'product_type_name', 'product_group_name',
                  'garment_group_name', 'colour_group_name', etc.
    """
    df = transactions.merge(
        articles[["article_id", level]],
        on="article_id",
        how="left",
    )
    df["week"] = df["t_dat"].dt.to_period("W").dt.start_time

    weekly = (
        df.groupby(["week", level])
        .agg(units=("article_id", "count"), revenue=("price", "sum"))
        .reset_index()
    )
    weekly = weekly.rename(columns={level: "item"})
    return weekly


@st.cache_data(show_spinner="Training Prophet models...")
def forecast_item(
    weekly: pd.DataFrame,
    item_name: str,
    periods: int = 8,
    yearly_seasonality: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Prophet on one item and return forecast + history.
    """
    series = weekly[weekly["item"] == item_name][["week", "units"]].copy()
    series = series.rename(columns={"week": "ds", "units": "y"})
    series = series.sort_values("ds")

    if len(series) < 12:
        return None, None  # too short for reliable forecast

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.fit(series)

    future = model.make_future_dataframe(periods=periods, freq="W")
    forecast = model.predict(future)

    return forecast, series


def get_top_items_by_volume(weekly: pd.DataFrame, top_n: int = 20) -> list:
    totals = weekly.groupby("item")["units"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def simple_trend_velocity(weekly: pd.DataFrame, recent_weeks: int = 4) -> pd.DataFrame:
    """
    Compute recent vs previous velocity for trend detection.
    """
    weekly = weekly.sort_values(["item", "week"])
    results = []

    for item, grp in weekly.groupby("item"):
        if len(grp) < recent_weeks * 2:
            continue
        recent = grp.tail(recent_weeks)["units"].mean()
        previous = grp.iloc[-(recent_weeks * 2) : -recent_weeks]["units"].mean()
        if previous == 0:
            velocity = np.nan
        else:
            velocity = (recent - previous) / previous

        results.append(
            {
                "item": item,
                "recent_avg_units": recent,
                "previous_avg_units": previous,
                "velocity": velocity,
                "total_units": grp["units"].sum(),
            }
        )

    df = pd.DataFrame(results)
    df["trend_flag"] = pd.cut(
        df["velocity"],
        bins=[-np.inf, -0.2, 0.2, np.inf],
        labels=["Declining", "Stable", "Accelerating"],
    )
    return df.sort_values("velocity", ascending=False)