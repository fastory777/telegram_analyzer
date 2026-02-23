"""Temporal analysis for Telegram messages."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# ---------------------------------------------------------------------------
# Heatmap: hour × weekday
# ---------------------------------------------------------------------------


def get_activity_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap of message counts by hour of day and day of week.

    Args:
        df: Messages DataFrame.

    Returns:
        Plotly Figure with a Heatmap trace.
    """
    pivot = (
        df.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="count")
        .pivot(index="weekday", columns="hour", values="count")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=[WEEKDAY_NAMES[i] for i in range(7)],
            colorscale="Viridis",
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y}</b> at %{x}<br>Messages: <b>%{z}</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Message Activity Heatmap (Hour × Day of Week)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=360,
    )
    return fig


# ---------------------------------------------------------------------------
# Activity timeline with rolling averages
# ---------------------------------------------------------------------------


def get_activity_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Plot daily message counts with 7-day and 30-day rolling averages.

    Args:
        df: Messages DataFrame.

    Returns:
        Plotly Figure.
    """
    daily = df.set_index("date").resample("D").size().reset_index()
    daily.columns = ["date", "count"]
    daily["roll_7d"] = daily["count"].rolling(7, min_periods=1).mean().round(1)
    daily["roll_30d"] = daily["count"].rolling(30, min_periods=1).mean().round(1)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["count"],
            name="Daily",
            marker_color="rgba(100, 149, 237, 0.4)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["roll_7d"],
            name="7-day avg",
            line=dict(color="#4169E1", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["roll_30d"],
            name="30-day avg",
            line=dict(color="#DC143C", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Activity Timeline",
        xaxis_title="Date",
        yaxis_title="Messages",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Weekday distribution
# ---------------------------------------------------------------------------


def get_weekday_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of total messages broken down by day of the week.

    Args:
        df: Messages DataFrame.

    Returns:
        Plotly Figure.
    """
    weekday_counts = (
        df.groupby("weekday")
        .size()
        .reindex(range(7), fill_value=0)
        .reset_index()
    )
    weekday_counts.columns = ["weekday", "count"]
    weekday_counts["weekday_name"] = weekday_counts["weekday"].apply(
        lambda x: WEEKDAY_NAMES[x]
    )

    fig = px.bar(
        weekday_counts,
        x="weekday_name",
        y="count",
        title="Messages by Day of Week",
        labels={"count": "Messages", "weekday_name": "Day of Week"},
        color="count",
        color_continuous_scale="Teal",
        category_orders={"weekday_name": WEEKDAY_NAMES},
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Hourly distribution
# ---------------------------------------------------------------------------


def get_hourly_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of total messages by hour of day.

    Args:
        df: Messages DataFrame.

    Returns:
        Plotly Figure.
    """
    hourly = (
        df.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hourly.columns = ["hour", "count"]
    hourly["label"] = hourly["hour"].apply(lambda h: f"{h:02d}:00")

    fig = px.bar(
        hourly,
        x="label",
        y="count",
        title="Messages by Hour of Day",
        labels={"count": "Messages", "label": "Hour"},
        color="count",
        color_continuous_scale="Sunset",
    )
    fig.update_layout(height=380, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


def detect_bursts(
    df: pd.DataFrame,
    threshold_multiplier: float = 2.5,
    window_days: int = 14,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Detect days with anomalously high message activity (bursts).

    A day is flagged as a burst when its count exceeds the rolling
    mean by `threshold_multiplier` × rolling standard deviation.

    Args:
        df: Messages DataFrame.
        threshold_multiplier: Multiplier for the std deviation threshold.
        window_days: Rolling window size in days.

    Returns:
        Tuple of (burst days DataFrame, Plotly Figure).
    """
    daily = df.set_index("date").resample("D").size().reset_index()
    daily.columns = ["date", "count"]

    rolling_mean = daily["count"].rolling(window_days, min_periods=1).mean()
    rolling_std = daily["count"].rolling(window_days, min_periods=1).std().fillna(0)

    daily["threshold"] = (rolling_mean + threshold_multiplier * rolling_std).round(1)
    daily["is_burst"] = daily["count"] > daily["threshold"]

    bursts = daily[daily["is_burst"]].copy()[["date", "count"]].reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["count"],
            name="Daily Messages",
            line=dict(color="#4169E1", width=1.5),
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["threshold"],
            name="Burst Threshold",
            line=dict(color="#FFA500", width=2, dash="dash"),
        )
    )
    if not bursts.empty:
        burst_daily = daily[daily["is_burst"]]
        fig.add_trace(
            go.Scatter(
                x=burst_daily["date"],
                y=burst_daily["count"],
                mode="markers",
                name="Burst Days",
                marker=dict(
                    color="#DC143C",
                    size=10,
                    symbol="star",
                    line=dict(width=1, color="white"),
                ),
            )
        )

    fig.update_layout(
        title="Activity Burst Detection",
        xaxis_title="Date",
        yaxis_title="Messages",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return bursts, fig


# ---------------------------------------------------------------------------
# Conversation density
# ---------------------------------------------------------------------------


def get_conversation_density(
    df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """
    Heatmap showing monthly message density for the top N chats.

    Args:
        df: Messages DataFrame.
        top_n: Number of top chats to include.

    Returns:
        Plotly Figure.
    """
    top_chats = df.groupby("chat_name").size().nlargest(top_n).index.tolist()
    subset = df[df["chat_name"].isin(top_chats)].copy()
    subset["period"] = subset["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        subset.groupby(["period", "chat_name"])
        .size()
        .reset_index(name="count")
    )

    fig = px.density_heatmap(
        monthly,
        x="period",
        y="chat_name",
        z="count",
        histfunc="sum",
        title=f"Conversation Density – Top {top_n} Chats",
        labels={"period": "Month", "chat_name": "Chat", "count": "Messages"},
        color_continuous_scale="Hot_r",
    )
    fig.update_layout(height=420)
    return fig


# ---------------------------------------------------------------------------
# Year-over-year comparison
# ---------------------------------------------------------------------------


def get_year_over_year(df: pd.DataFrame) -> go.Figure:
    """
    Compare monthly message counts across different years.

    Args:
        df: Messages DataFrame.

    Returns:
        Plotly line Figure.
    """
    df_copy = df.copy()
    df_copy["month_num"] = df_copy["date"].dt.month
    df_copy["year"] = df_copy["date"].dt.year.astype(str)

    monthly = (
        df_copy.groupby(["year", "month_num"])
        .size()
        .reset_index(name="count")
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly["month_name"] = monthly["month_num"].apply(lambda m: month_names[m - 1])

    fig = px.line(
        monthly,
        x="month_name",
        y="count",
        color="year",
        title="Year-over-Year Monthly Comparison",
        labels={"count": "Messages", "month_name": "Month", "year": "Year"},
        markers=True,
        category_orders={"month_name": month_names},
    )
    fig.update_layout(height=400, hovermode="x unified")
    return fig
