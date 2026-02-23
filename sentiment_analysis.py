"""Sentiment analysis for Telegram messages."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Library availability detection
# ---------------------------------------------------------------------------

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer

    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

try:
    from textblob import TextBlob as _TextBlob

    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False


def get_analyzer() -> Optional[Literal["vader", "textblob"]]:
    """
    Return the name of the available sentiment analyser.

    VADER is preferred over TextBlob.

    Returns:
        'vader', 'textblob', or None if neither is installed.
    """
    if _VADER_AVAILABLE:
        return "vader"
    if _TEXTBLOB_AVAILABLE:
        return "textblob"
    return None


# ---------------------------------------------------------------------------
# Single-text scoring
# ---------------------------------------------------------------------------


def analyze_text_sentiment(
    text: str,
    method: Optional[str] = None,
) -> dict:
    """
    Score the sentiment of a single text string.

    Args:
        text: Input text to analyse.
        method: 'vader' or 'textblob'. Auto-selects if None.

    Returns:
        Dict with keys: compound, positive, negative, neutral.
        All values in [-1, 1] or [0, 1] depending on key.
    """
    empty = {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return empty

    method = method or get_analyzer()
    if method is None:
        return empty

    if method == "vader" and _VADER_AVAILABLE:
        vader = _VaderAnalyzer()
        scores = vader.polarity_scores(text)
        return {
            "compound": round(scores["compound"], 4),
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral": round(scores["neu"], 4),
        }

    if method == "textblob" and _TEXTBLOB_AVAILABLE:
        polarity = _TextBlob(text).sentiment.polarity
        return {
            "compound": round(polarity, 4),
            "positive": round(max(0.0, polarity), 4),
            "negative": round(max(0.0, -polarity), 4),
            "neutral": round(1.0 - abs(polarity), 4),
        }

    return empty


# ---------------------------------------------------------------------------
# DataFrame-level analysis
# ---------------------------------------------------------------------------


def analyze_dataframe_sentiment(
    df: pd.DataFrame,
    sample_size: Optional[int] = 10_000,
    method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add sentiment scores to a messages DataFrame.

    For performance, only a random sample of `sample_size` rows is
    scored; the rest receive a compound score of 0 (neutral).

    Args:
        df: Messages DataFrame.
        sample_size: Max rows to score. None = score all (may be slow).
        method: Sentiment library to use ('vader' or 'textblob').

    Returns:
        DataFrame with added columns:
            - sentiment_compound: float in [-1, 1]
            - sentiment_label: 'Positive', 'Neutral', or 'Negative'
    """
    method = method or get_analyzer()
    if method is None:
        logger.warning("No sentiment library available. Install vaderSentiment or textblob.")
        df = df.copy()
        df["sentiment_compound"] = 0.0
        df["sentiment_label"] = "Neutral"
        return df

    df = df.copy()
    df["sentiment_compound"] = 0.0

    # Determine which rows to score
    if sample_size and len(df) > sample_size:
        sample_idx = df.sample(n=sample_size, random_state=42).index
    else:
        sample_idx = df.index

    if method == "vader" and _VADER_AVAILABLE:
        vader = _VaderAnalyzer()

        def _score_vader(text: str) -> float:
            try:
                if not isinstance(text, str) or len(text.strip()) < 3:
                    return 0.0
                return vader.polarity_scores(text)["compound"]
            except Exception:
                return 0.0

        df.loc[sample_idx, "sentiment_compound"] = df.loc[sample_idx, "text"].apply(
            _score_vader
        )

    elif method == "textblob" and _TEXTBLOB_AVAILABLE:

        def _score_textblob(text: str) -> float:
            try:
                if not isinstance(text, str) or len(text.strip()) < 3:
                    return 0.0
                return _TextBlob(text).sentiment.polarity
            except Exception:
                return 0.0

        df.loc[sample_idx, "sentiment_compound"] = df.loc[sample_idx, "text"].apply(
            _score_textblob
        )

    df["sentiment_compound"] = df["sentiment_compound"].fillna(0.0).round(4)
    df["sentiment_label"] = df["sentiment_compound"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
    )

    return df


# ---------------------------------------------------------------------------
# Per-contact sentiment
# ---------------------------------------------------------------------------


def get_sentiment_per_contact(
    df: pd.DataFrame,
    min_messages: int = 10,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Average sentiment score per chat/contact.

    Args:
        df: Messages DataFrame with 'sentiment_compound' column.
        min_messages: Minimum messages for a chat to be included.

    Returns:
        Tuple of (per-contact stats DataFrame, Plotly bar figure).
    """
    if "sentiment_compound" not in df.columns:
        return pd.DataFrame(), go.Figure()

    stats = (
        df.groupby("chat_name")
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            positive_pct=(
                "sentiment_label",
                lambda x: round((x == "Positive").mean() * 100, 1),
            ),
            negative_pct=(
                "sentiment_label",
                lambda x: round((x == "Negative").mean() * 100, 1),
            ),
            neutral_pct=(
                "sentiment_label",
                lambda x: round((x == "Neutral").mean() * 100, 1),
            ),
            message_count=("message_id", "count"),
        )
        .reset_index()
    )

    stats = stats[stats["message_count"] >= min_messages]
    stats["avg_sentiment"] = stats["avg_sentiment"].round(3)
    stats = stats.sort_values("avg_sentiment", ascending=False).head(25)

    fig = px.bar(
        stats,
        x="chat_name",
        y="avg_sentiment",
        title="Average Sentiment Score per Chat",
        labels={"avg_sentiment": "Avg Sentiment", "chat_name": "Chat"},
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data=["positive_pct", "negative_pct", "message_count"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=460, showlegend=False)

    return stats, fig


# ---------------------------------------------------------------------------
# Sentiment timeline
# ---------------------------------------------------------------------------


def get_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Weekly average sentiment over time with a 4-week rolling average.

    Args:
        df: Messages DataFrame with 'sentiment_compound' column.

    Returns:
        Plotly Figure.
    """
    if "sentiment_compound" not in df.columns:
        return go.Figure()

    weekly = (
        df[df["sentiment_compound"] != 0]
        .set_index("date")
        .resample("W")["sentiment_compound"]
        .mean()
        .reset_index()
    )
    weekly.columns = ["date", "avg_sentiment"]
    weekly["rolling_4w"] = weekly["avg_sentiment"].rolling(4, min_periods=1).mean()

    fig = go.Figure()

    # Coloured bars
    colors = weekly["avg_sentiment"].apply(
        lambda x: "rgba(46, 204, 113, 0.6)"
        if x >= 0
        else "rgba(231, 76, 60, 0.6)"
    )
    fig.add_trace(
        go.Bar(
            x=weekly["date"],
            y=weekly["avg_sentiment"],
            name="Weekly Avg",
            marker_color=colors,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["date"],
            y=weekly["rolling_4w"],
            name="4-week Rolling Avg",
            line=dict(color="#2980B9", width=2.5),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Sentiment Timeline (Weekly)",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Sentiment distribution
# ---------------------------------------------------------------------------


def get_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Pie chart of overall Positive / Neutral / Negative message split.

    Args:
        df: Messages DataFrame with 'sentiment_label' column.

    Returns:
        Plotly Figure.
    """
    if "sentiment_label" not in df.columns:
        return go.Figure()

    dist = df["sentiment_label"].value_counts().reset_index()
    dist.columns = ["label", "count"]

    color_map = {
        "Positive": "#2ECC71",
        "Neutral": "#95A5A6",
        "Negative": "#E74C3C",
    }

    fig = px.pie(
        dist,
        values="count",
        names="label",
        title="Overall Sentiment Distribution",
        color="label",
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside")
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# Top most positive / negative messages
# ---------------------------------------------------------------------------


def get_extreme_messages(
    df: pd.DataFrame,
    n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the top N most positive and most negative messages.

    Args:
        df: Messages DataFrame with 'sentiment_compound' column.
        n: Number of extreme messages to return per category.

    Returns:
        Tuple of (most positive DataFrame, most negative DataFrame).
    """
    if "sentiment_compound" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    scored = df[df["sentiment_compound"] != 0].copy()
    cols = ["chat_name", "sender", "date", "text", "sentiment_compound"]

    most_positive = (
        scored.nlargest(n, "sentiment_compound")[cols].reset_index(drop=True)
    )
    most_negative = (
        scored.nsmallest(n, "sentiment_compound")[cols].reset_index(drop=True)
    )

    return most_positive, most_negative
