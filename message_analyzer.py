"""Message analytics for Telegram data."""

from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Top contacts
# ---------------------------------------------------------------------------


def get_top_contacts(
    df: pd.DataFrame,
    n: int = 20,
    my_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Return the top N chats/contacts ranked by total message count.

    Args:
        df: Messages DataFrame.
        n: Number of contacts to return.
        my_name: If given, exclude self-messages from counts.

    Returns:
        Tuple of (summary DataFrame, Plotly bar figure).
    """
    work_df = df[df["sender"] != my_name] if my_name else df

    contact_counts = (
        work_df.groupby("chat_name")
        .agg(
            total_messages=("message_id", "count"),
            unique_senders=("sender", "nunique"),
            first_message=("date", "min"),
            last_message=("date", "max"),
        )
        .reset_index()
        .sort_values("total_messages", ascending=False)
        .head(n)
    )

    fig = px.bar(
        contact_counts,
        x="total_messages",
        y="chat_name",
        orientation="h",
        title=f"Top {n} Contacts by Message Count",
        labels={"total_messages": "Messages", "chat_name": "Contact / Chat"},
        color="total_messages",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, n * 28),
        showlegend=False,
    )

    return contact_counts, fig


# ---------------------------------------------------------------------------
# Conversation initiators
# ---------------------------------------------------------------------------


def get_conversation_initiators(
    df: pd.DataFrame,
    gap_hours: float = 4.0,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Determine who initiates conversations most often.

    A new conversation is detected after a silence gap of `gap_hours`.

    Args:
        df: Messages DataFrame.
        gap_hours: Hours of silence that constitute a new conversation.

    Returns:
        Tuple of (initiator counts DataFrame, Plotly pie figure).
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    gap = timedelta(hours=gap_hours)

    records: list[dict] = []
    for chat_name, group in df_sorted.groupby("chat_name"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        time_diffs = group["date"].diff()
        starts = group[(time_diffs > gap) | (time_diffs.isna())]
        for idx in starts.index:
            records.append(
                {"chat_name": chat_name, "initiator": group.loc[idx, "sender"]}
            )

    if not records:
        return pd.DataFrame(), go.Figure()

    init_df = pd.DataFrame(records)
    init_counts = (
        init_df.groupby("initiator")
        .size()
        .reset_index(name="conversations_started")
        .sort_values("conversations_started", ascending=False)
        .head(20)
    )

    fig = px.pie(
        init_counts,
        values="conversations_started",
        names="initiator",
        title="Conversation Initiators",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")

    return init_counts, fig


# ---------------------------------------------------------------------------
# Average message length
# ---------------------------------------------------------------------------


def get_avg_message_length(
    df: pd.DataFrame,
    min_messages: int = 10,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Compute average message character length per chat.

    Args:
        df: Messages DataFrame.
        min_messages: Minimum message count to include a chat.

    Returns:
        Tuple of (stats DataFrame, Plotly bar figure).
    """
    length_stats = (
        df.groupby("chat_name")
        .agg(
            avg_length=("text_length", "mean"),
            avg_words=("word_count", "mean"),
            total_messages=("message_id", "count"),
        )
        .reset_index()
    )
    length_stats = length_stats[length_stats["total_messages"] >= min_messages]
    length_stats = length_stats.sort_values("avg_length", ascending=False).head(20)
    length_stats["avg_length"] = length_stats["avg_length"].round(1)
    length_stats["avg_words"] = length_stats["avg_words"].round(1)

    fig = px.bar(
        length_stats,
        x="chat_name",
        y="avg_length",
        title="Average Message Length per Chat (characters)",
        labels={"avg_length": "Avg Chars", "chat_name": "Chat"},
        color="avg_words",
        color_continuous_scale="Blues",
        hover_data=["avg_words", "total_messages"],
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=450)

    return length_stats, fig


# ---------------------------------------------------------------------------
# Messages over time
# ---------------------------------------------------------------------------


def get_messages_over_time(
    df: pd.DataFrame,
    freq: str = "W",
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Aggregate message counts over time at a given frequency.

    Args:
        df: Messages DataFrame.
        freq: Pandas resample frequency string ('D', 'W', 'ME').

    Returns:
        Tuple of (time series DataFrame, Plotly line figure).
    """
    time_series = (
        df.set_index("date").resample(freq).size().reset_index()
    )
    time_series.columns = ["date", "message_count"]

    freq_labels = {"D": "Daily", "W": "Weekly", "ME": "Monthly", "M": "Monthly"}
    label = freq_labels.get(freq, freq)

    fig = px.line(
        time_series,
        x="date",
        y="message_count",
        title=f"{label} Message Count",
        labels={"message_count": "Messages", "date": "Date"},
    )
    fig.update_traces(line_color="#636EFA", line_width=2)
    fig.update_layout(hovermode="x unified", height=380)

    return time_series, fig


# ---------------------------------------------------------------------------
# Longest streak
# ---------------------------------------------------------------------------


def get_longest_streak(df: pd.DataFrame) -> dict:
    """
    Calculate the longest consecutive days with at least one message.

    Args:
        df: Messages DataFrame.

    Returns:
        Dict with 'streak' (int), 'start' (date), 'end' (date).
    """
    active_days = sorted(df["date"].dt.date.unique())

    if not active_days:
        return {"streak": 0, "start": None, "end": None}

    max_streak = 1
    current_streak = 1
    streak_start = active_days[0]
    best_start = active_days[0]
    best_end = active_days[0]

    for i in range(1, len(active_days)):
        delta = (active_days[i] - active_days[i - 1]).days
        if delta == 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                best_start = streak_start
                best_end = active_days[i]
        else:
            current_streak = 1
            streak_start = active_days[i]

    return {
        "streak": max_streak,
        "start": best_start,
        "end": best_end,
    }


# ---------------------------------------------------------------------------
# Night activity
# ---------------------------------------------------------------------------


def get_night_activity(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyse late-night messaging (22:00–05:59).

    Args:
        df: Messages DataFrame.

    Returns:
        Tuple of (per-sender night stats DataFrame, Plotly bar figure).
    """
    night_mask = (df["hour"] >= 22) | (df["hour"] < 6)
    night_df = df[night_mask].copy()

    night_hours = list(range(22, 24)) + list(range(0, 6))
    all_hours = pd.DataFrame({"hour": night_hours})
    night_by_hour = night_df.groupby("hour").size().reset_index(name="count")
    night_by_hour = all_hours.merge(night_by_hour, on="hour", how="left").fillna(0)
    night_by_hour["count"] = night_by_hour["count"].astype(int)

    fig = px.bar(
        night_by_hour,
        x="hour",
        y="count",
        title="Night Activity (22:00–05:59)",
        labels={"hour": "Hour of Day", "count": "Messages"},
        color="count",
        color_continuous_scale="Plasma",
    )
    fig.update_layout(height=380)

    night_senders = (
        night_df.groupby("sender")
        .size()
        .reset_index(name="night_messages")
        .sort_values("night_messages", ascending=False)
    )

    return night_senders, fig


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute high-level summary statistics for the entire dataset.

    Args:
        df: Messages DataFrame.

    Returns:
        Dict of summary statistics.
    """
    days_span = max((df["date"].max() - df["date"].min()).days, 1)

    top_chat = df.groupby("chat_name").size().idxmax() if len(df) else "N/A"
    top_sender = df.groupby("sender").size().idxmax() if len(df) else "N/A"

    return {
        "total_messages": len(df),
        "total_chats": int(df["chat_name"].nunique()),
        "total_contacts": int(df["sender"].nunique()),
        "date_range_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["date"].max().strftime("%Y-%m-%d"),
        "avg_messages_per_day": round(len(df) / days_span, 1),
        "total_words": int(df["word_count"].sum()),
        "avg_message_length": round(float(df["text_length"].mean()), 1),
        "most_active_chat": str(top_chat),
        "most_active_sender": str(top_sender),
        "days_span": days_span,
    }


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------


def get_monthly_breakdown(
    df: pd.DataFrame,
    top_n_chats: int = 5,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Return monthly message counts, broken down by top chats.

    Args:
        df: Messages DataFrame.
        top_n_chats: Number of top chats to show individually.

    Returns:
        Tuple of (monthly totals DataFrame, Plotly line figure).
    """
    # Overall monthly totals
    monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
    monthly["period"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    monthly = monthly.sort_values("period")

    # Per-chat monthly breakdown
    top_chats = df.groupby("chat_name").size().nlargest(top_n_chats).index.tolist()
    monthly_by_chat = df[df["chat_name"].isin(top_chats)].copy()
    monthly_by_chat["period"] = (
        monthly_by_chat["date"].dt.to_period("M").dt.to_timestamp()
    )
    monthly_by_chat = (
        monthly_by_chat.groupby(["period", "chat_name"])
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        monthly_by_chat,
        x="period",
        y="count",
        color="chat_name",
        title=f"Monthly Messages – Top {top_n_chats} Chats",
        labels={"count": "Messages", "period": "Month", "chat_name": "Chat"},
        markers=True,
    )
    fig.update_layout(hovermode="x unified", height=400)

    return monthly, fig


# ---------------------------------------------------------------------------
# Response time analysis
# ---------------------------------------------------------------------------


def get_response_time_stats(
    df: pd.DataFrame,
    chat_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Calculate average response times between messages in each chat.

    Args:
        df: Messages DataFrame.
        chat_name: If specified, analyse only this chat.

    Returns:
        Tuple of (response stats DataFrame, Plotly histogram figure).
    """
    work_df = df[df["chat_name"] == chat_name] if chat_name else df
    work_df = work_df.sort_values("date")

    records: list[dict] = []
    for cname, group in work_df.groupby("chat_name"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        for i in range(1, len(group)):
            prev_sender = group.loc[i - 1, "sender"]
            curr_sender = group.loc[i, "sender"]
            if prev_sender != curr_sender:
                diff_minutes = (
                    group.loc[i, "date"] - group.loc[i - 1, "date"]
                ).total_seconds() / 60.0
                # Cap at 24 hours to exclude silence gaps
                if 0 < diff_minutes <= 1440:
                    records.append(
                        {
                            "chat_name": cname,
                            "responder": curr_sender,
                            "response_minutes": round(diff_minutes, 1),
                        }
                    )

    if not records:
        return pd.DataFrame(), go.Figure()

    resp_df = pd.DataFrame(records)
    stats = (
        resp_df.groupby("chat_name")
        .agg(
            avg_response_min=("response_minutes", "median"),
            p25=("response_minutes", lambda x: x.quantile(0.25)),
            p75=("response_minutes", lambda x: x.quantile(0.75)),
            count=("response_minutes", "count"),
        )
        .reset_index()
        .sort_values("avg_response_min")
        .head(20)
    )
    stats["avg_response_min"] = stats["avg_response_min"].round(1)

    fig = px.histogram(
        resp_df[resp_df["response_minutes"] <= 120],
        x="response_minutes",
        nbins=60,
        title="Response Time Distribution (≤2 hours)",
        labels={"response_minutes": "Response Time (minutes)", "count": "Frequency"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(height=380)

    return stats, fig
