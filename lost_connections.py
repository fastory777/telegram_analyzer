"""Lost connections detection for Telegram data."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: find lost connections
# ---------------------------------------------------------------------------


def find_lost_connections(
    df: pd.DataFrame,
    min_messages: int = 10,
    inactive_months: int = 6,
    reference_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Identify contacts who have gone silent after a period of activity.

    A contact is "lost" when they exchanged at least `min_messages`
    total messages but have not sent or received any message in the
    last `inactive_months` months.

    Args:
        df: Messages DataFrame.
        min_messages: Minimum total messages to be considered.
        inactive_months: Number of months of inactivity required.
        reference_date: Reference point for inactivity (defaults to
            the last message date in the dataset).

    Returns:
        Tuple of (lost contacts DataFrame, Plotly scatter figure).
    """
    if reference_date is None:
        reference_date = df["date"].max()

    cutoff_date = reference_date - pd.DateOffset(months=inactive_months)

    # Overall chat-level stats
    all_stats = (
        df.groupby("chat_name")
        .agg(
            total_messages=("message_id", "count"),
            first_message=("date", "min"),
            last_message=("date", "max"),
            unique_senders=("sender", "nunique"),
        )
        .reset_index()
    )

    lost = all_stats[
        (all_stats["total_messages"] >= min_messages)
        & (all_stats["last_message"] < cutoff_date)
    ].copy()

    if lost.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                f"No lost connections found with current criteria<br>"
                f"(≥{min_messages} messages, inactive >{inactive_months} months)"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13),
        )
        fig.update_layout(
            title="Lost Connections",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return lost, fig

    lost["days_since_last"] = (reference_date - lost["last_message"]).dt.days
    lost["months_since_last"] = (lost["days_since_last"] / 30.44).round(1)
    lost["relationship_duration_days"] = (
        lost["last_message"] - lost["first_message"]
    ).dt.days
    lost = lost.sort_values("days_since_last", ascending=False).reset_index(drop=True)

    fig = px.scatter(
        lost,
        x="last_message",
        y="total_messages",
        size="total_messages",
        color="days_since_last",
        hover_name="chat_name",
        title=f"Lost Connections (inactive > {inactive_months} months)",
        labels={
            "last_message": "Last Message Date",
            "total_messages": "Total Messages",
            "days_since_last": "Days Inactive",
        },
        color_continuous_scale="Reds",
        size_max=40,
    )

    # Cutoff line
    fig.add_vline(
        x=cutoff_date.timestamp() * 1000,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Inactivity threshold ({inactive_months}mo)",
        annotation_position="top right",
    )
    fig.update_layout(height=480)

    return lost, fig


# ---------------------------------------------------------------------------
# Deleted accounts
# ---------------------------------------------------------------------------


def find_deleted_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find chats that contain messages from deleted Telegram accounts.

    Args:
        df: Messages DataFrame.

    Returns:
        DataFrame with columns: chat_name, message_count, first_message, last_message.
    """
    deleted_mask = df["is_deleted_account"].fillna(False)

    deleted_df = df[deleted_mask]
    if deleted_df.empty:
        return pd.DataFrame(
            columns=["chat_name", "message_count", "first_message", "last_message"]
        )

    stats = (
        deleted_df.groupby("chat_name")
        .agg(
            message_count=("message_id", "count"),
            first_message=("date", "min"),
            last_message=("date", "max"),
        )
        .reset_index()
        .sort_values("message_count", ascending=False)
    )

    return stats


# ---------------------------------------------------------------------------
# Individual relationship timeline
# ---------------------------------------------------------------------------


def get_relationship_timeline(df: pd.DataFrame, chat_name: str) -> go.Figure:
    """
    Plot the monthly message history for a specific chat.

    Args:
        df: Messages DataFrame.
        chat_name: The chat name to visualise.

    Returns:
        Plotly bar figure.
    """
    chat_df = df[df["chat_name"] == chat_name].copy()
    if chat_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data for '{chat_name}'",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    monthly = chat_df.set_index("date").resample("ME").size().reset_index()
    monthly.columns = ["date", "count"]

    fig = px.bar(
        monthly,
        x="date",
        y="count",
        title=f"Message History: {chat_name}",
        labels={"date": "Month", "count": "Messages"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Fading connections
# ---------------------------------------------------------------------------


def get_fading_connections(
    df: pd.DataFrame,
    window_months: int = 3,
    min_past_messages: int = 5,
    decline_pct: float = 50.0,
) -> pd.DataFrame:
    """
    Find connections that are becoming significantly less active.

    Compares message counts in the most recent `window_months` against
    the preceding `window_months` and flags chats with large declines.

    Args:
        df: Messages DataFrame.
        window_months: Size of each comparison window in months.
        min_past_messages: Minimum messages in past window to qualify.
        decline_pct: Percentage decline threshold to flag as "fading".

    Returns:
        DataFrame with columns: chat_name, past_count, recent_count, change_pct.
    """
    now = df["date"].max()
    recent_cutoff = now - pd.DateOffset(months=window_months)
    past_cutoff = recent_cutoff - pd.DateOffset(months=window_months)

    recent = (
        df[df["date"] >= recent_cutoff]
        .groupby("chat_name")
        .size()
        .reset_index(name="recent_count")
    )
    past = (
        df[(df["date"] >= past_cutoff) & (df["date"] < recent_cutoff)]
        .groupby("chat_name")
        .size()
        .reset_index(name="past_count")
    )

    comparison = past.merge(recent, on="chat_name", how="left").fillna(0)
    comparison["recent_count"] = comparison["recent_count"].astype(int)
    comparison["change_pct"] = (
        (comparison["recent_count"] - comparison["past_count"])
        / comparison["past_count"].replace(0, 1)
        * 100
    ).round(1)

    fading = comparison[
        (comparison["past_count"] >= min_past_messages)
        & (comparison["change_pct"] <= -decline_pct)
    ].sort_values("change_pct")

    return fading.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Communication phase detection
# ---------------------------------------------------------------------------


def detect_communication_phases(
    df: pd.DataFrame,
    chat_name: str,
    gap_days: int = 30,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Detect distinct phases in a conversation history.

    Phases are separated by gaps of at least `gap_days` days of silence.

    Args:
        df: Messages DataFrame.
        chat_name: The chat to analyse.
        gap_days: Minimum silence gap to count as a new phase.

    Returns:
        Tuple of (phases DataFrame, Plotly figure).
    """
    chat_df = df[df["chat_name"] == chat_name].sort_values("date").copy()
    if len(chat_df) < 2:
        return pd.DataFrame(), go.Figure()

    chat_df["date_only"] = chat_df["date"].dt.date
    daily = (
        chat_df.groupby("date_only").size().reset_index(name="count")
    )
    daily["date_only"] = pd.to_datetime(daily["date_only"])
    daily = daily.sort_values("date_only").reset_index(drop=True)

    # Identify phase boundaries
    gaps = daily["date_only"].diff().dt.days.fillna(0)
    phase_starts = daily.index[gaps > gap_days].tolist()
    phase_starts = [0] + phase_starts

    phases: list[dict] = []
    for i, start_idx in enumerate(phase_starts):
        end_idx = phase_starts[i + 1] - 1 if i + 1 < len(phase_starts) else len(daily) - 1
        phase_slice = daily.loc[start_idx:end_idx]
        phases.append(
            {
                "phase": i + 1,
                "start_date": phase_slice["date_only"].min(),
                "end_date": phase_slice["date_only"].max(),
                "messages": int(phase_slice["count"].sum()),
                "duration_days": (
                    phase_slice["date_only"].max() - phase_slice["date_only"].min()
                ).days
                + 1,
            }
        )

    phases_df = pd.DataFrame(phases)

    fig = px.bar(
        daily,
        x="date_only",
        y="count",
        title=f"Communication Phases: {chat_name}",
        labels={"date_only": "Date", "count": "Messages"},
        color_discrete_sequence=["#636EFA"],
    )

    # Add phase boundary lines
    for _, row in phases_df.iterrows():
        if row["phase"] > 1:
            fig.add_vline(
                x=str(row["start_date"]),
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Phase {int(row['phase'])}",
                annotation_position="top",
                annotation_font_size=10,
            )

    fig.update_layout(height=380)
    return phases_df, fig
