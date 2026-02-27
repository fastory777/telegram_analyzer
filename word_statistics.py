"""Word-frequency statistics for the Telegram Data Analyzer.

Public API
----------
compute_word_frequencies   Counter of word frequencies across a DataFrame.
get_top_words              Top-N word bar chart + DataFrame.
generate_wordcloud_figure  Plotly-embedded word cloud (requires ``wordcloud``).
get_repeated_messages      Verbatim-repeated message detection.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tokenizer import (
    tokenize,
    lemmatize_tokens,
    get_stopwords,
    PYMORPHY2_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core frequency computation
# ---------------------------------------------------------------------------


def compute_word_frequencies(
    df: pd.DataFrame,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
) -> Counter:
    """
    Count word frequencies across all messages in *df*.

    Args:
        df: Messages DataFrame with a ``text`` column.
        stopwords: Set of words to exclude.  Defaults to Russian + English.
        min_length: Minimum token character length (shorter tokens ignored).
        lemmatize: Apply lemmatization via pymorphy2 (Russian-aware).

    Returns:
        ``collections.Counter`` mapping token → count.
    """
    if stopwords is None:
        stopwords = get_stopwords()

    counter: Counter = Counter()
    for text in df["text"]:
        if not isinstance(text, str) or not text:
            continue
        tokens = tokenize(
            text,
            lowercase=True,
            remove_punctuation=True,
            min_length=min_length,
        )
        if lemmatize:
            tokens = lemmatize_tokens(tokens)
        counter.update(t for t in tokens if t not in stopwords)
    return counter


# ---------------------------------------------------------------------------
# Top-N words
# ---------------------------------------------------------------------------


def get_top_words(
    df: pd.DataFrame,
    n: int = 50,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
    chat_filter: Optional[str] = None,
    sender_filter: Optional[str] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Compute the top *n* most frequent words and return a bar chart.

    Args:
        df: Messages DataFrame.
        n: Number of words to show.
        stopwords: Words to exclude.
        min_length: Minimum token length.
        lemmatize: Apply lemmatization.
        chat_filter: Restrict to a single chat name (or None for all).
        sender_filter: Restrict to a single sender name (or None for all).
        date_start / date_end: Optional date range filter.

    Returns:
        ``(words_df, fig)`` — DataFrame with columns ``word`` / ``count``
        and a Plotly horizontal bar chart.
    """
    sub = df.copy()
    if chat_filter:
        sub = sub[sub["chat_name"] == chat_filter]
    if sender_filter:
        sub = sub[sub["sender"] == sender_filter]
    if date_start is not None:
        sub = sub[sub["date"] >= date_start]
    if date_end is not None:
        sub = sub[sub["date"] <= date_end]

    if stopwords is None:
        stopwords = get_stopwords()

    freq = compute_word_frequencies(
        sub, stopwords=stopwords, min_length=min_length, lemmatize=lemmatize
    )
    top = freq.most_common(n)

    if not top:
        fig = go.Figure()
        fig.update_layout(
            title="No words found — try reducing min word length or disabling stopwords"
        )
        return pd.DataFrame(columns=["word", "count"]), fig

    words_df = pd.DataFrame(top, columns=["word", "count"])

    fig = px.bar(
        words_df,
        x="count",
        y="word",
        orientation="h",
        title=f"Top {n} Most Frequent Words",
        labels={"count": "Occurrences", "word": "Word"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, min(n * 18, 1400)),
        showlegend=False,
        coloraxis_showscale=False,
    )
    return words_df, fig


# ---------------------------------------------------------------------------
# Word cloud
# ---------------------------------------------------------------------------


def generate_wordcloud_figure(
    df: pd.DataFrame,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
    max_words: int = 200,
) -> Optional[go.Figure]:
    """
    Generate a Plotly-embedded word cloud image.

    Requires the ``wordcloud`` library (``pip install wordcloud``).

    Args:
        df: Messages DataFrame.
        stopwords: Words to exclude.
        min_length: Minimum token length.
        lemmatize: Apply lemmatization.
        max_words: Maximum number of words in the cloud.

    Returns:
        Plotly Figure, or ``None`` if ``wordcloud`` is not installed.
    """
    if not WORDCLOUD_AVAILABLE:
        return None

    if stopwords is None:
        stopwords = get_stopwords()

    freq = compute_word_frequencies(
        df, stopwords=stopwords, min_length=min_length, lemmatize=lemmatize
    )
    if not freq:
        return None

    wc = WordCloud(
        width=900,
        height=450,
        max_words=max_words,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(freq)

    img_array = wc.to_array()

    fig = px.imshow(img_array, title="Word Cloud")
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Repeated messages
# ---------------------------------------------------------------------------


def get_repeated_messages(
    df: pd.DataFrame,
    min_count: int = 3,
    max_length: int = 300,
) -> pd.DataFrame:
    """
    Find verbatim-repeated messages (case-insensitive).

    Args:
        df: Messages DataFrame.
        min_count: Minimum number of repetitions to include.
        max_length: Maximum text length considered (long texts skipped).

    Returns:
        DataFrame sorted by ``count`` descending with columns:
        ``message``, ``count``, ``senders``, ``chats``,
        ``first_seen``, ``last_seen``.
    """
    sub = df[df["text"].str.len().le(max_length)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["_norm"] = sub["text"].str.lower().str.strip()

    # Count occurrences per normalised text
    counts = sub["_norm"].value_counts()
    frequent_texts = counts[counts >= min_count].index

    if len(frequent_texts) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    for norm_text in frequent_texts:
        group = sub[sub["_norm"] == norm_text]
        senders = sorted(set(group["sender"].tolist()))
        chats = sorted(set(group["chat_name"].tolist()))
        rows.append(
            {
                "message":    group["text"].iloc[0],
                "count":      len(group),
                "senders":    ", ".join(senders[:5]) + ("…" if len(senders) > 5 else ""),
                "chats":      ", ".join(chats[:5]) + ("…" if len(chats) > 5 else ""),
                "first_seen": group["date"].min(),
                "last_seen":  group["date"].max(),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("count", ascending=False).reset_index(drop=True)
    return result
