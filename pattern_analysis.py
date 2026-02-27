"""Message-pattern analysis for the Telegram Data Analyzer.

Computes bigrams, trigrams (and arbitrary n-grams) across all messages,
with stopword filtering and optional lemmatisation.

Public API
----------
compute_ngram_frequencies   Raw Counter of n-gram frequencies.
get_top_ngrams              Top-K n-grams as (DataFrame, bar chart).
get_top_bigrams             Convenience wrapper for n=2.
get_top_trigrams            Convenience wrapper for n=3.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tokenizer import tokenize, lemmatize_tokens, get_stopwords


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return all consecutive n-grams of *tokens*."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------


def compute_ngram_frequencies(
    df: pd.DataFrame,
    n: int,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
) -> Counter:
    """
    Count n-gram frequencies across all messages in *df*.

    Stopwords are removed *before* building n-grams so that common function
    words do not dominate bigram / trigram lists.

    Args:
        df: Messages DataFrame with a ``text`` column.
        n: N-gram size (2 = bigrams, 3 = trigrams, …).
        stopwords: Words to exclude before building n-grams.
        min_length: Minimum token character length.
        lemmatize: Apply lemmatisation via pymorphy2.

    Returns:
        ``collections.Counter`` mapping ``tuple[str, …]`` → count.
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
        filtered = [t for t in tokens if t not in stopwords]
        if len(filtered) >= n:
            counter.update(_get_ngrams(filtered, n))
    return counter


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def get_top_ngrams(
    df: pd.DataFrame,
    n: int,
    top_k: int = 30,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Return the top *top_k* n-grams as a DataFrame and a horizontal bar chart.

    Args:
        df: Messages DataFrame.
        n: N-gram size.
        top_k: Number of n-grams to show.
        stopwords: Words to exclude before n-gram construction.
        min_length: Minimum token length.
        lemmatize: Apply lemmatisation.

    Returns:
        ``(ngrams_df, fig)`` — DataFrame with columns ``ngram`` / ``count``
        and a Plotly horizontal bar chart.
    """
    freq = compute_ngram_frequencies(
        df, n=n, stopwords=stopwords, min_length=min_length, lemmatize=lemmatize
    )
    top = freq.most_common(top_k)

    ng_label = {2: "Bigrams", 3: "Trigrams"}.get(n, f"{n}-grams")
    singular  = ng_label.rstrip("s")

    if not top:
        fig = go.Figure()
        fig.update_layout(title=f"No {ng_label} found — try reducing min word length")
        return pd.DataFrame(columns=["ngram", "count"]), fig

    ngrams_df = pd.DataFrame(
        [(" ".join(g), c) for g, c in top],
        columns=["ngram", "count"],
    )

    color_scale = "Oranges" if n == 2 else "Purples"

    fig = px.bar(
        ngrams_df,
        x="count",
        y="ngram",
        orientation="h",
        title=f"Top {top_k} {ng_label}",
        labels={"count": "Occurrences", "ngram": singular},
        color="count",
        color_continuous_scale=color_scale,
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, min(top_k * 20, 1400)),
        coloraxis_showscale=False,
    )
    return ngrams_df, fig


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def get_top_bigrams(
    df: pd.DataFrame,
    top_k: int = 30,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
) -> tuple[pd.DataFrame, go.Figure]:
    """Return top *top_k* bigrams (n=2)."""
    return get_top_ngrams(
        df,
        n=2,
        top_k=top_k,
        stopwords=stopwords,
        min_length=min_length,
        lemmatize=lemmatize,
    )


def get_top_trigrams(
    df: pd.DataFrame,
    top_k: int = 30,
    stopwords: Optional[set[str]] = None,
    min_length: int = 3,
    lemmatize: bool = False,
) -> tuple[pd.DataFrame, go.Figure]:
    """Return top *top_k* trigrams (n=3)."""
    return get_top_ngrams(
        df,
        n=3,
        top_k=top_k,
        stopwords=stopwords,
        min_length=min_length,
        lemmatize=lemmatize,
    )
