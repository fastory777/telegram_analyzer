"""Advanced search engine for the Telegram Data Analyzer.

Search modes
------------
SUBSTRING   Classic substring containment (default).
EXACT_WORD  Whole-word match; Unicode-safe, works for Cyrillic + Latin.
PHRASE      Exact phrase in the correct word order.
AND         All space-separated terms must appear in the message.
OR          At least one space-separated term must appear.
REGEX       Raw regular expression (invalid patterns are caught gracefully).

Multi-input
-----------
Multiple query fields can be combined with AND, OR, or Proximity logic.
In Proximity mode all terms must appear within a configurable token window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import product as _iproduct
from typing import Optional

import pandas as pd

from tokenizer import lemmatize_word, strip_diacritics, PYMORPHY2_AVAILABLE

# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class SearchMode(str, Enum):
    SUBSTRING  = "Substring"
    EXACT_WORD = "Exact Word"
    PHRASE     = "Phrase"
    AND        = "AND (all words)"
    OR         = "OR (any word)"
    REGEX      = "Regex"


class MultiInputMode(str, Enum):
    AND       = "AND"
    OR        = "OR"
    PROXIMITY = "Proximity"


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------


@dataclass
class SearchOptions:
    # Text normalisation
    case_sensitive:     bool = False
    ignore_punctuation: bool = False
    ignore_diacritics:  bool = False
    lemmatize:          bool = False

    # Multi-input combination
    multi_mode:          MultiInputMode = MultiInputMode.AND
    proximity_distance:  int            = 5   # tokens

    # Pre-search row filters
    chat_filter:   Optional[list[str]]      = None  # None → all chats
    sender_filter: Optional[str]            = None
    date_start:    Optional[pd.Timestamp]   = None
    date_end:      Optional[pd.Timestamp]   = None

    # Context window
    context_before: int = 5
    context_after:  int = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _preprocess_text(text: str, opts: SearchOptions) -> str:
    """Normalise a message text string according to *opts*."""
    if not isinstance(text, str):
        return ""
    t = text
    if not opts.case_sensitive:
        t = t.lower()
    if opts.ignore_diacritics:
        t = strip_diacritics(t)
    if opts.ignore_punctuation:
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return t


def _preprocess_query(query: str, opts: SearchOptions, mode: SearchMode) -> str:
    """Normalise a query string according to *opts* and *mode*."""
    q = query.strip()
    if not opts.case_sensitive:
        q = q.lower()
    if opts.ignore_diacritics:
        q = strip_diacritics(q)
    if opts.ignore_punctuation and mode != SearchMode.REGEX:
        q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    if opts.lemmatize and mode in (SearchMode.EXACT_WORD, SearchMode.AND, SearchMode.OR):
        words = q.split()
        q = " ".join(lemmatize_word(w) for w in words)
    return q


def _word_boundary_pattern(word: str, case_sensitive: bool = True) -> re.Pattern:
    """
    Compile a Unicode-aware whole-word regex for *word*.

    Uses ``(?<!\\w)`` / ``(?!\\w)`` look-around instead of ``\\b`` so that
    Cyrillic characters (and other non-ASCII word chars) are handled correctly.
    """
    flags = re.UNICODE
    if not case_sensitive:
        flags |= re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", flags)


def _apply_filters(df: pd.DataFrame, opts: SearchOptions) -> pd.DataFrame:
    """Return a filtered view of *df* according to *opts*."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if opts.chat_filter:
        mask &= df["chat_name"].isin(opts.chat_filter)
    if opts.sender_filter and opts.sender_filter.strip():
        mask &= df["sender"].str.contains(
            opts.sender_filter.strip(), case=False, na=False
        )
    if opts.date_start is not None:
        mask &= df["date"] >= opts.date_start
    if opts.date_end is not None:
        mask &= df["date"] <= opts.date_end
    return df[mask]


def _proximity_match(text: str, terms: list[str], max_distance: int) -> bool:
    """
    Return True if every term in *terms* appears within *max_distance* tokens
    of each other in *text*.
    """
    tokens = re.findall(r"\w+", text, re.UNICODE)
    if not tokens:
        return False
    positions: list[list[int]] = []
    for term in terms:
        pos = [i for i, t in enumerate(tokens) if t == term]
        if not pos:
            return False
        positions.append(pos)
    for combo in _iproduct(*positions):
        if max(combo) - min(combo) <= max_distance:
            return True
    return False


# ---------------------------------------------------------------------------
# Single-query search
# ---------------------------------------------------------------------------


def search_single(
    df: pd.DataFrame,
    query: str,
    mode: SearchMode,
    opts: SearchOptions,
) -> pd.DataFrame:
    """
    Search *df* for a single *query* string.

    Args:
        df: Messages DataFrame (already filtered by :func:`_apply_filters`).
        query: Search query string.
        mode: Which search mode to apply.
        opts: Normalisation options.

    Returns:
        Subset of *df* whose ``text`` column matches *query*.

    Raises:
        ValueError: If *mode* is ``REGEX`` and *query* is an invalid pattern.
    """
    if not query.strip():
        return df.iloc[0:0]

    proc: pd.Series = df["text"].apply(lambda t: _preprocess_text(t, opts))

    if mode == SearchMode.REGEX:
        try:
            flags = re.UNICODE | (0 if opts.case_sensitive else re.IGNORECASE)
            pat = re.compile(query.strip(), flags)
            mask = proc.apply(lambda t: bool(pat.search(t)))
        except re.error as exc:
            raise ValueError(f"Invalid regex: {exc}") from exc

    elif mode == SearchMode.SUBSTRING:
        q = _preprocess_query(query, opts, mode)
        mask = proc.str.contains(re.escape(q), regex=True, na=False)

    elif mode == SearchMode.EXACT_WORD:
        q = _preprocess_query(query, opts, mode)
        # Already lowercased if not case-sensitive; pass case_sensitive=True
        pat = _word_boundary_pattern(q, case_sensitive=True)
        mask = proc.apply(lambda t: bool(pat.search(t)))

    elif mode == SearchMode.PHRASE:
        q = _preprocess_query(query, opts, mode)
        mask = proc.str.contains(re.escape(q), regex=True, na=False)

    elif mode == SearchMode.AND:
        terms = [_preprocess_query(w, opts, mode) for w in query.split() if w.strip()]
        if not terms:
            return df.iloc[0:0]
        mask = proc.apply(lambda t: all(term in t for term in terms))

    elif mode == SearchMode.OR:
        terms = [_preprocess_query(w, opts, mode) for w in query.split() if w.strip()]
        if not terms:
            return df.iloc[0:0]
        mask = proc.apply(lambda t: any(term in t for term in terms))

    else:
        mask = pd.Series(False, index=df.index)

    return df[mask]


# ---------------------------------------------------------------------------
# Multi-query search  (main public entry point)
# ---------------------------------------------------------------------------


def search(
    df: pd.DataFrame,
    queries: list[str],
    mode: SearchMode,
    opts: SearchOptions,
) -> pd.DataFrame:
    """
    Multi-input search — combine several query fields.

    Each non-empty element of *queries* represents one search-input field.
    Results are combined according to ``opts.multi_mode``:

    * ``AND``       — all queries must match (intersection).
    * ``OR``        — any query may match (union).
    * ``Proximity`` — all terms must appear within ``opts.proximity_distance``
                      tokens of each other in the same message.

    Args:
        df: Full messages DataFrame.
        queries: One query string per input field.
        mode: Search mode applied to each individual query.
        opts: Normalisation and filter options.

    Returns:
        Filtered DataFrame sorted by ``date``.

    Raises:
        ValueError: If ``mode`` is ``REGEX`` and any query is invalid.
    """
    queries = [q.strip() for q in queries if q.strip()]
    if not queries:
        return df.iloc[0:0]

    filtered = _apply_filters(df, opts)
    if filtered.empty:
        return filtered

    # Fast path: single query
    if len(queries) == 1:
        return search_single(filtered, queries[0], mode, opts)

    # Proximity: all terms evaluated together
    if opts.multi_mode == MultiInputMode.PROXIMITY:
        proc = filtered["text"].apply(lambda t: _preprocess_text(t, opts))
        terms = [_preprocess_query(q, opts, mode) for q in queries]
        mask = proc.apply(
            lambda t: _proximity_match(t, terms, opts.proximity_distance)
        )
        return filtered[mask]

    # AND / OR across multiple independent query fields
    result_sets: list[pd.Index] = [
        search_single(filtered, q, mode, opts).index for q in queries
    ]
    if opts.multi_mode == MultiInputMode.AND:
        combined: pd.Index = result_sets[0]
        for s in result_sets[1:]:
            combined = combined.intersection(s)
    else:  # OR
        combined = result_sets[0]
        for s in result_sets[1:]:
            combined = combined.union(s)

    return filtered.loc[filtered.index.intersection(combined)]


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def get_context_window(
    df: pd.DataFrame,
    match_indices: pd.Index,
    before: int = 5,
    after: int = 5,
) -> pd.DataFrame:
    """
    Collect context rows around each matched row (within the same chat).

    Args:
        df: Full (or filtered) messages DataFrame, sorted by date.
        match_indices: Row indices of matched messages.
        before: Messages to include before each match.
        after: Messages to include after each match.

    Returns:
        DataFrame containing all context rows with an added ``is_match``
        boolean column.
    """
    if match_indices.empty:
        empty = df.iloc[0:0].copy()
        empty["is_match"] = False
        return empty

    match_set: set[int] = set(match_indices.tolist())
    positional = list(df.index)
    pos_map: dict[int, int] = {idx: p for p, idx in enumerate(positional)}

    context_idx: set[int] = set()
    for idx in match_set:
        if idx not in pos_map:
            continue
        p = pos_map[idx]
        chat = df.at[idx, "chat_name"]
        lo = max(0, p - before)
        hi = min(len(positional), p + after + 1)
        for pi in range(lo, hi):
            ci = positional[pi]
            if df.at[ci, "chat_name"] == chat:
                context_idx.add(ci)

    ctx = df.loc[sorted(context_idx)].copy()
    ctx["is_match"] = ctx.index.isin(match_set)
    return ctx


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------


def highlight_text(
    text: str,
    queries: list[str],
    mode: SearchMode,
    opts: SearchOptions,
) -> str:
    """
    Return HTML with matched portions wrapped in golden ``<mark>`` tags.

    Args:
        text: Plain message text.
        queries: List of query strings (one per input field).
        mode: Search mode (controls which parts of each query to highlight).
        opts: Search options (case sensitivity).

    Returns:
        HTML-escaped string with ``<mark>`` tags around matches.
    """
    import html as _html

    if not text or not queries:
        return _html.escape(str(text))

    result = _html.escape(str(text))
    flags = re.UNICODE | (0 if opts.case_sensitive else re.IGNORECASE)

    _mark = (
        '<mark style="background:#FFD700;color:#000;'
        'padding:0 2px;border-radius:3px;">{}</mark>'
    )

    for query in queries:
        q = query.strip()
        if not q:
            continue
        try:
            if mode == SearchMode.REGEX:
                pat = re.compile(q, flags)
                result = pat.sub(lambda m: _mark.format(m.group()), result)

            elif mode == SearchMode.EXACT_WORD:
                pat = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)", flags)
                result = pat.sub(lambda m: _mark.format(m.group()), result)

            elif mode in (SearchMode.AND, SearchMode.OR):
                for term in q.split():
                    if term.strip():
                        pat = re.compile(re.escape(term.strip()), flags)
                        result = pat.sub(lambda m: _mark.format(m.group()), result)

            else:  # SUBSTRING, PHRASE
                pat = re.compile(re.escape(q), flags)
                result = pat.sub(lambda m: _mark.format(m.group()), result)

        except re.error:
            pass  # skip bad patterns silently

    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def get_match_stats(results: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-chat match counts and date range for *results*.

    Args:
        results: DataFrame returned by :func:`search`.

    Returns:
        DataFrame with columns: ``chat_name``, ``matches``,
        ``first_match``, ``last_match``.
    """
    if results.empty:
        return pd.DataFrame()
    return (
        results.groupby("chat_name")
        .agg(
            matches=("text", "count"),
            first_match=("date", "min"),
            last_match=("date", "max"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )
