"""Keyword search engine for Telegram messages."""

import re
from typing import List, Optional

import pandas as pd


def _build_pattern(keywords: List[str], case_sensitive: bool = False) -> Optional[re.Pattern]:
    """
    Compile a regex pattern from a list of keyword strings.

    Args:
        keywords: List of keyword strings to search for.
        case_sensitive: Whether to match case-sensitively.

    Returns:
        Compiled regex pattern, or None if keywords list is empty.
    """
    cleaned = [k.strip() for k in keywords if k.strip()]
    if not cleaned:
        return None
    escaped = [re.escape(k) for k in cleaned]
    pattern_str = "|".join(escaped)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern_str, flags)


def search_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    context_size: int = 5,
    case_sensitive: bool = False,
    max_results: int = 500,
) -> list[dict]:
    """
    Search for keywords across all chats with surrounding context.

    For each match, `context_size` messages before and after (within
    the same chat) are returned alongside the matching message.

    Args:
        df: Full messages DataFrame sorted by date.
        keywords: List of keyword strings to search for.
        context_size: Number of messages to include before/after each match.
        case_sensitive: Whether search is case-sensitive.
        max_results: Maximum number of results to return.

    Returns:
        List of result dicts, each containing:
            - chat_name: str
            - match_date: datetime
            - match_sender: str
            - match_text: str
            - matched_keywords: list of matched keyword strings
            - context: list of context message dicts
    """
    pattern = _build_pattern(keywords, case_sensitive)
    if pattern is None:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    # Identify matching row indices
    matches_mask = df["text"].str.contains(pattern, regex=True, na=False)
    match_indices = df[matches_mask].index.tolist()

    results: list[dict] = []
    seen: set[int] = set()

    for idx in match_indices:
        if idx in seen:
            continue
        seen.add(idx)

        chat_name = df.loc[idx, "chat_name"]

        # Collect context within the same chat
        start = max(0, idx - context_size)
        end = min(len(df) - 1, idx + context_size)
        context_slice = df.loc[start:end]
        context_slice = context_slice[context_slice["chat_name"] == chat_name]

        context_messages: list[dict] = []
        for ctx_idx, row in context_slice.iterrows():
            context_messages.append(
                {
                    "is_match": ctx_idx == idx,
                    "sender": row["sender"],
                    "text": row["text"],
                    "date": row["date"],
                }
            )

        # Determine which keywords actually matched
        matched = [
            k.strip()
            for k in keywords
            if k.strip()
            and re.search(
                re.escape(k.strip()),
                df.loc[idx, "text"],
                0 if case_sensitive else re.IGNORECASE,
            )
        ]

        results.append(
            {
                "chat_name": chat_name,
                "match_date": df.loc[idx, "date"],
                "match_sender": df.loc[idx, "sender"],
                "match_text": df.loc[idx, "text"],
                "matched_keywords": matched,
                "context": context_messages,
            }
        )

        if len(results) >= max_results:
            break

    return results


def highlight_keywords(
    text: str,
    keywords: List[str],
    case_sensitive: bool = False,
) -> str:
    """
    Wrap matched keywords in an HTML <mark> tag for highlighting.

    Args:
        text: Plain text to highlight within.
        keywords: List of keyword strings to highlight.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        HTML string with keywords wrapped in <mark> tags.
    """
    # Escape HTML entities first
    import html as html_mod
    text = html_mod.escape(text)

    flags = 0 if case_sensitive else re.IGNORECASE
    for keyword in keywords:
        kw = keyword.strip()
        if not kw:
            continue
        escaped_kw = re.escape(html_mod.escape(kw))
        text = re.sub(
            escaped_kw,
            lambda m: (
                f'<mark style="background:#FFD700;color:#000;padding:0 2px;'
                f'border-radius:3px;">{m.group()}</mark>'
            ),
            text,
            flags=flags,
        )
    return text


def get_keyword_stats(
    df: pd.DataFrame,
    keywords: List[str],
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Return per-chat occurrence counts for the given keywords.

    Args:
        df: Messages DataFrame.
        keywords: List of keyword strings.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        DataFrame with columns: chat_name, occurrences, first_occurrence, last_occurrence.
    """
    pattern = _build_pattern(keywords, case_sensitive)
    if pattern is None:
        return pd.DataFrame()

    matches = df[df["text"].str.contains(pattern, regex=True, na=False)]

    if matches.empty:
        return pd.DataFrame()

    stats = (
        matches.groupby("chat_name")
        .agg(
            occurrences=("message_id", "count"),
            first_occurrence=("date", "min"),
            last_occurrence=("date", "max"),
        )
        .reset_index()
        .sort_values("occurrences", ascending=False)
    )

    return stats


def get_keyword_timeline(
    df: pd.DataFrame,
    keywords: List[str],
    freq: str = "ME",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """
    Count keyword occurrences per time period.

    Args:
        df: Messages DataFrame.
        keywords: List of keyword strings.
        freq: Pandas resample frequency string.
        case_sensitive: Whether matching is case-sensitive.

    Returns:
        DataFrame with columns: period, count.
    """
    pattern = _build_pattern(keywords, case_sensitive)
    if pattern is None:
        return pd.DataFrame()

    matches = df[df["text"].str.contains(pattern, regex=True, na=False)].copy()
    if matches.empty:
        return pd.DataFrame()

    timeline = matches.set_index("date").resample(freq).size().reset_index()
    timeline.columns = ["period", "count"]
    return timeline
