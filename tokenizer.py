"""Text tokenization utilities for the Telegram Data Analyzer.

Provides Unicode-aware tokenization, stopword removal,
inverted index construction, and optional Russian lemmatization
via pymorphy2 (install separately: ``pip install pymorphy2``).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully
# ---------------------------------------------------------------------------

try:
    import pymorphy2 as _pymorphy2

    _morph = _pymorphy2.MorphAnalyzer()
    PYMORPHY2_AVAILABLE = True
except Exception:
    PYMORPHY2_AVAILABLE = False

try:
    import nltk as _nltk

    try:
        from nltk.corpus import stopwords as _sw

        _RU_STOPS: set[str] = set(_sw.words("russian"))
        _EN_STOPS: set[str] = set(_sw.words("english"))
    except LookupError:
        _nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _sw

        _RU_STOPS = set(_sw.words("russian"))
        _EN_STOPS = set(_sw.words("english"))
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    _RU_STOPS = set()
    _EN_STOPS = set()

# ---------------------------------------------------------------------------
# Built-in fallback stopwords (used when NLTK is not installed)
# ---------------------------------------------------------------------------

_BASIC_RU_STOPS: set[str] = {
    "и", "в", "не", "на", "я", "с", "что", "а", "по", "это",
    "он", "как", "но", "то", "все", "она", "так", "его", "за",
    "от", "у", "из", "мне", "ты", "мы", "вы", "они", "нас",
    "вас", "был", "была", "было", "были", "есть", "нет", "да",
    "ну", "же", "ли", "еще", "уже", "вот", "бы", "себя",
    "свой", "для", "до", "или", "об", "при", "ко", "без",
    "над", "под", "через", "после", "перед", "между", "те",
    "эти", "этот", "эта", "этим", "если", "когда", "тогда",
    "там", "тут", "здесь", "очень", "просто", "только",
    "можно", "надо", "нужно", "хочу", "буду", "будет", "тебе",
    "меня", "тебя", "него", "неё", "них", "им", "ей", "ему",
    "её", "их", "мой", "твой", "наш", "ваш", "который",
    "которая", "которое", "которые", "ещё", "сейчас",
    "потому", "этого", "этой", "того", "той", "тот",
    "чего", "кто", "где", "куда", "зачем", "почему",
    "будто", "чтобы", "чтоб", "тоже", "такой", "такая",
    "какой", "какая", "вообще", "всё", "больше", "меньше",
    "сам", "сама", "само", "сами", "весь", "вся", "всё",
}

_BASIC_EN_STOPS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "is", "was",
    "are", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "shall", "that", "this", "these", "those",
    "it", "its", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "how", "when",
    "where", "why", "not", "so", "if", "as", "than", "then",
    "up", "out", "no", "just", "more", "also", "into", "about",
    "over", "after", "before", "between", "both", "through",
    "during", "each", "been", "being", "same", "too", "very",
    "own", "other", "such", "now", "here", "there", "one",
    "all", "some", "any", "only", "even", "still", "already",
    "again", "well", "back", "get", "got", "go", "going",
}


def get_stopwords(
    include_russian: bool = True,
    include_english: bool = True,
) -> set[str]:
    """Return a combined stopword set for the requested languages."""
    stops: set[str] = set()
    if include_russian:
        stops |= _RU_STOPS if _RU_STOPS else _BASIC_RU_STOPS
    if include_english:
        stops |= _EN_STOPS if _EN_STOPS else _BASIC_EN_STOPS
    return stops


# ---------------------------------------------------------------------------
# Core tokenization
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Remove diacritic marks (accents, combining characters) from text."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def tokenize(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_diacritics: bool = False,
    min_length: int = 1,
) -> list[str]:
    """
    Tokenize *text* into words, Unicode-safe.

    Args:
        text: Input string.
        lowercase: Convert to lowercase before tokenizing.
        remove_punctuation: Strip non-word characters first.
        remove_diacritics: Remove accent / combining marks.
        min_length: Minimum token character length.

    Returns:
        List of word tokens.
    """
    if not text:
        return []
    if lowercase:
        text = text.lower()
    if remove_diacritics:
        text = strip_diacritics(text)
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = _TOKEN_PATTERN.findall(text)
    if min_length > 1:
        tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


@lru_cache(maxsize=50_000)
def lemmatize_word(word: str) -> str:
    """
    Lemmatize a single word using pymorphy2 (Russian-aware).

    Falls back to the original word if pymorphy2 is not installed.
    Results are cached (LRU, 50 000 entries) for performance.
    """
    if not PYMORPHY2_AVAILABLE:
        return word
    try:
        parsed = _morph.parse(word)
        if parsed:
            return parsed[0].normal_form
    except Exception:
        pass
    return word


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens."""
    if not PYMORPHY2_AVAILABLE:
        return tokens
    return [lemmatize_word(t) for t in tokens]


# ---------------------------------------------------------------------------
# Inverted index (optional performance layer)
# ---------------------------------------------------------------------------


def build_inverted_index(
    df: pd.DataFrame,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    lemmatize: bool = False,
    min_length: int = 2,
) -> dict[str, set[int]]:
    """
    Build a word → set-of-row-indices inverted index.

    The index maps each word token to the set of integer DataFrame row indices
    where that token appears.  Used to accelerate exact-word searches.

    Args:
        df: Messages DataFrame with a ``text`` column.
        lowercase: Normalise tokens to lowercase.
        remove_punctuation: Strip punctuation before tokenising.
        lemmatize: Apply lemmatization (requires pymorphy2).
        min_length: Minimum token length to index.

    Returns:
        ``dict[str, set[int]]``
    """
    index: dict[str, set[int]] = defaultdict(set)
    for idx, text in df["text"].items():
        if not isinstance(text, str) or not text:
            continue
        tokens = tokenize(
            text,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            min_length=min_length,
        )
        if lemmatize:
            tokens = lemmatize_tokens(tokens)
        for token in set(tokens):
            index[token].add(int(idx))
    return dict(index)
