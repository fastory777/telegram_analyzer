"""Utility functions for the Telegram Data Analyzer."""

import re
import os
from datetime import datetime
from typing import Union, Optional
import pandas as pd


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.

    Supports multiple Telegram date formats.

    Args:
        date_str: Date string to parse.

    Returns:
        Parsed datetime or None if unparseable.
    """
    if not date_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def normalize_text(text_field) -> str:
    """
    Normalize Telegram message text.

    Telegram exports text as either a plain string or a list of mixed
    strings and dicts (for styled text, mentions, links, etc.).

    Args:
        text_field: Raw text field from Telegram JSON.

    Returns:
        Plain string with concatenated content.
    """
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        parts: list[str] = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def clean_username(name: str) -> str:
    """
    Clean and normalize a username.

    Args:
        name: Raw username string.

    Returns:
        Cleaned username string.
    """
    if not name:
        return "Unknown"
    return str(name).strip()


def get_time_of_day(hour: int) -> str:
    """
    Classify an hour (0-23) into a named time-of-day period.

    Args:
        hour: Hour integer (0-23).

    Returns:
        One of 'Morning', 'Afternoon', 'Evening', 'Night'.
    """
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def format_number(n: Union[int, float]) -> str:
    """
    Format a number with thousand separators.

    Args:
        n: Number to format.

    Returns:
        Formatted string, e.g. '1,234,567'.
    """
    return f"{int(n):,}"


def find_json_files(directory: str) -> list[str]:
    """
    Recursively find all messages.json files under a directory.

    Args:
        directory: Root directory to search.

    Returns:
        List of absolute paths to messages.json files.
    """
    json_files: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file == "messages.json":
                json_files.append(os.path.join(root, file))
    return json_files


def get_export_metadata(export_path: str) -> dict:
    """
    Get metadata about the export directory.

    Args:
        export_path: Path to Telegram export directory.

    Returns:
        Dict with 'total_chats', 'export_path', 'json_files'.
    """
    json_files = find_json_files(export_path)
    return {
        "total_chats": len(json_files),
        "export_path": export_path,
        "json_files": json_files,
    }


def is_deleted_account(sender: str, sender_id: str) -> bool:
    """
    Determine if a message sender is a deleted Telegram account.

    Args:
        sender: Display name of the sender.
        sender_id: Telegram user ID string.

    Returns:
        True if the account appears to be deleted.
    """
    sender_lower = sender.lower() if sender else ""
    id_lower = sender_id.lower() if sender_id else ""
    return (
        "deleted account" in sender_lower
        or "deleted_account" in sender_lower
        or "deleted" in id_lower
    )


def truncate_text(text: str, max_len: int = 100) -> str:
    """
    Truncate text to a maximum length, appending '...' if truncated.

    Args:
        text: Input text.
        max_len: Maximum character length.

    Returns:
        Potentially truncated string.
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."
