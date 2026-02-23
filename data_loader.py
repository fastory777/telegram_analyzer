"""Data loader for Telegram exported JSON data.

Supports both export formats produced by Telegram Desktop:

* **Unified export** – a single ``result.json`` at the root of the export
  directory.  All chats live under ``result.json → chats → list``.
  (This is the format produced by modern Telegram Desktop versions.)

* **Per-chat export** – individual ``messages.json`` files inside
  ``chats/chat_XXX/`` sub-directories.
  (Legacy format; kept for backward-compatibility.)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import pandas as pd

from utils import normalize_text, parse_date, clean_username, find_json_files, is_deleted_account

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level JSON loading
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Optional[dict | list]:
    """Read and return parsed JSON from *path*, or None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Encoding fallback failed for %s: %s", path, exc)
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode error in %s: %s", path, exc)
    except FileNotFoundError as exc:
        logger.warning("File not found: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error loading %s: %s", path, exc)
    return None


# ---------------------------------------------------------------------------
# Message parsing (shared by both loaders)
# ---------------------------------------------------------------------------


def _parse_message(msg: dict, chat_name: str, chat_type: str, chat_id) -> Optional[dict]:
    """
    Convert a single raw message dict into a flat row dict.

    Returns None for non-message entries, empty texts, or bad dates.
    """
    if msg.get("type") != "message":
        return None

    text = normalize_text(msg.get("text", ""))
    if not text.strip():
        return None

    parsed_date = parse_date(msg.get("date", ""))
    if parsed_date is None:
        return None

    sender = clean_username(msg.get("from") or msg.get("actor") or "Unknown")
    from_id = str(msg.get("from_id") or msg.get("actor_id") or "")

    return {
        "chat_name": str(chat_name),
        "chat_type": chat_type,
        "chat_id": chat_id,
        "message_id": msg.get("id"),
        "sender": sender,
        "sender_id": from_id,
        "text": text,
        "date": parsed_date,
        "year": parsed_date.year,
        "month": parsed_date.month,
        "day": parsed_date.day,
        "hour": parsed_date.hour,
        "weekday": parsed_date.weekday(),
        "weekday_name": parsed_date.strftime("%A"),
        "text_length": len(text),
        "word_count": len(text.split()),
        "is_deleted_account": is_deleted_account(sender, from_id),
    }


def _parse_chat_block(chat: dict) -> list[dict]:
    """Parse all valid messages from one chat dict."""
    chat_name = chat.get("name") or f"Chat {chat.get('id', '?')}"
    chat_type = chat.get("type", "unknown")
    chat_id = chat.get("id")

    rows: list[dict] = []
    for msg in chat.get("messages", []):
        row = _parse_message(msg, chat_name, chat_type, chat_id)
        if row is not None:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Format A: unified result.json  (modern Telegram Desktop)
# ---------------------------------------------------------------------------


def _load_result_json(export_path: str) -> list[dict]:
    """
    Load all messages from the unified ``result.json`` file.

    The file lives at the root of the export directory and contains every
    chat under ``chats.list`` (and optionally ``left_chats.list``).
    """
    result_path = os.path.join(export_path, "result.json")
    if not os.path.exists(result_path):
        return []

    data = _read_json(result_path)
    if data is None or not isinstance(data, dict):
        return []

    all_rows: list[dict] = []

    for section_key in ("chats", "left_chats"):
        section = data.get(section_key)
        if not section:
            continue

        # The list of chats can live directly in the value or under a "list" key
        if isinstance(section, list):
            chat_list = section
        elif isinstance(section, dict):
            chat_list = section.get("list", [])
        else:
            continue

        for chat in chat_list:
            rows = _parse_chat_block(chat)
            all_rows.extend(rows)
            logger.debug(
                "result.json: parsed %d messages from '%s'",
                len(rows),
                chat.get("name", "<unnamed>"),
            )

    logger.info("result.json: total rows parsed = %d", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Format B: per-chat messages.json  (legacy / fallback)
# ---------------------------------------------------------------------------


def _load_per_chat_jsons(export_path: str) -> list[dict]:
    """
    Recursively find and parse individual ``messages.json`` files.

    Used when no ``result.json`` is present at the root.
    """
    json_files = find_json_files(export_path)
    if not json_files:
        return []

    all_rows: list[dict] = []
    for json_file in json_files:
        data = _read_json(json_file)
        if data is None or not isinstance(data, dict):
            continue
        rows = _parse_chat_block(data)
        all_rows.extend(rows)
        logger.info("messages.json: %d messages from %s", len(rows), json_file)

    return all_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_telegram_export(export_path: str) -> pd.DataFrame:
    """
    Load all Telegram chat messages from *export_path* into a DataFrame.

    Automatically detects whether the export uses the unified
    ``result.json`` format or the legacy per-chat ``messages.json`` format.

    Args:
        export_path: Root directory of the Telegram Desktop export.

    Returns:
        A sorted, unified DataFrame with one row per message.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no
            recognisable Telegram export data.
        ValueError: If parsing yields zero valid messages.
    """
    # Strip accidental surrounding quotes (common copy-paste artifact)
    export_path = export_path.strip().strip('"').strip("'")

    if not os.path.isdir(export_path):
        raise FileNotFoundError(
            f"Directory not found: {export_path}\n"
            "Please enter the full path to your Telegram export folder."
        )

    # Try the unified result.json first (modern format)
    all_rows = _load_result_json(export_path)

    # Fall back to per-chat messages.json files (legacy format)
    if not all_rows:
        logger.info("No result.json data; trying per-chat messages.json files.")
        all_rows = _load_per_chat_jsons(export_path)

    if not all_rows:
        raise FileNotFoundError(
            f"No Telegram export data found in: {export_path}\n\n"
            "Expected either:\n"
            "  • A 'result.json' file at the root of the export folder\n"
            "  • Individual 'messages.json' files inside 'chats/' sub-folders\n\n"
            "Make sure you exported in JSON format (not HTML)."
        )

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Loaded %d messages from %d chats.",
        len(df),
        df["chat_name"].nunique(),
    )
    return df


def load_contacts(export_path: str) -> Optional[pd.DataFrame]:
    """
    Load the contacts list from ``result.json`` or ``contacts.json``.

    Args:
        export_path: Path to the Telegram export root directory.

    Returns:
        DataFrame of contacts, or None if unavailable.
    """
    # Modern format: contacts are embedded in result.json
    result_path = os.path.join(export_path, "result.json")
    if os.path.exists(result_path):
        data = _read_json(result_path)
        if isinstance(data, dict):
            contacts = data.get("contacts", {})
            if isinstance(contacts, dict):
                contact_list = contacts.get("list", [])
            elif isinstance(contacts, list):
                contact_list = contacts
            else:
                contact_list = []
            if contact_list:
                return pd.DataFrame(contact_list)

    # Legacy format: standalone contacts.json
    contacts_path = os.path.join(export_path, "contacts.json")
    if os.path.exists(contacts_path):
        data = _read_json(contacts_path)
        if isinstance(data, dict):
            contact_list = data.get("contacts", data.get("list", []))
            if contact_list:
                return pd.DataFrame(contact_list)

    return None


def get_my_name(df: pd.DataFrame) -> str:
    """
    Heuristically detect the user's own Telegram name.

    The account owner is the sender who appears in the most distinct
    personal chats (since in a 1-on-1 chat both parties always appear,
    the account owner shows up across many chats while contacts each
    appear in only one).

    Args:
        df: Messages DataFrame.

    Returns:
        Most likely self-name, or empty string if undetermined.
    """
    personal = df[df["chat_type"].isin({"personal_chat", "bot_chat", "saved_messages"})]
    if personal.empty:
        personal = df

    sender_chat_counts = (
        personal.groupby("sender")["chat_name"]
        .nunique()
        .sort_values(ascending=False)
    )

    if not sender_chat_counts.empty:
        top = str(sender_chat_counts.index[0])
        if top and top not in ("Unknown", "Deleted Account"):
            return top

    return ""
