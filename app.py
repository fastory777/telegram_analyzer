"""
Telegram Data Analyzer — Main Streamlit Application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd
import streamlit as st

# Ensure project root is on the path when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_search import (
    SearchMode,
    SearchOptions,
    MultiInputMode,
    get_context_window,
    get_match_stats,
    highlight_text,
    search,
)
from data_loader import load_telegram_export, get_my_name
from keyword_search import search_keywords, get_keyword_stats, highlight_keywords
from lost_connections import (
    find_deleted_accounts,
    find_lost_connections,
    get_fading_connections,
    get_relationship_timeline,
    detect_communication_phases,
)
from message_analyzer import (
    get_avg_message_length,
    get_conversation_initiators,
    get_longest_streak,
    get_messages_over_time,
    get_monthly_breakdown,
    get_night_activity,
    get_response_time_stats,
    get_summary_stats,
    get_top_contacts,
)
from pattern_analysis import get_top_bigrams, get_top_trigrams
from report_generator import generate_report
from sentiment_analysis import (
    analyze_dataframe_sentiment,
    get_analyzer,
    get_extreme_messages,
    get_sentiment_distribution,
    get_sentiment_per_contact,
    get_sentiment_timeline,
)
from social_graph import (
    build_social_graph,
    detect_communities,
    export_graph_image,
    get_graph_metrics,
    plot_social_graph,
)
from temporal_analysis import (
    detect_bursts,
    get_activity_heatmap,
    get_activity_timeline,
    get_conversation_density,
    get_hourly_distribution,
    get_weekday_distribution,
    get_year_over_year,
)
from tokenizer import get_stopwords, PYMORPHY2_AVAILABLE
from word_statistics import (
    WORDCLOUD_AVAILABLE,
    generate_wordcloud_figure,
    get_repeated_messages,
    get_top_words,
)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Telegram Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .metric-block {
        background: #1e1e2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .metric-block .val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #7c3aed;
    }
    .metric-block .lbl {
        font-size: .78rem;
        color: #94a3b8;
        margin-top: .3rem;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    /* Search result context blocks */
    .ctx-block {
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        overflow: hidden;
        margin: .6rem 0;
    }
    .ctx-row {
        padding: 5px 10px;
        font-size: .88rem;
        border-bottom: 1px solid #1a1a2e;
    }
    .ctx-row:last-child { border-bottom: none; }
    .ctx-match {
        background: rgba(255,215,0,0.10);
        border-left: 3px solid #FFD700;
    }
    .ctx-sender { color: #7c3aed; font-weight: 600; font-size: .78rem; }
    .ctx-time   { color: #64748b; font-size: .73rem; }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 16px;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=3)
def _load_data(export_path: str) -> pd.DataFrame:
    """Cached data loader — invalidates when path changes."""
    return load_telegram_export(export_path)


@st.cache_data(show_spinner=False, max_entries=2)
def _run_sentiment(df_hash: int, df: pd.DataFrame, sample: int) -> pd.DataFrame:
    """Cached sentiment analysis — keyed by DataFrame hash."""
    return analyze_dataframe_sentiment(df, sample_size=sample)


# ---------------------------------------------------------------------------
# Helper: metric card
# ---------------------------------------------------------------------------


def _metric(col, label: str, value: str) -> None:
    with col:
        st.markdown(
            f'<div class="metric-block"><div class="val">{value}</div>'
            f'<div class="lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def build_sidebar() -> dict:
    """Render the sidebar and return a configuration dict."""
    sb = st.sidebar

    sb.title("📊 Telegram Analyzer")
    sb.caption("🔒 All processing happens locally — your data never leaves this machine.")
    sb.markdown("---")

    # ── Data source ──────────────────────────────────────────────────────────
    sb.subheader("Data Source")
    export_path = sb.text_input(
        "Export directory path",
        placeholder="e.g. C:/Users/you/Downloads/Telegram Export",
        help="The folder produced by Telegram Desktop's export function.",
    )
    load_btn = sb.button("⬆ Load Data", type="primary", use_container_width=True)
    sb.markdown("---")

    # ── Identity ─────────────────────────────────────────────────────────────
    sb.subheader("Identity")
    detected_name = st.session_state.get("detected_name", "")
    my_name = sb.text_input(
        "Your Telegram name",
        value=detected_name,
        placeholder="As shown in chat history",
        help="Used to separate your messages from others in graphs and stats.",
    )
    sb.markdown("---")

    # Defaults for optional controls (only rendered when data is loaded)
    date_start: Optional[object] = None
    date_end: Optional[object] = None
    min_messages: int = 5
    graph_min: int = 10
    lc_min: int = 10
    lc_months: int = 6
    run_sentiment: bool = False
    use_ru_stopwords: bool = True
    use_en_stopwords: bool = True
    lemmatize: bool = False
    min_word_length: int = 3
    proximity_distance: int = 5

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        min_dt = df["date"].min().date()
        max_dt = df["date"].max().date()

        # ── Filters ──────────────────────────────────────────────────────────
        sb.subheader("Filters")
        date_start = sb.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt)
        date_end   = sb.date_input("To",   value=max_dt, min_value=min_dt, max_value=max_dt)
        min_messages = sb.slider("Min messages (charts)", 1, 200, 5)
        sb.markdown("---")

        # ── Social Graph ─────────────────────────────────────────────────────
        sb.subheader("Social Graph")
        graph_min = sb.slider("Min shared messages", 1, 100, 10)
        sb.markdown("---")

        # ── Lost Connections ─────────────────────────────────────────────────
        sb.subheader("Lost Connections")
        lc_min    = sb.slider("Min total messages", 5, 200, 10)
        lc_months = sb.slider("Inactive months threshold", 1, 36, 6)
        sb.markdown("---")

        # ── Text Analysis settings ───────────────────────────────────────────
        sb.subheader("Text Analysis")
        use_ru_stopwords = sb.checkbox("Remove Russian stopwords", value=True)
        use_en_stopwords = sb.checkbox("Remove English stopwords", value=True)
        lem_label = "Lemmatize (pymorphy2)"
        if not PYMORPHY2_AVAILABLE:
            lem_label += " — not installed"
        lemmatize = sb.checkbox(
            lem_label,
            value=False,
            disabled=not PYMORPHY2_AVAILABLE,
            help="pip install pymorphy2  to enable Russian lemmatization.",
        )
        min_word_length   = sb.slider("Min word length", 2, 8, 3)
        proximity_distance = sb.slider(
            "Proximity distance (tokens)", 2, 20, 5,
            help="Max token gap in Advanced Search Proximity mode.",
        )
        sb.markdown("---")

        # ── Sentiment ────────────────────────────────────────────────────────
        sb.subheader("Sentiment")
        run_sentiment = sb.checkbox("Enable sentiment analysis", value=False)
        sb.markdown("---")

        if sb.button("📄 Generate HTML Report", use_container_width=True):
            st.session_state["gen_report"] = True

    return {
        "export_path":        export_path,
        "load_btn":           load_btn,
        "my_name":            my_name,
        "date_start":         date_start,
        "date_end":           date_end,
        "min_messages":       min_messages,
        "graph_min":          graph_min,
        "lc_min":             lc_min,
        "lc_months":          lc_months,
        "run_sentiment":      run_sentiment,
        "use_ru_stopwords":   use_ru_stopwords,
        "use_en_stopwords":   use_en_stopwords,
        "lemmatize":          lemmatize,
        "min_word_length":    min_word_length,
        "proximity_distance": proximity_distance,
    }


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------


def tab_overview(df: pd.DataFrame, stats: dict) -> None:
    st.header("Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    _metric(c1, "Messages", f"{stats['total_messages']:,}")
    _metric(c2, "Chats",    f"{stats['total_chats']:,}")
    _metric(c3, "Contacts", f"{stats['total_contacts']:,}")
    _metric(c4, "Avg/Day",  f"{stats['avg_messages_per_day']:,}")
    _metric(c5, "Words",    f"{stats['total_words']:,}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.info(f"**Period:** {stats['date_range_start']} → {stats['date_range_end']}")
    col2.info(f"**Most active chat:** {stats['most_active_chat']}")
    col3.info(f"**Avg message length:** {stats['avg_message_length']} chars")

    _, timeline_fig = get_messages_over_time(df, "W")
    st.plotly_chart(timeline_fig, use_container_width=True, key="overview_timeline")

    col_a, col_b = st.columns(2)
    with col_a:
        _, top_fig = get_top_contacts(df, n=10)
        st.plotly_chart(top_fig, use_container_width=True, key="overview_top_contacts")
    with col_b:
        heatmap = get_activity_heatmap(df)
        st.plotly_chart(heatmap, use_container_width=True, key="overview_heatmap")

    streak = get_longest_streak(df)
    if streak["streak"] > 0:
        st.success(
            f"🔥 **Longest messaging streak:** {streak['streak']} consecutive days"
            f"  ({streak['start']} → {streak['end']})"
        )


# ---------------------------------------------------------------------------
# Tab: Message Statistics
# ---------------------------------------------------------------------------


def tab_message_stats(df: pd.DataFrame, my_name: str, min_messages: int) -> None:
    st.header("Message Statistics")

    inner = st.tabs(
        ["Top Contacts", "Initiators", "Avg Length", "Night Activity", "Response Time"]
    )

    with inner[0]:
        n_show = st.slider("Show top N contacts", 5, 50, 20, key="tc_n")
        contacts_df, contacts_fig = get_top_contacts(df, n=n_show, my_name=my_name)
        st.plotly_chart(contacts_fig, use_container_width=True, key="msg_top_contacts")
        st.dataframe(contacts_df, use_container_width=True, hide_index=True)
        csv = contacts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv,
            file_name="top_contacts.csv",
            mime="text/csv",
        )

    with inner[1]:
        init_df, init_fig = get_conversation_initiators(df)
        if not init_df.empty:
            st.plotly_chart(init_fig, use_container_width=True, key="msg_initiators")
            st.dataframe(init_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to determine conversation initiators.")

    with inner[2]:
        len_df, len_fig = get_avg_message_length(df, min_messages=min_messages)
        st.plotly_chart(len_fig, use_container_width=True, key="msg_avg_length")
        st.dataframe(len_df, use_container_width=True, hide_index=True)

    with inner[3]:
        night_df, night_fig = get_night_activity(df)
        st.plotly_chart(night_fig, use_container_width=True, key="msg_night")
        if not night_df.empty:
            st.dataframe(night_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No night-time messages found.")

    with inner[4]:
        st.caption(
            "Median response time between different senders within a chat "
            "(capped at 24 hours per exchange)."
        )
        resp_df, resp_fig = get_response_time_stats(df)
        if not resp_df.empty:
            st.plotly_chart(resp_fig, use_container_width=True, key="msg_response_time")
            st.dataframe(resp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for response time analysis.")


# ---------------------------------------------------------------------------
# Tab: Advanced Search
# ---------------------------------------------------------------------------


def _render_search_results(
    results: pd.DataFrame,
    queries: list[str],
    mode: SearchMode,
    opts: SearchOptions,
    df_for_context: pd.DataFrame,
) -> None:
    """Render search results with context windows and export buttons."""
    if results.empty:
        st.warning("No matches found. Try a different query or mode.")
        return

    n_matches = len(results)
    st.success(f"Found **{n_matches:,}** matching message{'s' if n_matches != 1 else ''}.")

    # Per-chat stats
    stats_df = get_match_stats(results)
    if not stats_df.empty:
        with st.expander("Matches per chat", expanded=False):
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Export buttons
    col_csv, col_json, _ = st.columns([1, 1, 5])
    export_cols = [c for c in ["date", "chat_name", "sender", "text"] if c in results.columns]
    with col_csv:
        csv_bytes = results[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ CSV",
            data=csv_bytes,
            file_name="search_results.csv",
            mime="text/csv",
            key="adv_dl_csv",
        )
    with col_json:
        json_bytes = (
            results[export_cols]
            .to_json(orient="records", force_ascii=False, indent=2)
            .encode("utf-8")
        )
        st.download_button(
            "⬇ JSON",
            data=json_bytes,
            file_name="search_results.json",
            mime="application/json",
            key="adv_dl_json",
        )

    st.markdown("---")

    # Build context once for all matches
    ctx_df = get_context_window(
        df_for_context,
        results.index,
        before=opts.context_before,
        after=opts.context_after,
    )

    display_cap = 200
    for i, idx in enumerate(results.index[:display_cap]):
        if idx not in df_for_context.index:
            continue
        row = df_for_context.loc[idx]
        header_str = (
            f"[{row['chat_name']}]  "
            f"{pd.Timestamp(row['date']).strftime('%Y-%m-%d %H:%M')}  —  "
            f"{str(row['sender'])[:40]}"
        )
        with st.expander(header_str, expanded=False):
            # Context rows: same chat, within the window
            chat = row["chat_name"]
            window = ctx_df[ctx_df["chat_name"] == chat]
            # Slice around the match position
            all_idx_list = list(df_for_context.index)
            try:
                pos = all_idx_list.index(idx)
            except ValueError:
                pos = 0
            lo_bound = all_idx_list[max(0, pos - opts.context_before)]
            hi_bound = all_idx_list[min(len(all_idx_list) - 1, pos + opts.context_after)]
            window = window[
                (window.index >= lo_bound) & (window.index <= hi_bound)
            ]

            rows_html = ""
            for ctx_idx, ctx_row in window.iterrows():
                is_match = bool(ctx_row.get("is_match", False))
                highlighted = highlight_text(str(ctx_row["text"]), queries, mode, opts)
                extra_cls = "ctx-match" if is_match else ""
                rows_html += (
                    f'<div class="ctx-row {extra_cls}">'
                    f'<span class="ctx-time">'
                    f'{pd.Timestamp(ctx_row["date"]).strftime("%H:%M")}'
                    f'</span> '
                    f'<span class="ctx-sender">{ctx_row["sender"]}</span>&nbsp; '
                    f"{highlighted}</div>"
                )
            st.markdown(f'<div class="ctx-block">{rows_html}</div>', unsafe_allow_html=True)

    if n_matches > display_cap:
        st.info(
            f"Showing first {display_cap} of {n_matches} results. "
            "Narrow your search to see fewer results."
        )


def tab_advanced_search(df: pd.DataFrame, config: dict) -> None:
    st.header("Advanced Search")

    # ── Mode selectors ────────────────────────────────────────────────────────
    col_mode, col_multi = st.columns(2)
    with col_mode:
        mode_label = st.selectbox(
            "Search mode",
            options=[m.value for m in SearchMode],
            key="adv_mode",
        )
        mode = SearchMode(mode_label)
    with col_multi:
        multi_label = st.selectbox(
            "Multi-input combination",
            options=[m.value for m in MultiInputMode],
            key="adv_multi_mode",
        )
        multi_mode = MultiInputMode(multi_label)

    # ── Dynamic query inputs ──────────────────────────────────────────────────
    st.markdown("**Search queries** — each field is one term / pattern / phrase")

    if "adv_num_queries" not in st.session_state:
        st.session_state["adv_num_queries"] = 1
    n_inputs: int = st.session_state["adv_num_queries"]

    btn_col1, lbl_col, btn_col2 = st.columns([1, 3, 1])
    with btn_col1:
        if st.button("− Remove", key="adv_minus", disabled=(n_inputs <= 1)):
            removed_key = f"adv_q_{n_inputs - 1}"
            st.session_state.pop(removed_key, None)
            st.session_state["adv_num_queries"] = max(1, n_inputs - 1)
            st.rerun()
    with lbl_col:
        st.caption(f"{n_inputs} input field{'s' if n_inputs != 1 else ''} active")
    with btn_col2:
        if st.button("+ Add", key="adv_plus", disabled=(n_inputs >= 10)):
            st.session_state["adv_num_queries"] = min(10, n_inputs + 1)
            st.rerun()

    queries: list[str] = []
    placeholders = {
        SearchMode.SUBSTRING:  "word or phrase",
        SearchMode.EXACT_WORD: "exact whole word",
        SearchMode.PHRASE:     "exact phrase in order",
        SearchMode.AND:        "word1 word2  (all must appear)",
        SearchMode.OR:         "word1 word2  (any must appear)",
        SearchMode.REGEX:      "regex pattern",
    }
    for i in range(n_inputs):
        q = st.text_input(
            f"Input {i + 1}",
            key=f"adv_q_{i}",
            placeholder=f"{placeholders.get(mode, 'query')} {i + 1}",
            label_visibility="collapsed",
        )
        queries.append(q)

    # ── Search options ────────────────────────────────────────────────────────
    with st.expander("Search options", expanded=False):
        oc1, oc2, oc3, oc4 = st.columns(4)
        case_sensitive = oc1.checkbox("Case sensitive",    value=False, key="adv_case")
        ignore_punct   = oc2.checkbox("Ignore punctuation", value=False, key="adv_punct")
        ignore_diac    = oc3.checkbox("Ignore diacritics",  value=False, key="adv_diac")
        lem_lbl = "Lemmatize" + ("" if PYMORPHY2_AVAILABLE else " (needs pymorphy2)")
        do_lem = oc4.checkbox(lem_lbl, value=False, key="adv_lem", disabled=not PYMORPHY2_AVAILABLE)

        cc1, cc2, cc3 = st.columns(3)
        ctx_before = cc1.number_input("Context before (msgs)", 0, 20, 5, key="adv_ctx_b")
        ctx_after  = cc2.number_input("Context after (msgs)",  0, 20, 5, key="adv_ctx_a")
        prox_dist  = cc3.number_input(
            "Proximity distance (tokens)", 2, 50,
            config.get("proximity_distance", 5),
            key="adv_prox",
            help="Only used in Proximity mode.",
        )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("Filters (chat / sender / date)", expanded=False):
        all_chats = sorted(df["chat_name"].unique().tolist())
        selected_chats = st.multiselect(
            "Limit to chats",
            options=all_chats,
            default=[],
            key="adv_chat_filter",
            placeholder="All chats",
        )
        sender_filter = st.text_input(
            "Sender contains",
            value="",
            key="adv_sender_filter",
            placeholder="Leave empty for all senders",
        )
        fc1, fc2 = st.columns(2)
        min_dt = df["date"].min().date()
        max_dt = df["date"].max().date()
        adv_ds = fc1.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt, key="adv_ds")
        adv_de = fc2.date_input("To",   value=max_dt, min_value=min_dt, max_value=max_dt, key="adv_de")

    # ── Search button ─────────────────────────────────────────────────────────
    do_search = st.button("🔍 Search", type="primary", key="adv_search_btn")

    if do_search:
        active_queries = [q.strip() for q in queries if q.strip()]
        if not active_queries:
            st.warning("Enter at least one search term.")
            return

        opts = SearchOptions(
            case_sensitive=case_sensitive,
            ignore_punctuation=ignore_punct,
            ignore_diacritics=ignore_diac,
            lemmatize=do_lem,
            multi_mode=multi_mode,
            proximity_distance=int(prox_dist),
            chat_filter=selected_chats if selected_chats else None,
            sender_filter=sender_filter.strip() or None,
            date_start=pd.Timestamp(adv_ds),
            date_end=pd.Timestamp(adv_de) + pd.Timedelta(days=1),
            context_before=int(ctx_before),
            context_after=int(ctx_after),
        )

        with st.spinner(f"Searching {len(df):,} messages…"):
            try:
                results = search(df, active_queries, mode, opts)
            except ValueError as exc:
                st.error(f"Search error: {exc}")
                return

        _render_search_results(results, active_queries, mode, opts, df)

    # ── Mode reference ────────────────────────────────────────────────────────
    with st.expander("Mode reference", expanded=False):
        st.markdown(
            """
| Mode | Behaviour |
|---|---|
| **Substring** | Message contains the query anywhere (classic search) |
| **Exact Word** | Whole-word match — `илья` will NOT match `василья` or `ильяс` |
| **Phrase** | Exact phrase in correct word order |
| **AND (all words)** | All space-separated terms must appear (anywhere in the message) |
| **OR (any word)** | At least one space-separated term must appear |
| **Regex** | Raw Python regex; use `(?<!\\w)…(?!\\w)` for word boundaries |

**Multi-input combination**

| Combination | Behaviour |
|---|---|
| **AND** | All input fields must match the same message |
| **OR** | Any input field may match |
| **Proximity** | All terms must appear within N tokens of each other |
"""
        )


# ---------------------------------------------------------------------------
# Tab: Word Statistics
# ---------------------------------------------------------------------------


def tab_word_statistics(df: pd.DataFrame, config: dict) -> None:
    st.header("Word Statistics")

    stopwords    = get_stopwords(
        include_russian=config["use_ru_stopwords"],
        include_english=config["use_en_stopwords"],
    )
    lemmatize    = config["lemmatize"]
    min_word_len = config["min_word_length"]

    all_contacts = sorted(df["sender"].unique().tolist())
    all_chats    = sorted(df["chat_name"].unique().tolist())

    sub = st.tabs(["Top Words", "Per Contact", "Word Cloud", "Repeated Messages"])

    # ── Top Words ─────────────────────────────────────────────────────────────
    with sub[0]:
        tw1, tw2, tw3 = st.columns([2, 2, 1])
        with tw1:
            chat_sel = st.selectbox(
                "Filter by chat",
                options=["(all chats)"] + all_chats,
                key="ws_chat",
            )
            chat_filter = None if chat_sel == "(all chats)" else chat_sel
        with tw2:
            sender_sel = st.selectbox(
                "Filter by sender",
                options=["(all senders)"] + all_contacts,
                key="ws_sender",
            )
            sender_filter = None if sender_sel == "(all senders)" else sender_sel
        with tw3:
            n_words = st.slider("Top N", 10, 100, 50, key="ws_n")

        with st.spinner("Computing word frequencies…"):
            words_df, words_fig = get_top_words(
                df,
                n=n_words,
                stopwords=stopwords,
                min_length=min_word_len,
                lemmatize=lemmatize,
                chat_filter=chat_filter,
                sender_filter=sender_filter,
            )

        st.plotly_chart(words_fig, use_container_width=True, key="ws_top_words")
        if not words_df.empty:
            st.dataframe(words_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                data=words_df.to_csv(index=False).encode("utf-8"),
                file_name="top_words.csv",
                mime="text/csv",
                key="ws_csv",
            )

    # ── Per Contact ───────────────────────────────────────────────────────────
    with sub[1]:
        contact = st.selectbox("Select sender", options=all_contacts, key="ws_contact")
        n_contact_words = st.slider("Top N words", 10, 100, 30, key="ws_contact_n")

        with st.spinner(f"Computing frequencies for {contact}…"):
            c_df, c_fig = get_top_words(
                df,
                n=n_contact_words,
                stopwords=stopwords,
                min_length=min_word_len,
                lemmatize=lemmatize,
                sender_filter=contact,
            )

        st.plotly_chart(c_fig, use_container_width=True, key="ws_contact_words")
        if not c_df.empty:
            st.dataframe(c_df, use_container_width=True, hide_index=True)

    # ── Word Cloud ────────────────────────────────────────────────────────────
    with sub[2]:
        if not WORDCLOUD_AVAILABLE:
            st.warning(
                "The **wordcloud** library is not installed.\n\n"
                "```\npip install wordcloud\n```"
            )
        else:
            wc_max = st.slider("Max words in cloud", 50, 500, 200, key="ws_wc_max")
            if st.button("Generate Word Cloud", key="ws_gen_wc"):
                with st.spinner("Generating word cloud…"):
                    wc_fig = generate_wordcloud_figure(
                        df,
                        stopwords=stopwords,
                        min_length=min_word_len,
                        lemmatize=lemmatize,
                        max_words=wc_max,
                    )
                if wc_fig is not None:
                    st.plotly_chart(wc_fig, use_container_width=True, key="ws_wordcloud")
                else:
                    st.info("Not enough words to generate a word cloud.")

    # ── Repeated Messages ─────────────────────────────────────────────────────
    with sub[3]:
        st.caption("Messages that appear verbatim multiple times (case-insensitive).")
        rm1, rm2 = st.columns(2)
        rm_min    = rm1.slider("Min repetitions", 2, 20, 3, key="ws_rm_min")
        rm_maxlen = rm2.slider("Max message length (chars)", 20, 500, 200, key="ws_rm_len")

        with st.spinner("Scanning for repeated messages…"):
            repeated_df = get_repeated_messages(df, min_count=rm_min, max_length=rm_maxlen)

        if repeated_df.empty:
            st.info(f"No messages repeated ≥{rm_min} times.")
        else:
            st.metric("Repeated message patterns found", len(repeated_df))
            st.dataframe(repeated_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                data=repeated_df.to_csv(index=False).encode("utf-8"),
                file_name="repeated_messages.csv",
                mime="text/csv",
                key="ws_rm_csv",
            )


# ---------------------------------------------------------------------------
# Tab: Message Patterns
# ---------------------------------------------------------------------------


def tab_message_patterns(df: pd.DataFrame, config: dict) -> None:
    st.header("Message Patterns")

    stopwords    = get_stopwords(
        include_russian=config["use_ru_stopwords"],
        include_english=config["use_en_stopwords"],
    )
    lemmatize    = config["lemmatize"]
    min_word_len = config["min_word_length"]

    sub = st.tabs(["Bigrams", "Trigrams"])

    with sub[0]:
        bc1, bc2 = st.columns([1, 3])
        bg_k = bc1.slider("Top K bigrams", 10, 50, 30, key="mp_bg_k")
        bc2.caption("Two-word combinations most frequently appearing together.")

        with st.spinner("Computing bigrams…"):
            bg_df, bg_fig = get_top_bigrams(
                df,
                top_k=bg_k,
                stopwords=stopwords,
                min_length=min_word_len,
                lemmatize=lemmatize,
            )

        st.plotly_chart(bg_fig, use_container_width=True, key="mp_bigrams")
        if not bg_df.empty:
            st.dataframe(bg_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                data=bg_df.to_csv(index=False).encode("utf-8"),
                file_name="bigrams.csv",
                mime="text/csv",
                key="mp_bg_csv",
            )

    with sub[1]:
        tc1, tc2 = st.columns([1, 3])
        tg_k = tc1.slider("Top K trigrams", 10, 50, 30, key="mp_tg_k")
        tc2.caption("Three-word combinations most frequently appearing together.")

        with st.spinner("Computing trigrams…"):
            tg_df, tg_fig = get_top_trigrams(
                df,
                top_k=tg_k,
                stopwords=stopwords,
                min_length=min_word_len,
                lemmatize=lemmatize,
            )

        st.plotly_chart(tg_fig, use_container_width=True, key="mp_trigrams")
        if not tg_df.empty:
            st.dataframe(tg_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download CSV",
                data=tg_df.to_csv(index=False).encode("utf-8"),
                file_name="trigrams.csv",
                mime="text/csv",
                key="mp_tg_csv",
            )


# ---------------------------------------------------------------------------
# Tab: Social Graph
# ---------------------------------------------------------------------------


def tab_social_graph(
    df: pd.DataFrame,
    my_name: str,
    graph_min: int,
    date_start,
    date_end,
) -> None:
    st.header("Social Graph")

    col_info, col_opt = st.columns([3, 1])
    with col_info:
        st.caption(
            "Nodes = people · Edge thickness ∝ message count · "
            "Node size ∝ total messages · Colour = number of connections"
        )
    with col_opt:
        show_metrics = st.checkbox("Show metrics table", value=True)

    if not my_name:
        st.warning("Set your name in the sidebar for a more accurate graph.")

    with st.spinner("Building social graph…"):
        G = build_social_graph(
            df,
            my_name=my_name,
            min_messages=graph_min,
            date_start=pd.Timestamp(date_start) if date_start else None,
            date_end=pd.Timestamp(date_end) if date_end else None,
        )
        graph_fig = plot_social_graph(G, my_name=my_name)

    st.plotly_chart(graph_fig, use_container_width=True, key="social_graph")

    m1, m2, m3 = st.columns(3)
    m1.metric("Nodes (people)", G.number_of_nodes())
    m2.metric("Edges (connections)", G.number_of_edges())
    communities = detect_communities(G)
    m3.metric("Communities detected", len(set(communities.values())))

    if show_metrics and G.number_of_nodes() > 0:
        st.subheader("Centrality Metrics")
        metrics_df = get_graph_metrics(G)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("📥 Export graph as PNG"):
            png_path = "social_graph.png"
            export_graph_image(G, png_path)
            if os.path.exists(png_path):
                with open(png_path, "rb") as fh:
                    st.download_button(
                        "Download social_graph.png",
                        data=fh.read(),
                        file_name="social_graph.png",
                        mime="image/png",
                    )

    with col_exp2:
        if G.number_of_nodes() > 0:
            metrics_df2 = get_graph_metrics(G)
            csv = metrics_df2.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export metrics CSV",
                data=csv,
                file_name="graph_metrics.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Tab: Lost Connections
# ---------------------------------------------------------------------------


def tab_lost_connections(df: pd.DataFrame, lc_min: int, lc_months: int) -> None:
    st.header("Lost Connections")

    with st.spinner("Analysing connection history…"):
        lost_df, lost_fig = find_lost_connections(df, lc_min, lc_months)
        deleted_df = find_deleted_accounts(df)
        fading_df  = get_fading_connections(df)

    sub = st.tabs(["Lost Connections", "Deleted Accounts", "Fading Connections"])

    with sub[0]:
        st.metric("Lost connections found", len(lost_df))
        st.plotly_chart(lost_fig, use_container_width=True, key="lost_scatter")

        if not lost_df.empty:
            display_cols = [
                c for c in [
                    "chat_name", "total_messages", "last_message",
                    "months_since_last", "relationship_duration_days",
                ]
                if c in lost_df.columns
            ]
            st.dataframe(lost_df[display_cols], use_container_width=True, hide_index=True)

            st.subheader("Contact Timeline")
            selected = st.selectbox(
                "Select contact to view history",
                lost_df["chat_name"].tolist(),
                key="lc_select",
            )
            if selected:
                tl_fig = get_relationship_timeline(df, selected)
                st.plotly_chart(tl_fig, use_container_width=True, key="lost_timeline")

                phase_df, phase_fig = detect_communication_phases(df, selected)
                if not phase_df.empty:
                    with st.expander("Communication phases"):
                        st.plotly_chart(phase_fig, use_container_width=True, key="lost_phases")
                        st.dataframe(phase_df, use_container_width=True, hide_index=True)

    with sub[1]:
        if deleted_df.empty:
            st.info("No deleted accounts detected in your export.")
        else:
            st.metric("Deleted accounts found", len(deleted_df))
            st.dataframe(deleted_df, use_container_width=True, hide_index=True)

    with sub[2]:
        st.caption("Contacts whose activity dropped by ≥50% in the most recent 3-month period.")
        if fading_df.empty:
            st.info("No significantly fading connections detected.")
        else:
            st.metric("Fading connections", len(fading_df))
            st.dataframe(fading_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab: Temporal Analysis
# ---------------------------------------------------------------------------


def tab_temporal(df: pd.DataFrame) -> None:
    st.header("Temporal Analysis")

    sub = st.tabs(["Heatmap", "Timeline", "Distributions", "Burst Detection", "Year-over-Year"])

    with sub[0]:
        st.plotly_chart(get_activity_heatmap(df),     use_container_width=True, key="temp_heatmap")
        st.plotly_chart(get_conversation_density(df), use_container_width=True, key="temp_density")

    with sub[1]:
        st.plotly_chart(get_activity_timeline(df),    use_container_width=True, key="temp_timeline")

    with sub[2]:
        col_wd, col_hr = st.columns(2)
        with col_wd:
            st.plotly_chart(get_weekday_distribution(df), use_container_width=True, key="temp_weekday")
        with col_hr:
            st.plotly_chart(get_hourly_distribution(df),  use_container_width=True, key="temp_hourly")

    with sub[3]:
        burst_thresh = st.slider(
            "Burst threshold multiplier (× rolling std-dev)",
            1.0, 5.0, 2.5, 0.25,
            key="burst_thresh",
        )
        burst_df, burst_fig = detect_bursts(df, threshold_multiplier=burst_thresh)
        st.plotly_chart(burst_fig, use_container_width=True, key="temp_bursts")
        if not burst_df.empty:
            st.caption(f"{len(burst_df)} burst day(s) detected:")
            st.dataframe(burst_df, use_container_width=True, hide_index=True)

    with sub[4]:
        st.plotly_chart(get_year_over_year(df), use_container_width=True, key="temp_yoy")


# ---------------------------------------------------------------------------
# Tab: Sentiment Analysis
# ---------------------------------------------------------------------------


def tab_sentiment(df: pd.DataFrame) -> None:
    st.header("Sentiment Analysis")

    analyzer_name = get_analyzer()
    if not analyzer_name:
        st.error(
            "No sentiment library found. Install one:\n"
            "```\npip install vaderSentiment\n```"
        )
        return

    st.info(
        f"Using **{analyzer_name.upper()}**. "
        "Up to 10,000 messages are scored for performance; "
        "the rest are treated as neutral."
    )

    if "sentiment_df" not in st.session_state:
        if st.button("▶ Run Sentiment Analysis", type="primary"):
            with st.spinner("Analysing sentiment…"):
                df_hash = hash(tuple(df["message_id"].iloc[:100].tolist()))
                st.session_state["sentiment_df"] = _run_sentiment(df_hash, df, 10_000)
            st.rerun()
        return

    sdf: pd.DataFrame = st.session_state["sentiment_df"]

    if "sentiment_label" in sdf.columns:
        vc = sdf["sentiment_label"].value_counts()
        total_scored = int(sdf["sentiment_compound"].ne(0).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored messages", f"{total_scored:,}")
        c2.metric(
            "😊 Positive",
            f"{vc.get('Positive', 0):,}",
            f"{vc.get('Positive', 0) / max(total_scored, 1) * 100:.1f}%",
        )
        c3.metric("😐 Neutral", f"{vc.get('Neutral', 0):,}")
        c4.metric(
            "😞 Negative",
            f"{vc.get('Negative', 0):,}",
            f"-{vc.get('Negative', 0) / max(total_scored, 1) * 100:.1f}%",
            delta_color="inverse",
        )

    sub = st.tabs(["Per Contact", "Timeline", "Distribution", "Extreme Messages"])

    with sub[0]:
        per_df, per_fig = get_sentiment_per_contact(sdf)
        st.plotly_chart(per_fig, use_container_width=True, key="sent_per_contact")
        if not per_df.empty:
            st.dataframe(per_df, use_container_width=True, hide_index=True)

    with sub[1]:
        tl_fig = get_sentiment_timeline(sdf)
        st.plotly_chart(tl_fig, use_container_width=True, key="sent_timeline")

    with sub[2]:
        dist_fig = get_sentiment_distribution(sdf)
        st.plotly_chart(dist_fig, use_container_width=True, key="sent_distribution")

    with sub[3]:
        pos_df, neg_df = get_extreme_messages(sdf, n=10)
        col_p, col_n = st.columns(2)
        with col_p:
            st.markdown("**Most Positive Messages**")
            if not pos_df.empty:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
        with col_n:
            st.markdown("**Most Negative Messages**")
            if not neg_df.empty:
                st.dataframe(neg_df, use_container_width=True, hide_index=True)

    if st.button("🗑 Clear sentiment data"):
        del st.session_state["sentiment_df"]
        st.rerun()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_and_offer_report(df: pd.DataFrame, stats: dict, my_name: str) -> None:
    """Build all figures, call report_generator, offer download."""
    with st.spinner("Generating HTML report…"):
        try:
            figures = {
                "timeline":         get_messages_over_time(df, "W")[1],
                "top_contacts":     get_top_contacts(df, n=20)[1],
                "heatmap":          get_activity_heatmap(df),
                "temporal":         get_activity_timeline(df),
                "weekday":          get_weekday_distribution(df),
                "hourly":           get_hourly_distribution(df),
                "monthly":          get_monthly_breakdown(df)[1],
                "initiators":       get_conversation_initiators(df)[1],
                "avg_length":       get_avg_message_length(df)[1],
                "bursts":           detect_bursts(df)[1],
                "lost_connections": find_lost_connections(df)[1],
                "social_graph":     plot_social_graph(
                    build_social_graph(df, my_name=my_name), my_name=my_name
                ),
            }
            tables = {
                "top_contacts":     get_top_contacts(df, n=20)[0],
                "lost_connections": find_lost_connections(df)[0],
                "deleted_accounts": find_deleted_accounts(df),
                "graph_metrics":    get_graph_metrics(
                    build_social_graph(df, my_name=my_name)
                ),
            }
            if "sentiment_df" in st.session_state:
                sdf = st.session_state["sentiment_df"]
                figures["sentiment_per_contact"]  = get_sentiment_per_contact(sdf)[1]
                figures["sentiment_timeline"]     = get_sentiment_timeline(sdf)
                figures["sentiment_distribution"] = get_sentiment_distribution(sdf)

            report_path = generate_report(df, stats, figures, tables, my_name=my_name)

            with open(report_path, "rb") as fh:
                st.sidebar.download_button(
                    "📥 Download Report HTML",
                    data=fh.read(),
                    file_name="telegram_analysis_report.html",
                    mime="text/html",
                )
            st.sidebar.success("Report ready — click to download!")

        except Exception as exc:
            st.sidebar.error(f"Report generation failed: {exc}")


# ---------------------------------------------------------------------------
# Welcome / landing page
# ---------------------------------------------------------------------------


def show_welcome() -> None:
    st.title("📊 Telegram Data Analyzer")
    st.markdown(
        """
Welcome! This tool analyses your **Telegram Desktop** export data — **100% locally**.

---

### Privacy Guarantee

- ✅ Everything runs on your computer
- ✅ No internet connection required after installation
- ✅ No data is sent anywhere

---

### Quick Start

1. **Export your Telegram data**
   - Open **Telegram Desktop**
   - Go to **Settings → Advanced → Export Telegram Data**
   - Select **JSON** format
   - Wait for the export to finish

2. **Enter the export folder path** in the sidebar

3. **Click "Load Data"**

---

### Features

| Tab | What you get |
|-----|-------------|
| **Overview** | Key metrics, timeline, top contacts, activity heatmap |
| **Message Stats** | Initiators, avg length, night activity, response times |
| **Advanced Search** | 6 modes · multi-input AND/OR/Proximity · CSV+JSON export |
| **Social Graph** | Interactive network, centrality metrics, community detection |
| **Lost Connections** | Inactive contacts, deleted accounts, fading connections |
| **Temporal** | Heatmap, timeline, burst detection, year-over-year |
| **Sentiment** | Per-contact scores, timeline, positive/negative distribution |
| **Word Statistics** | Top words, per-contact words, word cloud, repeated messages |
| **Message Patterns** | Most common bigrams and trigrams |

---
"""
    )
    st.info("Enter your export directory path in the sidebar to begin.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = build_sidebar()

    # ── Load data on button press ────────────────────────────────────────────
    if config["load_btn"]:
        path = config["export_path"].strip().strip('"').strip("'")
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not path:
            st.sidebar.error("Please enter an export directory path.")
        elif not os.path.isdir(path):
            st.sidebar.error(
                f"Directory not found: {path}\n\n"
                "Paste the **folder** path, not the path to result.json inside it."
            )
        else:
            with st.spinner("Loading Telegram export…"):
                try:
                    df = _load_data(path)
                    st.session_state["df"]            = df
                    st.session_state["export_path"]   = path
                    st.session_state["detected_name"] = get_my_name(df)
                    st.session_state.pop("sentiment_df", None)
                    st.sidebar.success(
                        f"Loaded **{len(df):,}** messages "
                        f"from **{df['chat_name'].nunique()}** chats!"
                    )
                    st.rerun()
                except FileNotFoundError as exc:
                    st.sidebar.error(str(exc))
                except ValueError as exc:
                    st.sidebar.error(str(exc))
                except Exception as exc:
                    st.sidebar.error(f"Unexpected error: {exc}")
        return

    # ── No data yet → show landing page ─────────────────────────────────────
    if "df" not in st.session_state or st.session_state["df"] is None:
        show_welcome()
        return

    df_full: pd.DataFrame = st.session_state["df"]
    my_name: str = config["my_name"]

    # ── Apply date filter ────────────────────────────────────────────────────
    date_start = config["date_start"]
    date_end   = config["date_end"]

    if date_start and date_end:
        df = df_full[
            (df_full["date"].dt.date >= date_start)
            & (df_full["date"].dt.date <= date_end)
        ].copy()
    else:
        df = df_full.copy()

    if df.empty:
        st.error("No messages match the current date filter. Adjust the date range.")
        return

    # ── Summary stats ────────────────────────────────────────────────────────
    stats = get_summary_stats(df)

    # ── Report generation ────────────────────────────────────────────────────
    if st.session_state.pop("gen_report", False):
        _generate_and_offer_report(df, stats, my_name)

    # ── Main tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "📈 Overview",
            "💬 Message Stats",
            "🔍 Advanced Search",
            "🕸 Social Graph",
            "👻 Lost Connections",
            "⏱ Temporal",
            "😊 Sentiment",
            "📊 Word Statistics",
            "🔤 Message Patterns",
        ]
    )

    with tabs[0]:
        tab_overview(df, stats)

    with tabs[1]:
        tab_message_stats(df, my_name, config["min_messages"])

    with tabs[2]:
        tab_advanced_search(df, config)

    with tabs[3]:
        tab_social_graph(df, my_name, config["graph_min"], date_start, date_end)

    with tabs[4]:
        tab_lost_connections(df, config["lc_min"], config["lc_months"])

    with tabs[5]:
        tab_temporal(df)

    with tabs[6]:
        if config["run_sentiment"]:
            tab_sentiment(df)
        else:
            st.info(
                "Enable **Sentiment Analysis** in the sidebar to activate this tab.\n\n"
                "*(Requires vaderSentiment or textblob to be installed.)*"
            )

    with tabs[7]:
        tab_word_statistics(df, config)

    with tabs[8]:
        tab_message_patterns(df, config)


if __name__ == "__main__":
    main()
