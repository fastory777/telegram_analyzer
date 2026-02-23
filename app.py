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
    /* Dark surface cards */
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
    div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 10px; }
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

    # ── Date filters (only shown after data loads) ───────────────────────────
    date_start: Optional[object] = None
    date_end: Optional[object] = None
    min_messages: int = 5
    keywords_input: str = ""
    graph_min: int = 10
    lc_min: int = 10
    lc_months: int = 6
    run_sentiment: bool = False

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        min_dt = df["date"].min().date()
        max_dt = df["date"].max().date()

        sb.subheader("Filters")
        date_start = sb.date_input("From", value=min_dt, min_value=min_dt, max_value=max_dt)
        date_end = sb.date_input("To", value=max_dt, min_value=min_dt, max_value=max_dt)
        min_messages = sb.slider("Min messages (charts)", 1, 200, 5)
        sb.markdown("---")

        sb.subheader("Keyword Search")
        keywords_input = sb.text_input(
            "Keywords",
            placeholder="word1, word2, phrase",
            help="Comma-separated. Leave empty to skip.",
        )
        sb.markdown("---")

        sb.subheader("Social Graph")
        graph_min = sb.slider("Min shared messages", 1, 100, 10)
        sb.markdown("---")

        sb.subheader("Lost Connections")
        lc_min = sb.slider("Min total messages", 5, 200, 10)
        lc_months = sb.slider("Inactive months threshold", 1, 36, 6)
        sb.markdown("---")

        sb.subheader("Sentiment")
        run_sentiment = sb.checkbox("Enable sentiment analysis", value=False)
        sb.markdown("---")

        if sb.button("📄 Generate HTML Report", use_container_width=True):
            st.session_state["gen_report"] = True

    return {
        "export_path": export_path,
        "load_btn": load_btn,
        "my_name": my_name,
        "date_start": date_start,
        "date_end": date_end,
        "min_messages": min_messages,
        "keywords_input": keywords_input,
        "graph_min": graph_min,
        "lc_min": lc_min,
        "lc_months": lc_months,
        "run_sentiment": run_sentiment,
    }


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def tab_overview(df: pd.DataFrame, stats: dict) -> None:
    st.header("Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    _metric(c1, "Messages", f"{stats['total_messages']:,}")
    _metric(c2, "Chats", f"{stats['total_chats']:,}")
    _metric(c3, "Contacts", f"{stats['total_contacts']:,}")
    _metric(c4, "Avg/Day", f"{stats['avg_messages_per_day']:,}")
    _metric(c5, "Words", f"{stats['total_words']:,}")

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


def tab_message_stats(df: pd.DataFrame, my_name: str, min_messages: int) -> None:
    st.header("Message Statistics")

    inner = st.tabs(
        ["Top Contacts", "Initiators", "Avg Length", "Night Activity", "Response Time"]
    )

    # ── Top contacts ─────────────────────────────────────────────────────────
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

    # ── Conversation initiators ───────────────────────────────────────────────
    with inner[1]:
        init_df, init_fig = get_conversation_initiators(df)
        if not init_df.empty:
            st.plotly_chart(init_fig, use_container_width=True, key="msg_initiators")
            st.dataframe(init_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to determine conversation initiators.")

    # ── Average message length ────────────────────────────────────────────────
    with inner[2]:
        len_df, len_fig = get_avg_message_length(df, min_messages=min_messages)
        st.plotly_chart(len_fig, use_container_width=True, key="msg_avg_length")
        st.dataframe(len_df, use_container_width=True, hide_index=True)

    # ── Night activity ────────────────────────────────────────────────────────
    with inner[3]:
        night_df, night_fig = get_night_activity(df)
        st.plotly_chart(night_fig, use_container_width=True, key="msg_night")
        if not night_df.empty:
            st.dataframe(night_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No night-time messages found.")

    # ── Response time ─────────────────────────────────────────────────────────
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


def tab_keyword_search(df: pd.DataFrame, keywords_input: str) -> None:
    st.header("Keyword Search")

    col_kw, col_ctx, col_cs = st.columns([4, 1, 1])
    with col_kw:
        kw_str = st.text_input(
            "Keywords (comma-separated)",
            value=keywords_input,
            placeholder="birthday, trip, project",
            label_visibility="collapsed",
        )
    with col_ctx:
        ctx_size = st.number_input("Context", 1, 20, 5, label_visibility="visible")
    with col_cs:
        case_sensitive = st.checkbox("Case sensitive")

    do_search = st.button("🔍 Search", type="primary")

    if do_search and kw_str.strip():
        keywords = [k.strip() for k in kw_str.split(",") if k.strip()]

        with st.spinner(f"Searching {len(df):,} messages…"):
            results = search_keywords(
                df, keywords, context_size=ctx_size, case_sensitive=case_sensitive
            )
            stats_df = get_keyword_stats(df, keywords, case_sensitive=case_sensitive)

        if not results:
            st.warning("No matches found. Try different keywords or disable case-sensitive mode.")
            return

        st.success(f"Found **{len(results)}** match{'es' if len(results) != 1 else ''}.")

        if not stats_df.empty:
            with st.expander("Occurrences per chat", expanded=False):
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        display_cap = 150

        for result in results[:display_cap]:
            header_str = (
                f"[{result['chat_name']}]  "
                f"{result['match_date'].strftime('%Y-%m-%d %H:%M')}  —  "
                f"{result['match_sender'][:40]}"
            )
            with st.expander(header_str, expanded=False):
                rows_html = ""
                for ctx_msg in result["context"]:
                    is_match = ctx_msg["is_match"]
                    highlighted = highlight_keywords(ctx_msg["text"], keywords, case_sensitive)
                    extra_cls = "ctx-match" if is_match else ""
                    rows_html += (
                        f'<div class="ctx-row {extra_cls}">'
                        f'<span class="ctx-time">{ctx_msg["date"].strftime("%H:%M")}</span> '
                        f'<span class="ctx-sender">{ctx_msg["sender"]}</span>&nbsp; '
                        f"{highlighted}</div>"
                    )
                st.markdown(
                    f'<div class="ctx-block">{rows_html}</div>',
                    unsafe_allow_html=True,
                )

        if len(results) > display_cap:
            st.info(
                f"Showing first {display_cap} of {len(results)} results. "
                "Narrow your search to see fewer results."
            )

    elif do_search:
        st.warning("Please enter at least one keyword.")


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


def tab_lost_connections(df: pd.DataFrame, lc_min: int, lc_months: int) -> None:
    st.header("Lost Connections")

    with st.spinner("Analysing connection history…"):
        lost_df, lost_fig = find_lost_connections(df, lc_min, lc_months)
        deleted_df = find_deleted_accounts(df)
        fading_df = get_fading_connections(df)

    sub = st.tabs(["Lost Connections", "Deleted Accounts", "Fading Connections"])

    # ── Lost ─────────────────────────────────────────────────────────────────
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

    # ── Deleted accounts ─────────────────────────────────────────────────────
    with sub[1]:
        if deleted_df.empty:
            st.info("No deleted accounts detected in your export.")
        else:
            st.metric("Deleted accounts found", len(deleted_df))
            st.dataframe(deleted_df, use_container_width=True, hide_index=True)

    # ── Fading connections ───────────────────────────────────────────────────
    with sub[2]:
        st.caption("Contacts whose activity dropped by ≥50% in the most recent 3-month period.")
        if fading_df.empty:
            st.info("No significantly fading connections detected.")
        else:
            st.metric("Fading connections", len(fading_df))
            st.dataframe(fading_df, use_container_width=True, hide_index=True)


def tab_temporal(df: pd.DataFrame) -> None:
    st.header("Temporal Analysis")

    sub = st.tabs(["Heatmap", "Timeline", "Distributions", "Burst Detection", "Year-over-Year"])

    with sub[0]:
        st.plotly_chart(get_activity_heatmap(df), use_container_width=True, key="temp_heatmap")
        st.plotly_chart(get_conversation_density(df), use_container_width=True, key="temp_density")

    with sub[1]:
        st.plotly_chart(get_activity_timeline(df), use_container_width=True, key="temp_timeline")

    with sub[2]:
        col_wd, col_hr = st.columns(2)
        with col_wd:
            st.plotly_chart(get_weekday_distribution(df), use_container_width=True, key="temp_weekday")
        with col_hr:
            st.plotly_chart(get_hourly_distribution(df), use_container_width=True, key="temp_hourly")

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

    # Summary metrics
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
        c3.metric(
            "😐 Neutral",
            f"{vc.get('Neutral', 0):,}",
        )
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
                "timeline": get_messages_over_time(df, "W")[1],
                "top_contacts": get_top_contacts(df, n=20)[1],
                "heatmap": get_activity_heatmap(df),
                "temporal": get_activity_timeline(df),
                "weekday": get_weekday_distribution(df),
                "hourly": get_hourly_distribution(df),
                "monthly": get_monthly_breakdown(df)[1],
                "initiators": get_conversation_initiators(df)[1],
                "avg_length": get_avg_message_length(df)[1],
                "bursts": detect_bursts(df)[1],
                "lost_connections": find_lost_connections(df)[1],
                "social_graph": plot_social_graph(
                    build_social_graph(df, my_name=my_name), my_name=my_name
                ),
            }

            tables = {
                "top_contacts": get_top_contacts(df, n=20)[0],
                "lost_connections": find_lost_connections(df)[0],
                "deleted_accounts": find_deleted_accounts(df),
                "graph_metrics": get_graph_metrics(
                    build_social_graph(df, my_name=my_name)
                ),
            }

            # Add sentiment if already computed
            if "sentiment_df" in st.session_state:
                sdf = st.session_state["sentiment_df"]
                figures["sentiment_per_contact"] = get_sentiment_per_contact(sdf)[1]
                figures["sentiment_timeline"] = get_sentiment_timeline(sdf)
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
| **Keyword Search** | Full-text search across all chats with context |
| **Social Graph** | Interactive network, centrality metrics, community detection |
| **Lost Connections** | Inactive contacts, deleted accounts, fading connections |
| **Temporal** | Heatmap, timeline, burst detection, year-over-year |
| **Sentiment** | Per-contact scores, timeline, positive/negative distribution |

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
        if not path:
            st.sidebar.error("Please enter an export directory path.")
        elif not os.path.isdir(path):
            st.sidebar.error(f"Directory not found:\n{path}")
        else:
            with st.spinner("Loading Telegram export…"):
                try:
                    df = _load_data(path)
                    st.session_state["df"] = df
                    st.session_state["export_path"] = path
                    # Auto-detect user name
                    st.session_state["detected_name"] = get_my_name(df)
                    # Clear stale sentiment cache
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
    date_end = config["date_end"]

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

    # ── Summary stats (computed once) ────────────────────────────────────────
    stats = get_summary_stats(df)

    # ── Report generation (triggered by sidebar button) ───────────────────────
    if st.session_state.pop("gen_report", False):
        _generate_and_offer_report(df, stats, my_name)

    # ── Main tabs ────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            "📈 Overview",
            "💬 Message Stats",
            "🔍 Keyword Search",
            "🕸 Social Graph",
            "👻 Lost Connections",
            "⏱ Temporal",
            "😊 Sentiment",
        ]
    )

    with tabs[0]:
        tab_overview(df, stats)

    with tabs[1]:
        tab_message_stats(df, my_name, config["min_messages"])

    with tabs[2]:
        tab_keyword_search(df, config["keywords_input"])

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


if __name__ == "__main__":
    main()
