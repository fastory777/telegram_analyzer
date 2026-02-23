"""HTML report generator using Jinja2."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fig_to_html(fig: go.Figure, div_id: Optional[str] = None) -> str:
    """
    Convert a Plotly Figure to an embeddable HTML div string.

    The Plotly JS library is NOT included (it is included once in the
    template head via CDN).

    Args:
        fig: Plotly Figure object.
        div_id: Optional HTML element ID for the div.

    Returns:
        HTML string of the rendered chart div.
    """
    try:
        return pio.to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={"responsive": True},
        )
    except Exception as exc:
        logger.warning(f"Failed to convert figure to HTML: {exc}")
        return "<p style='color:gray;'>Chart could not be rendered.</p>"


def df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """
    Convert a DataFrame to a styled HTML table string.

    Args:
        df: DataFrame to convert.
        max_rows: Maximum number of rows to include.

    Returns:
        HTML string of the table.
    """
    if df is None or df.empty:
        return "<p style='color:#888;'>No data available.</p>"

    return df.head(max_rows).to_html(
        classes="report-table",
        index=False,
        border=0,
        na_rep="—",
        escape=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    stats: dict,
    figures: dict[str, go.Figure],
    tables: dict[str, pd.DataFrame],
    output_path: str = "telegram_analysis_report.html",
    my_name: str = "",
) -> str:
    """
    Generate a self-contained HTML analysis report.

    The report embeds all charts as interactive Plotly divs and includes
    a summary statistics dashboard, rendered via a Jinja2 template.

    Args:
        df: Full messages DataFrame (used for extra inline stats).
        stats: Summary statistics dict from `get_summary_stats()`.
        figures: Dict mapping section key → Plotly Figure.
        tables: Dict mapping section key → DataFrame.
        output_path: Destination path for the HTML file.
        my_name: The user's own Telegram name (for personalisation).

    Returns:
        Absolute path of the written HTML file.
    """
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    if not os.path.isdir(template_dir):
        raise FileNotFoundError(
            f"Templates directory not found: {template_dir}"
        )

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Convert figures → HTML strings
    html_figures: dict[str, str] = {}
    for key, fig in figures.items():
        html_figures[key] = fig_to_html(fig, div_id=f"chart-{key}")

    # Convert tables → HTML strings
    html_tables: dict[str, str] = {}
    for key, table in tables.items():
        if isinstance(table, pd.DataFrame):
            html_tables[key] = df_to_html_table(table)
        else:
            html_tables[key] = str(table)

    # Prepare top-contacts list for the summary section
    top_contacts_list: list[dict] = []
    if "top_contacts" in tables and isinstance(tables["top_contacts"], pd.DataFrame):
        for _, row in tables["top_contacts"].head(10).iterrows():
            top_contacts_list.append(
                {
                    "name": row.get("chat_name", ""),
                    "count": int(row.get("total_messages", 0)),
                }
            )

    template = env.get_template("index.html")

    rendered = template.render(
        stats=stats,
        figures=html_figures,
        tables=html_tables,
        top_contacts_list=top_contacts_list,
        my_name=my_name or "You",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_messages=stats.get("total_messages", 0),
        total_chats=stats.get("total_chats", 0),
        total_contacts=stats.get("total_contacts", 0),
        date_range_start=stats.get("date_range_start", "N/A"),
        date_range_end=stats.get("date_range_end", "N/A"),
        avg_messages_per_day=stats.get("avg_messages_per_day", 0),
        most_active_chat=stats.get("most_active_chat", "N/A"),
        total_words=stats.get("total_words", 0),
    )

    output_path = os.path.abspath(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    logger.info(f"Report written to {output_path}")
    return output_path
