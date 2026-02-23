"""Social graph builder using NetworkX and Plotly."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_social_graph(
    df: pd.DataFrame,
    my_name: str = "",
    min_messages: int = 5,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> nx.Graph:
    """
    Build a weighted undirected social interaction graph.

    Nodes represent users; edges represent shared conversation,
    weighted by the total number of messages in that chat.

    Args:
        df: Messages DataFrame.
        my_name: The user's own name (to label self-node distinctly).
        min_messages: Minimum shared messages for an edge to be included.
        date_start: Optional start date filter.
        date_end: Optional end date filter.

    Returns:
        NetworkX Graph with node attribute 'message_count' and
        edge attribute 'weight'.
    """
    filtered = df.copy()
    if date_start is not None:
        filtered = filtered[filtered["date"] >= date_start]
    if date_end is not None:
        filtered = filtered[filtered["date"] <= date_end]

    G: nx.Graph = nx.Graph()

    # Focus on personal chats (excludes group chats to keep graph clean)
    personal_types = {"personal_chat", "bot_chat", "saved_messages"}
    personal = filtered[filtered["chat_type"].isin(personal_types)]

    # Fallback: if no personal chats detected, use all chats
    if personal.empty:
        personal = filtered

    for chat_name, group in personal.groupby("chat_name"):
        senders = group["sender"].unique().tolist()
        msg_count = len(group)

        # Accumulate node message counts
        for sender in senders:
            sender_msgs = int((group["sender"] == sender).sum())
            if G.has_node(sender):
                G.nodes[sender]["message_count"] = (
                    G.nodes[sender].get("message_count", 0) + sender_msgs
                )
            else:
                G.add_node(sender, message_count=sender_msgs, is_self=(sender == my_name))

        # Add / update edges between every pair of senders in this chat
        if msg_count >= min_messages and len(senders) >= 2:
            for i in range(len(senders)):
                for j in range(i + 1, len(senders)):
                    u, v = senders[i], senders[j]
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += msg_count
                        G[u][v]["chats"].append(chat_name)
                    else:
                        G.add_edge(u, v, weight=msg_count, chats=[chat_name])

    # Prune edges below threshold
    weak_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_messages
    ]
    G.remove_edges_from(weak_edges)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


# ---------------------------------------------------------------------------
# Plotly visualisation
# ---------------------------------------------------------------------------


def plot_social_graph(G: nx.Graph, my_name: str = "") -> go.Figure:
    """
    Render the social interaction graph as an interactive Plotly figure.

    Node size is proportional to message count.
    Edge thickness is proportional to edge weight.
    Node colour reflects degree (number of connections).

    Args:
        G: NetworkX Graph as produced by `build_social_graph`.
        my_name: Label for the self-node (highlighted differently).

    Returns:
        Plotly Figure object.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available. Try lowering the minimum messages threshold.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    # Choose layout based on graph size
    n_nodes = G.number_of_nodes()
    if n_nodes <= 4:
        pos = nx.circular_layout(G)
    elif n_nodes <= 30:
        pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42, weight="weight")
    else:
        pos = nx.kamada_kawai_layout(G, weight="weight")

    # --- Edge traces (one per edge for variable width) ---
    edge_traces: list[go.Scatter] = []
    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_weight = max(weights) if weights else 1

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 1)
        width = 1.0 + 8.0 * (weight / max_weight)
        opacity = 0.25 + 0.55 * (weight / max_weight)
        chats = ", ".join(data.get("chats", [])[:3])

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=f"rgba(100,120,220,{opacity:.2f})"),
                hoverinfo="text",
                hovertext=f"{u} ↔ {v}<br>Messages: {weight:,}<br>Chats: {chats}",
                showlegend=False,
            )
        )

    # --- Node trace ---
    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_hover: list[str] = []
    node_sizes: list[float] = []
    node_colors: list[int] = []
    node_labels: list[str] = []

    msg_counts = [G.nodes[n].get("message_count", 1) for n in G.nodes]
    max_msg = max(msg_counts) if msg_counts else 1

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        msg_count = data.get("message_count", 1)
        degree = G.degree(node)
        is_self = node == my_name

        node_sizes.append(20 + 45 * (msg_count / max_msg))
        node_colors.append(degree)
        node_labels.append(node if n_nodes <= 40 else "")
        node_hover.append(
            f"<b>{node}</b>{'  ★ (you)' if is_self else ''}<br>"
            f"Messages: {msg_count:,}<br>"
            f"Connections: {degree}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_hover,
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="#ddd"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="YlOrRd",
            colorbar=dict(
                title="Connections",
                thickness=12,
                len=0.5,
            ),
            line=dict(width=2, color="#fff"),
        ),
        showlegend=False,
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(text="Social Interaction Graph", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=620,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------


def get_graph_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Calculate centrality and structural metrics for each graph node.

    Args:
        G: NetworkX Graph.

    Returns:
        DataFrame with one row per node, sorted by message count desc.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    degree_centrality = nx.degree_centrality(G)

    try:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes}

    try:
        closeness = nx.closeness_centrality(G)
    except Exception:
        closeness = {n: 0.0 for n in G.nodes}

    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes}

    rows: list[dict] = []
    for node, data in G.nodes(data=True):
        rows.append(
            {
                "contact": node,
                "message_count": data.get("message_count", 0),
                "connections": G.degree(node),
                "degree_centrality": round(degree_centrality.get(node, 0), 4),
                "betweenness_centrality": round(betweenness.get(node, 0), 4),
                "closeness_centrality": round(closeness.get(node, 0), 4),
                "eigenvector_centrality": round(eigenvector.get(node, 0), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("message_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """
    Detect communities using greedy modularity optimisation.

    Args:
        G: NetworkX Graph.

    Returns:
        Dict mapping node name → community integer index.
    """
    if G.number_of_nodes() < 3:
        return {node: 0 for node in G.nodes}

    try:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        community_map: dict[str, int] = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[str(node)] = i
        return community_map
    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return {str(node): 0 for node in G.nodes}


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------


def export_graph_image(G: nx.Graph, output_path: str = "social_graph.png") -> None:
    """
    Export the social graph as a static PNG using matplotlib.

    Requires matplotlib to be installed.

    Args:
        G: NetworkX Graph.
        output_path: Destination file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.error("matplotlib is required for PNG export.")
        return

    if G.number_of_nodes() == 0:
        logger.warning("Graph is empty; skipping PNG export.")
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, seed=42, weight="weight")

    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 5.0 * (w / max_w) for w in weights]

    msg_counts = [G.nodes[n].get("message_count", 1) for n in G.nodes]
    max_msg = max(msg_counts) if msg_counts else 1
    node_sizes = [200 + 3000 * (c / max_msg) for c in msg_counts]

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_size=node_sizes,
        width=edge_widths,
        node_color=msg_counts,
        cmap=cm.YlOrRd,
        font_size=8,
        alpha=0.85,
        edge_color="gray",
        font_color="black",
    )

    ax.set_title("Social Interaction Graph", fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Graph exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save graph image: {e}")
    finally:
        plt.close(fig)
