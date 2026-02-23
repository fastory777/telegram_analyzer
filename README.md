# Telegram Data Analyzer

A local, privacy-first analytics dashboard for your Telegram Desktop export data.

> **Your data never leaves your machine.** All processing happens locally.
> No API calls, no cloud uploads, no external services.

---

## Features

| Feature | Details |
|---|---|
| **Message Analytics** | Top contacts, initiators, avg length, night activity, response times |
| **Keyword Search** | Full-text search with ±5-message context, per-chat stats |
| **Social Graph** | Interactive NetworkX + Plotly graph, centrality metrics, community detection |
| **Lost Connections** | Inactive contacts, deleted accounts, fading connections, communication phases |
| **Temporal Analysis** | Hour×Day heatmap, activity timeline, burst detection, year-over-year |
| **Sentiment Analysis** | VADER / TextBlob, per-contact scores, timeline, extreme messages |
| **HTML Report** | Self-contained, interactive, embeds all charts |

---

## Installation

### Prerequisites

- Python 3.11 or newer
- A Telegram Desktop export in JSON format

### Steps

```bash
# 1. Clone or download this project
git clone https://github.com/yourname/telegram-data-analyzer.git
cd telegram-data-analyzer/telegram_analyzer

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Export Your Telegram Data

1. Open **Telegram Desktop** (not the mobile app)
2. Click the ☰ menu → **Settings**
3. Navigate to **Advanced → Export Telegram Data**
4. Configure what to export:
   - ✅ Personal chats
   - ✅ Contacts
   - Format: **JSON** ← this is required
   - Media: optional (not used by this app)
5. Click **Export** and wait for completion
6. Note the output folder path (usually `~/Downloads/Telegram Desktop/`)

The resulting folder structure will look like:

```
Telegram Desktop/          ← this is your export_path
├── chats/
│   ├── ChatExport_2024-01-01/
│   │   └── messages.json
│   ├── ChatExport_2024-01-02/
│   │   └── messages.json
│   └── ...
├── contacts.json
└── ...
```

---

## Running the Application

```bash
# From the telegram_analyzer/ directory
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`.

To specify a different port:

```bash
streamlit run app.py --server.port 8888
```

---

## Usage Guide

### 1. Load Your Data

- Enter the full path to your Telegram export folder in the **sidebar**
- Click **"Load Data"**
- The app detects your Telegram username automatically (or you can enter it manually)

### 2. Apply Filters

Use the sidebar controls to:
- Set a **date range** for the analysis
- Set **minimum message counts** to filter out low-activity chats

### 3. Explore the Tabs

#### Overview
Displays total messages, chats, contacts, words, and a weekly activity timeline.
A top-contacts bar chart and an hour×weekday heatmap give an immediate picture of your communication patterns.

#### Message Stats
- **Top Contacts**: Ranked bar chart with CSV export
- **Initiators**: Who starts conversations — pie chart
- **Avg Length**: Average character count per chat
- **Night Activity**: Bar chart of messages between 22:00 and 06:00
- **Response Time**: Median response latency distribution

#### Keyword Search
- Enter comma-separated keywords (e.g. `birthday, trip, project`)
- Set context window size (default 5 messages before/after)
- Matching messages are highlighted in gold
- See occurrence counts per chat

#### Social Graph
- Interactive graph where nodes are people and edges show communication weight
- Node size = total messages; edge thickness = shared message count
- Adjust the minimum-messages threshold to declutter
- Export as PNG or download centrality metrics as CSV

#### Lost Connections
- **Lost**: Contacts with ≥N messages but silent for ≥M months
- **Deleted Accounts**: Chats containing messages from deleted Telegram accounts
- **Fading**: Contacts whose activity dropped ≥50% in the last 3-month period
- Click a contact to view their message history timeline and communication phases

#### Temporal Analysis
- **Heatmap**: Hour of day × Day of week message density
- **Timeline**: Daily counts with 7-day and 30-day rolling averages
- **Distributions**: Weekday and hourly message histograms
- **Burst Detection**: Statistical outlier days (configurable threshold)
- **Year-over-Year**: Monthly comparison across years

#### Sentiment Analysis
Enable in the sidebar, then click **"Run Sentiment Analysis"**:
- Scores up to 10,000 messages with VADER (or TextBlob if VADER is absent)
- Per-contact average sentiment bar chart
- Weekly sentiment timeline
- Positive/Neutral/Negative distribution pie
- Top 10 most positive and most negative messages

### 4. Generate a Report

Click **"Generate HTML Report"** in the sidebar to produce a self-contained
`telegram_analysis_report.html` that you can open in any browser or share with
someone who doesn't have Python installed. All charts are embedded and interactive.

---

## Expected UI (Description)

```
┌─────────────────────────────────────────────────────────┐
│ Sidebar                │ Main area                       │
│                        │                                 │
│ [Export path input]    │  TABS: Overview | Messages |   │
│ [Load Data btn]        │        Search | Graph | Lost   │
│ [Your name]            │        Temporal | Sentiment    │
│ [Date range]           │                                 │
│ [Min messages slider]  │  ┌──────────────────────────┐  │
│ [Keyword input]        │  │   Key metric cards        │  │
│ [Graph settings]       │  └──────────────────────────┘  │
│ [Lost conn. settings]  │  ┌──────────────────────────┐  │
│ [Sentiment toggle]     │  │   Interactive Plotly chart│  │
│ [Generate Report btn]  │  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
telegram_analyzer/
│
├── app.py                  # Streamlit UI — entry point
├── data_loader.py          # JSON parsing and DataFrame construction
├── message_analyzer.py     # Analytics: top contacts, streaks, lengths…
├── keyword_search.py       # Full-text search with context windows
├── social_graph.py         # NetworkX graph + Plotly renderer + PNG export
├── lost_connections.py     # Inactive / deleted / fading contact detection
├── temporal_analysis.py    # Heatmaps, timelines, burst detection
├── sentiment_analysis.py   # VADER / TextBlob scoring
├── report_generator.py     # Jinja2-based HTML report builder
├── utils.py                # Shared helpers (date parsing, text normalisation…)
│
├── templates/
│   └── index.html          # Jinja2 report template (dark theme, Plotly CDN)
│
├── requirements.txt
└── README.md
```

---

## Performance Notes

| Dataset size | Typical load time | Sentiment scoring |
|---|---|---|
| < 10,000 messages | < 2 seconds | < 5 seconds |
| 10,000 – 100,000 | 3–15 seconds | 15–45 seconds |
| 100,000+ | 15–60 seconds | ~60 seconds (10k sample) |

Large exports are handled efficiently through:
- Streaming JSON parsing
- Vectorised pandas operations
- Sentiment sampling (10,000 message cap, configurable)
- `@st.cache_data` for repeated renders

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "No messages.json files found" | Make sure you exported in **JSON** format, not HTML |
| App is slow with large exports | Reduce the date range filter, or disable sentiment analysis |
| Graph is empty | Lower the "Min shared messages" slider in the sidebar |
| Sentiment tab not appearing | Install `vaderSentiment`: `pip install vaderSentiment` |
| Report generation fails | Check that the `templates/` folder exists next to `app.py` |
| Wrong username auto-detected | Override it manually in the sidebar "Your Telegram name" field |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web UI |
| `pandas` | Data manipulation |
| `networkx` | Social graph computation |
| `plotly` | Interactive charts and graph rendering |
| `matplotlib` | Static PNG graph export |
| `vaderSentiment` | Primary sentiment analyser |
| `textblob` | Fallback sentiment analyser |
| `jinja2` | HTML report templating |
| `kaleido` | Plotly static image export |
| `scipy`, `numpy` | Numerical utilities |

---

## Future Improvements

- [ ] **Group chat support** — separate personal vs group chat analytics
- [ ] **Media analysis** — photo/video/sticker frequency over time
- [ ] **Multi-language sentiment** — support for non-English text (lang-detect + multilingual models)
- [ ] **Contact clustering** — automatic grouping of contacts by communication patterns
- [ ] **Message threading** — track reply chains within conversations
- [ ] **Emoji analysis** — most-used emojis per contact / over time
- [ ] **Export to CSV/JSON** — bulk data export for external analysis
- [ ] **Mobile-friendly layout** — responsive Streamlit theme
- [ ] **Comparison mode** — compare two contacts or two time periods side-by-side
- [ ] **Memory between sessions** — persist filter settings across restarts

---

## License

MIT — use freely, modify freely, no warranty.
