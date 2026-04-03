import json
import os
import re
import glob

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from flask import send_from_directory, abort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH     = os.path.join(BASE_DIR, "data", "proteins.json")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata.json")
GRID_BASE     = os.path.join(BASE_DIR, "data", "clustering")

# ---------------------------------------------------------------------------
# Load protein data (includes PMID-level UniProt metadata)
# ---------------------------------------------------------------------------
def load_data():
    with open(JSON_PATH) as f:
        raw = json.load(f)

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    rows = []
    for pmid, entry in raw.items():
        uniprot_ids   = entry.get("Uniprot_IDS", []) or []
        protein_names = entry.get("Protein_names", []) or []
        organisms     = entry.get("Organisms", []) or []
        sequences     = entry.get("Sequences", []) or []
        n_collected   = entry.get("Number_of_proteins_collected", 0)

        # Paper-level metadata from metadata.json
        m = meta.get(str(pmid), {})
        groups   = m.get("groups", [])
        title    = m.get("title", "")
        pub_date = m.get("pub_date", "")
        source   = m.get("source", "")
        url      = m.get("url", "")

        for protein in entry.get("proteins", []):
            row = {"pmid": pmid}
            row.update({k: (v if v is not None else "") for k, v in protein.items()})
            row["n_uniprot_entries"]    = len(uniprot_ids)
            row["uniprot_ids"]          = ", ".join(uniprot_ids)
            row["n_proteins_collected"] = n_collected
            row["groups"]               = ", ".join(groups) if groups else "unknown"
            row["title"]                = title
            row["pub_date"]             = pub_date
            row["source"]               = source
            row["url"]                  = url
            rows.append(row)

    return pd.DataFrame(rows), raw, meta

df, RAW, META = load_data()

ALL_GROUPS = sorted({
    g for entry in META.values() for g in entry.get("groups", [])
})

# Column sets
EXTRACTION_FIELDS = [
    "pmid", "enzyme_name", "organism_source", "strain", "expression_strain",
    "plasmid", "molecular_weight", "medium_name", "inducer",
    "induction_temperature", "lysis_buffer", "elution_buffer", "desalting_process",
]
PAPER_FIELDS      = ["title", "pub_date", "source", "groups"]
UNIPROT_FIELDS    = ["uniprot_ids", "n_uniprot_entries", "n_proteins_collected"]
TABLE_FIELDS      = EXTRACTION_FIELDS + PAPER_FIELDS + UNIPROT_FIELDS

# ---------------------------------------------------------------------------
# Scan grid directory structure
# ---------------------------------------------------------------------------
def scan_grid():
    models, mins, thresholds, fields = set(), set(), set(), set()
    pattern = os.path.join(GRID_BASE, "model=*", "min=*", "t=*_cluster.html")
    for fpath in glob.glob(pattern):
        parts = fpath.split(os.sep)
        for p in parts:
            if p.startswith("model="):
                models.add(p.replace("model=", ""))
            elif p.startswith("min="):
                mins.add(p.replace("min=", ""))
        fname = os.path.basename(fpath)
        m = re.match(r"t=([\d.]+)_(.+)_cluster\.html$", fname)
        if m:
            thresholds.add(m.group(1))
            fields.add(m.group(2))
    return (
        sorted(models),
        sorted(mins, key=float),
        sorted(thresholds, key=float),
        sorted(fields),
    )

MODELS, MINS, THRESHOLDS, CLUSTER_FIELDS = scan_grid()
METRIC_TYPES = ["silhouette", "davies_bouldin", "n_clusters"]

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ── Global ── */
body { background-color: #f4f6f9; }

/* ── Navbar ── */
.navbar-brand { font-size: 1.2rem; font-weight: 700; letter-spacing: 0.02em; }
.navbar-subtitle { font-size: 0.75rem; opacity: 0.75; display: block; line-height: 1.2; }

/* ── Tabs ── */
.nav-tabs .nav-link          { color: #495057; font-weight: 500; border-radius: 6px 6px 0 0; }
.nav-tabs .nav-link.active   { color: #1a73e8; font-weight: 700; border-bottom: 3px solid #1a73e8; }
.nav-tabs .nav-link:hover    { color: #1a73e8; }

/* ── Filter panel ── */
.filter-panel {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 16px 20px 12px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ── Contact avatar ── */
.avatar-circle {
    width: 56px; height: 56px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: 700; color: #fff;
    margin-bottom: 10px;
}
.person-card { transition: transform .15s, box-shadow .15s; }
.person-card:hover { transform: translateY(-3px); box-shadow: 0 6px 18px rgba(0,0,0,.12) !important; }

/* ── Accordion ── */
.accordion-button { font-weight: 600; }
.accordion-item   { border-left: 4px solid #1a73e8 !important; margin-bottom: 6px; border-radius: 6px !important; }
"""

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
    title="ProtoPure",
)
app.index_string = app.index_string.replace(
    "</head>", f"<style>{CUSTOM_CSS}</style></head>"
)
server = app.server


@server.route("/grid_files/<path:filepath>")
def serve_grid_file(filepath):
    full = os.path.realpath(os.path.join(GRID_BASE, filepath))
    if not full.startswith(os.path.realpath(GRID_BASE)):
        abort(403)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(os.path.dirname(full), os.path.basename(full))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def make_dropdown(label, id_, options, value=None, multi=False, clearable=True):
    return dbc.Col([
        html.Label(label, className="fw-semibold small mb-1"),
        dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in options],
            value=value if value is not None else (options[0] if options else None),
            multi=multi,
            clearable=clearable,
            style={"fontSize": "13px"},
        ),
    ])


def detail_field(label, value):
    """Single labeled field for the detail panel."""
    if not value:
        return None
    return html.Div([
        html.Span(label + ": ", className="fw-semibold text-muted small"),
        html.Span(str(value), className="small"),
    ], className="mb-1")


# ---------------------------------------------------------------------------
# Tab layouts
# ---------------------------------------------------------------------------
DEFAULT_COLS = ["pmid", "enzyme_name", "organism_source"]

FIELD_LABELS = {c: c.replace("_", " ").title() for c in TABLE_FIELDS}
FIELD_LABELS.update({
    "pmid": "PMID", "enzyme_name": "Enzyme Name", "organism_source": "Organism",
    "expression_strain": "Expression Strain", "uniprot_ids": "UniProt IDs",
    "n_uniprot_entries": "# UniProt", "n_proteins_collected": "# Proteins",
    "pub_date": "Publication Date", "source": "Journal",
})


def proteins_tab():
    search_fields = [
        {"label": "PMID",              "value": "pmid"},
        {"label": "Enzyme name",       "value": "enzyme_name"},
        {"label": "Organism",          "value": "organism_source"},
        {"label": "Expression strain", "value": "expression_strain"},
        {"label": "Plasmid",           "value": "plasmid"},
        {"label": "Inducer",           "value": "inducer"},
        {"label": "UniProt ID",        "value": "uniprot_ids"},
        {"label": "Journal",           "value": "source"},
        {"label": "Paper title",       "value": "title"},
    ]

    col_options = [{"label": FIELD_LABELS.get(c, c), "value": c} for c in TABLE_FIELDS]

    return dbc.Container([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Filter by group", className="fw-semibold small mb-1"),
                    dcc.Dropdown(
                        id="filter-group",
                        options=[{"label": g, "value": g} for g in ALL_GROUPS],
                        multi=True,
                        placeholder="All groups",
                        style={"fontSize": "13px"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Search by", className="fw-semibold small mb-1"),
                    dcc.Dropdown(
                        id="search-field",
                        options=search_fields,
                        value="enzyme_name",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Search value", className="fw-semibold small mb-1"),
                    dbc.InputGroup([
                        dbc.Input(id="search-value", placeholder="Type to filter…", debounce=True, size="sm"),
                        dbc.Button("✕ Clear", id="clear-search", size="sm", color="secondary", outline=True),
                    ]),
                ], width=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("Show columns", className="fw-semibold small mb-1"),
                    dcc.Dropdown(
                        id="col-selector",
                        options=col_options,
                        value=DEFAULT_COLS,
                        multi=True,
                        placeholder="Select columns…",
                        style={"fontSize": "13px"},
                    ),
                ], width=12),
            ]),
        ], className="filter-panel"),
        dbc.Row([
            dbc.Col(html.Div(id="protein-count", className="text-muted small"), width=10),
            dbc.Col(
                dbc.Button("⬇ Download CSV", id="download-btn", size="sm", color="success", outline=True),
                width=2, className="text-end",
            ),
        ], className="mb-2 align-items-center"),
        dcc.Download(id="download-csv"),
        # Table
        dash_table.DataTable(
            id="protein-table",
            columns=[{"name": FIELD_LABELS.get(c, c), "id": c} for c in DEFAULT_COLS],
            data=df[DEFAULT_COLS].to_dict("records"),
            page_size=25,
            page_action="native",
            sort_action="native",
            filter_action="none",
            row_selectable="single",
            selected_rows=[],
            style_table={"overflowX": "auto"},
            style_cell={
                "fontSize": "12px",
                "padding": "6px 10px",
                "textAlign": "left",
                "maxWidth": "220px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            },
            style_header={
                "fontWeight": "700",
                "backgroundColor": "#1a3a5c",
                "color": "#ffffff",
                "borderBottom": "2px solid #1a3a5c",
                "fontSize": "11px",
                "textTransform": "uppercase",
                "letterSpacing": "0.04em",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#eef4fd"},
                {"if": {"state": "selected"}, "backgroundColor": "#cfe2ff", "border": "1px solid #9ec5fe"},
            ],
            tooltip_data=[
                {col: {"value": str(row.get(col, "")), "type": "markdown"} for col in TABLE_FIELDS}
                for row in df[TABLE_FIELDS].to_dict("records")
            ],
            tooltip_delay=0,
            tooltip_duration=None,
        ),
        # Detail panel (shown on row click)
        html.Div(id="detail-panel", className="mt-3"),
    ], fluid=True, className="pt-3")


def clustering_tab():
    return dbc.Container([
        dbc.Row([
            make_dropdown("Model", "dd-model", MODELS, value=MODELS[0] if MODELS else None),
            make_dropdown("Min community size", "dd-min", MINS, value=MINS[0] if MINS else None),
            make_dropdown("Threshold", "dd-threshold", THRESHOLDS, value=THRESHOLDS[0] if THRESHOLDS else None),
            make_dropdown("Field", "dd-field", CLUSTER_FIELDS, value=CLUSTER_FIELDS[0] if CLUSTER_FIELDS else None),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id="plot-type",
                    options=[
                        {"label": " UMAP cluster plot", "value": "cluster"},
                        {"label": " Cluster distribution", "value": "distribution"},
                    ],
                    value="cluster",
                    inline=True,
                    className="mb-2",
                ),
            ], width=8),
            dbc.Col([
                html.Label("Show top N clusters", className="fw-semibold small mb-1"),
                dcc.Dropdown(
                    id="top-n-clusters",
                    options=[{"label": str(n), "value": n} for n in [10, 15, 20, 30, 50]] +
                            [{"label": "All", "value": 0}],
                    value=20,
                    clearable=False,
                    style={"fontSize": "13px"},
                ),
            ], width=2),
        ], className="mb-2 align-items-end"),
        dbc.Row([dbc.Col(html.Div(id="plot-status", className="text-danger small mb-1"))]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id="cluster-graph",
                    config={"displayModeBar": True, "toImageButtonOptions": {"format": "svg"}},
                    style={"height": "650px"},
                ),
                width=12,
            ),
        ]),
        html.Div(id="cluster-point-detail", className="mt-2"),
        html.Hr(),
        html.H6("Cluster metrics for selected parameters", className="mt-2 mb-2 fw-semibold"),
        dbc.Row([dbc.Col(html.Div(id="metrics-table-container"), width=12)]),
    ], fluid=True, className="pt-3")


def grid_metrics_tab():
    metrics_dir = os.path.join(GRID_BASE, "metrics")
    available = []
    if os.path.isdir(metrics_dir):
        for fname in sorted(os.listdir(metrics_dir)):
            m = re.match(r"(silhouette|davies_bouldin|n_clusters)_heatmap__(.+)\.png$", fname)
            if m:
                available.append((m.group(1), m.group(2)))
    model_shorts = sorted(set(x[1] for x in available))

    return dbc.Container([
        dbc.Row([
            make_dropdown("Model", "dd-metrics-model", model_shorts,
                          value=model_shorts[0] if model_shorts else None),
            make_dropdown("Metric", "dd-metrics-type", METRIC_TYPES, value="silhouette"),
        ], className="mb-3 g-3"),
        dbc.Row([dbc.Col(html.Div(id="metrics-image-container"), width=12)]),
    ], fluid=True, className="pt-3")


AVATAR_COLORS = ["#1a73e8", "#e8711a", "#1aa85c", "#8e1ae8"]

def person_card(name, email, role, departments, color="#1a73e8"):
    initials = "".join(p[0].upper() for p in name.split()[:2])
    return dbc.Card([
        dbc.CardBody([
            html.Div(initials, className="avatar-circle", style={"backgroundColor": color}),
            html.H6(name, className="mb-0 fw-bold"),
            html.A(email, href=f"mailto:{email}", className="text-muted small d-block mb-2"),
            html.Span(role, className="badge rounded-pill mb-2",
                      style={"backgroundColor": color, "fontSize": "11px"}),
            html.Ul([html.Li(d, className="small text-muted") for d in departments],
                    className="mb-0 ps-3") if departments else None,
        ])
    ], className="h-100 shadow-sm person-card border-0")


TEAM = [
    dict(
        name="Ricardo Almada Monter",
        email="ralmadamonter@ucsd.edu",
        role="Graduate Student Researcher",
        departments=["Department of Chemistry & Biochemistry, UC San Diego"],
    ),
    dict(
        name="Jose Martinez Lomeli",
        email="lomeli90@gmail.com",
        role="Independent Researcher",
        departments=[],
    ),
    dict(
        name="Erika Garay",
        email="ecgaray@health.ucsd.edu",
        role="Staff Scientist",
        departments=[
            "Skaggs School of Pharmacy and Pharmaceutical Sciences, UC San Diego",
        ],
    ),
    dict(
        name="Adrian Jinich, PhD",
        email="ajinich@health.ucsd.edu",
        role="Assistant Professor",
        departments=[
            "Skaggs School of Pharmacy and Pharmaceutical Sciences, UC San Diego",
            "Department of Chemistry & Biochemistry, UC San Diego",
        ],
    ),
]


def contact_cards():
    return dbc.Row(
        [dbc.Col(person_card(**m, color=AVATAR_COLORS[i % len(AVATAR_COLORS)]), width=3)
         for i, m in enumerate(TEAM)],
        className="g-4",
    )


def contact_tab():
    return dbc.Container([
        html.H3("Research Team", className="mt-3 mb-4"),
        contact_cards(),
    ], fluid=True, className="pt-3")


def readme_tab():
    return dbc.Container([
        html.H3("ProtoPure — Dashboard Guide", className="mt-3 mb-1"),
        html.P(
            "ProtoPure is a LLM-enhanced tool for the systematic extraction and comparison of "
            "protein purification conditions from scientific literature, designed to support "
            "biochemical workflow design. Click a section below to expand it.",
            className="text-muted mb-4",
        ),
        dbc.Accordion([

            # ── Extraction Data ───────────────────────────────────────────────
            dbc.AccordionItem(title="Extraction Data", children=[
                html.P(
                    "Shows all proteins extracted by the LLM pipeline, one row per protein. "
                    "Each row combines the extracted purification conditions with paper metadata "
                    "and linked UniProt entries.",
                    className="mb-3",
                ),
                html.H6("Filters", className="fw-bold"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("Filter by group  ", className="fw-semibold"),
                        "Select one or more protein families (azoreductases, sdrs, sams, etc.) "
                        "to restrict the table to papers from that collection. Multiple groups "
                        "can be selected at once.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Search by  ", className="fw-semibold"),
                        "Choose which field to search: PMID, Enzyme name, Organism, Expression strain, "
                        "Plasmid, Inducer, UniProt ID, Journal, or Paper title.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Search value  ", className="fw-semibold"),
                        "Type any text to filter rows. The search is case-insensitive and matches "
                        "partial strings (e.g. 'coli' matches 'Escherichia coli').",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("✕ Clear  ", className="fw-semibold"),
                        "Resets both the search value and the group filter, returning to the full dataset.",
                    ]),
                ], flush=True, className="mb-3"),
                html.H6("Table", className="fw-bold mt-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("Hover over any cell  ", className="fw-semibold"),
                        "to see the full text if it is truncated.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Click a column header  ", className="fw-semibold"),
                        "to sort by that column.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Click a row  ", className="fw-semibold"),
                        "to open the detail panel below the table, showing the full extracted "
                        "protein fields, all linked UniProt entries (with links to uniprot.org), "
                        "and the paper metadata (title, journal, date, group badges, PubMed link, "
                        "and full-text link where available).",
                    ]),
                ], flush=True, className="mb-3"),
                html.H6("⬇ Download CSV", className="fw-bold mt-2"),
                html.P(
                    "Downloads the currently visible (filtered) table as a CSV file. "
                    "Apply any filters first — the download reflects exactly what is shown.",
                    className="mb-0",
                ),
            ]),

            # ── Clustering Explorer ───────────────────────────────────────────
            dbc.AccordionItem(title="Clustering Explorer", children=[
                html.P(
                    "Displays interactive clustering results from a grid search over embedding "
                    "models, community-size thresholds, and similarity thresholds. Each "
                    "combination produces per-field UMAP plots and cluster-size distributions "
                    "rendered natively in Plotly — hover, zoom, and pan are fully interactive.",
                    className="mb-3",
                ),
                html.H6("Parameter dropdowns", className="fw-bold"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("Model  ", className="fw-semibold"),
                        "The sentence-embedding model used to encode field values "
                        "(e.g. PubMedBERT, BioBERT, SapBERT).",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Min community size  ", className="fw-semibold"),
                        "Minimum number of entries required to form a cluster. "
                        "Lower values produce more (smaller) clusters.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Threshold  ", className="fw-semibold"),
                        "Cosine similarity threshold for grouping entries into a cluster. "
                        "Higher values require closer matches (tighter clusters).",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Field  ", className="fw-semibold"),
                        "Which extraction field to cluster: enzyme name, organism, lysis buffer, "
                        "inducer, etc. Clustering is performed independently per field.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Show top N clusters  ", className="fw-semibold"),
                        "Limits the number of distinctly coloured clusters shown. "
                        "The largest N clusters (by entry count) each get a unique colour; "
                        "all remaining clusters are grouped into a single light-gray "
                        "\"Other\" trace so they remain visible without adding colour noise. "
                        "Choose \"All\" to colour every cluster individually (slow for large grids).",
                    ]),
                ], flush=True, className="mb-3"),
                html.H6("Plot type", className="fw-bold mt-2"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("UMAP cluster plot  ", className="fw-semibold"),
                        "2D projection of all field values coloured by cluster membership. "
                        "Hover over any point to see the original text, PMID, and cluster label. "
                        "Noise points (not assigned to any cluster) are shown in light gray at "
                        "low opacity. The plot title shows the total number of clusters and noise points.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Cluster distribution  ", className="fw-semibold"),
                        "Bar chart of the top-N clusters sorted by size (largest first). "
                        "Useful for quickly seeing which conditions are most common across the dataset.",
                    ]),
                ], flush=True, className="mb-3"),
                html.H6("Recommended starting parameters", className="fw-bold mt-2"),
                html.P([
                    "For a clear overview of enzyme name clustering, try: ",
                    html.Span("NeuML/pubmedbert-base-embeddings", className="font-monospace small"),
                    " · min = 10 · threshold = 0.7 · top N = 20. "
                    "This yields ~167 clusters with well-sized groups and manageable noise.",
                ], className="mb-3"),
                html.H6("Metrics table", className="fw-bold mt-2"),
                html.P(
                    "Shows Silhouette score (cosine) and Davies-Bouldin index for every field "
                    "under the selected model / min-size / threshold combination. "
                    "Higher Silhouette (highlighted green above 0.6) indicates well-separated clusters. "
                    "Lower Davies-Bouldin indicates more compact, better-separated clusters.",
                    className="mb-0",
                ),
            ]),

            # ── Grid Metrics ──────────────────────────────────────────────────
            dbc.AccordionItem(title="Grid Metrics", children=[
                html.P(
                    "Displays summary heatmap images that compare clustering quality across "
                    "all threshold / min-community-size combinations for a given model.",
                    className="mb-3",
                ),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("Model  ", className="fw-semibold"),
                        "Select the embedding model whose heatmaps you want to view.",
                    ]),
                    dbc.ListGroupItem([
                        html.Span("Metric  ", className="fw-semibold"),
                        html.Ul([
                            html.Li([html.Span("silhouette  ", className="fw-semibold"),
                                     "— ranges from -1 to 1; higher is better."]),
                            html.Li([html.Span("davies_bouldin  ", className="fw-semibold"),
                                     "— non-negative; lower is better."]),
                            html.Li([html.Span("n_clusters  ", className="fw-semibold"),
                                     "— number of clusters found at each parameter combination."]),
                        ], className="mb-0 mt-1"),
                    ]),
                ], flush=True),
            ]),

            # ── Contact ───────────────────────────────────────────────────────
            dbc.AccordionItem(title="Contact", children=[contact_cards()]),

        ], start_collapsed=True, always_open=True),
    ], fluid=True, className="pt-3")


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Span("⚗ ProtoPure", className="navbar-brand text-white"),
                html.Span("LLM-Enhanced Systematic Extraction and Comparison of Protein Purification Conditions",
                          className="navbar-subtitle text-white"),
            ]),
        ], fluid=True),
        color="#1a3a5c",
        dark=True,
        className="mb-3 rounded shadow-sm px-3 py-2",
    ),
    dbc.Tabs([
        dbc.Tab(label="Extraction Data",    tab_id="tab-proteins"),
        dbc.Tab(label="Clustering Explorer",tab_id="tab-clustering"),
        dbc.Tab(label="Grid Metrics",       tab_id="tab-grid-metrics"),
        dbc.Tab(label="README",             tab_id="tab-readme"),
        dbc.Tab(label="Contact",            tab_id="tab-contact"),
    ], id="main-tabs", active_tab="tab-proteins"),
    html.Div(id="tab-content", className="mt-2"),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(tab):
    if tab == "tab-proteins":
        return proteins_tab()
    elif tab == "tab-readme":
        return readme_tab()
    elif tab == "tab-contact":
        return contact_tab()
    elif tab == "tab-clustering":
        return clustering_tab()
    elif tab == "tab-grid-metrics":
        return grid_metrics_tab()
    return html.Div()


@app.callback(
    Output("protein-table", "data"),
    Output("protein-table", "columns"),
    Output("protein-table", "tooltip_data"),
    Output("protein-count", "children"),
    Input("filter-group", "value"),
    Input("search-field", "value"),
    Input("search-value", "value"),
    Input("col-selector", "value"),
)
def filter_proteins(groups, search_field, search_value, selected_cols):
    filtered = df.copy()
    if groups:
        mask = filtered["groups"].apply(
            lambda g: any(sel in g.split(", ") for sel in groups)
        )
        filtered = filtered[mask]
    if search_value and search_field and search_field in filtered.columns:
        filtered = filtered[
            filtered[search_field].astype(str).str.contains(search_value, case=False, na=False)
        ]

    cols = selected_cols if selected_cols else DEFAULT_COLS
    # Always include pmid in data even if hidden, so row-click detail still works
    data_cols = list(dict.fromkeys(["pmid"] + cols))
    col_defs = [{"name": FIELD_LABELS.get(c, c), "id": c} for c in cols]

    records = filtered[data_cols].to_dict("records")
    tooltips = [
        {c: {"value": str(row.get(c, "")), "type": "markdown"} for c in cols}
        for row in records
    ]
    label = f"Showing {len(filtered):,} of {len(df):,} proteins"
    return records, col_defs, tooltips, label


@app.callback(
    Output("detail-panel", "children"),
    Input("protein-table", "selected_rows"),
    State("protein-table", "data"),
)
def show_detail(selected_rows, table_data):
    if not selected_rows or not table_data:
        return html.Div()

    row = table_data[selected_rows[0]]
    pmid = row.get("pmid", "")
    entry = RAW.get(pmid, {})

    uniprot_ids   = entry.get("Uniprot_IDS", []) or []
    protein_names = entry.get("Protein_names", []) or []
    organisms     = entry.get("Organisms", []) or []
    sequences     = entry.get("Sequences", []) or []

    # --- Extracted protein card ---
    extraction_items = []
    for field in EXTRACTION_FIELDS[1:]:  # skip pmid
        val = row.get(field, "")
        if val:
            item = detail_field(field.replace("_", " ").title(), val)
            if item:
                extraction_items.append(item)

    # Paper metadata from META
    m = META.get(str(pmid), {})

    pubmed_link = html.A(
        f"PubMed: {pmid}",
        href=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        target="_blank",
        className="small",
    )

    # --- Extracted protein card ---
    extracted_card = dbc.Card([
        dbc.CardHeader(html.Span("Extracted Protein", className="fw-bold")),
        dbc.CardBody(extraction_items or [html.Span("No extraction data.", className="text-muted small")]),
    ], className="mb-3", color="light")

    # --- Paper metadata card ---
    paper_links = [pubmed_link]
    if m.get("url"):
        paper_links += [
            html.Span(" · ", className="text-muted mx-1"),
            html.A("Full text", href=m["url"], target="_blank", className="small"),
        ]

    paper_body = []
    if m.get("title"):
        paper_body.append(html.P(m["title"], className="fw-semibold small mb-2"))
    meta_line = "  ·  ".join(filter(None, [m.get("source", ""), m.get("pub_date", "")]))
    if meta_line:
        paper_body.append(html.P(meta_line, className="text-muted small mb-1"))
    if m.get("groups"):
        paper_body.append(html.P(
            [html.Span("Groups: ", className="fw-semibold")] +
            [dbc.Badge(g, color="primary", className="me-1") for g in m["groups"]],
            className="mb-0"
        ))

    paper_card = dbc.Card([
        dbc.CardHeader(html.Div(paper_links)),
        dbc.CardBody(paper_body or [html.Span("No paper metadata available.", className="text-muted small")]),
    ], className="mb-3")

    # --- UniProt entries card ---
    if uniprot_ids:
        uniprot_rows = []
        for i, uid in enumerate(uniprot_ids):
            name = protein_names[i] if i < len(protein_names) else "—"
            org  = organisms[i]     if i < len(organisms)     else "—"
            seq  = sequences[i]     if i < len(sequences)     else None
            seq_info = f"{len(seq)} aa" if seq else "—"

            uniprot_rows.append(
                dbc.ListGroupItem([
                    dbc.Row([
                        dbc.Col([
                            html.A(uid,
                                   href=f"https://www.uniprot.org/uniprot/{uid}",
                                   target="_blank",
                                   className="fw-bold small me-2"),
                            html.Span(org, className="text-muted small"),
                        ], width=4),
                        dbc.Col(html.Span(name, className="small"), width=6),
                        dbc.Col(html.Span(seq_info, className="text-muted small"), width=2),
                    ], align="center"),
                ])
            )

        uniprot_card = dbc.Card([
            dbc.CardHeader(html.Span(
                f"UniProt Entries for PMID {pmid} ({len(uniprot_ids)} entries)",
                className="fw-bold",
            )),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Span("UniProt ID / Organism", className="fw-semibold small"), width=4),
                    dbc.Col(html.Span("Protein Name", className="fw-semibold small"), width=6),
                    dbc.Col(html.Span("Sequence", className="fw-semibold small"), width=2),
                ], className="px-3 mb-1"),
                dbc.ListGroup(uniprot_rows, flush=True),
            ]),
        ], className="mb-3")
    else:
        uniprot_card = dbc.Card([
            dbc.CardHeader("UniProt Entries"),
            dbc.CardBody(html.Span("No UniProt entries linked to this PMID.", className="text-muted small")),
        ], className="mb-3", color="light")

    # --- All proteins from this PMID (conditions table) ---
    all_proteins = entry.get("proteins", []) or []
    CONDITION_COLS = [
        "enzyme_name", "organism_source", "strain", "expression_strain",
        "plasmid", "molecular_weight", "medium_name", "inducer",
        "induction_temperature", "lysis_buffer", "elution_buffer", "desalting_process",
    ]
    cond_records = [
        {c: (p.get(c) or "") for c in CONDITION_COLS}
        for p in all_proteins
    ]
    # Mark the selected protein
    sel_idx = next(
        (i for i, p in enumerate(all_proteins)
         if all(str(p.get(c, "") or "") == str(row.get(c, "") or "") for c in CONDITION_COLS[:3])),
        None,
    )
    cond_col_defs = [
        {"name": c.replace("_", " ").title(), "id": c} for c in CONDITION_COLS
    ]
    conditions_card = dbc.Card([
        dbc.CardHeader(
            html.Span(
                f"All purification conditions from PMID {pmid}  ({len(all_proteins)} protein{'s' if len(all_proteins) != 1 else ''})",
                className="fw-bold",
            )
        ),
        dbc.CardBody(
            dash_table.DataTable(
                columns=cond_col_defs,
                data=cond_records,
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontSize": "12px",
                    "padding": "5px 10px",
                    "textAlign": "left",
                    "maxWidth": "260px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                style_header={
                    "fontWeight": "700",
                    "backgroundColor": "#1a3a5c",
                    "color": "#ffffff",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.04em",
                },
                style_data_conditional=(
                    [{"if": {"row_index": "odd"}, "backgroundColor": "#eef4fd"}] +
                    ([{"if": {"row_index": sel_idx}, "backgroundColor": "#cfe2ff",
                       "border": "1px solid #9ec5fe"}] if sel_idx is not None else [])
                ),
                tooltip_data=[
                    {c: {"value": str(r.get(c, "")), "type": "markdown"} for c in CONDITION_COLS}
                    for r in cond_records
                ],
                tooltip_delay=0,
                tooltip_duration=None,
            ) if cond_records else html.Span("No protein conditions available.", className="text-muted small")
        ),
    ], className="mb-3")

    return html.Div([
        html.Hr(),
        html.H6("Selected Row Detail", className="fw-semibold mb-3"),
        dbc.Row([
            dbc.Col(extracted_card, width=5),
            dbc.Col(uniprot_card, width=7),
        ]),
        dbc.Row([
            dbc.Col(paper_card, width=12),
        ]),
        dbc.Row([
            dbc.Col(conditions_card, width=12),
        ]),
    ])


# Qualitative color palette — high-contrast, colorblind-friendly base
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79",
    "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd",
    "#e6550d", "#31a354", "#756bb1", "#636363", "#6baed6",
]
_NOISE_COLOR = "#c0c0c0"


def _load_cluster_csv(model, min_val, threshold):
    """Return the ALL_FIELDS dataframe or None."""
    path = os.path.join(GRID_BASE, f"model={model}", f"min={min_val}",
                        f"t={threshold}_ALL_FIELDS.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _umap_figure(cdf, field_name, top_n=20):
    """Build a UMAP scatter figure from a filtered cluster DataFrame."""
    cdf = cdf.reset_index(drop=True)
    noise_mask = cdf["cluster_id"] == -1
    non_noise  = cdf[~noise_mask]

    # Rank clusters by size; optionally cap at top_n
    cluster_sizes = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
    if top_n and top_n > 0:
        top_ids = set(cluster_sizes.index[:top_n])
    else:
        top_ids = set(cluster_sizes.index)

    fig = go.Figure()

    # NOISE — thin gray, low opacity, drawn first
    if noise_mask.any():
        nd = cdf[noise_mask]
        fig.add_trace(go.Scattergl(
            x=nd["x"], y=nd["y"],
            mode="markers",
            name="Noise",
            marker=dict(color=_NOISE_COLOR, size=4, opacity=0.25),
            hovertemplate="<b>Noise</b><br>%{customdata[0]}<br>PMID: %{text}<extra></extra>",
            customdata=list(zip(nd["value"].str[:80].tolist(),
                                nd["protein_index"].astype(str).tolist())),
            text=nd["key"].astype(str),
            showlegend=True,
        ))

    # "Other clusters" bucket — light gray, slightly more visible than noise
    other_mask = ~noise_mask & ~cdf["cluster_id"].isin(top_ids)
    if other_mask.any():
        od = cdf[other_mask]
        fig.add_trace(go.Scattergl(
            x=od["x"], y=od["y"],
            mode="markers",
            name=f"Other ({len(cluster_sizes) - len(top_ids)} clusters)",
            marker=dict(color="#adb5bd", size=5, opacity=0.35),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}<br>PMID: %{text}<extra></extra>",
            customdata=list(zip(
                ("Cluster " + od["cluster_id"].astype(str)).tolist(),
                od["protein_index"].astype(str).tolist(),
                od["value"].str[:80].tolist(),
            )),
            text=od["key"].astype(str),
            showlegend=True,
        ))

    # Top-N named clusters — distinct colors, larger markers
    for rank, cid in enumerate(cluster_sizes.index[:len(top_ids)]):
        cd = non_noise[non_noise["cluster_id"] == cid]
        label = cd["cluster_label_short"].iloc[0][:35] if len(cd) else f"C{cid}"
        color = _PALETTE[rank % len(_PALETTE)]
        fig.add_trace(go.Scattergl(
            x=cd["x"], y=cd["y"],
            mode="markers",
            name=f"[{cid}] {label}",
            marker=dict(color=color, size=7, opacity=0.80,
                        line=dict(width=0.4, color="rgba(255,255,255,0.6)")),
            hovertemplate=(
                "<b>[%{meta}] %{customdata[0]}</b><br>"
                "%{customdata[2]}<br>"
                "PMID: %{text}<extra></extra>"
            ),
            meta=cid,
            customdata=list(zip(
                cd["cluster_label_short"].str[:50].tolist(),
                cd["protein_index"].astype(str).tolist(),
                cd["value"].str[:100].tolist(),
            )),
            text=cd["key"].astype(str),
        ))

    n_total   = len(cluster_sizes)
    n_shown   = len(top_ids)
    n_noise   = noise_mask.sum()
    title_txt = (
        f"UMAP — <b>{field_name.replace('_', ' ').title()}</b>"
        f"  ·  top {n_shown}/{n_total} clusters shown  ·  {n_noise:,} noise pts"
    )
    fig.update_layout(
        title=dict(text=title_txt, font=dict(size=13)),
        plot_bgcolor="#f9fafc",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(
            title=dict(text="Cluster", font=dict(size=11)),
            font=dict(size=10),
            itemsizing="constant",
            bordercolor="#dee2e6", borderwidth=1,
            tracegroupgap=1,
        ),
        margin=dict(l=20, r=200, t=50, b=20),
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
    )
    return fig


def _distribution_figure(cdf, field_name, top_n=20):
    """Build a cluster-size bar chart from a filtered cluster DataFrame."""
    cdf = cdf.reset_index(drop=True)
    counts = (
        cdf[cdf["cluster_id"] != -1]
        .groupby(["cluster_id", "cluster_label_short"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if top_n and top_n > 0:
        counts = counts.head(top_n)

    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(counts))]
    labels = counts["cluster_label_short"].str[:45]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts["count"],
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.6)",
        marker_line_width=0.8,
        opacity=0.88,
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Cluster Sizes — <b>{field_name.replace('_', ' ').title()}</b>"
                 + (f"  (top {top_n})" if top_n else ""),
            font=dict(size=13),
        ),
        plot_bgcolor="#f9fafc",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickangle=-45, tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e5e7eb", zeroline=False,
            title="# entries",
        ),
        margin=dict(l=50, r=20, t=50, b=160),
        hoverlabel=dict(bgcolor="white", font_size=12),
        bargap=0.25,
    )
    return fig


@app.callback(
    Output("cluster-graph", "figure"),
    Output("plot-status", "children"),
    Input("dd-model", "value"),
    Input("dd-min", "value"),
    Input("dd-threshold", "value"),
    Input("dd-field", "value"),
    Input("plot-type", "value"),
    Input("top-n-clusters", "value"),
)
def update_cluster_plot(model, min_val, threshold, field, plot_type, top_n):
    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f9fafc")
    if not all([model, min_val, threshold, field, plot_type]):
        return empty_fig, "Select all parameters above."

    cdf = _load_cluster_csv(model, min_val, threshold)
    if cdf is None:
        return empty_fig, f"Data file not found for model={model} min={min_val} t={threshold}"

    field_df = cdf[cdf["field"] == field]
    if field_df.empty:
        return empty_fig, f"No data for field '{field}' in this parameter combination."

    n = top_n or 0
    if plot_type == "cluster":
        return _umap_figure(field_df, field, top_n=n), ""
    else:
        return _distribution_figure(field_df, field, top_n=n), ""


_CONDITION_COLS = [
    "enzyme_name", "organism_source", "strain", "expression_strain",
    "plasmid", "molecular_weight", "medium_name", "inducer",
    "induction_temperature", "lysis_buffer", "elution_buffer", "desalting_process",
]


@app.callback(
    Output("cluster-point-detail", "children"),
    Input("cluster-graph", "clickData"),
    State("dd-field", "value"),
)
def show_cluster_point_detail(click_data, field):
    if not click_data:
        return html.Div()

    point = click_data["points"][0]
    pmid = str(point.get("text", ""))
    customdata = point.get("customdata", [])

    # customdata layout: [label_or_value, protein_index, value_text]  (noise: [value, protein_index])
    try:
        protein_index = int(customdata[1])
    except (IndexError, ValueError, TypeError):
        protein_index = 0

    entry = RAW.get(pmid, {})
    proteins = entry.get("proteins", []) or []
    if not proteins:
        return html.Div(f"No protein data for PMID {pmid}.", className="text-muted small mt-2")

    protein_index = min(protein_index, len(proteins) - 1)
    protein = proteins[protein_index]

    # Field value that was clicked (used for context header)
    clicked_value = str(customdata[0] if customdata else "")

    # Conditions table: condition → value, skip empty
    cond_rows = [
        {"Condition": c.replace("_", " ").title(), "Value": str(protein.get(c) or "")}
        for c in _CONDITION_COLS
        if protein.get(c)
    ]

    m = META.get(pmid, {})
    pubmed_link = html.A(f"PMID {pmid}", href=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                         target="_blank", className="small")
    paper_info = "  ·  ".join(filter(None, [m.get("source", ""), m.get("pub_date", "")]))

    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.Span("Purification Conditions — ", className="fw-bold"),
                pubmed_link,
                html.Span(f"  ·  {paper_info}", className="text-muted small") if paper_info else None,
                html.Span(f"  ·  {field.replace('_', ' ').title()}: ", className="text-muted small ms-2"),
                html.Span(f'"{clicked_value[:80]}"', className="small fst-italic"),
            ]),
            dbc.CardBody(
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in ["Condition", "Value"]],
                    data=cond_rows,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "fontSize": "12px",
                        "padding": "5px 10px",
                        "textAlign": "left",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "Condition"},
                         "fontWeight": "600", "width": "200px", "minWidth": "200px",
                         "backgroundColor": "#f8f9fa"},
                        {"if": {"column_id": "Value"},
                         "whiteSpace": "normal", "height": "auto"},
                    ],
                    style_header={
                        "fontWeight": "700",
                        "backgroundColor": "#1a3a5c",
                        "color": "#ffffff",
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.04em",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#eef4fd"},
                    ],
                ) if cond_rows else html.Span("No conditions recorded for this protein.", className="text-muted small")
            ),
        ], className="mb-3"),
    ])


@app.callback(
    Output("metrics-table-container", "children"),
    Input("dd-model", "value"),
    Input("dd-min", "value"),
    Input("dd-threshold", "value"),
)
def update_metrics_table(model, min_val, threshold):
    if not all([model, min_val, threshold]):
        return html.Div()

    csv_path = os.path.join(
        GRID_BASE, f"model={model}", f"min={min_val}", f"t={threshold}_FIELD_CLUSTER_METRICS.csv"
    )
    if not os.path.isfile(csv_path):
        return html.Div("Metrics file not found.", className="text-muted small")

    metrics_df = pd.read_csv(csv_path).round(4)
    return dash_table.DataTable(
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in metrics_df.columns],
        data=metrics_df.to_dict("records"),
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "5px 10px", "textAlign": "left"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            {
                "if": {"filter_query": "{silhouette_cosine} > 0.6", "column_id": "silhouette_cosine"},
                "color": "#198754", "fontWeight": "bold",
            },
            {
                "if": {"filter_query": "{silhouette_cosine} < 0.3", "column_id": "silhouette_cosine"},
                "color": "#dc3545",
            },
        ],
    )


@app.callback(
    Output("metrics-image-container", "children"),
    Input("dd-metrics-model", "value"),
    Input("dd-metrics-type", "value"),
)
def update_metrics_image(model_short, metric_type):
    if not model_short or not metric_type:
        return html.Div()

    filename  = f"{metric_type}_heatmap__{model_short}.png"
    full_path = os.path.join(GRID_BASE, "metrics", filename)

    if not os.path.isfile(full_path):
        return html.Div(f"Image not found: {filename}", className="text-muted small")

    return html.Img(
        src=f"/grid_files/metrics/{filename}",
        style={"maxWidth": "100%", "border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "8px"},
    )


# ---------------------------------------------------------------------------
@app.callback(
    Output("search-value", "value"),
    Output("filter-group", "value"),
    Input("clear-search", "n_clicks"),
    prevent_initial_call=True,
)
def clear_filters(_):
    return "", []


@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("protein-table", "data"),
    prevent_initial_call=True,
)
def download_csv(_, table_data):
    filtered_df = pd.DataFrame(table_data)
    return dcc.send_data_frame(filtered_df.to_csv, "llm_extractor_results.csv", index=False)


# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
