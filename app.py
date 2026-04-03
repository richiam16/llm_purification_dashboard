import json
import os
import re
import glob

import pandas as pd
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
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="LLM Extractor Dashboard",
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
def proteins_tab():
    col_defs = [{"name": c.replace("_", " ").title(), "id": c} for c in TABLE_FIELDS]

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

    return dbc.Container([
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
        ], className="mb-3"),
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
            columns=col_defs,
            data=df[TABLE_FIELDS].to_dict("records"),
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
                "fontWeight": "bold",
                "backgroundColor": "#f0f4f8",
                "borderBottom": "2px solid #dee2e6",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
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
            ]),
        ]),
        dbc.Row([dbc.Col(html.Div(id="plot-status", className="text-danger small mb-1"))]),
        dbc.Row([
            dbc.Col(
                html.Iframe(
                    id="cluster-iframe",
                    src="",
                    style={"width": "100%", "height": "820px", "border": "1px solid #dee2e6", "borderRadius": "4px"},
                ),
                width=12,
            ),
        ]),
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


def section(title, body_items):
    """Reusable card section for the README page."""
    return dbc.Card([
        dbc.CardHeader(html.H5(title, className="mb-0")),
        dbc.CardBody(body_items),
    ], className="mb-4")


def readme_tab():
    return dbc.Container([
        html.H3("Dashboard Guide", className="mt-3 mb-1"),
        html.P(
            "This dashboard explores protein purification conditions extracted from scientific "
            "literature using Large Language Models (LLMs). Use the tabs at the top to navigate "
            "between pages.",
            className="text-muted mb-4",
        ),

        # ── Extraction Data ──────────────────────────────────────────────────
        section("Extraction Data", [
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

        # ── Clustering Explorer ───────────────────────────────────────────────
        section("Clustering Explorer", [
            html.P(
                "Displays interactive clustering results from a grid search over embedding "
                "models, community-size thresholds, and similarity thresholds. Each "
                "combination produces per-field UMAP plots and cluster-size distributions.",
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
            ], flush=True, className="mb-3"),
            html.H6("Plot type", className="fw-bold mt-2"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Span("UMAP cluster plot  ", className="fw-semibold"),
                    "2D projection of all field values coloured by cluster membership. "
                    "Hover over points to see the original text, PMID, and cluster label. "
                    "Points labelled NOISE were not assigned to any cluster.",
                ]),
                dbc.ListGroupItem([
                    html.Span("Cluster distribution  ", className="fw-semibold"),
                    "Bar chart showing the number of entries per cluster for the selected field.",
                ]),
            ], flush=True, className="mb-3"),
            html.H6("Metrics table", className="fw-bold mt-2"),
            html.P(
                "Shows Silhouette score (cosine) and Davies-Bouldin index for every field "
                "under the selected model / min-size / threshold combination. "
                "Higher Silhouette (highlighted green above 0.6) indicates well-separated clusters. "
                "Lower Davies-Bouldin indicates more compact, better-separated clusters.",
                className="mb-0",
            ),
        ]),

        # ── Grid Metrics ──────────────────────────────────────────────────────
        section("Grid Metrics", [
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

    ], fluid=True, className="pt-3")


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="LLM Extractor — Results Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-3 rounded",
    ),
    dbc.Tabs([
        dbc.Tab(label="Extraction Data",    tab_id="tab-proteins"),
        dbc.Tab(label="Clustering Explorer",tab_id="tab-clustering"),
        dbc.Tab(label="Grid Metrics",       tab_id="tab-grid-metrics"),
        dbc.Tab(label="README",             tab_id="tab-readme"),
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
    elif tab == "tab-clustering":
        return clustering_tab()
    elif tab == "tab-grid-metrics":
        return grid_metrics_tab()
    return html.Div()


@app.callback(
    Output("protein-table", "data"),
    Output("protein-table", "tooltip_data"),
    Output("protein-count", "children"),
    Input("filter-group", "value"),
    Input("search-field", "value"),
    Input("search-value", "value"),
)
def filter_proteins(groups, search_field, search_value):
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

    records = filtered[TABLE_FIELDS].to_dict("records")
    tooltips = [
        {col: {"value": str(row.get(col, "")), "type": "markdown"} for col in TABLE_FIELDS}
        for row in records
    ]
    label = f"Showing {len(filtered):,} of {len(df):,} proteins"
    return records, tooltips, label


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
    ])


@app.callback(
    Output("cluster-iframe", "src"),
    Output("plot-status", "children"),
    Input("dd-model", "value"),
    Input("dd-min", "value"),
    Input("dd-threshold", "value"),
    Input("dd-field", "value"),
    Input("plot-type", "value"),
)
def update_cluster_plot(model, min_val, threshold, field, plot_type):
    if not all([model, min_val, threshold, field, plot_type]):
        return "", "Select all parameters above."

    filename = (
        f"t={threshold}_{field}_cluster.html"
        if plot_type == "cluster"
        else f"t={threshold}_clusters_distribution_{field}.html"
    )
    rel_path  = f"model={model}/min={min_val}/{filename}"
    full_path = os.path.join(GRID_BASE, f"model={model}", f"min={min_val}", filename)

    if not os.path.isfile(full_path):
        return "", f"File not found: {rel_path}"
    return f"/grid_files/{rel_path}", ""


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
