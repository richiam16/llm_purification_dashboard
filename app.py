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
JSON_PATH = os.path.join(BASE_DIR, "data", "proteins.json")
GRID_BASE = os.path.join(BASE_DIR, "data", "clustering")

# ---------------------------------------------------------------------------
# Load protein data
# ---------------------------------------------------------------------------
def load_protein_df():
    with open(JSON_PATH) as f:
        raw = json.load(f)
    rows = []
    for pmid, entry in raw.items():
        for protein in entry.get("proteins", []):
            row = {"pmid": pmid}
            row.update({k: (v if v is not None else "") for k, v in protein.items()})
            rows.append(row)
    return pd.DataFrame(rows)

df = load_protein_df()

PROTEIN_FIELDS = [
    "pmid", "enzyme_name", "organism_source", "strain", "expression_strain",
    "plasmid", "molecular_weight", "medium_name", "inducer",
    "induction_temperature", "lysis_buffer", "elution_buffer", "desalting_process",
]

# ---------------------------------------------------------------------------
# Scan grid directory structure
# ---------------------------------------------------------------------------
def scan_grid():
    models, mins, thresholds, fields = set(), set(), set(), set()
    pattern = os.path.join(GRID_BASE, "model=*", "min=*", "t=*_cluster.html")
    for fpath in glob.glob(pattern):
        parts = fpath.split(os.sep)
        # extract model and min from parent dirs
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

# Metric heatmap image names
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
    """Serve clustering HTML files and images for iframe / img embedding."""
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


def proteins_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Search enzyme name", className="fw-semibold small mb-1"),
                dbc.Input(id="filter-enzyme", placeholder="e.g. Azoreductase", debounce=True, size="sm"),
            ], width=4),
            dbc.Col([
                html.Label("Search organism", className="fw-semibold small mb-1"),
                dbc.Input(id="filter-organism", placeholder="e.g. Escherichia coli", debounce=True, size="sm"),
            ], width=4),
            dbc.Col([
                html.Label("Search expression strain", className="fw-semibold small mb-1"),
                dbc.Input(id="filter-strain", placeholder="e.g. BL21", debounce=True, size="sm"),
            ], width=4),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(html.Div(id="protein-count", className="text-muted small"), width=12),
        ], className="mb-2"),
        dash_table.DataTable(
            id="protein-table",
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in PROTEIN_FIELDS],
            data=df[PROTEIN_FIELDS].to_dict("records"),
            page_size=25,
            page_action="native",
            sort_action="native",
            filter_action="none",  # we handle filtering manually
            style_table={"overflowX": "auto"},
            style_cell={
                "fontSize": "12px",
                "padding": "6px 10px",
                "textAlign": "left",
                "maxWidth": "280px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f8f9fa",
                "borderBottom": "2px solid #dee2e6",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            ],
            tooltip_data=[
                {col: {"value": str(row[col]), "type": "markdown"} for col in PROTEIN_FIELDS}
                for row in df[PROTEIN_FIELDS].to_dict("records")
            ],
            tooltip_delay=0,
            tooltip_duration=None,
        ),
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
        dbc.Row([
            dbc.Col(html.Div(id="plot-status", className="text-danger small mb-1")),
        ]),
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
        dbc.Row([
            dbc.Col(html.Div(id="metrics-table-container"), width=12),
        ]),
    ], fluid=True, className="pt-3")


def grid_metrics_tab():
    # Build short model name → filename mapping from what exists
    metrics_dir = os.path.join(GRID_BASE, "metrics")
    available = []
    if os.path.isdir(metrics_dir):
        for f in sorted(os.listdir(metrics_dir)):
            m = re.match(r"(silhouette|davies_bouldin|n_clusters)_heatmap__(.+)\.png$", f)
            if m:
                available.append((m.group(1), m.group(2), f))

    model_shorts = sorted(set(x[1] for x in available))

    return dbc.Container([
        dbc.Row([
            make_dropdown("Model", "dd-metrics-model", model_shorts,
                          value=model_shorts[0] if model_shorts else None),
            make_dropdown("Metric", "dd-metrics-type", METRIC_TYPES, value="silhouette"),
        ], className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(html.Div(id="metrics-image-container"), width=12),
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
        dbc.Tab(label="Extraction Data", tab_id="tab-proteins"),
        dbc.Tab(label="Clustering Explorer", tab_id="tab-clustering"),
        dbc.Tab(label="Grid Metrics", tab_id="tab-grid-metrics"),
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
    elif tab == "tab-clustering":
        return clustering_tab()
    elif tab == "tab-grid-metrics":
        return grid_metrics_tab()
    return html.Div()


@app.callback(
    Output("protein-table", "data"),
    Output("protein-count", "children"),
    Input("filter-enzyme", "value"),
    Input("filter-organism", "value"),
    Input("filter-strain", "value"),
)
def filter_proteins(enzyme, organism, strain):
    filtered = df.copy()
    if enzyme:
        filtered = filtered[filtered["enzyme_name"].str.contains(enzyme, case=False, na=False)]
    if organism:
        filtered = filtered[filtered["organism_source"].str.contains(organism, case=False, na=False)]
    if strain:
        filtered = filtered[filtered["expression_strain"].str.contains(strain, case=False, na=False)]
    count = len(filtered)
    label = f"Showing {count:,} of {len(df):,} proteins"
    return filtered[PROTEIN_FIELDS].to_dict("records"), label


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

    if plot_type == "cluster":
        filename = f"t={threshold}_{field}_cluster.html"
    else:
        filename = f"t={threshold}_clusters_distribution_{field}.html"

    rel_path = f"model={model}/min={min_val}/{filename}"
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

    csv_path = os.path.join(GRID_BASE, f"model={model}", f"min={min_val}", f"t={threshold}_FIELD_CLUSTER_METRICS.csv")
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

    filename = f"{metric_type}_heatmap__{model_short}.png"
    rel_path = f"metrics/{filename}"
    full_path = os.path.join(GRID_BASE, "metrics", filename)

    if not os.path.isfile(full_path):
        return html.Div(f"Image not found: {filename}", className="text-muted small")

    return html.Img(
        src=f"/grid_files/{rel_path}",
        style={"maxWidth": "100%", "border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "8px"},
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
