# LLM Extractor Dashboard

Interactive Dash dashboard for exploring protein purification extraction results and clustering grid outputs from [llm_extractor](https://github.com/jinichlab/llm_extractor).

## Setup

```bash
conda env create -f environment_dashboard.yml
conda activate llm_dashboard
```

## Data

The `data/` directory uses symlinks pointing to the extraction pipeline outputs:

```
data/
├── proteins.json  →  llm_project/additional_code/test_uniprot_id.json
└── clustering/    →  llm_project/joint_clustering/join_clustering_grid_second
```

If you clone this on a new machine, update the symlinks to point to your local data paths:

```bash
ln -sf /path/to/proteins.json data/proteins.json
ln -sf /path/to/clustering_grid data/clustering
```

## Run

```bash
python app.py
```

Then open `http://localhost:8050` in your browser.

## Tabs

### Extraction Data
Searchable table of all extracted proteins (8 000+ rows). Filter by enzyme name, organism, or expression strain. Hover over truncated cells to see the full text.

### Clustering Explorer
Select a **model**, **min community size**, **threshold**, and **field** to view:
- **UMAP cluster plot** — 2D projection coloured by community cluster
- **Cluster distribution** — bar chart of cluster sizes

The metrics table below shows Silhouette and Davies-Bouldin scores for the selected parameters.

### Grid Metrics
Heatmap images summarising clustering quality (silhouette, Davies-Bouldin, n\_clusters) across all threshold/min combinations for a selected model.
