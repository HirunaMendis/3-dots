# Data Storm v7.0 — Latent Potential Pipeline

## Overview
This repository implements a Bronze → Silver → Gold lakehouse-style pipeline for the Data Storm v7.0 preliminary round.

The main script `datastorm_eda_pipeline.py`:
- ingests raw CSVs into `bronze/`
- applies reusable data quality checks and quarantines bad rows into `silver/rejected/`
- produces cleaned Silver data in `silver/`
- builds feature-engineered Gold data in `gold/`
- generates EDA plots in `plots/`
- writes the final prediction output to `output/`

## Repository Structure

- `datastorm_eda_pipeline.py` — main pipeline script
- `data/` — raw input CSV files
- `bronze/` — raw parquet copies of ingested raw files
- `silver/` — cleaned Silver-layer parquet datasets
- `silver/rejected/` — quarantined rejected records with failure reasons
- `gold/` — feature-engineered model-ready dataset(s)
- `plots/` — generated EDA figures
- `output/` — final prediction CSV

## Required Packages

Install the Python dependencies before running:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn requests tqdm
```

## Running the Pipeline

1. Place the raw input CSVs in the `data/` directory.
2. Run the pipeline:

```bash
python datastorm_eda_pipeline.py
```

3. After successful execution:
- cleaned Silver parquet files are available in `silver/`
- rejected bad records are in `silver/rejected/`
- the Gold feature store is written to `gold/outlet_gold.parquet`
- EDA plots are saved in `plots/`
- prediction CSV is saved in `output/`

## Output File

The final predictions are saved to:

```
output/3 dots_predictions.csv
```

The output contains:
- `row_id`
- `Outlet_ID`
- `Maximum_Monthly_Liters`

## Notes

- The script is currently designed to produce one heuristic latent potential model for January 2026.
- POI scraping is implemented but disabled by default. To enable it, set `SCRAPE_POI = True` in `datastorm_eda_pipeline.py`.
- If a sample submission file is present in `data/`, the script will filter output to that subset.

## Suggested Improvements

For a complete competition submission, add:
- `README.md` (this file)
- a PDF summary report documenting data forensics, POI acquisition, methodology, and GenAI usage
- a more modular code structure with separate scripts or notebooks for Bronze, Silver, Gold, and modeling
- explicit validation and documentation of the latent potential logic

## How the Pipeline Matches the Challenge

- Bronze layer: raw ingestion preserved in `bronze/`
- Silver layer: reusable data quality checks, cleaned records, and rejected quarantined data
- Gold layer: enriched feature store for modeling
- Deliverables: prediction CSV plus clear pipeline structure

## Contact

Update `TEAM_NAME` in `datastorm_eda_pipeline.py` before generating your final submission file.
