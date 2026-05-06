# FloodSense Lokoja

**A Python-based flood risk intelligence and decision support system for Lokoja, Kogi State, Nigeria.**

FloodSense is not just a flood map. It is a reproducible workflow that identifies flood-prone areas, validates modelled flood zones with Sentinel-1 SAR evidence, estimates exposed people and infrastructure, ranks intervention priorities, and presents results in an interactive WebGIS dashboard.

---

## Study area

- City: Lokoja, Kogi State, Nigeria
- Validation event: October 2022 Niger–Benue flood
- Projected CRS: EPSG:32632 (UTM Zone 32N)

---

## Project concept

| Stage | What it does |
|---|---|
| DEM analysis | Studies the shape of the land |
| Rainfall analysis | Estimates how much water enters the system |
| Flow accumulation | Identifies where water gathers |
| Flood susceptibility | Predicts likely flood-prone areas |
| Sentinel-1 validation | Checks predictions against satellite flood evidence |
| Exposure analysis | Counts who and what may be affected |
| Priority index | Decides where intervention should happen first |
| WebGIS dashboard | Presents results to decision-makers |

---

## Folder structure

```
config/
  config.yaml               Master configuration — paths, weights, CRS, scenarios

data/
  raw/                      Original downloaded datasets (boundary, dem, sentinel1, ...)
  interim/                  Clipped, reprojected, and cleaned datasets
  processed/                Analysis-ready rasters, vectors, tables, dashboard layers

notebooks/                  Portfolio Jupyter notebooks — one per analysis stage
scripts/                    Executable workflow scripts (00–07 + run_pipeline.py)
src/floodsense/             Reusable Python package (imported by scripts and notebooks)
dashboard/                  Streamlit + Folium WebGIS app
outputs/                    Maps, tables, validation results, reports
docs/                       Methodology, validation report, portfolio summary
tests/                      Unit tests for core module logic
```

---

## Installation

```bash
# Option A — pip (recommended for quick start)
pip install -e .

# Option B — conda environment
conda env create -f environment.yml
conda activate floodsense-lokoja
pip install -e .
```

---

## Data download (required first step)

Before running any analysis, place your boundary file in `data/raw/boundary/`.

Easiest option — geoBoundaries (no login needed):

```bash
curl -L "https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM2/" \
     -o data/raw/boundary/nga_adm2.geojson
```

See `scripts/00_download_data.py` for full instructions on all download options.

---

## Execution order

```bash
pip install -e .

# Stage 00: boundary setup and directory creation
python scripts/00_download_data.py

# Stage 01: clip, reproject, and clean all datasets
python scripts/01_prepare_data.py

# Stage 02: DEM terrain and hydrological analysis
python scripts/02_run_hydrology.py

# Stage 03: flood susceptibility model
python scripts/03_build_susceptibility_model.py

# Stage 04: Sentinel-1 SAR validation
python scripts/04_run_sar_validation.py

# Stage 05: exposure analysis
python scripts/05_run_exposure_analysis.py

# Stage 06: flood intervention priority index
python scripts/06_build_priority_index.py

# Stage 07: export WebGIS dashboard layers
python scripts/07_export_dashboard_layers.py

# Launch dashboard
streamlit run dashboard/app.py
```

Run all stages at once:

```bash
python scripts/run_pipeline.py --stage all
```

Run a single stage:

```bash
python scripts/run_pipeline.py --stage susceptibility
```

---

## Expected outputs

| File | Description |
|---|---|
| `data/processed/rasters/susceptibility_score.tif` | Raw flood susceptibility 0–1 score |
| `data/processed/rasters/susceptibility_class.tif` | Classified 1–5 risk map |
| `data/processed/rasters/observed_flood_extent.tif` | SAR-derived flood extent |
| `data/processed/vectors/high_risk_flood_zones.gpkg` | High + Very High risk polygons |
| `data/processed/vectors/exposed_buildings.gpkg` | Buildings in flood zones |
| `data/processed/vectors/exposed_roads.gpkg` | Roads in flood zones |
| `data/processed/tables/exposure_summary.csv` | Community-level exposure table |
| `outputs/validation/validation_metrics.csv` | SAR validation accuracy metrics |
| `data/processed/tables/priority_ranking.csv` | Ranked community priority table |
| `data/processed/dashboard_layers/*.geojson` | WebGIS-ready layers |

---

## Limitations

- The susceptibility model uses transparent weighted overlay, not hydraulic modelling.
- SAR flood extraction uses ratio/change detection — calibrate threshold for Lokoja conditions.
- Results depend on DEM quality, boundary accuracy, and input dataset resolution.
- Land-cover risk scores assume ESA WorldCover class scheme.

## Future improvements

- Return-period rainfall scenarios (10-year, 50-year, 100-year).
- Ward/community-level aggregation for administrative reporting.
- Critical infrastructure exposure (hospitals, schools, markets).
- Calibrated SAR threshold with permanent-water masking.
- Cloud-hosted WebGIS deployment.
