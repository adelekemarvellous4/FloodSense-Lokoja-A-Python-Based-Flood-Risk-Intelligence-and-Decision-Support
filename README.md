# FloodSense Lokoja

**FloodSense Lokoja: A Python-Based Flood Risk Intelligence and Decision Support System** is a portfolio-grade geospatial project for Lokoja, Kogi State, Nigeria.

This is not just a flood map. It is a reproducible flood-risk decision-support workflow that identifies flood-prone areas, validates modelled flood zones with Sentinel-1 SAR evidence, estimates exposed people and infrastructure, ranks intervention priorities, and exports WebGIS-ready layers.

## Study Area

- City: Lokoja
- State: Kogi
- Country: Nigeria
- Default projected CRS: EPSG:32632

## Project Concept

FloodSense works like a flood detective system:

- DEM analysis studies the shape of the land.
- Rainfall analysis estimates how much water enters the system.
- Flow accumulation identifies where water gathers.
- Flood susceptibility predicts likely flood-prone areas.
- Sentinel-1 validation checks predictions against satellite flood evidence.
- Exposure analysis counts who and what may be affected.
- Priority indexing decides where intervention should happen first.
- WebGIS dashboard presents results to decision-makers.

## Folder Structure

```text
data/raw/                 Raw input datasets
data/interim/             Clipped, cleaned, and reprojected datasets
data/processed/           Analysis-ready rasters, vectors, tables, dashboard layers
notebooks/                Portfolio-readable workflow notebooks
src/floodsense/           Reusable Python package
scripts/                  Thin executable workflow scripts
dashboard/                Streamlit/Folium WebGIS dashboard
outputs/                  Maps, tables, validation outputs, dashboard exports
docs/                     Methodology and portfolio documentation
tests/                    Unit tests for reusable logic
```

## Expected Raw Data Placement

- Boundary files: `data/raw/boundary/`
- DEM files: `data/raw/dem/`
- Rainfall files: `data/raw/rainfall/`
- Sentinel-1 files: `data/raw/sentinel1/`
- Landcover files: `data/raw/landcover/`
- Building files: `data/raw/buildings/`
- Road files: `data/raw/roads/`
- Population files: `data/raw/population/`

Boundary data is required. Other datasets are optional by stage; missing optional data is reported clearly and skipped.

## Installation

```bash
pip install -e .
```

Or create the Conda environment:

```bash
conda env create -f environment.yml
conda activate floodsense-lokoja
pip install -e .
```

## Execution Order

```bash
pip install -e .

python scripts/01_prepare_data.py
python scripts/02_run_hydrology.py
python scripts/03_build_susceptibility_model.py
python scripts/04_run_sar_validation.py
python scripts/05_run_exposure_analysis.py
python scripts/06_build_priority_index.py
python scripts/07_export_dashboard_layers.py
streamlit run dashboard/app.py
```

Full pipeline option:

```bash
python scripts/run_pipeline.py --stage all
```

Run one stage:

```bash
python scripts/run_pipeline.py --stage susceptibility
```

## Expected Outputs

- `data/processed/rasters/susceptibility_score.tif`
- `data/processed/rasters/susceptibility_class.tif`
- `data/processed/rasters/observed_flood_extent.tif`
- `data/processed/vectors/high_risk_flood_zones.gpkg`
- `data/processed/vectors/exposed_buildings.gpkg`
- `data/processed/vectors/exposed_roads.gpkg`
- `data/processed/tables/exposure_summary.csv`
- `outputs/validation/validation_metrics.csv`
- `data/processed/tables/priority_ranking.csv`
- `data/processed/dashboard_layers/*.geojson`

## Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard loads exported GeoJSON layers and CSV tables when available. Missing layers are handled gracefully.

## Limitations

- The first susceptibility model is a transparent weighted overlay, not a full hydraulic model.
- Sentinel-1 flood extraction uses starter ratio/change detection and should be calibrated with local flood dates.
- Land-cover risk interpretation depends on the class scheme of the input land-cover product.
- Flood results depend heavily on input data quality, DEM resolution, and boundary accuracy.

## Future Improvements

- Calibrated SAR thresholding with permanent-water masking.
- Rainfall return-period scenarios for 10-year, 50-year, and 100-year events.
- Ward/community-level aggregation.
- Critical infrastructure exposure.
- Model sensitivity analysis and weight calibration.
- Cloud-hosted WebGIS deployment.
