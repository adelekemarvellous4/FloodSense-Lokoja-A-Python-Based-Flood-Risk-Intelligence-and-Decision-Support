# Dashboard Guide

Run the dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard reads GeoJSON layers from `data/processed/dashboard_layers/` and CSV tables from `outputs/dashboard/`, `data/processed/tables/`, and `outputs/validation/`.

Expected dashboard layers:

- `lokoja_boundary.geojson`
- `flood_susceptibility_zones.geojson`
- `observed_flood_extent.geojson`
- `exposed_buildings.geojson`
- `exposed_roads.geojson`
- `priority_zones.geojson`

Missing layers are shown as unavailable instead of causing the dashboard to fail.
