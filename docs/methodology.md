# Methodology

FloodSense Lokoja is organized as a reproducible flood-risk decision-support workflow for Lokoja, Kogi State, Nigeria.

## Workflow

1. Define the Lokoja study boundary and projected CRS.
2. Clean, clip, and reproject all input datasets.
3. Derive DEM-based terrain and hydrological indicators.
4. Build a weighted flood susceptibility raster.
5. Classify susceptibility into Very Low, Low, Moderate, High, and Very High risk classes.
6. Validate High and Very High zones against historical Sentinel-1 SAR flood evidence.
7. Intersect high-risk zones with population, buildings, and roads.
8. Build the Flood Intervention Priority Index from available exposure indicators.
9. Export WebGIS-ready GeoJSON layers and CSV tables.

The workflow is intentionally modular so each stage can be rerun independently.
