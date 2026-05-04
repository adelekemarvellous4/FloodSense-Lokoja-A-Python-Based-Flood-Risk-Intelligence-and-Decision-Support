# Exposure Analysis

Exposure analysis estimates who and what may be affected by high-risk flood zones.

## Process

1. Convert High and Very High susceptibility classes to flood-zone polygons.
2. Intersect flood zones with building footprints.
3. Intersect flood zones with roads and calculate affected length in kilometers.
4. Sum population raster cells inside flood-zone polygons.
5. Save vector outputs and summary tables.

Outputs include `exposed_buildings.gpkg`, `exposed_roads.gpkg`, `population_exposure.csv`, and `exposure_summary.csv`.
