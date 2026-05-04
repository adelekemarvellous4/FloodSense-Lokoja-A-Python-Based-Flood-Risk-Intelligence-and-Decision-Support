"""Exposure analysis functions for population, buildings, roads, and infrastructure."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from floodsense.mapping import plot_overlay_preview
from floodsense.paths import first_existing_file, get_project_root
from floodsense.raster_utils import raster_to_polygon
from floodsense.vector_utils import calculate_line_lengths_km, read_vector, save_vector


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def extract_high_risk_flood_zones(classified_raster_path: str | Path, output_vector_path: str | Path) -> Path:
    """Convert High and Very High susceptibility classes to polygons."""
    return raster_to_polygon(classified_raster_path, _resolve(output_vector_path), target_values=[4, 5])


def calculate_building_exposure(
    flood_zones_path: str | Path, buildings_path: str | Path, output_path: str | Path
) -> Path:
    """Intersect building footprints with high-risk flood zones."""
    flood = read_vector(flood_zones_path)
    buildings = read_vector(buildings_path)
    if buildings.crs != flood.crs:
        buildings = buildings.to_crs(flood.crs)
    exposed = gpd.overlay(buildings, flood[["geometry"]], how="intersection")
    save_vector(exposed, _resolve(output_path))
    return _resolve(output_path)


def calculate_road_exposure(flood_zones_path: str | Path, roads_path: str | Path, output_path: str | Path) -> Path:
    """Intersect roads with high-risk flood zones and calculate affected length."""
    flood = read_vector(flood_zones_path)
    roads = read_vector(roads_path)
    if roads.crs != flood.crs:
        roads = roads.to_crs(flood.crs)
    exposed = gpd.overlay(roads, flood[["geometry"]], how="intersection")
    exposed = calculate_line_lengths_km(exposed)
    save_vector(exposed, _resolve(output_path))
    return _resolve(output_path)


def calculate_population_exposure(
    flood_mask_path: str | Path, population_raster_path: str | Path, output_csv: str | Path
) -> Path:
    """Estimate exposed population by summing population cells inside high-risk flood polygons."""
    flood = read_vector(flood_mask_path)
    with rasterio.open(population_raster_path) as src:
        population = src.read(1).astype("float32")
        if src.nodata is not None:
            population[population == src.nodata] = np.nan
        flood = flood.to_crs(src.crs)
        mask = geometry_mask(
            [geom for geom in flood.geometry if geom is not None],
            out_shape=population.shape,
            transform=src.transform,
            invert=True,
        )
    exposed_population = float(np.nansum(population[mask]))
    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"area": "high_risk_flood_zones", "exposed_population": exposed_population}]).to_csv(
        output_csv, index=False
    )
    return output_csv


def summarize_exposure(
    buildings_path: str | Path | None,
    roads_path: str | Path | None,
    population_csv: str | Path | None,
    output_csv: str | Path,
) -> Path:
    """Create an exposure summary table from available exposure outputs."""
    row = {"area": "high_risk_flood_zones"}
    if buildings_path and Path(buildings_path).exists():
        row["exposed_buildings"] = len(read_vector(buildings_path))
    if roads_path and Path(roads_path).exists():
        roads = read_vector(roads_path)
        row["affected_road_length_km"] = float(roads.get("length_km", pd.Series(dtype=float)).sum())
    if population_csv and Path(population_csv).exists():
        population = pd.read_csv(population_csv)
        if "exposed_population" in population.columns:
            row["exposed_population"] = float(population["exposed_population"].sum())
    output_csv = _resolve(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_csv, index=False)
    return output_csv


def run_exposure_workflow(config: dict) -> None:
    """Run flood exposure analysis."""
    paths = config["paths"]
    rasters = _resolve(paths["processed_rasters"])
    vectors = _resolve(paths["processed_vectors"])
    tables = _resolve(paths["processed_tables"])
    maps = _resolve(paths["outputs_maps"])
    classified = rasters / "susceptibility_class.tif"
    if not classified.exists():
        print("[exposure] Missing susceptibility_class.tif. Run susceptibility workflow first.")
        return
    flood_zones = extract_high_risk_flood_zones(classified, vectors / "high_risk_flood_zones.gpkg")
    buildings_out = None
    roads_out = None
    population_out = None
    buildings = _resolve(paths["interim_cleaned"]) / "buildings.gpkg"
    if buildings.exists():
        buildings_out = calculate_building_exposure(
            flood_zones, buildings, vectors / "exposed_buildings.gpkg"
        )
    else:
        print("[exposure] Buildings data missing. Skipping building exposure.")
    roads = _resolve(paths["interim_cleaned"]) / "roads.gpkg"
    if roads.exists():
        roads_out = calculate_road_exposure(flood_zones, roads, vectors / "exposed_roads.gpkg")
    else:
        print("[exposure] Roads data missing. Skipping road exposure.")
    population = _resolve(paths["interim_reprojected"]) / "population.tif"
    if population.exists():
        population_out = calculate_population_exposure(
            flood_zones, population, tables / "population_exposure.csv"
        )
    else:
        print("[exposure] Population raster missing. Skipping population exposure.")
    summary = summarize_exposure(
        buildings_out, roads_out, population_out, tables / "exposure_summary.csv"
    )
    boundary = vectors / "lokoja_boundary.gpkg"
    if boundary.exists():
        plot_overlay_preview(boundary, flood_zones, maps / "exposure_preview.png", "High-Risk Flood Zones")
    print(f"[exposure] Exposure summary written: {summary}")
