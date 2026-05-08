"""Exposure analysis for FloodSense Lokoja."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from floodsense.mapping import plot_overlay_preview
from floodsense.paths import ensure_dir, first_existing_file, get_project_root
from floodsense.raster_utils import raster_to_polygon
from floodsense.vector_utils import (
    calculate_line_lengths_km, clean_geometries, read_vector, save_vector
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


# ── OSM auto-download ─────────────────────────────────────────────────────────

def _download_osm_buildings(boundary_gdf: gpd.GeoDataFrame, out_path: Path) -> Path | None:
    """Download building footprints from OpenStreetMap via osmnx."""
    try:
        import osmnx as ox
        print("  [exposure] Downloading buildings from OpenStreetMap...")
        boundary_geo = boundary_gdf.to_crs("EPSG:4326")
        poly = boundary_geo.geometry.unary_union
        gdf  = ox.features_from_polygon(poly, tags={"building": True})
        gdf  = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf  = gdf[["geometry"]].reset_index(drop=True)
        gdf  = gdf.to_crs(boundary_gdf.crs)
        gdf  = clean_geometries(gdf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GPKG")
        print(f"  [exposure] Buildings downloaded: {len(gdf):,} footprints → {out_path.name}")
        return out_path
    except ImportError:
        print("  [exposure] osmnx not installed — pip install osmnx")
    except Exception as exc:
        print(f"  [exposure] OSM buildings download failed: {exc}")
    return None


def _download_osm_roads(boundary_gdf: gpd.GeoDataFrame, out_path: Path) -> Path | None:
    """Download road network from OpenStreetMap via osmnx."""
    try:
        import osmnx as ox
        print("  [exposure] Downloading roads from OpenStreetMap...")
        boundary_geo = boundary_gdf.to_crs("EPSG:4326")
        poly = boundary_geo.geometry.unary_union
        G    = ox.graph_from_polygon(poly, network_type="drive")
        _, edges = ox.graph_to_gdfs(G)
        edges = edges[["highway", "name", "length", "geometry"]].copy()
        edges = edges.to_crs(boundary_gdf.crs)
        total_km = edges["length"].sum() / 1000
        out_path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_file(out_path, driver="GPKG")
        print(f"  [exposure] Roads downloaded: {len(edges):,} segments "
              f"({total_km:.1f} km total) → {out_path.name}")
        return out_path
    except ImportError:
        print("  [exposure] osmnx not installed — pip install osmnx")
    except Exception as exc:
        print(f"  [exposure] OSM roads download failed: {exc}")
    return None


# ── Flood zone extraction ─────────────────────────────────────────────────────

def extract_high_risk_flood_zones(
    classified_raster_path: str | Path,
    output_vector_path: str | Path,
) -> Path:
    """
    Convert High (class 4) and Very High (class 5) susceptibility cells
    to vector polygons. These polygons are the flood exposure mask used
    in all downstream intersection operations.
    """
    out = _resolve(output_vector_path)
    raster_to_polygon(classified_raster_path, out, target_values=[4, 5])
    with rasterio.open(classified_raster_path) as src:
        data     = src.read(1)
        res      = src.res
        px_km2   = abs(res[0] * res[1]) / 1_000_000
        n_pixels = int(np.isin(data, [4, 5]).sum())
    area_km2 = n_pixels * px_km2
    print(f"  [exposure] High/Very High flood zone: {n_pixels:,} pixels  "
          f"({area_km2:.1f} km²) → {out.name}")
    return out


# ── Building exposure ─────────────────────────────────────────────────────────

def calculate_building_exposure(
    flood_zones_path: str | Path,
    buildings_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Intersect building footprints with high-risk flood zones.

    Each building that overlaps any High or Very High risk polygon
    is counted as exposed. Partial overlaps are included — even a
    building that is 10% within the flood zone is at risk.
    """
    flood     = read_vector(flood_zones_path)
    buildings = read_vector(buildings_path)
    if str(buildings.crs) != str(flood.crs):
        buildings = buildings.to_crs(flood.crs)
    buildings = clean_geometries(buildings)

    exposed = gpd.overlay(buildings, flood[["geometry"]], how="intersection")
    exposed = exposed[exposed.geometry.is_valid & ~exposed.geometry.is_empty].copy()
    exposed = exposed.reset_index(drop=True)

    out = _resolve(output_path)
    save_vector(exposed, out)
    print(f"  [exposure] Exposed buildings: {len(exposed):,} → {out.name}")
    return out


# ── Road exposure ─────────────────────────────────────────────────────────────

def calculate_road_exposure(
    flood_zones_path: str | Path,
    roads_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Clip road network to high-risk flood zones and calculate affected length.

    Road segments are split at flood zone boundaries so partial segments
    are measured accurately. Length is reported in km.
    """
    flood = read_vector(flood_zones_path)
    roads = read_vector(roads_path)
    if str(roads.crs) != str(flood.crs):
        roads = roads.to_crs(flood.crs)
    roads = clean_geometries(roads)

    exposed = gpd.overlay(roads, flood[["geometry"]], how="intersection")
    exposed = exposed[exposed.geometry.is_valid & ~exposed.geometry.is_empty].copy()
    exposed = calculate_line_lengths_km(exposed)
    exposed = exposed.reset_index(drop=True)

    total_km = float(exposed["length_km"].sum()) if "length_km" in exposed.columns else 0.0

    out = _resolve(output_path)
    save_vector(exposed, out)
    print(f"  [exposure] Affected roads: {len(exposed):,} segments  "
          f"({total_km:.1f} km) → {out.name}")
    return out


# ── Population exposure ───────────────────────────────────────────────────────

def calculate_population_exposure(
    flood_zones_path: str | Path,
    population_raster_path: str | Path,
    output_csv: str | Path,
) -> Path:
    """
    Estimate exposed population using zonal statistics.

    WorldPop raster values represent people per pixel. Summing all
    population pixels whose centroids fall within High/Very High
    flood zones gives the estimated exposed population count.
    """
    flood = read_vector(flood_zones_path)

    with rasterio.open(population_raster_path) as src:
        population = src.read(1).astype("float32")
        nodata     = src.nodata
        if nodata is not None:
            population[population == nodata] = np.nan

        flood_reproj = flood.to_crs(src.crs)
        flood_mask = geometry_mask(
            [geom for geom in flood_reproj.geometry if geom is not None],
            out_shape=population.shape,
            transform=src.transform,
            invert=True,
        )

    exposed_pop = float(np.nansum(population[flood_mask]))
    valid_cells = int(np.sum(flood_mask & np.isfinite(population)))

    out = _resolve(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "area":               "high_risk_flood_zones",
        "exposed_population": round(exposed_pop),
        "valid_population_cells": valid_cells,
    }]).to_csv(out, index=False)

    print(f"  [exposure] Exposed population: {exposed_pop:,.0f} people "
          f"({valid_cells:,} population cells)")
    return out


# ── Exposure summary ──────────────────────────────────────────────────────────

def summarize_exposure(
    buildings_path: str | Path | None,
    roads_path:     str | Path | None,
    population_csv: str | Path | None,
    output_csv:     str | Path,
) -> Path:
    """Compile all exposure outputs into one summary CSV."""
    row: dict = {"area": "Lokoja LGA — High/Very High flood zones"}

    if buildings_path and Path(buildings_path).exists():
        n = len(read_vector(buildings_path))
        row["exposed_buildings"] = n
    else:
        row["exposed_buildings"] = "not available"

    if roads_path and Path(roads_path).exists():
        roads = read_vector(roads_path)
        km = float(roads["length_km"].sum()) if "length_km" in roads.columns else 0.0
        row["affected_road_length_km"] = round(km, 2)
    else:
        row["affected_road_length_km"] = "not available"

    if population_csv and Path(population_csv).exists():
        pop_df = pd.read_csv(population_csv)
        if "exposed_population" in pop_df.columns:
            row["exposed_population"] = int(pop_df["exposed_population"].iloc[0])
    else:
        row["exposed_population"] = "not available"

    out = _resolve(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    return out


def print_exposure_summary(
    buildings_path, roads_path, population_csv, flood_zones_path
) -> None:
    """Print a clean exposure summary table."""
    print("\n" + "─" * 55)
    print("  Exposure analysis — Lokoja LGA")
    print("  Flood zone: High + Very High susceptibility classes")
    print("─" * 55)

    # Flood zone area
    if flood_zones_path and Path(flood_zones_path).exists():
        gdf = read_vector(flood_zones_path)
        if gdf.crs and gdf.crs.is_projected:
            area = gdf.geometry.area.sum() / 1_000_000
        else:
            area = gdf.to_crs("EPSG:32632").geometry.area.sum() / 1_000_000
        print(f"  Flood zone area        : {area:>10.1f} km²")

    # Buildings
    if buildings_path and Path(buildings_path).exists():
        n = len(read_vector(buildings_path))
        print(f"  Exposed buildings      : {n:>10,}")
    else:
        print(f"  Exposed buildings      : {'not available':>10}")

    # Roads
    if roads_path and Path(roads_path).exists():
        roads = read_vector(roads_path)
        km = float(roads["length_km"].sum()) if "length_km" in roads.columns else 0.0
        print(f"  Affected road length   : {km:>10.1f} km")
    else:
        print(f"  Affected road length   : {'not available':>10}")

    # Population
    if population_csv and Path(population_csv).exists():
        pop = pd.read_csv(population_csv)
        if "exposed_population" in pop.columns:
            n = int(pop["exposed_population"].iloc[0])
            print(f"  Exposed population     : {n:>10,} people")
    else:
        print(f"  Exposed population     : {'not available':>10}")

    print("─" * 55)
    print("  Note: Based on High (class 4) + Very High (class 5)")
    print("  susceptibility zones from the weighted overlay model.")


# ── Main workflow ─────────────────────────────────────────────────────────────

def run_exposure_workflow(config: dict) -> None:
    """
    Run flood exposure analysis for Lokoja LGA.

    Stages:
      1. Extract High/Very High flood zone polygons
      2. Download buildings and roads from OSM if not already present
      3. Intersect buildings with flood zones
      4. Clip roads to flood zones, calculate affected length
      5. Estimate exposed population using WorldPop raster
      6. Compile exposure summary table
      7. Save preview map and summary
    """
    print("\n[exposure] Starting exposure analysis")

    paths   = config["paths"]
    rasters = ensure_dir(_resolve(paths["processed_rasters"]))
    vectors = ensure_dir(_resolve(paths["processed_vectors"]))
    tables  = ensure_dir(_resolve(paths["processed_tables"]))
    cleaned = ensure_dir(_resolve(paths["interim_cleaned"]))
    maps    = ensure_dir(_resolve(paths["outputs_maps"]))

    classified = rasters / "susceptibility_class.tif"
    if not classified.exists():
        print("[exposure] susceptibility_class.tif not found.")
        print("  Run script 03 first.")
        return

    # Stage 1 — flood zone polygons
    print("\n[exposure] Stage 1/5 — Extracting high-risk flood zones")
    flood_zones = vectors / "high_risk_flood_zones.gpkg"
    extract_high_risk_flood_zones(classified, flood_zones)

    # Load boundary for OSM downloads
    boundary_path = vectors / "lokoja_boundary.gpkg"
    if not boundary_path.exists():
        boundary_path = _resolve(paths["interim_cleaned"]) / "lokoja_boundary.gpkg"
    boundary = read_vector(boundary_path) if boundary_path.exists() else None

    # Stage 2 — buildings
    print("\n[exposure] Stage 2/5 — Building exposure")
    buildings_raw = cleaned / "buildings.gpkg"
    if not buildings_raw.exists() and boundary is not None:
        _download_osm_buildings(boundary, buildings_raw)

    buildings_out = None
    if buildings_raw.exists():
        buildings_out = calculate_building_exposure(
            flood_zones, buildings_raw, vectors / "exposed_buildings.gpkg"
        )
    else:
        print("  [exposure] Buildings not available — skipping")

    # Stage 3 — roads
    print("\n[exposure] Stage 3/5 — Road exposure")
    roads_raw = cleaned / "roads.gpkg"
    if not roads_raw.exists() and boundary is not None:
        _download_osm_roads(boundary, roads_raw)

    roads_out = None
    if roads_raw.exists():
        roads_out = calculate_road_exposure(
            flood_zones, roads_raw, vectors / "exposed_roads.gpkg"
        )
    else:
        print("  [exposure] Roads not available — skipping")

    # Stage 4 — population
    print("\n[exposure] Stage 4/5 — Population exposure")
    population = _resolve(paths["interim_reprojected"]) / "population.tif"
    population_out = None
    if population.exists():
        population_out = calculate_population_exposure(
            flood_zones, population, tables / "population_exposure.csv"
        )
    else:
        print("  [exposure] Population raster not found — skipping")

    # Stage 5 — summary
    print("\n[exposure] Stage 5/5 — Compiling exposure summary")
    summary_path = tables / "exposure_summary.csv"
    summarize_exposure(buildings_out, roads_out, population_out, summary_path)
    print(f"  Exposure summary saved: {summary_path.name}")

    # Copy summary to outputs/tables too
    out_tables = ensure_dir(_resolve(paths.get("outputs_tables", "outputs/tables")))
    import shutil
    shutil.copy2(summary_path, out_tables / "exposure_summary.csv")

    # Preview map
    if boundary_path.exists():
        plot_overlay_preview(
            boundary_path, flood_zones,
            maps / "exposure_flood_zones_preview.png",
            "Lokoja — High and Very High flood susceptibility zones"
        )

    print_exposure_summary(buildings_out, roads_out, population_out, flood_zones)

    print("\n[exposure] Complete")
    print(f"  Flood zones : {flood_zones}")
    print(f"  Summary     : {summary_path}")
    print("\n  Next step: python scripts/06_build_priority_index.py")