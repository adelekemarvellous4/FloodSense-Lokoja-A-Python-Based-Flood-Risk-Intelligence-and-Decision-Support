"""Reusable vector processing utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    return gpd.read_file(path)


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empty geometries and repair invalid geometries where possible."""
    if gdf.empty:
        return gdf.copy()
    cleaned = gdf.copy()
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty].copy()
    if cleaned.empty:
        return cleaned
    cleaned["geometry"] = cleaned.geometry.make_valid()
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty].copy()
    return cleaned.reset_index(drop=True)


def reproject_vector(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to a target CRS."""
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS. Define its CRS before reprojection.")
    return gdf.to_crs(target_crs)


def dissolve_boundary(gdf: gpd.GeoDataFrame, dissolve_field: str | None = None) -> gpd.GeoDataFrame:
    """Dissolve boundary features into one or grouped polygons."""
    if gdf.empty:
        return gdf.copy()
    if dissolve_field and dissolve_field in gdf.columns:
        dissolved = gdf.dissolve(by=dissolve_field).reset_index()
    else:
        dissolved = gdf.dissolve().reset_index(drop=True)
    return clean_geometries(dissolved)


def clip_vector(gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip vector features to a boundary."""
    if gdf.empty or boundary_gdf.empty:
        return gdf.iloc[0:0].copy()
    if gdf.crs != boundary_gdf.crs:
        gdf = reproject_vector(gdf, str(boundary_gdf.crs))
    return clean_geometries(gpd.clip(gdf, boundary_gdf))


def save_vector(gdf: gpd.GeoDataFrame, output_path: str | Path, driver: str | None = None) -> Path:
    """Save a vector dataset as GPKG, GeoJSON, or another supported driver."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if driver is None:
        suffix = output_path.suffix.lower()
        driver = "GeoJSON" if suffix in {".geojson", ".json"} else "GPKG"
    if output_path.exists():
        output_path.unlink()
    gdf.to_file(output_path, driver=driver)
    return output_path


def calculate_line_lengths_km(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a length_km field to line features."""
    result = gdf.copy()
    if result.empty:
        result["length_km"] = []
        return result
    if result.crs is None or not result.crs.is_projected:
        raise ValueError("Line length calculation requires a projected CRS.")
    result["length_km"] = result.geometry.length / 1000.0
    return result
