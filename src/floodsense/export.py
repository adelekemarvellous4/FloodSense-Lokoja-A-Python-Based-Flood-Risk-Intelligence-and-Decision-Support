"""Dashboard and reporting export utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd

from floodsense.paths import get_project_root
from floodsense.raster_utils import raster_to_polygon


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def export_geojson(gdf: gpd.GeoDataFrame, output_path: str | Path, simplify_tolerance: float | None = None) -> Path:
    """Export a GeoDataFrame as GeoJSON, optionally simplified."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = gdf.copy()
    if simplify_tolerance is not None and not output.empty:
        output["geometry"] = output.geometry.simplify(simplify_tolerance, preserve_topology=True)
    if output_path.exists():
        output_path.unlink()
    output.to_file(output_path, driver="GeoJSON")
    return output_path


def write_metadata(metadata_dict: dict[str, Any], output_path: str | Path) -> Path:
    """Write dashboard metadata JSON."""
    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return output_path


def _vector_to_geojson(source: Path, target: Path) -> None:
    if not source.exists():
        print(f"[export] Missing layer: {source.name}. Skipping.")
        return
    export_geojson(gpd.read_file(source), target)


def _copy_csv(source: Path, target: Path) -> None:
    if not source.exists():
        print(f"[export] Missing table: {source.name}. Skipping.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def export_dashboard_layers(config: dict) -> None:
    """Export dashboard-ready GeoJSON layers and CSV tables."""
    paths = config["paths"]
    dashboard = _resolve(paths["dashboard_layers"])
    dashboard.mkdir(parents=True, exist_ok=True)
    vectors = _resolve(paths["processed_vectors"])
    rasters = _resolve(paths["processed_rasters"])

    _vector_to_geojson(vectors / "lokoja_boundary.gpkg", dashboard / "lokoja_boundary.geojson")
    _vector_to_geojson(vectors / "high_risk_flood_zones.gpkg", dashboard / "flood_susceptibility_zones.geojson")
    _vector_to_geojson(vectors / "exposed_buildings.gpkg", dashboard / "exposed_buildings.geojson")
    _vector_to_geojson(vectors / "exposed_roads.gpkg", dashboard / "exposed_roads.geojson")
    _vector_to_geojson(vectors / "priority_zones.gpkg", dashboard / "priority_zones.geojson")

    observed = rasters / "observed_flood_extent.tif"
    if observed.exists():
        raster_to_polygon(observed, dashboard / "observed_flood_extent.geojson", target_values=[1])
    else:
        print("[export] observed_flood_extent.tif missing. Skipping observed flood GeoJSON.")

    output_dashboard = _resolve(paths.get("outputs_dashboard", "outputs/dashboard"))
    tables = _resolve(paths["processed_tables"])
    validation = _resolve(paths.get("outputs_validation", "outputs/validation"))
    outputs_tables = _resolve(paths.get("outputs_tables", "outputs/tables"))
    _copy_csv(tables / "exposure_summary.csv", output_dashboard / "exposure_summary.csv")
    _copy_csv(validation / "validation_metrics.csv", output_dashboard / "validation_metrics.csv")
    _copy_csv(tables / "priority_ranking.csv", output_dashboard / "priority_ranking.csv")
    _copy_csv(outputs_tables / "priority_ranking.csv", output_dashboard / "priority_ranking.csv")

    metadata = {
        "project": config.get("project", {}),
        "layers": sorted(path.name for path in dashboard.glob("*.geojson")),
        "tables": sorted(path.name for path in output_dashboard.glob("*.csv")),
    }
    write_metadata(metadata, dashboard / "metadata.json")
    print(f"[export] Dashboard layers written to: {dashboard}")
