"""Data cleaning, clipping, reprojection, and alignment utilities."""

from __future__ import annotations

from pathlib import Path

from floodsense.paths import ensure_dir, first_existing_file, get_project_root
from floodsense.raster_utils import clip_raster_to_boundary, reproject_raster
from floodsense.vector_utils import (
    clean_geometries,
    clip_vector,
    dissolve_boundary,
    read_vector,
    reproject_vector,
    save_vector,
)


VECTOR_EXTENSIONS = [".gpkg", ".geojson", ".json", ".shp"]
RASTER_EXTENSIONS = [".tif", ".tiff"]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else get_project_root() / path


def prepare_boundary(
    raw_boundary_path: str | Path, output_gpkg: str | Path, output_geojson: str | Path, target_crs: str
) -> Path:
    """Clean, dissolve, reproject, and save the Lokoja boundary."""
    boundary = read_vector(raw_boundary_path)
    boundary = clean_geometries(boundary)
    boundary = dissolve_boundary(boundary)
    boundary = reproject_vector(boundary, target_crs)
    save_vector(boundary, _resolve(output_gpkg))
    save_vector(boundary, _resolve(output_geojson), driver="GeoJSON")
    print(f"[preprocessing] Boundary prepared: {output_gpkg}")
    return _resolve(output_gpkg)


def preprocess_vector_dataset(
    input_path: str | Path, boundary_path: str | Path, output_path: str | Path, target_crs: str
) -> Path:
    """Clean, reproject, clip, and save a vector dataset."""
    gdf = reproject_vector(clean_geometries(read_vector(input_path)), target_crs)
    boundary = read_vector(boundary_path)
    clipped = clip_vector(gdf, boundary)
    save_vector(clipped, _resolve(output_path))
    print(f"[preprocessing] Vector processed: {output_path}")
    return _resolve(output_path)


def preprocess_raster_dataset(
    input_path: str | Path,
    boundary_path: str | Path,
    clipped_output: str | Path,
    reprojected_output: str | Path,
    target_crs: str,
) -> Path:
    """Clip a raster to the boundary and reproject it."""
    boundary = read_vector(boundary_path)
    clipped = clip_raster_to_boundary(input_path, boundary, _resolve(clipped_output))
    reprojected = reproject_raster(clipped, _resolve(reprojected_output), target_crs)
    print(f"[preprocessing] Raster processed: {reprojected_output}")
    return reprojected


def run_preprocessing_workflow(config: dict) -> None:
    """Run the full preprocessing workflow."""
    paths = config["paths"]
    crs = config["project"].get("crs", "EPSG:32632")
    for folder in paths.values():
        ensure_dir(folder)

    boundary_input = first_existing_file(paths["raw_boundary"], VECTOR_EXTENSIONS)
    if boundary_input is None:
        raise FileNotFoundError(
            "Boundary is required. Place Lokoja boundary data in data/raw/boundary/."
        )

    boundary_gpkg = Path(paths["processed_vectors"]) / "lokoja_boundary.gpkg"
    boundary_geojson = Path(paths["dashboard_layers"]) / "lokoja_boundary.geojson"
    boundary_path = prepare_boundary(boundary_input, boundary_gpkg, boundary_geojson, crs)

    vector_inputs = {
        "buildings": paths.get("raw_buildings"),
        "roads": paths.get("raw_roads"),
        "rivers": paths.get("raw_rivers", "data/raw/rivers"),
    }
    for name, folder in vector_inputs.items():
        input_file = first_existing_file(folder, VECTOR_EXTENSIONS)
        if input_file is None:
            print(f"[preprocessing] Optional vector dataset missing: {name}. Skipping.")
            continue
        preprocess_vector_dataset(
            input_file,
            boundary_path,
            Path(paths["interim_cleaned"]) / f"{name}.gpkg",
            crs,
        )

    raster_inputs = {
        "dem": paths.get("raw_dem"),
        "rainfall": paths.get("raw_rainfall"),
        "landcover": paths.get("raw_landcover"),
        "population": paths.get("raw_population"),
    }
    for name, folder in raster_inputs.items():
        input_file = first_existing_file(folder, RASTER_EXTENSIONS)
        if input_file is None:
            print(f"[preprocessing] Optional raster dataset missing: {name}. Skipping.")
            continue
        preprocess_raster_dataset(
            input_file,
            boundary_path,
            Path(paths["interim_clipped"]) / f"{name}_clipped.tif",
            Path(paths["interim_reprojected"]) / f"{name}.tif",
            crs,
        )
