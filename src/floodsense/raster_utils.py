"""Reusable raster processing utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import shape


def get_raster_profile(path: str | Path) -> dict:
    """Return a raster profile."""
    with rasterio.open(path) as src:
        return src.profile.copy()


def clip_raster_to_boundary(
    raster_path: str | Path, boundary_gdf: gpd.GeoDataFrame, output_path: str | Path
) -> Path:
    """Clip a raster to a boundary GeoDataFrame."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        boundary = boundary_gdf.to_crs(src.crs) if boundary_gdf.crs != src.crs else boundary_gdf
        geometries = [geom.__geo_interface__ for geom in boundary.geometry if geom is not None]
        data, transform = mask(src, geometries, crop=True)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
    return output_path


def reproject_raster(
    input_path: str | Path, output_path: str | Path, target_crs: str, resolution: float | None = None
) -> Path:
    """Reproject a raster to a target CRS."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform, width=width, height=height)
        with rasterio.open(output_path, "w", **profile) as dst:
            for index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, index),
                    destination=rasterio.band(dst, index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                )
    return output_path


def align_raster_to_reference(
    input_path: str | Path, reference_path: str | Path, output_path: str | Path
) -> Path:
    """Align a raster to the grid, CRS, transform, and size of a reference raster."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref, rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            count=1,
        )
        destination = np.full((ref.height, ref.width), src.nodata or 0, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest,
        )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination, 1)
    return output_path


def normalize_array(array: np.ndarray, inverse: bool = False, nodata: float | int | None = None) -> np.ndarray:
    """Normalize numeric raster values to 0-1 while preserving nodata as NaN."""
    arr = array.astype("float32", copy=True)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype="float32")
    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)
    if np.isclose(max_value, min_value):
        normalized = np.zeros(arr.shape, dtype="float32")
        normalized[~finite] = np.nan
    else:
        normalized = ((arr - min_value) / (max_value - min_value)).astype("float32")
    if inverse:
        normalized = 1.0 - normalized
        normalized[~finite] = np.nan
    return normalized


def classify_array(
    array: np.ndarray, bins: list[float] | np.ndarray, labels: list[int] | np.ndarray | None = None
) -> np.ndarray:
    """Classify an array using bin edges."""
    arr = array.astype("float32", copy=False)
    classes = np.digitize(arr, bins, right=True) + 1
    classes[~np.isfinite(arr)] = 0
    if labels is not None:
        label_array = np.asarray(labels)
        mapped = np.zeros(classes.shape, dtype=label_array.dtype)
        valid = (classes > 0) & (classes <= len(label_array))
        mapped[valid] = label_array[classes[valid] - 1]
        return mapped
    return classes.astype("uint8")


def save_raster(
    array: np.ndarray, profile: dict, output_path: str | Path, nodata: float | int | None = None
) -> Path:
    """Save a single-band array as a raster."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=str(array.dtype), compress="deflate")
    if nodata is not None:
        out_profile["nodata"] = nodata
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(array, 1)
    return output_path


def raster_to_polygon(
    raster_path: str | Path, output_path: str | Path, target_values: list[int] | None = None
) -> Path:
    """Convert selected raster classes to polygons."""
    output_path = Path(output_path)
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        mask_array = data != src.nodata if src.nodata is not None else np.ones(data.shape, dtype=bool)
        if target_values is not None:
            mask_array &= np.isin(data, target_values)
        records = [
            {"value": int(value), "geometry": shape(geometry)}
            for geometry, value in shapes(data.astype("int32"), mask=mask_array, transform=src.transform)
        ]
        gdf = gpd.GeoDataFrame(records, crs=src.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    driver = "GeoJSON" if output_path.suffix.lower() in {".geojson", ".json"} else "GPKG"
    gdf.to_file(output_path, driver=driver)
    return output_path
