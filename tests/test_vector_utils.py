"""Tests for vector utility functions."""

import geopandas as gpd
from shapely.geometry import Point, Polygon

from floodsense.vector_utils import clean_geometries, reproject_vector


def test_clean_geometries_repairs_invalid_polygon() -> None:
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[invalid], crs="EPSG:4326")
    cleaned = clean_geometries(gdf)
    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0].is_valid


def test_reproject_vector_changes_crs() -> None:
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(6.74, 7.8)], crs="EPSG:4326")
    projected = reproject_vector(gdf, "EPSG:32632")
    assert projected.crs.to_string() == "EPSG:32632"
