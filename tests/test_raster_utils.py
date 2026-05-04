"""Tests for raster utility functions."""

import numpy as np

from floodsense.raster_utils import classify_array, normalize_array


def test_normalize_array_scales_values() -> None:
    array = np.array([1, 2, 3], dtype="float32")
    result = normalize_array(array)
    np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0], dtype="float32"))


def test_normalize_array_inverse() -> None:
    array = np.array([1, 2, 3], dtype="float32")
    result = normalize_array(array, inverse=True)
    np.testing.assert_allclose(result, np.array([1.0, 0.5, 0.0], dtype="float32"))


def test_classify_array_uses_bins() -> None:
    array = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype="float32")
    result = classify_array(array, bins=[0.2, 0.4, 0.6, 0.8])
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5], dtype="uint8"))
