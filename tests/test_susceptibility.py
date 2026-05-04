"""Tests for flood susceptibility modelling functions."""

import numpy as np

from floodsense.susceptibility import calculate_weighted_susceptibility, classify_susceptibility


def test_calculate_weighted_susceptibility() -> None:
    factors = {
        "elevation": np.array([[0.0, 1.0]], dtype="float32"),
        "slope": np.array([[1.0, 0.0]], dtype="float32"),
    }
    weights = {"elevation": 0.5, "slope": 0.5}
    score = calculate_weighted_susceptibility(factors, weights)
    np.testing.assert_allclose(score, np.array([[0.5, 0.5]], dtype="float32"))


def test_classify_susceptibility() -> None:
    score = np.array([0.1, 0.35, 0.55, 0.75, 0.95], dtype="float32")
    result = classify_susceptibility(score)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5], dtype="uint8"))
