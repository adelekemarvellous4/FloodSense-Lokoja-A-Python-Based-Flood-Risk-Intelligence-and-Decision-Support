"""Tests for priority index functions."""

import pandas as pd

from floodsense.priority_index import classify_priority, normalize_series


def test_normalize_series_scales_values() -> None:
    result = normalize_series(pd.Series([10, 20, 30]))
    assert result.tolist() == [0.0, 0.5, 1.0]


def test_classify_priority_thresholds() -> None:
    assert classify_priority(0.1) == "Low"
    assert classify_priority(0.3) == "Moderate"
    assert classify_priority(0.6) == "High"
    assert classify_priority(0.9) == "Critical"
