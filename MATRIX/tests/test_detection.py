"""Unit tests for error and outlier detection module."""

import numpy as np
import pytest

from core.error_detection import (
    ErrorDetector,
    aggregate_detected_entries
)


def test_missing_value_detection():
    matrix = np.array([
        [1.0, 2.0, np.nan],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0]
    ])
    col_names = ["C1", "C2", "C3"]
    entries = ErrorDetector.detect_missing(matrix, col_names)

    assert len(entries) == 2
    assert {(e.row_idx, e.col_idx) for e in entries} == {(0, 2), (1, 1)}
    for e in entries:
        assert e.entry_type == "Missing"


def test_robust_zscore_detection():
    # Regular data with an obvious spike
    data = np.array([
        [10.0, 100.0],
        [10.2, 102.0],
        [9.9, 101.0],
        [10.1, 99.0],
        [10.0, 1000.0],  # 1000 is an extreme outlier
        [10.3, 103.0],
        [9.8, 98.0]
    ])
    col_names = ["X1", "X2"]
    entries = ErrorDetector.detect_robust_zscore(data, col_names, threshold=3.5)

    assert len(entries) == 1
    assert entries[0].row_idx == 4
    assert entries[0].col_idx == 1
    assert entries[0].original_val == 1000.0


def test_iqr_fence_detection():
    data = np.array([
        [5.0], [5.2], [4.8], [5.1], [5.0], [5.3], [4.9], [55.0]
    ])
    entries = ErrorDetector.detect_iqr_fence(data, ["Value"], multiplier=1.5)

    assert len(entries) == 1
    assert entries[0].row_idx == 7
    assert entries[0].original_val == 55.0


def test_aggregate_detection_with_user_override():
    matrix = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 50.0, 6.0],  # 50 is statistical outlier
        [7.0, 8.0, 9.0]
    ])
    col_names = ["A", "B", "C"]
    user_flags = {(2, 0)}  # User marks (2, 0) as corrupt

    entries, status_grid, counts = aggregate_detected_entries(
        matrix=matrix,
        col_names=col_names,
        method="robust_zscore",
        zscore_threshold=3.0,
        user_corrupted_cells=user_flags
    )

    assert counts["missing"] >= 1
    assert counts["user_corrupted"] == 1
    assert status_grid[2, 0] == 3  # Status code 3 for user marked
