"""Unit tests for reconstruction and error correction algorithms."""

import numpy as np
import pytest

from core.reconstruction import (
    MatrixReconstructor,
    run_matrix_reconstruction
)


def test_iterative_truncated_svd_recovery():
    # Construct a rank-1 clean matrix: u @ v^T
    rng = np.random.default_rng(42)
    u = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    v = np.array([[2.0, 4.0, 6.0, 8.0]])
    clean = u @ v  # Perfect rank-1 matrix

    # Mask out some cells
    target_mask = np.zeros(clean.shape, dtype=bool)
    target_mask[1, 1] = True  # True value is 2*4 = 8.0
    target_mask[3, 2] = True  # True value is 4*6 = 24.0

    corrupted = clean.copy()
    corrupted[target_mask] = np.nan

    reconstructed, it, converged, rel_change = MatrixReconstructor.iterative_truncated_svd(
        corrupted, target_mask, rank=1, max_iter=30, tol=1e-5
    )

    # Reconstructed values should be very close to ground truth on low-rank matrix
    assert np.isclose(reconstructed[1, 1], clean[1, 1], atol=0.5)
    assert np.isclose(reconstructed[3, 2], clean[3, 2], atol=0.5)

    # Non-target cells must remain completely unchanged
    assert np.array_equal(reconstructed[~target_mask], clean[~target_mask])


def test_soft_impute_recovery():
    rng = np.random.default_rng(42)
    A = rng.normal(10, 2, size=(10, 6))
    target_mask = np.zeros(A.shape, dtype=bool)
    target_mask[0, 0] = True
    target_mask[2, 3] = True

    reconstructed, it, converged, rel_change = MatrixReconstructor.soft_impute(
        A, target_mask, max_iter=20, tol=1e-3
    )

    assert not np.isnan(reconstructed).any()
    assert np.array_equal(reconstructed[~target_mask], A[~target_mask])


def test_run_reconstruction_pipeline():
    matrix = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 100.0, 6.0],  # 100.0 is an outlier
        [7.0, 8.0, 9.0]
    ])
    status_grid = np.array([
        [0, 0, 1],
        [0, 2, 0],
        [0, 0, 0]
    ])
    col_names = ["C1", "C2", "C3"]

    result = run_matrix_reconstruction(
        matrix=matrix,
        status_grid=status_grid,
        col_names=col_names,
        method="iterative_svd",
        svd_rank=1
    )

    assert len(result.cell_records) == 2
    assert not np.isnan(result.reconstructed_matrix).any()
