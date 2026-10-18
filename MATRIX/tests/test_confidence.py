"""Unit tests for confidence estimation and pattern analysis modules."""

import numpy as np
import pytest

from core.pattern_analysis import PatternAnalyzer
from core.reconstruction import run_matrix_reconstruction
from core.confidence import ConfidenceEstimator


def test_svd_energy_spectrum():
    # Matrix with clear dominant 1st singular value
    A = np.outer(np.arange(1, 11), np.array([1, 2, 3, 4]))
    svd_res = PatternAnalyzer.analyze_svd_spectrum(A)

    assert len(svd_res.singular_values) == 4
    # Dominant singular value should carry almost all energy (>95%)
    assert svd_res.cumulative_energy[0] > 0.90
    assert svd_res.suggested_rank_90 == 1
    assert "Low-Rank" in svd_res.suitability_summary


def test_confidence_estimation_bounds():
    rng = np.random.default_rng(42)
    # 15x5 matrix
    u = rng.uniform(1, 5, size=(15, 2))
    v = rng.uniform(1, 3, size=(2, 5))
    clean = u @ v

    status_grid = np.zeros(clean.shape, dtype=int)
    status_grid[2, 1] = 1  # Missing
    status_grid[5, 3] = 2  # Outlier
    clean[2, 1] = np.nan
    clean[5, 3] = 999.0

    col_names = [f"Col_{i}" for i in range(5)]
    recon_res = run_matrix_reconstruction(
        matrix=clean,
        status_grid=status_grid,
        col_names=col_names,
        method="iterative_svd",
        svd_rank=2
    )

    conf_rep = ConfidenceEstimator.estimate_confidence(
        matrix=clean,
        target_mask=recon_res.target_mask,
        reconstruction_result=recon_res,
        col_names=col_names,
        review_threshold=0.70,
        n_bootstrap_trials=3
    )

    assert len(conf_rep.records) == 2
    for rec in conf_rep.records:
        assert 0.0 <= rec.confidence_score <= 1.0
        assert rec.confidence_level in ["High", "Moderate", "Low (Needs Review)"]

    assert 0.0 <= conf_rep.mean_confidence <= 1.0
    assert len(conf_rep.limitations) > 0
