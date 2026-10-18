"""Error and Outlier Detection Module for Matrix Data."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class DetectedEntry:
    """Represents a flagged cell in the matrix."""
    row_idx: int
    col_idx: int
    row_label: str
    col_name: str
    original_val: Optional[float]
    entry_type: str              # 'Missing', 'Potential Outlier', 'User Corrupted'
    detection_method: str        # 'NaN Check', 'Isolation Forest', 'Robust Z-Score', 'IQR Fence', 'Range Check', 'Manual'
    anomaly_score: float         # Magnitude of deviation or algorithm score
    flagged_for_correction: bool # Whether user/system designated for reconstruction
    description: str

    def to_dict(self) -> Dict:
        return asdict(self)


class ErrorDetector:
    """Performs multi-strategy error, outlier, and missing-value detection."""

    @staticmethod
    def detect_missing(
        matrix: np.ndarray,
        col_names: List[str],
        row_labels: Optional[List[str]] = None
    ) -> List[DetectedEntry]:
        """Identify missing (NaN/null/inf) values in the matrix."""
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        flagged = []

        for r in range(n_rows):
            for c in range(n_cols):
                val = matrix[r, c]
                if np.isnan(val) or np.isinf(val):
                    flagged.append(
                        DetectedEntry(
                            row_idx=r,
                            col_idx=c,
                            row_label=labels[r],
                            col_name=col_names[c],
                            original_val=None,
                            entry_type="Missing",
                            detection_method="Missing (NaN/Inf)",
                            anomaly_score=1.0,
                            flagged_for_correction=True,
                            description=f"Missing value (NaN/Null) at {labels[r]}, column '{col_names[c]}'"
                        )
                    )
        return flagged

    @staticmethod
    def detect_robust_zscore(
        matrix: np.ndarray,
        col_names: List[str],
        row_labels: Optional[List[str]] = None,
        threshold: float = 3.5
    ) -> List[DetectedEntry]:
        """Detect outliers using Modified Robust Z-Score based on Median Absolute Deviation (MAD).
        
        Formula: M_i = 0.6745 * |x_i - median| / MAD.
        Standard threshold is 3.5 for strong deviation.
        """
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        flagged = []

        for c in range(n_cols):
            col_data = matrix[:, c]
            valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
            valid_vals = col_data[valid_mask]

            if len(valid_vals) < 3:
                continue

            med = np.median(valid_vals)
            mad = np.median(np.abs(valid_vals - med))

            if mad < 1e-9:
                # If MAD is 0 (constant or identical median values), use mean absolute deviation
                mean_val = np.mean(valid_vals)
                mad = np.mean(np.abs(valid_vals - mean_val))
                if mad < 1e-9:
                    continue

            for r in range(n_rows):
                val = matrix[r, c]
                if np.isnan(val) or np.isinf(val):
                    continue

                mod_z = 0.6745 * np.abs(val - med) / mad
                if mod_z > threshold:
                    flagged.append(
                        DetectedEntry(
                            row_idx=r,
                            col_idx=c,
                            row_label=labels[r],
                            col_name=col_names[c],
                            original_val=float(val),
                            entry_type="Potential Outlier",
                            detection_method=f"Robust Z-Score (MAD > {threshold})",
                            anomaly_score=float(mod_z),
                            flagged_for_correction=True,
                            description=(
                                f"Robust Z-score is {mod_z:.2f} (median: {med:.2f}, MAD: {mad:.2f}). "
                                f"Significantly deviates from typical column distribution."
                            )
                        )
                    )
        return flagged

    @staticmethod
    def detect_iqr_fence(
        matrix: np.ndarray,
        col_names: List[str],
        row_labels: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> List[DetectedEntry]:
        """Detect outliers using Tukey's Interquartile Range (IQR) fence rule."""
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        flagged = []

        for c in range(n_cols):
            col_data = matrix[:, c]
            valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
            valid_vals = col_data[valid_mask]

            if len(valid_vals) < 4:
                continue

            q25, q75 = np.percentile(valid_vals, [25, 75])
            iqr = q75 - q25

            if iqr < 1e-9:
                continue

            lower_bound = q25 - multiplier * iqr
            upper_bound = q75 + multiplier * iqr

            for r in range(n_rows):
                val = matrix[r, c]
                if np.isnan(val) or np.isinf(val):
                    continue

                if val < lower_bound or val > upper_bound:
                    dist = max(lower_bound - val, val - upper_bound)
                    score = dist / iqr
                    flagged.append(
                        DetectedEntry(
                            row_idx=r,
                            col_idx=c,
                            row_label=labels[r],
                            col_name=col_names[c],
                            original_val=float(val),
                            entry_type="Potential Outlier",
                            detection_method=f"IQR Fence ({multiplier}x IQR)",
                            anomaly_score=float(score),
                            flagged_for_correction=True,
                            description=(
                                f"Value {val:.2f} is outside bounds [{lower_bound:.2f}, {upper_bound:.2f}] "
                                f"(Q1: {q25:.2f}, Q3: {q75:.2f}, IQR: {iqr:.2f})."
                            )
                        )
                    )
        return flagged

    @staticmethod
    def detect_isolation_forest(
        matrix: np.ndarray,
        col_names: List[str],
        row_labels: Optional[List[str]] = None,
        contamination: float = 0.05,
        random_state: int = 42
    ) -> List[DetectedEntry]:
        """Multivariate anomaly detection using Scikit-Learn IsolationForest.
        
        Pins down specific anomalous cells within outlying rows.
        """
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        flagged = []

        if n_rows < 5:
            # Not enough samples for multivariate tree isolation
            return []

        # Impute temporarily with column medians for IsolationForest training
        imputed_matrix = matrix.copy()
        for c in range(n_cols):
            vals = imputed_matrix[:, c]
            valid = vals[~np.isnan(vals)]
            med = np.median(valid) if len(valid) > 0 else 0.0
            imputed_matrix[np.isnan(vals), c] = med

        iso = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        preds = iso.fit_predict(imputed_matrix)
        scores = -iso.score_samples(imputed_matrix)  # Higher score = more anomalous

        # Outlier rows
        outlier_rows = np.where(preds == -1)[0]

        # For outlier rows, identify which feature cells contributed most to anomaly
        col_medians = np.median(imputed_matrix, axis=0)
        col_mads = np.median(np.abs(imputed_matrix - col_medians), axis=0)
        col_mads[col_mads < 1e-6] = 1.0

        for r in outlier_rows:
            row_score = scores[r]
            # Identify which features are >= 2.5 MAD away from column center
            z_scores = 0.6745 * np.abs(imputed_matrix[r] - col_medians) / col_mads

            # If none reached 2.5, pick top deviant feature
            max_c = int(np.argmax(z_scores))
            top_cols = [c for c in range(n_cols) if z_scores[c] >= 2.2]
            if not top_cols:
                top_cols = [max_c]

            for c in top_cols:
                val = matrix[r, c]
                if np.isnan(val) or np.isinf(val):
                    continue

                flagged.append(
                    DetectedEntry(
                        row_idx=r,
                        col_idx=c,
                        row_label=labels[r],
                        col_name=col_names[c],
                        original_val=float(val),
                        entry_type="Potential Outlier",
                        detection_method=f"Isolation Forest (Row Anomaly Score: {row_score:.3f})",
                        anomaly_score=float(z_scores[c]),
                        flagged_for_correction=True,
                        description=(
                            f"Isolation Forest identified row as multivariate anomaly (score: {row_score:.3f}). "
                            f"Feature '{col_names[c]}' has relative deviation {z_scores[c]:.2f}."
                        )
                    )
                )
        return flagged

    @staticmethod
    def detect_custom_range(
        matrix: np.ndarray,
        col_names: List[str],
        row_labels: Optional[List[str]] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> List[DetectedEntry]:
        """Detect values strictly outside user-specified domain bounds."""
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        flagged = []

        if min_val is None and max_val is None:
            return []

        for r in range(n_rows):
            for c in range(n_cols):
                val = matrix[r, c]
                if np.isnan(val) or np.isinf(val):
                    continue

                violated = False
                reason = ""
                if min_val is not None and val < min_val:
                    violated = True
                    reason = f"Value {val} is below minimum allowed bound {min_val}"
                elif max_val is not None and val > max_val:
                    violated = True
                    reason = f"Value {val} exceeds maximum allowed bound {max_val}"

                if violated:
                    flagged.append(
                        DetectedEntry(
                            row_idx=r,
                            col_idx=c,
                            row_label=labels[r],
                            col_name=col_names[c],
                            original_val=float(val),
                            entry_type="Potential Outlier",
                            detection_method="Domain Range Boundary",
                            anomaly_score=abs(val - (min_val if val < (min_val or 0) else (max_val or 0))),
                            flagged_for_correction=True,
                            description=reason
                        )
                    )
        return flagged


def aggregate_detected_entries(
    matrix: np.ndarray,
    col_names: List[str],
    row_labels: Optional[List[str]] = None,
    method: str = "robust_zscore",
    zscore_threshold: float = 3.5,
    iqr_multiplier: float = 1.5,
    iso_contamination: float = 0.05,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    user_corrupted_cells: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[List[DetectedEntry], np.ndarray, Dict[str, int]]:
    """Execute detection strategy and combine findings into a unified matrix map.
    
    Status mapping for visualization and downstream reconstruction:
      0: CLEAN
      1: MISSING
      2: OUTLIER (detected statistically)
      3: USER_CORRUPTED (manually flagged)
    """
    n_rows, n_cols = matrix.shape
    labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]

    # 1. Always detect missing values first
    missing_entries = ErrorDetector.detect_missing(matrix, col_names, labels)

    # 2. Run statistical outlier detection
    outlier_entries = []
    if method == "robust_zscore":
        outlier_entries = ErrorDetector.detect_robust_zscore(
            matrix, col_names, labels, threshold=zscore_threshold
        )
    elif method == "isolation_forest":
        outlier_entries = ErrorDetector.detect_isolation_forest(
            matrix, col_names, labels, contamination=iso_contamination
        )
    elif method == "iqr_fence":
        outlier_entries = ErrorDetector.detect_iqr_fence(
            matrix, col_names, labels, multiplier=iqr_multiplier
        )
    elif method == "custom_range":
        outlier_entries = ErrorDetector.detect_custom_range(
            matrix, col_names, labels, min_val=range_min, max_val=range_max
        )

    # Status grid (0=clean, 1=missing, 2=outlier, 3=user_marked)
    status_grid = np.zeros((n_rows, n_cols), dtype=int)
    entry_dict: Dict[Tuple[int, int], DetectedEntry] = {}

    for entry in missing_entries:
        key = (entry.row_idx, entry.col_idx)
        entry_dict[key] = entry
        status_grid[entry.row_idx, entry.col_idx] = 1

    for entry in outlier_entries:
        key = (entry.row_idx, entry.col_idx)
        if key not in entry_dict:  # Missing takes precedence
            entry_dict[key] = entry
            status_grid[entry.row_idx, entry.col_idx] = 2

    # 3. Add user-marked corrupted cells
    if user_corrupted_cells:
        for r, c in user_corrupted_cells:
            if 0 <= r < n_rows and 0 <= c < n_cols:
                key = (r, c)
                val = matrix[r, c]
                is_missing = np.isnan(val) or np.isinf(val)
                entry_dict[key] = DetectedEntry(
                    row_idx=r,
                    col_idx=c,
                    row_label=labels[r],
                    col_name=col_names[c],
                    original_val=None if is_missing else float(val),
                    entry_type="User Corrupted",
                    detection_method="Manual User Override",
                    anomaly_score=1.0,
                    flagged_for_correction=True,
                    description=f"Manually flagged as corrupted by user at {labels[r]}, {col_names[c]}"
                )
                status_grid[r, c] = 3

    all_entries = sorted(entry_dict.values(), key=lambda e: (e.row_idx, e.col_idx))

    counts = {
        "total_cells": int(matrix.size),
        "clean": int(np.sum(status_grid == 0)),
        "missing": int(np.sum(status_grid == 1)),
        "outliers": int(np.sum(status_grid == 2)),
        "user_corrupted": int(np.sum(status_grid == 3)),
        "total_flagged": len(all_entries)
    }

    return all_entries, status_grid, counts
