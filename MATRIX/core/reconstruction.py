"""Matrix Reconstruction and Error Correction Algorithms."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


@dataclass
class ReconstructedCellRecord:
    """Audit record for an individual cell reconstruction."""
    row_idx: int
    col_idx: int
    row_label: str
    col_name: str
    original_val: Optional[float]
    reconstructed_val: float
    delta: Optional[float]
    reason: str
    accepted: bool = True
    confidence_score: float = 0.0
    confidence_level: str = "Medium"
    uncertainty_std: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "Row": self.row_label,
            "Column": self.col_name,
            "Original": self.original_val if self.original_val is not None else np.nan,
            "Reconstructed": round(self.reconstructed_val, 4),
            "Delta": round(self.delta, 4) if self.delta is not None else np.nan,
            "Reason": self.reason,
            "Confidence": f"{int(self.confidence_score * 100)}%",
            "Confidence_Raw": self.confidence_score,
            "Uncertainty_Std": round(self.uncertainty_std, 4),
            "Level": self.confidence_level,
            "Accepted": self.accepted
        }


@dataclass
class ReconstructionResult:
    """Stores full reconstruction output and metadata."""
    reconstructed_matrix: np.ndarray
    original_matrix: np.ndarray
    target_mask: np.ndarray
    cell_records: List[ReconstructedCellRecord]
    method_name: str
    parameters: Dict[str, any]
    iterations_run: int
    converged: bool
    final_frobenius_change: float


class MatrixReconstructor:
    """Implements Iterative Truncated SVD, SoftImpute, KNN, and Baseline imputation."""

    @staticmethod
    def _initialize_matrix(
        matrix: np.ndarray,
        target_mask: np.ndarray,
        strategy: str = "median"
    ) -> np.ndarray:
        """Create initial completion matrix using column median/mean for target cells."""
        X = matrix.copy()
        n_rows, n_cols = X.shape

        for c in range(n_cols):
            col_vals = X[:, c]
            valid_mask = (~np.isnan(col_vals)) & (~target_mask[:, c])
            valid_vals = col_vals[valid_mask]

            if len(valid_vals) > 0:
                fill_val = np.median(valid_vals) if strategy == "median" else np.mean(valid_vals)
            else:
                fill_val = 0.0

            # Fill target and missing cells
            fill_positions = target_mask[:, c] | np.isnan(col_vals)
            X[fill_positions, c] = fill_val

        return X

    @classmethod
    def iterative_truncated_svd(
        cls,
        matrix: np.ndarray,
        target_mask: np.ndarray,
        rank: int = 2,
        max_iter: int = 50,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, int, bool, float]:
        """Iterative Truncated SVD (Hard-Impute) Matrix Completion.
        
        Algorithm:
          1. Initialize X^(0) on target cells with column medians.
          2. Iteratively compute SVD: X^(t) = U S V^T.
          3. Reconstruct low-rank projection: X_hat = U_k S_k V_k^T.
          4. Update only target entries: X^(t+1) = P_Omega(Original) + P_Omega_perp(X_hat).
          5. Repeat until ||X^(t+1) - X^(t)||_F / ||X^(t)||_F < tol.
        """
        n_rows, n_cols = matrix.shape
        k = max(1, min(rank, min(n_rows, n_cols)))

        # Observed mask (entries that must stay fixed)
        observed_mask = ~target_mask & ~np.isnan(matrix)

        X = cls._initialize_matrix(matrix, target_mask, strategy="median")
        converged = False
        rel_change = 1.0

        for it in range(1, max_iter + 1):
            X_prev = X.copy()

            # Center columns before SVD to prevent mean offset distortion
            col_means = np.mean(X, axis=0)
            X_centered = X - col_means

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                break

            # Truncate to rank k
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]

            # Projected low-rank approximation
            X_hat = (U_k * S_k) @ Vt_k + col_means

            # Update ONLY the target cells; keep observed cells intact
            X[target_mask] = X_hat[target_mask]
            X[observed_mask] = matrix[observed_mask]

            norm_prev = np.linalg.norm(X_prev, ord="fro")
            if norm_prev > 1e-9:
                rel_change = np.linalg.norm(X - X_prev, ord="fro") / norm_prev
            else:
                rel_change = 0.0

            if rel_change < tol:
                converged = True
                break

        return X, it, converged, float(rel_change)

    @classmethod
    def soft_impute(
        cls,
        matrix: np.ndarray,
        target_mask: np.ndarray,
        shrinkage_lambda: Optional[float] = None,
        max_iter: int = 50,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, int, bool, float]:
        """SoftImpute Matrix Completion via Singular Value Soft-Thresholding.
        
        Solves nuclear norm regularized matrix recovery:
          S_lambda(sigma_i) = max(0, sigma_i - lambda)
        """
        n_rows, n_cols = matrix.shape
        observed_mask = ~target_mask & ~np.isnan(matrix)

        X = cls._initialize_matrix(matrix, target_mask, strategy="median")

        # Determine adaptive shrinkage lambda if not provided
        if shrinkage_lambda is None:
            _, S_init, _ = np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)
            # Default lambda: 15% of largest singular value
            shrinkage_lambda = float(S_init[0] * 0.15) if len(S_init) > 0 else 1.0

        converged = False
        rel_change = 1.0

        for it in range(1, max_iter + 1):
            X_prev = X.copy()
            col_means = np.mean(X, axis=0)
            X_centered = X - col_means

            try:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                break

            # Soft threshold singular values
            S_shrunk = np.maximum(0.0, S - shrinkage_lambda)

            # Reconstruct
            X_hat = (U * S_shrunk) @ Vt + col_means

            # Replace targets
            X[target_mask] = X_hat[target_mask]
            X[observed_mask] = matrix[observed_mask]

            norm_prev = np.linalg.norm(X_prev, ord="fro")
            rel_change = (
                np.linalg.norm(X - X_prev, ord="fro") / norm_prev if norm_prev > 1e-9 else 0.0
            )

            if rel_change < tol:
                converged = True
                break

        return X, it, converged, float(rel_change)

    @classmethod
    def knn_impute(
        cls,
        matrix: np.ndarray,
        target_mask: np.ndarray,
        n_neighbors: int = 5
    ) -> Tuple[np.ndarray, int, bool, float]:
        """Scikit-Learn KNN Imputation baseline."""
        n_rows, n_cols = matrix.shape
        k = min(n_neighbors, max(1, n_rows - 1))

        # Prepare matrix where targets are set to NaN for KNNImputer
        working = matrix.copy()
        working[target_mask] = np.nan

        imputer = KNNImputer(n_neighbors=k, weights="distance")
        try:
            imputed = imputer.fit_transform(working)
        except Exception:
            # Fallback to column median if KNN cannot find neighbors
            imputer_fallback = SimpleImputer(strategy="median")
            imputed = imputer_fallback.fit_transform(working)

        # Ensure observed cells remain exact
        observed_mask = ~target_mask & ~np.isnan(matrix)
        imputed[observed_mask] = matrix[observed_mask]

        return imputed, 1, True, 0.0

    @classmethod
    def baseline_impute(
        cls,
        matrix: np.ndarray,
        target_mask: np.ndarray,
        strategy: str = "median"
    ) -> Tuple[np.ndarray, int, bool, float]:
        """Simple baseline imputer (column median or mean)."""
        imputer = SimpleImputer(strategy=strategy)
        working = matrix.copy()
        working[target_mask] = np.nan
        imputed = imputer.fit_transform(working)

        observed_mask = ~target_mask & ~np.isnan(matrix)
        imputed[observed_mask] = matrix[observed_mask]

        return imputed, 1, True, 0.0


def run_matrix_reconstruction(
    matrix: np.ndarray,
    status_grid: np.ndarray,
    col_names: List[str],
    row_labels: Optional[List[str]] = None,
    method: str = "iterative_svd",
    reconstruct_missing: bool = True,
    correct_outliers: bool = True,
    correct_user_corrupted: bool = True,
    svd_rank: int = 2,
    soft_lambda: Optional[float] = None,
    knn_neighbors: int = 5,
    baseline_strategy: str = "median",
    max_iter: int = 50,
    tol: float = 1e-4
) -> ReconstructionResult:
    """Execute matrix reconstruction for selected target cells."""
    n_rows, n_cols = matrix.shape
    labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]

    # Build target mask based on user selections
    target_mask = np.zeros((n_rows, n_cols), dtype=bool)

    if reconstruct_missing:
        target_mask |= (status_grid == 1) | np.isnan(matrix)

    if correct_outliers:
        target_mask |= (status_grid == 2)

    if correct_user_corrupted:
        target_mask |= (status_grid == 3)

    params: Dict[str, any] = {"method": method}

    # Dispatch to chosen algorithm
    if method == "iterative_svd":
        params["rank"] = svd_rank
        params["max_iter"] = max_iter
        params["tol"] = tol
        reconstructed, it, converged, rel_change = MatrixReconstructor.iterative_truncated_svd(
            matrix, target_mask, rank=svd_rank, max_iter=max_iter, tol=tol
        )
    elif method == "soft_impute":
        params["shrinkage_lambda"] = soft_lambda
        params["max_iter"] = max_iter
        params["tol"] = tol
        reconstructed, it, converged, rel_change = MatrixReconstructor.soft_impute(
            matrix, target_mask, shrinkage_lambda=soft_lambda, max_iter=max_iter, tol=tol
        )
    elif method == "knn":
        params["n_neighbors"] = knn_neighbors
        reconstructed, it, converged, rel_change = MatrixReconstructor.knn_impute(
            matrix, target_mask, n_neighbors=knn_neighbors
        )
    elif method == "baseline":
        params["strategy"] = baseline_strategy
        reconstructed, it, converged, rel_change = MatrixReconstructor.baseline_impute(
            matrix, target_mask, strategy=baseline_strategy
        )
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    # Generate cell records for all modified cells
    cell_records: List[ReconstructedCellRecord] = []
    target_indices = np.argwhere(target_mask)

    for r, c in target_indices:
        orig = matrix[r, c]
        is_nan = np.isnan(orig) or np.isinf(orig)
        recon_val = float(reconstructed[r, c])
        orig_val = None if is_nan else float(orig)
        delta = None if is_nan else float(recon_val - orig_val)

        status_code = status_grid[r, c]
        if status_code == 1 or is_nan:
            reason = "Missing Value"
        elif status_code == 2:
            reason = "Potential Outlier"
        elif status_code == 3:
            reason = "User Corrupted"
        else:
            reason = "Flagged Correction"

        cell_records.append(
            ReconstructedCellRecord(
                row_idx=r,
                col_idx=c,
                row_label=labels[r],
                col_name=col_names[c],
                original_val=orig_val,
                reconstructed_val=recon_val,
                delta=delta,
                reason=reason,
                accepted=True
            )
        )

    return ReconstructionResult(
        reconstructed_matrix=reconstructed,
        original_matrix=matrix.copy(),
        target_mask=target_mask,
        cell_records=cell_records,
        method_name=method,
        parameters=params,
        iterations_run=it,
        converged=converged,
        final_frobenius_change=rel_change
    )
