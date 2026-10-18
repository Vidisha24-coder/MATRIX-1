"""Pattern Analysis and SVD Spectrum Analysis Module."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class SVDAnalysisResult:
    """Stores SVD singular values, energy spectrum, and rank recommendations."""
    singular_values: np.ndarray
    variance_explained_ratio: np.ndarray
    cumulative_energy: np.ndarray
    suggested_rank_80: int
    suggested_rank_90: int
    suggested_rank_95: int
    elbow_rank: int
    effective_rank: float
    condition_number: float
    suitability_summary: str
    reconstruction_justification: str


class PatternAnalyzer:
    """Analyzes matrix distributions, correlations, missingness, and low-rank structure."""

    @staticmethod
    def compute_summary_statistics(df_numeric: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive column-wise distribution metrics."""
        summary = []
        for col in df_numeric.columns:
            s = df_numeric[col].dropna()
            n_total = len(df_numeric)
            n_valid = len(s)
            missing_count = n_total - n_valid
            missing_pct = (missing_count / n_total) * 100 if n_total > 0 else 0.0

            if n_valid > 0:
                mean_val = float(s.mean())
                std_val = float(s.std(ddof=1)) if n_valid > 1 else 0.0
                med_val = float(s.median())
                mad_val = float(np.median(np.abs(s - med_val)))
                min_val = float(s.min())
                max_val = float(s.max())
                skew_val = float(s.skew()) if n_valid > 2 else 0.0
            else:
                mean_val = std_val = med_val = mad_val = min_val = max_val = skew_val = np.nan

            summary.append({
                "Column": col,
                "Valid Count": n_valid,
                "Missing Count": missing_count,
                "Missing %": round(missing_pct, 1),
                "Mean": round(mean_val, 3),
                "Std Dev": round(std_val, 3),
                "Median": round(med_val, 3),
                "MAD": round(mad_val, 3),
                "Min": round(min_val, 3),
                "Max": round(max_val, 3),
                "Skewness": round(skew_val, 2)
            })

        return pd.DataFrame(summary)

    @staticmethod
    def compute_correlation_matrix(df_numeric: pd.DataFrame, method: str = "pearson") -> Tuple[pd.DataFrame, float]:
        """Compute pairwise correlation matrix and average absolute inter-feature correlation."""
        corr = df_numeric.corr(method=method)
        # Average off-diagonal correlation
        n = corr.shape[0]
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            off_diag = np.abs(corr.to_numpy())[mask]
            valid_off_diag = off_diag[~np.isnan(off_diag)]
            avg_corr = float(np.mean(valid_off_diag)) if len(valid_off_diag) > 0 else 0.0
        else:
            avg_corr = 0.0

        return corr, avg_corr

    @staticmethod
    def analyze_svd_spectrum(
        matrix: np.ndarray,
        col_names: Optional[List[str]] = None,
        max_rank: Optional[int] = None
    ) -> SVDAnalysisResult:
        """Perform Singular Value Decomposition (SVD) on imputed baseline to evaluate rank structure.
        
        Evaluates singular value decay, cumulative energy, and recommends optimal reconstruction rank.
        """
        n_rows, n_cols = matrix.shape
        min_dim = min(n_rows, n_cols)

        # Baseline imputation (column median) for SVD spectrum estimation
        imputed = matrix.copy()
        for c in range(n_cols):
            col_vals = imputed[:, c]
            valid = col_vals[~np.isnan(col_vals) & ~np.isinf(col_vals)]
            fill_val = np.median(valid) if len(valid) > 0 else 0.0
            imputed[np.isnan(col_vals) | np.isinf(col_vals), c] = fill_val

        # Center matrix by column means for accurate principal component variance
        col_means = np.mean(imputed, axis=0)
        centered = imputed - col_means

        # Compute full SVD
        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback with small regularization on diagonal
            _, S, _ = np.linalg.svd(centered + 1e-6 * np.eye(n_rows, n_cols), full_matrices=False)

        # S is sorted descending
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-12:
            variance_ratio = np.ones_like(S) / len(S)
            cum_energy = np.linspace(1 / len(S), 1.0, len(S))
        else:
            variance_ratio = (S ** 2) / total_energy
            cum_energy = np.cumsum(variance_ratio)

        # Suggested ranks for energy thresholds
        rank_80 = int(np.searchsorted(cum_energy, 0.80) + 1)
        rank_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        rank_95 = int(np.searchsorted(cum_energy, 0.95) + 1)

        rank_80 = min(rank_80, min_dim)
        rank_90 = min(rank_90, min_dim)
        rank_95 = min(rank_95, min_dim)

        # Scree elbow detection (maximum second difference / change in slope)
        if len(S) >= 3:
            diffs = np.diff(S)
            second_diffs = np.diff(diffs)
            elbow_rank = int(np.argmax(second_diffs) + 2)
            elbow_rank = min(elbow_rank, min_dim)
        else:
            elbow_rank = 1

        # Effective rank (Entropy-based: exp(-sum(p * log(p))))
        p = S / np.sum(S) if np.sum(S) > 0 else np.ones_like(S) / len(S)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))

        # Condition number
        cond_num = float(S[0] / S[-1]) if S[-1] > 1e-12 else float("inf")

        # Suitability synthesis
        top3_energy = float(cum_energy[min(2, len(cum_energy)-1)] * 100)
        is_low_rank = (rank_90 <= max(2, int(min_dim * 0.5))) or (top3_energy >= 85.0)

        if is_low_rank:
            suitability_summary = (
                f"Strong Low-Rank Structure: Top {min(3, len(S))} singular values account for {top3_energy:.1f}% "
                f"of total variance. Effective rank is {effective_rank:.1f} / {min_dim}."
            )
            reconstruction_justification = (
                f"Because the matrix energy is heavily concentrated in the top singular components "
                f"(rank {rank_90} captures >=90% energy), Truncated SVD or SoftImpute matrix completion is "
                f"strongly justified. The matrix columns share substantial latent subspace factors, "
                f"allowing unobserved or corrupted cells to be recovered from the low-dimensional projection."
            )
        else:
            suitability_summary = (
                f"Diffuse / High-Rank Spectrum: Variance is spread across many components. "
                f"Rank {rank_90} needed for 90% energy. Effective rank is {effective_rank:.1f} / {min_dim}."
            )
            reconstruction_justification = (
                f"The singular values decay relatively slowly, suggesting complex or weakly correlated features. "
                f"While low-rank SVD can still recover global trends, KNN-based local imputation or a slightly "
                f"higher rank (k={rank_90}) is recommended to avoid excessive smoothing."
            )

        return SVDAnalysisResult(
            singular_values=S,
            variance_explained_ratio=variance_ratio,
            cumulative_energy=cum_energy,
            suggested_rank_80=rank_80,
            suggested_rank_90=rank_90,
            suggested_rank_95=rank_95,
            elbow_rank=elbow_rank,
            effective_rank=effective_rank,
            condition_number=cond_num,
            suitability_summary=suitability_summary,
            reconstruction_justification=reconstruction_justification
        )
