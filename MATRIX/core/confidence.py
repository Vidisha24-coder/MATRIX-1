"""Confidence and Uncertainty Estimation Module for Matrix Reconstruction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.reconstruction import ReconstructionResult, MatrixReconstructor, ReconstructedCellRecord


@dataclass
class ConfidenceReport:
    """Detailed summary of confidence calculations and limitations."""
    records: List[ReconstructedCellRecord]
    mean_confidence: float
    median_confidence: float
    min_confidence: float
    max_confidence: float
    n_high_confidence: int
    n_moderate_confidence: int
    n_low_confidence: int
    review_threshold: float
    method_description: str
    limitations: List[str]


class ConfidenceEstimator:
    """Estimates reconstruction uncertainty using Monte Carlo masking and multi-method ensemble agreement."""

    @classmethod
    def estimate_confidence(
        cls,
        matrix: np.ndarray,
        target_mask: np.ndarray,
        reconstruction_result: ReconstructionResult,
        col_names: List[str],
        row_labels: Optional[List[str]] = None,
        review_threshold: float = 0.70,
        n_bootstrap_trials: int = 5,
        mask_fraction: float = 0.08,
        seed: int = 42
    ) -> ConfidenceReport:
        """Estimate cell-level reconstruction confidence.
        
        Methodology:
          1. Monte Carlo Jackknife Masking:
             Repeatedly hide an additional 8% of observed cells, re-run reconstruction,
             and measure variance (std dev) across bootstrap runs.
          2. Multi-Method Ensemble Agreement:
             Compute alternate reconstruction using distance-weighted KNN imputer.
             Evaluate normalized agreement between SVD and KNN estimates.
          3. Subspace Projection Scale:
             Normalize uncertainty relative to column standard deviation.
        
        IMPORTANT: This yields an empirical consistency heuristic (0.0 to 1.0),
        NOT a calibrated frequentist or Bayesian probability.
        """
        n_rows, n_cols = matrix.shape
        labels = row_labels if row_labels else [f"Row_{i}" for i in range(n_rows)]
        rng = np.random.default_rng(seed)

        # Baseline column scales for normalization
        col_stds = np.zeros(n_cols)
        for c in range(n_cols):
            vals = matrix[:, c]
            valid = vals[~np.isnan(vals) & ~target_mask[:, c]]
            col_stds[c] = float(np.std(valid)) if len(valid) > 1 else 1.0
            if col_stds[c] < 1e-6:
                col_stds[c] = 1.0

        target_indices = np.argwhere(target_mask)
        if len(target_indices) == 0:
            return ConfidenceReport(
                records=[],
                mean_confidence=1.0,
                median_confidence=1.0,
                min_confidence=1.0,
                max_confidence=1.0,
                n_high_confidence=0,
                n_moderate_confidence=0,
                n_low_confidence=0,
                review_threshold=review_threshold,
                method_description="No target cells required reconstruction.",
                limitations=["No cells reconstructed."]
            )

        # 1. Monte Carlo Perturbation / Masking Trials
        bootstrap_estimates: Dict[Tuple[int, int], List[float]] = {
            (r, c): [] for r, c in target_indices
        }

        # Observed entries available for secondary masking
        observed_positions = [
            (r, c) for r in range(n_rows) for c in range(n_cols)
            if not target_mask[r, c] and not np.isnan(matrix[r, c])
        ]

        n_to_mask = max(1, int(len(observed_positions) * mask_fraction))
        primary_method = reconstruction_result.method_name
        params = reconstruction_result.parameters

        for trial in range(n_bootstrap_trials):
            # Create perturbed target mask
            trial_mask = target_mask.copy()
            if len(observed_positions) > 0:
                masked_obs_idx = rng.choice(len(observed_positions), size=min(n_to_mask, len(observed_positions)), replace=False)
                for idx in masked_obs_idx:
                    r_m, c_m = observed_positions[idx]
                    trial_mask[r_m, c_m] = True

            # Run trial reconstruction
            if primary_method == "iterative_svd":
                trial_X, _, _, _ = MatrixReconstructor.iterative_truncated_svd(
                    matrix, trial_mask, rank=params.get("rank", 2), max_iter=20, tol=1e-3
                )
            elif primary_method == "soft_impute":
                trial_X, _, _, _ = MatrixReconstructor.soft_impute(
                    matrix, trial_mask, shrinkage_lambda=params.get("shrinkage_lambda"), max_iter=20, tol=1e-3
                )
            elif primary_method == "knn":
                trial_X, _, _, _ = MatrixReconstructor.knn_impute(
                    matrix, trial_mask, n_neighbors=params.get("n_neighbors", 5)
                )
            else:
                trial_X, _, _, _ = MatrixReconstructor.baseline_impute(
                    matrix, trial_mask, strategy=params.get("strategy", "median")
                )

            for r, c in target_indices:
                bootstrap_estimates[(r, c)].append(float(trial_X[r, c]))

        # 2. Multi-Method Ensemble Reference (KNN baseline)
        knn_ref_X, _, _, _ = MatrixReconstructor.knn_impute(matrix, target_mask, n_neighbors=min(5, max(1, n_rows - 1)))

        # 3. Compute Stability and Confidence Scores for Each Target Cell
        updated_records: List[ReconstructedCellRecord] = []
        scores: List[float] = []

        record_lookup = {
            (rec.row_idx, rec.col_idx): rec for rec in reconstruction_result.cell_records
        }

        for r, c in target_indices:
            key = (r, c)
            rec = record_lookup.get(key)
            if not rec:
                continue

            recon_val = rec.reconstructed_val
            trials = bootstrap_estimates[key]

            # Empirical std dev across masking runs
            uncert_std = float(np.std(trials)) if len(trials) > 1 else 0.0

            # Stability score: relative to feature standard deviation
            col_std = col_stds[c]
            rel_uncert = uncert_std / col_std
            # High stability gives score close to 1.0; high variance degrades score
            stability_score = float(np.exp(-1.5 * rel_uncert))

            # Ensemble agreement score with KNN baseline
            knn_val = float(knn_ref_X[r, c])
            diff = abs(recon_val - knn_val)
            scale = abs(recon_val) + abs(knn_val) + col_std + 1e-6
            ensemble_agreement = float(max(0.0, 1.0 - (diff / scale)))

            # Composite confidence score (60% bootstrap stability, 40% ensemble agreement)
            raw_score = 0.60 * stability_score + 0.40 * ensemble_agreement
            # Bound cleanly between 0.10 and 0.98
            score = float(np.clip(raw_score, 0.10, 0.98))

            if score >= 0.80:
                level = "High"
            elif score >= review_threshold:
                level = "Moderate"
            else:
                level = "Low (Needs Review)"

            rec.confidence_score = score
            rec.confidence_level = level
            rec.uncertainty_std = uncert_std

            updated_records.append(rec)
            scores.append(score)

        scores_arr = np.array(scores) if scores else np.array([1.0])
        mean_conf = float(np.mean(scores_arr))
        med_conf = float(np.median(scores_arr))
        min_conf = float(np.min(scores_arr))
        max_conf = float(np.max(scores_arr))

        n_high = int(np.sum(scores_arr >= 0.80))
        n_mod = int(np.sum((scores_arr >= review_threshold) & (scores_arr < 0.80)))
        n_low = int(np.sum(scores_arr < review_threshold))

        method_desc = (
            "Confidence is estimated using a dual-evidence framework: "
            "(1) Monte Carlo Jackknife stability across random secondary maskings, and "
            "(2) cross-algorithm ensemble agreement against distance-weighted KNN. "
            "Scores range from 0% (high variance / disagreement) to 100% (high stability and consensus)."
        )

        limitations = [
            "Empirical Heuristic: Confidence scores are stability indices, not calibrated Bayesian posterior probabilities or coverage intervals.",
            "Missingness Mechanism: Assumes missing entries are Missing at Random (MAR) or Missing Completely at Random (MCAR). Non-random structural block missingness may bias estimates.",
            "Subspace Rank Dependency: If the chosen rank k deviates significantly from true latent rank, systematic subspace bias can occur.",
            "Small Sample Uncertainty: In very small matrices (< 10 rows), resampling variance has higher finite-sample variance."
        ]

        return ConfidenceReport(
            records=updated_records,
            mean_confidence=mean_conf,
            median_confidence=med_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            n_high_confidence=n_high,
            n_moderate_confidence=n_mod,
            n_low_confidence=n_low,
            review_threshold=review_threshold,
            method_description=method_desc,
            limitations=limitations
        )
