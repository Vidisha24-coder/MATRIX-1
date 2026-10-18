"""Data Loading, Validation, and Synthetic Matrix Generation Module."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import io
import numpy as np
import pandas as pd


@dataclass
class MatrixDataset:
    """Encapsulates raw dataset, selected numeric matrix, and metadata."""
    raw_df: pd.DataFrame
    numeric_df: pd.DataFrame
    numeric_columns: List[str]
    non_numeric_columns: List[str]
    matrix_array: np.ndarray
    original_copy: pd.DataFrame
    dataset_name: str


def load_file_data(
    file_or_path: Union[str, io.BytesIO],
    file_name: str = "dataset"
) -> pd.DataFrame:
    """Load CSV or Excel data into a pandas DataFrame.
    
    Handles encoding, delimiter detection, and duplicate column names.
    """
    if file_or_path is None:
        raise ValueError("No file provided.")

    is_excel = (
        file_name.lower().endswith(".xlsx") or
        file_name.lower().endswith(".xls")
    )

    try:
        if is_excel:
            df = pd.read_excel(file_or_path)
        else:
            # Try UTF-8 first, fallback to latin-1
            try:
                if isinstance(file_or_path, (str, io.BytesIO)):
                    df = pd.read_csv(file_or_path)
                else:
                    df = pd.read_csv(file_or_path)
            except UnicodeDecodeError:
                if isinstance(file_or_path, io.BytesIO):
                    file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Failed to parse file '{file_name}': {str(e)}")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    # Deduplicate column names if needed
    cols = []
    seen = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in seen:
            seen[col_str] += 1
            cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            cols.append(col_str)
    df.columns = cols

    return df


def extract_numeric_matrix(
    df: pd.DataFrame,
    selected_columns: Optional[List[str]] = None,
    dataset_name: str = "matrix_data"
) -> MatrixDataset:
    """Validate and extract a numeric matrix from DataFrame.
    
    Replaces infinite values with NaN and ensures proper shape.
    """
    raw_df = df.copy()

    # Determine candidate numeric columns
    detected_numeric = []
    non_numeric = []
    for col in df.columns:
        # Check if column is already numeric or coercible to numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        # If at least 40% of non-null entries are numeric, treat as numeric candidate
        valid_count = converted.notna().sum()
        total_non_null = df[col].notna().sum()
        if total_non_null > 0 and (valid_count / total_non_null) >= 0.4:
            detected_numeric.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            detected_numeric.append(col)
        else:
            non_numeric.append(col)

    if selected_columns is None:
        target_cols = detected_numeric
    else:
        target_cols = [c for c in selected_columns if c in df.columns]

    if not target_cols:
        raise ValueError(
            "No numeric columns found or selected. "
            "Please select at least 2 numeric columns for matrix analysis."
        )

    # Convert selected columns to float and replace inf with NaN
    numeric_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce").astype(float)
        # Replace infinite values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        numeric_df[col] = series

    # Guardrails: matrix shape
    n_rows, n_cols = numeric_df.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError(
            f"Matrix dimension ({n_rows}x{n_cols}) is too small. "
            "Matrix must have at least 2 rows and 2 columns for SVD and reconstruction."
        )

    # Check if completely empty
    if numeric_df.isna().all().all():
        raise ValueError("The selected matrix contains only missing values.")

    matrix_array = numeric_df.to_numpy(dtype=float, copy=True)

    return MatrixDataset(
        raw_df=raw_df,
        numeric_df=numeric_df,
        numeric_columns=target_cols,
        non_numeric_columns=[c for c in df.columns if c not in target_cols],
        matrix_array=matrix_array,
        original_copy=numeric_df.copy(deep=True),
        dataset_name=dataset_name,
    )


# =====================================================================
# Synthetic Dataset Generators for Testing and Demonstration
# =====================================================================

def generate_sensor_network_matrix(
    n_timesteps: int = 35,
    n_sensors: int = 10,
    true_rank: int = 3,
    missing_rate: float = 0.08,
    error_rate: float = 0.04,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Generate synthetic environmental sensor data with low-rank physics."""
    rng = np.random.default_rng(seed)

    # Generate latent temporal factors (e.g., diurnal cycle, ambient trend, wind shift)
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    f1 = 22.0 + 8.0 * np.sin(t)
    f2 = 5.0 * np.cos(t / 2.0)
    f3 = 2.0 * np.sin(2.5 * t)
    factors = np.vstack([f1, f2, f3]).T  # Shape: (n_timesteps, 3)

    # Sensor sensitivity / spatial mixing weights
    weights = rng.uniform(0.6, 1.4, size=(true_rank, n_sensors))
    base_offsets = rng.uniform(-3.0, 3.0, size=(1, n_sensors))

    # Low-rank clean matrix
    clean_matrix = factors @ weights + base_offsets
    # Add subtle measurement noise
    noise = rng.normal(0, 0.4, size=clean_matrix.shape)
    matrix = clean_matrix + noise

    ground_truth = matrix.copy()
    mask_missing = np.zeros(matrix.shape, dtype=bool)
    mask_error = np.zeros(matrix.shape, dtype=bool)

    # Inject missing values (NaN)
    total_cells = matrix.size
    n_missing = int(total_cells * missing_rate)
    all_indices = [(r, c) for r in range(matrix.shape[0]) for c in range(matrix.shape[1])]
    rng.shuffle(all_indices)

    missing_idx = all_indices[:n_missing]
    for r, c in missing_idx:
        matrix[r, c] = np.nan
        mask_missing[r, c] = True

    # Inject corrupted outliers (sensor glitch, spike, zero stuck)
    remaining_idx = all_indices[n_missing:n_missing + int(total_cells * error_rate)]
    for r, c in remaining_idx:
        corruption_type = rng.choice(["high_spike", "low_spike", "freeze_zero"])
        if corruption_type == "high_spike":
            matrix[r, c] = matrix[r, c] + rng.uniform(25.0, 50.0)
        elif corruption_type == "low_spike":
            matrix[r, c] = matrix[r, c] - rng.uniform(25.0, 40.0)
        else:
            matrix[r, c] = 0.0
        mask_error[r, c] = True

    sensor_names = [f"Sensor_{i+1:02d}" for i in range(n_sensors)]
    df = pd.DataFrame(matrix, columns=sensor_names)
    df.insert(0, "Timestamp", [f"T+{i:02d}:00" for i in range(n_timesteps)])

    meta = {
        "ground_truth": ground_truth,
        "mask_missing": mask_missing,
        "mask_error": mask_error,
        "true_rank": true_rank,
        "name": "Synthetic Sensor Network (Low-Rank Environmental Data)"
    }
    return df, meta


def generate_collaborative_ratings_matrix(
    n_users: int = 24,
    n_items: int = 8,
    true_rank: int = 2,
    missing_rate: float = 0.12,
    error_rate: float = 0.04,
    seed: int = 101
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Generate synthetic user-item ratings (1.0 to 5.0 stars) with latent taste factors."""
    rng = np.random.default_rng(seed)

    # User latent affinities (e.g., preference for action vs drama)
    user_factors = rng.uniform(0.5, 2.5, size=(n_users, true_rank))
    item_factors = rng.uniform(0.5, 2.0, size=(true_rank, n_items))

    clean_ratings = user_factors @ item_factors
    # Normalize to 1.0 - 5.0 scale
    clean_ratings = 1.0 + 4.0 * (clean_ratings - clean_ratings.min()) / (clean_ratings.max() - clean_ratings.min() + 1e-6)
    matrix = clean_ratings.copy()

    total_cells = matrix.size
    all_indices = [(r, c) for r in range(n_users) for c in range(n_items)]
    rng.shuffle(all_indices)

    mask_missing = np.zeros(matrix.shape, dtype=bool)
    mask_error = np.zeros(matrix.shape, dtype=bool)

    n_missing = int(total_cells * missing_rate)
    for r, c in all_indices[:n_missing]:
        matrix[r, c] = np.nan
        mask_missing[r, c] = True

    n_error = int(total_cells * error_rate)
    for r, c in all_indices[n_missing:n_missing + n_error]:
        # Corrupted rating outside valid 1-5 range or inverted extreme
        matrix[r, c] = rng.choice([25.0, 99.0, -10.0, 50.0])
        mask_error[r, c] = True

    item_names = [f"Item_{chr(65+i)}" for i in range(n_items)]
    df = pd.DataFrame(np.round(matrix, 2), columns=item_names)
    df.insert(0, "User_ID", [f"User_{i+1:03d}" for i in range(n_users)])

    meta = {
        "ground_truth": clean_ratings,
        "mask_missing": mask_missing,
        "mask_error": mask_error,
        "true_rank": true_rank,
        "name": "Collaborative Ratings Matrix (1-5 Star Ratings)"
    }
    return df, meta


def generate_financial_metrics_matrix(
    n_assets: int = 18,
    n_indicators: int = 7,
    true_rank: int = 2,
    missing_rate: float = 0.07,
    error_rate: float = 0.03,
    seed: int = 202
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Generate financial asset indicators (P/E, Beta, Volatility, ROI, etc.)."""
    rng = np.random.default_rng(seed)

    market_factor = rng.normal(0, 1, size=(n_assets, 1))
    sector_factor = rng.normal(0, 1, size=(n_assets, 1))
    factors = np.hstack([market_factor, sector_factor])

    loadings = rng.uniform(0.5, 2.0, size=(2, n_indicators))
    means = np.array([18.0, 1.15, 0.22, 12.5, 4.2, 1.45, 8.8])[:n_indicators]
    stds = np.array([5.0, 0.35, 0.08, 4.0, 1.5, 0.40, 2.5])[:n_indicators]

    clean = (factors @ loadings) * stds + means
    matrix = clean.copy()

    total_cells = matrix.size
    all_indices = [(r, c) for r in range(n_assets) for c in range(n_indicators)]
    rng.shuffle(all_indices)

    mask_missing = np.zeros(matrix.shape, dtype=bool)
    mask_error = np.zeros(matrix.shape, dtype=bool)

    n_missing = int(total_cells * missing_rate)
    for r, c in all_indices[:n_missing]:
        matrix[r, c] = np.nan
        mask_missing[r, c] = True

    n_error = int(total_cells * error_rate)
    for r, c in all_indices[n_missing:n_missing + n_error]:
        # Outlier multiplier
        matrix[r, c] = matrix[r, c] * rng.choice([5.0, 8.0, -3.0])
        mask_error[r, c] = True

    indicator_names = ["PE_Ratio", "Beta", "Volatility", "ROI_Pct", "Debt_Ratio", "Quick_Ratio", "Margin_Pct"][:n_indicators]
    df = pd.DataFrame(np.round(matrix, 2), columns=indicator_names)
    df.insert(0, "Asset_Ticker", [f"EQ_{i+1:02d}" for i in range(n_assets)])

    meta = {
        "ground_truth": clean,
        "mask_missing": mask_missing,
        "mask_error": mask_error,
        "true_rank": true_rank,
        "name": "Financial Asset Multi-Metric Matrix"
    }
    return df, meta


def inject_synthetic_errors_into_df(
    df_numeric: pd.DataFrame,
    missing_fraction: float = 0.10,
    error_fraction: float = 0.05,
    outlier_scale: float = 5.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Inject controlled missingness and outliers into any numeric DataFrame for benchmarking."""
    rng = np.random.default_rng(seed)
    arr = df_numeric.to_numpy(dtype=float, copy=True)
    n_rows, n_cols = arr.shape
    total_cells = arr.size

    ground_truth = arr.copy()
    mask_missing = np.zeros(arr.shape, dtype=bool)
    mask_error = np.zeros(arr.shape, dtype=bool)

    # Find already observed positions
    observed_idx = [(r, c) for r in range(n_rows) for c in range(n_cols) if not np.isnan(arr[r, c])]
    rng.shuffle(observed_idx)

    n_missing = int(len(observed_idx) * missing_fraction)
    for r, c in observed_idx[:n_missing]:
        arr[r, c] = np.nan
        mask_missing[r, c] = True

    rem = observed_idx[n_missing:]
    n_error = int(len(observed_idx) * error_fraction)
    for r, c in rem[:n_error]:
        col_std = np.nanstd(ground_truth[:, c])
        if col_std == 0 or np.isnan(col_std):
            col_std = 1.0
        delta = rng.choice([-1.0, 1.0]) * outlier_scale * col_std
        arr[r, c] = arr[r, c] + delta
        mask_error[r, c] = True

    corrupted_df = pd.DataFrame(arr, columns=df_numeric.columns, index=df_numeric.index)
    meta = {
        "ground_truth": ground_truth,
        "mask_missing": mask_missing,
        "mask_error": mask_error,
        "name": "Custom Injected Matrix"
    }
    return corrupted_df, meta
