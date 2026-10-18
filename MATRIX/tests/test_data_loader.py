"""Unit tests for data loader and synthetic generator module."""

import io
import pytest
import numpy as np
import pandas as pd

from core.data_loader import (
    load_file_data,
    extract_numeric_matrix,
    generate_sensor_network_matrix,
    generate_collaborative_ratings_matrix,
    generate_financial_metrics_matrix,
    inject_synthetic_errors_into_df
)


def test_load_csv_from_buffer():
    csv_content = "ID,Sensor_A,Sensor_B\n1,10.5,20.2\n2,11.0,21.5\n3,10.8,20.9\n"
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    df = load_file_data(buffer, file_name="test.csv")

    assert len(df) == 3
    assert list(df.columns) == ["ID", "Sensor_A", "Sensor_B"]


def test_load_duplicate_columns():
    csv_content = "A,A,B\n1,2,3\n4,5,6\n"
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    df = load_file_data(buffer, file_name="dup.csv")

    assert len(df.columns) == 3
    # Ensure all column names are distinct
    assert len(set(df.columns)) == 3
    assert "A" in df.columns
    assert "B" in df.columns


def test_empty_file_error():
    buffer = io.BytesIO(b"")
    with pytest.raises(ValueError, match="empty"):
        load_file_data(buffer, file_name="empty.csv")


def test_extract_numeric_matrix_guardrails():
    # Only non-numeric
    df_non_num = pd.DataFrame({"Text1": ["a", "b"], "Text2": ["c", "d"]})
    with pytest.raises(ValueError, match="No numeric columns found"):
        extract_numeric_matrix(df_non_num)

    # Matrix too small (< 2x2)
    df_small = pd.DataFrame({"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="too small"):
        extract_numeric_matrix(df_small)


def test_synthetic_sensor_generation():
    df, meta = generate_sensor_network_matrix(n_timesteps=20, n_sensors=8, missing_rate=0.1, error_rate=0.05, seed=42)
    assert len(df) == 20
    assert "Timestamp" in df.columns
    assert len([c for c in df.columns if c.startswith("Sensor_")]) == 8
    assert meta["mask_missing"].sum() > 0
    assert meta["mask_error"].sum() > 0


def test_synthetic_error_injection():
    df_clean = pd.DataFrame({
        "A": np.linspace(10, 50, 20),
        "B": np.linspace(20, 100, 20)
    })
    corrupted_df, meta = inject_synthetic_errors_into_df(
        df_clean, missing_fraction=0.15, error_fraction=0.10, outlier_scale=5.0, seed=42
    )

    assert corrupted_df.isna().sum().sum() > 0
    assert meta["mask_error"].sum() > 0
