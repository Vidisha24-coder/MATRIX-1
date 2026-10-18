"""Plotly Interactive Visualizations Component with Ultra-Modern Styling."""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from core.pattern_analysis import SVDAnalysisResult
from core.confidence import ConfidenceReport
from core.reconstruction import ReconstructedCellRecord

# Modern Plotly Base Theme
PLOTLY_FONT = dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color="#334155")


def plot_matrix_status_heatmap(
    status_grid: np.ndarray,
    col_names: List[str],
    row_labels: Optional[List[str]] = None
) -> go.Figure:
    """Plot interactive heatmap showing data quality and detected error states."""
    n_rows, n_cols = status_grid.shape
    if row_labels and len(row_labels) == n_rows:
        labels = list(row_labels)
    else:
        labels = [f"Row {i}" for i in range(n_rows)]

    if col_names and len(col_names) == n_cols:
        cols = list(col_names)
    else:
        cols = [f"Col {j}" for j in range(n_cols)]

    # Modern Harmonious Palette: Valid=Emerald, Missing=Amber, Outlier=Rose, User=Violet
    colorscale = [
        [0.00, "#10b981"],  # Valid / Clean (0)
        [0.33, "#f59e0b"],  # Missing (1)
        [0.66, "#ef4444"],  # Statistical Outlier (2)
        [1.00, "#a855f7"]   # User Marked Corrupt (3)
    ]

    status_text_map = {
        0: "Valid / Clean",
        1: "Missing (NaN)",
        2: "Statistical Outlier",
        3: "User Marked Corrupt"
    }

    custom_text = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        r_str = labels[r] if r < len(labels) else f"Row {r}"
        for c in range(n_cols):
            c_str = cols[c] if c < len(cols) else f"Col {c}"
            st = status_grid[r, c]
            custom_text[r, c] = (
                f"<b>Sample / Row:</b> {r_str}<br>"
                f"<b>Feature:</b> {c_str}<br>"
                f"<b>Quality Status:</b> {status_text_map.get(st, 'Unknown')}"
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=status_grid,
            x=cols,
            y=labels,
            text=custom_text,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            xgap=1,
            ygap=1
        )
    )

    fig.update_layout(
        title=dict(text="Matrix Quality & Anomaly Detection Map", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Matrix Columns / Features", tickangle=-30),
        yaxis=dict(title="Samples / Rows", autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=min(600, max(360, n_rows * 20)),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc"
    )
    return fig


def plot_svd_energy_spectrum(svd_result: SVDAnalysisResult, chosen_rank: int = 2) -> go.Figure:
    """Plot Scree plot of singular values and cumulative energy curve with rank indicator."""
    s_vals = svd_result.singular_values
    cum_energy = svd_result.cumulative_energy * 100
    ranks = list(range(1, len(s_vals) + 1))

    fig = go.Figure()

    # Singular values bar chart
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=s_vals,
            name="Singular Value (σ)",
            marker=dict(
                color="#4ba3e3",
                line=dict(color="#3896d8", width=1)
            ),
            opacity=0.85,
            yaxis="y1"
        )
    )

    # Cumulative energy line
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=cum_energy,
            name="Cumulative Variance (%)",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=3, shape="spline"),
            marker=dict(size=8, color="#d97706", line=dict(color="#ffffff", width=1.5)),
            yaxis="y2"
        )
    )

    # Vertical line at chosen rank
    fig.add_vline(
        x=chosen_rank,
        line_width=2,
        line_dash="dash",
        line_color="#10b981",
        annotation_text=f"Selected Rank k={chosen_rank}",
        annotation_position="top left",
        annotation_font=dict(color="#059669", weight="bold")
    )

    # 90% variance threshold line
    fig.add_hline(
        y=90.0,
        line_width=1.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="90% Variance Target",
        annotation_position="bottom right",
        annotation_font=dict(color="#64748b"),
        yref="y2"
    )

    fig.update_layout(
        title=dict(text="Singular Value Spectrum (Scree Plot) & Cumulative Energy", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Subspace Rank (k)", tickmode="linear", dtick=1, gridcolor="#f1f5f9"),
        yaxis=dict(title="Singular Value (σ)", side="left", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(
            title="Cumulative Energy (%)",
            side="right",
            overlaying="y",
            range=[0, 105],
            showgrid=False
        ),
        legend=dict(x=0.55, y=0.30, bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff"
    )
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Plot pairwise feature correlation matrix heatmap."""
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Feature Correlation Matrix"
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff"
    )
    return fig


def plot_confidence_distribution(
    confidence_report: ConfidenceReport,
    review_threshold: float = 0.70
) -> go.Figure:
    """Plot histogram of reconstruction confidence scores with review threshold cutoff."""
    scores = [r.confidence_score * 100 for r in confidence_report.records]

    if not scores:
        fig = go.Figure()
        fig.add_annotation(text="No reconstructed cells to evaluate", showarrow=False)
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=15,
            marker=dict(color="#0284c7", line=dict(color="#0369a1", width=1)),
            opacity=0.85,
            name="Confidence Scores"
        )
    )

    # Threshold cutoff line
    thresh_pct = review_threshold * 100
    fig.add_vline(
        x=thresh_pct,
        line_width=2.5,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Review Cutoff ({int(thresh_pct)}%)",
        annotation_position="top right",
        annotation_font=dict(color="#b91c1c", weight="bold")
    )

    fig.update_layout(
        title=dict(text="Confidence Score Distribution Across Reconstructed Cells", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Estimated Confidence (%)", range=[0, 100], gridcolor="#f1f5f9"),
        yaxis=dict(title="Number of Cells", gridcolor="#f1f5f9"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff"
    )
    return fig


def plot_corrections_scatter(cell_records: List[ReconstructedCellRecord]) -> go.Figure:
    """Plot scatter of original vs reconstructed values for outlier and corrupted cells."""
    records_with_orig = [r for r in cell_records if r.original_val is not None]

    if not records_with_orig:
        fig = go.Figure()
        fig.add_annotation(
            text="No previous original values to compare (all targets were missing NaNs)",
            showarrow=False
        )
        return fig

    df_plot = pd.DataFrame([
        {
            "Original": r.original_val,
            "Reconstructed": r.reconstructed_val,
            "Confidence": r.confidence_score * 100,
            "Cell": f"{r.row_label}, {r.col_name}",
            "Reason": r.reason
        }
        for r in records_with_orig
    ])

    fig = px.scatter(
        df_plot,
        x="Original",
        y="Reconstructed",
        color="Confidence",
        hover_name="Cell",
        hover_data={"Reason": True, "Confidence": ":.1f%"},
        color_continuous_scale="Viridis",
        title="Original vs. Reconstructed Values (Flagged Cells)"
    )

    # Add 45-degree reference line
    min_val = min(df_plot["Original"].min(), df_plot["Reconstructed"].min())
    max_val = max(df_plot["Original"].max(), df_plot["Reconstructed"].max())
    fig.add_shape(
        type="line",
        x0=min_val,
        y0=min_val,
        x1=max_val,
        y1=max_val,
        line=dict(color="#94a3b8", dash="dash"),
    )

    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff"
    )
    return fig


def plot_feature_distribution_comparison(
    orig_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    col_name: str
) -> go.Figure:
    """Compare feature distribution before and after error correction."""
    fig = go.Figure()

    orig_s = orig_df[col_name].dropna()
    corr_s = corrected_df[col_name].dropna()

    fig.add_trace(
        go.Box(
            y=orig_s,
            name="Original (with outliers/NaNs)",
            marker_color="#ef4444",
            boxpoints="outliers",
            jitter=0.3
        )
    )

    fig.add_trace(
        go.Box(
            y=corr_s,
            name="Reconstructed / Corrected",
            marker_color="#10b981",
            boxpoints="outliers",
            jitter=0.3
        )
    )

    fig.update_layout(
        title=dict(text=f"Feature Distribution Before vs After: '{col_name}'", font=dict(size=15, weight="bold")),
        yaxis_title="Value Range",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        font=PLOTLY_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff"
    )
    return fig
