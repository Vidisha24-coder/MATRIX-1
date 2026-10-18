"""Matrix Reconstruction and Error Correction - Ultra-Modern Interactive Application."""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.data_loader import (
    MatrixDataset,
    load_file_data,
    extract_numeric_matrix,
    generate_sensor_network_matrix,
    generate_collaborative_ratings_matrix,
    generate_financial_metrics_matrix,
    inject_synthetic_errors_into_df
)
from core.error_detection import (
    aggregate_detected_entries,
    DetectedEntry
)
from core.pattern_analysis import (
    PatternAnalyzer,
    SVDAnalysisResult
)
from core.reconstruction import (
    run_matrix_reconstruction,
    ReconstructionResult,
    ReconstructedCellRecord
)
from core.confidence import (
    ConfidenceEstimator,
    ConfidenceReport
)
from core.report_generator import ReportGenerator
import importlib
import components.ui_helpers
import components.plots
importlib.reload(components.ui_helpers)
importlib.reload(components.plots)

from components.ui_helpers import (
    inject_custom_css,
    render_top_navbar,
    render_hero_banner,
    render_step_flow_bar,
    render_kpi_card
)
from components.plots import (
    plot_matrix_status_heatmap,
    plot_svd_energy_spectrum,
    plot_confidence_distribution,
    plot_corrections_scatter,
    plot_feature_distribution_comparison,
    plot_correlation_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Matrix Reconstruction & Error Correction",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern CSS
inject_custom_css()

# Session State Initialization
if "matrix_dataset" not in st.session_state:
    st.session_state.matrix_dataset = None
if "detected_entries" not in st.session_state:
    st.session_state.detected_entries = []
if "status_grid" not in st.session_state:
    st.session_state.status_grid = None
if "detection_counts" not in st.session_state:
    st.session_state.detection_counts = {}
if "user_corrupted_cells" not in st.session_state:
    st.session_state.user_corrupted_cells = set()
if "svd_result" not in st.session_state:
    st.session_state.svd_result = None
if "reconstruction_result" not in st.session_state:
    st.session_state.reconstruction_result = None
if "confidence_report" not in st.session_state:
    st.session_state.confidence_report = None
if "cell_records" not in st.session_state:
    st.session_state.cell_records = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "1. Data Input"

# Render top Talegri navbar & hero header
render_top_navbar()
render_hero_banner()

# Sidebar: Global Workflow Navigation
st.sidebar.markdown("### 🧭 Architecture Workflow")
steps = [
    "1. Data Input",
    "2. Detect Errors",
    "3. Pattern Analysis",
    "4. Reconstruct & Correct",
    "5. Estimate Confidence",
    "6. Results Dashboard & Export"
]

selected_step = st.sidebar.radio(
    "Select Stage:",
    steps,
    index=steps.index(st.session_state.current_tab) if st.session_state.current_tab in steps else 0
)
st.session_state.current_tab = selected_step

# Step index for visual stepper bar
step_num = int(selected_step[0])
render_step_flow_bar(current_step=step_num)

# Quick Sidebar Status Badges
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Live Engine Status")
if st.session_state.matrix_dataset is not None:
    ds = st.session_state.matrix_dataset
    n_r, n_c = ds.numeric_df.shape
    st.sidebar.success(f"**Matrix Active:** {n_r} × {n_c} ({n_r * n_c} cells)\n\n*{ds.dataset_name}*")
else:
    st.sidebar.info("Awaiting Matrix Ingestion in Step 1.")

if st.session_state.detected_entries:
    c = st.session_state.detection_counts
    st.sidebar.warning(
        f"**Flagged Cells:** {c.get('total_flagged', 0)}\n\n"
        f"• Missing: **{c.get('missing', 0)}**\n\n"
        f"• Outliers: **{c.get('outliers', 0)}**\n\n"
        f"• User-Flagged: **{c.get('user_corrupted', 0)}**"
    )

if st.session_state.confidence_report is not None:
    cr = st.session_state.confidence_report
    st.sidebar.success(
        f"**Reconstructed:** {len(cr.records)} cells\n\n"
        f"• Mean Confidence: **{cr.mean_confidence*100:.1f}%**\n\n"
        f"• Review Needed: **{cr.n_low_confidence}**"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Google DeepMind Advanced Agentic Coding • System Architecture Flow v2.0")


# ==============================================================================
# TAB 1: DATA INPUT
# ==============================================================================
if selected_step == "1. Data Input":
    st.markdown("### 📥 Step 1: Data Ingestion & Matrix Configuration")
    st.write(
        "Upload a dataset containing your matrix (CSV or Excel) or select one of the built-in benchmarks. "
        "The original data is preserved unchanged for audit and downloads."
    )

    # Fast 1-Click Benchmark Loaders
    st.markdown("#### ⚡ Quick-Start Benchmarks")
    b_col1, b_col2, b_col3 = st.columns(3)
    
    with b_col1:
        if st.button("🌡️ Load Sensor Network (24×6)", use_container_width=True):
            df_sensors, _ = generate_sensor_network_matrix(n_timesteps=24, n_sensors=6, seed=42)
            dataset = extract_numeric_matrix(df_sensors, dataset_name="Sensor_Network_Benchmark")
            if "Timestamp" in df_sensors.columns:
                dataset.numeric_df.index = df_sensors["Timestamp"].astype(str)
            st.session_state.matrix_dataset = dataset
            st.session_state.detected_entries = []
            st.session_state.reconstruction_result = None
            st.session_state.confidence_report = None
            st.session_state.user_corrupted_cells = set()
            st.session_state.current_tab = "2. Detect Errors"
            st.rerun()

    with b_col2:
        if st.button("⭐ Load Ratings Matrix (20×8)", use_container_width=True):
            df_ratings, _ = generate_collaborative_ratings_matrix(n_users=20, n_items=8, seed=101)
            dataset = extract_numeric_matrix(df_ratings, dataset_name="Customer_Ratings_Benchmark")
            if "User_ID" in df_ratings.columns:
                dataset.numeric_df.index = df_ratings["User_ID"].astype(str)
            st.session_state.matrix_dataset = dataset
            st.session_state.detected_entries = []
            st.session_state.reconstruction_result = None
            st.session_state.confidence_report = None
            st.session_state.user_corrupted_cells = set()
            st.session_state.current_tab = "2. Detect Errors"
            st.rerun()

    with b_col3:
        if st.button("📈 Load Financial Metrics (16×7)", use_container_width=True):
            df_fin, _ = generate_financial_metrics_matrix(n_assets=16, n_indicators=7, seed=202)
            dataset = extract_numeric_matrix(df_fin, dataset_name="Financial_Metrics_Benchmark")
            if "Asset_Ticker" in df_fin.columns:
                dataset.numeric_df.index = df_fin["Asset_Ticker"].astype(str)
            st.session_state.matrix_dataset = dataset
            st.session_state.detected_entries = []
            st.session_state.reconstruction_result = None
            st.session_state.confidence_report = None
            st.session_state.user_corrupted_cells = set()
            st.session_state.current_tab = "2. Detect Errors"
            st.rerun()

    st.markdown("---")

    data_source = st.radio(
        "Or Choose Custom Ingestion Method:",
        [
            "Upload Custom File (CSV / Excel)",
            "Inject Synthetic Errors into Clean Data (Benchmark Mode)"
        ],
        horizontal=True
    )

    df_raw = None
    dataset_name = "matrix_data"

    if data_source == "Upload Custom File (CSV / Excel)":
        uploaded_file = st.file_uploader(
            "Upload matrix file (.csv or .xlsx)",
            type=["csv", "xlsx", "xls"],
            help="Upload tabular data. Numeric columns will be extracted into the analysis matrix."
        )
        if uploaded_file is not None:
            try:
                df_raw = load_file_data(uploaded_file, file_name=uploaded_file.name)
                dataset_name = uploaded_file.name.split(".")[0]
            except Exception as e:
                st.error(f"Error loading file: {e}")

    else:
        st.info("Synthetic Error Injection allows you to take clean sample data and inject known missing/corrupted cells to test reconstruction accuracy.")
        clean_choice = st.selectbox(
            "Select Base Clean Dataset:",
            ["Environmental Sensors", "Financial Indicators"]
        )
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            missing_pct = st.slider("Missing Fraction (%)", 0, 30, 10) / 100.0
        with col_i2:
            error_pct = st.slider("Outlier Corruption Fraction (%)", 0, 15, 5) / 100.0
        with col_i3:
            scale_mult = st.slider("Outlier Magnitude (x Std)", 2.0, 10.0, 5.0)

        if st.button("Generate Corrupted Matrix", type="primary"):
            if clean_choice == "Environmental Sensors":
                df_clean, _ = generate_sensor_network_matrix(30, 10, missing_rate=0, error_rate=0, seed=42)
            else:
                df_clean, _ = generate_financial_metrics_matrix(16, 7, missing_rate=0, error_rate=0, seed=202)

            num_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
            corrupted_num, meta = inject_synthetic_errors_into_df(
                df_clean[num_cols], missing_fraction=missing_pct, error_fraction=error_pct, outlier_scale=scale_mult
            )
            df_raw = df_clean.copy()
            for c in num_cols:
                df_raw[c] = corrupted_num[c]
            dataset_name = f"Injected_Benchmark_M{int(missing_pct*100)}_E{int(error_pct*100)}"
            st.success("Successfully generated matrix with controlled ground-truth errors!")

    # If data is uploaded or injected, configure numeric matrix
    if df_raw is not None:
        st.markdown("---")
        st.subheader("Matrix Preview & Column Configuration")

        col_prev1, col_prev2 = st.columns([3, 1])
        with col_prev1:
            st.write(f"**Original Raw Data Preview** ({df_raw.shape[0]} rows × {df_raw.shape[1]} columns):")
            st.dataframe(df_raw.head(10), use_container_width=True)

        candidate_numeric = []
        for c in df_raw.columns:
            s_num = pd.to_numeric(df_raw[c], errors="coerce")
            if s_num.notna().sum() > 0:
                candidate_numeric.append(c)

        with col_prev2:
            st.markdown("**Matrix Column Configuration**")
            selected_cols = st.multiselect(
                "Select numeric matrix features:",
                options=df_raw.columns.tolist(),
                default=candidate_numeric,
                help="Only selected numeric columns will be included in the SVD reconstruction matrix."
            )

            id_col = st.selectbox(
                "Row Label Column (Optional):",
                options=["Auto (Row Index)"] + df_raw.columns.tolist(),
                index=0
            )

        if st.button("Confirm Matrix and Proceed to Step 2 ➔", type="primary"):
            try:
                dataset = extract_numeric_matrix(
                    df=df_raw,
                    selected_columns=selected_cols,
                    dataset_name=dataset_name
                )
                if id_col != "Auto (Row Index)" and id_col in df_raw.columns:
                    dataset.numeric_df.index = df_raw[id_col].astype(str)

                st.session_state.matrix_dataset = dataset
                st.session_state.detected_entries = []
                st.session_state.reconstruction_result = None
                st.session_state.confidence_report = None
                st.session_state.user_corrupted_cells = set()
                st.success(f"Matrix successfully configured: {dataset.numeric_df.shape[0]} rows × {dataset.numeric_df.shape[1]} columns.")
                st.session_state.current_tab = "2. Detect Errors"
                st.rerun()
            except ValueError as e:
                st.error(f"Matrix validation error: {e}")


# ==============================================================================
# TAB 2: DETECT ERRORS
# ==============================================================================
elif selected_step == "2. Detect Errors":
    st.markdown("### 🔍 Step 2: Detect Errors & Unreliable Entries")
    st.write(
        "Identify and distinguish **Missing Values (NaN)**, **Potential Statistical Outliers**, and **User-Marked Corruptions**. "
        "Adjust statistical thresholds as appropriate for your domain. *Statistical outliers are candidates for inspection, not automatically discarded.*"
    )

    if st.session_state.matrix_dataset is None:
        st.warning("Please load or select a matrix in Step 1 first.")
    else:
        dataset = st.session_state.matrix_dataset
        matrix = dataset.matrix_array
        col_names = dataset.numeric_columns
        row_labels = [str(idx) for idx in dataset.numeric_df.index]

        col_opt1, col_opt2 = st.columns([1, 2])

        with col_opt1:
            st.markdown("#### Detection Settings")
            detection_method = st.selectbox(
                "Outlier Detection Method:",
                [
                    ("robust_zscore", "Robust Z-Score (Median Absolute Deviation)"),
                    ("isolation_forest", "Isolation Forest (Scikit-Learn ML)"),
                    ("iqr_fence", "Interquartile Range (IQR Fence)"),
                    ("custom_range", "Domain Value Bounds (Min/Max)")
                ],
                format_func=lambda x: x[1]
            )[0]

            z_thresh = 3.5
            iqr_mult = 1.5
            iso_contam = 0.05
            min_bound = None
            max_bound = None

            if detection_method == "robust_zscore":
                z_thresh = st.slider(
                    "Robust Z-Score Threshold (MAD):",
                    min_value=2.0, max_value=6.0, value=3.5, step=0.1,
                    help="Cells with |0.6745 * (x - median) / MAD| > threshold will be flagged as potential outliers."
                )
            elif detection_method == "isolation_forest":
                iso_contam = st.slider(
                    "Isolation Forest Contamination Rate:",
                    min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                    help="Expected proportion of anomalies in the dataset."
                )
            elif detection_method == "iqr_fence":
                iqr_mult = st.slider(
                    "IQR Multiplier (k):",
                    min_value=1.0, max_value=3.5, value=1.5, step=0.1,
                    help="Flag values outside [Q1 - k*IQR, Q3 + k*IQR]. 1.5 = moderate outliers; 3.0 = extreme."
                )
            elif detection_method == "custom_range":
                c_min, c_max = st.columns(2)
                with c_min:
                    use_min = st.checkbox("Min Bound")
                    if use_min:
                        min_bound = st.number_input("Minimum Allowed Value", value=0.0)
                with c_max:
                    use_max = st.checkbox("Max Bound")
                    if use_max:
                        max_bound = st.number_input("Maximum Allowed Value", value=100.0)

            # Manual Cell Flagging Tool
            with st.expander("🛠️ Manual Cell Corrupt/Flag Tool"):
                st.caption("Explicitly mark a cell as corrupted by specifying row & column:")
                manual_r = st.selectbox("Row Label:", options=row_labels)
                manual_c = st.selectbox("Column:", options=col_names)
                r_idx = row_labels.index(manual_r)
                c_idx = col_names.index(manual_c)

                cur_flagged = (r_idx, c_idx) in st.session_state.user_corrupted_cells
                if cur_flagged:
                    if st.button("Unflag this Cell", key="unflag_btn"):
                        st.session_state.user_corrupted_cells.remove((r_idx, c_idx))
                        st.rerun()
                else:
                    if st.button("Flag Cell as Corrupted", key="flag_btn"):
                        st.session_state.user_corrupted_cells.add((r_idx, c_idx))
                        st.rerun()

            if st.button("Run Error Detection", type="primary", use_container_width=True):
                entries, status_grid, counts = aggregate_detected_entries(
                    matrix=matrix,
                    col_names=col_names,
                    row_labels=row_labels,
                    method=detection_method,
                    zscore_threshold=z_thresh,
                    iqr_multiplier=iqr_mult,
                    iso_contamination=iso_contam,
                    range_min=min_bound,
                    range_max=max_bound,
                    user_corrupted_cells=st.session_state.user_corrupted_cells
                )
                st.session_state.detected_entries = entries
                st.session_state.status_grid = status_grid
                st.session_state.detection_counts = counts
                st.success(f"Detection completed! Flagged {counts['total_flagged']} unreliable cells.")

        # Display detection findings
        with col_opt2:
            if st.session_state.status_grid is not None:
                counts = st.session_state.detection_counts
                st.markdown("#### Detection Quality Summary")

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    render_kpi_card("Clean Observed", f"{counts['clean']}", f"{(counts['clean']/counts['total_cells'])*100:.1f}% of cells", "success", "✅")
                with k2:
                    render_kpi_card("Missing (NaN)", f"{counts['missing']}", "Unobserved cells", "warning", "⚠️")
                with k3:
                    render_kpi_card("Statistical Outliers", f"{counts['outliers']}", "Anomalous entries", "danger", "🚨")
                with k4:
                    render_kpi_card("User Marked", f"{counts['user_corrupted']}", "Manual overrides", "purple", "🛠️")

                fig_status = plot_matrix_status_heatmap(st.session_state.status_grid, col_names, row_labels)
                st.plotly_chart(fig_status, use_container_width=True)

        # Flagged Entries Audit Table
        if st.session_state.detected_entries:
            st.markdown("---")
            st.subheader(f"Flagged Unreliable Cells ({len(st.session_state.detected_entries)})")
            st.caption("Inspect each cell, the anomaly detection reason, and its original recorded value.")

            entries_data = []
            for e in st.session_state.detected_entries:
                entries_data.append({
                    "Row": e.row_label,
                    "Column": e.col_name,
                    "Type": e.entry_type,
                    "Original Value": f"{e.original_val:.3f}" if e.original_val is not None else "NaN",
                    "Detection Method": e.detection_method,
                    "Anomaly Magnitude": f"{e.anomaly_score:.2f}",
                    "Explanation": e.description
                })
            df_entries = pd.DataFrame(entries_data)
            st.dataframe(df_entries, use_container_width=True)

            if st.button("Proceed to Pattern Analysis ➔", type="primary"):
                st.session_state.current_tab = "3. Pattern Analysis"
                st.rerun()


# ==============================================================================
# TAB 3: PATTERN ANALYSIS
# ==============================================================================
elif selected_step == "3. Pattern Analysis":
    st.markdown("### 📊 Step 3: Pattern Analysis & SVD Energy Spectrum")
    st.write(
        "Evaluate underlying matrix patterns: singular value decay ($A \\approx U \\Sigma V^T$), cumulative energy ratio, "
        "inter-feature correlation, and distributional statistics. This establishes why low-rank matrix completion is justified."
    )

    if st.session_state.matrix_dataset is None:
        st.warning("Please load a dataset in Step 1 first.")
    else:
        dataset = st.session_state.matrix_dataset
        matrix = dataset.matrix_array
        col_names = dataset.numeric_columns

        # Compute SVD spectrum and summary metrics
        svd_res = PatternAnalyzer.analyze_svd_spectrum(matrix, col_names)
        st.session_state.svd_result = svd_res

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            render_kpi_card("Suggested Rank (90% Energy)", f"k = {svd_res.suggested_rank_90}", "Optimal subspace rank", "success", "🎯")
        with col_s2:
            render_kpi_card("Scree Elbow Rank", f"k = {svd_res.elbow_rank}", "Point of max curvature", "cyan", "📐")
        with col_s3:
            render_kpi_card("Effective Matrix Rank", f"{svd_res.effective_rank:.2f}", f"Out of {min(matrix.shape)} max", "purple", "✨")
        with col_s4:
            render_kpi_card("Condition Number", f"{svd_res.condition_number:.1f}", "Subspace stability ratio", "warning", "⚖️")

        # Scientific justification callout
        st.markdown(
            f"""
            <div class="callout-box callout-info">
                <div>💡</div>
                <div>
                    <b>Reconstruction Feasibility & Mathematical Justification:</b><br>
                    {svd_res.suitability_summary}<br>
                    <span style="font-size:0.84rem; opacity:0.9;">{svd_res.reconstruction_justification}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            fig_svd = plot_svd_energy_spectrum(svd_res, chosen_rank=svd_res.suggested_rank_90)
            st.plotly_chart(fig_svd, use_container_width=True)

        with col_p2:
            corr_matrix, avg_corr = PatternAnalyzer.compute_correlation_matrix(dataset.numeric_df)
            st.caption(f"Average pairwise absolute correlation: **{avg_corr:.2f}**")
            fig_corr = plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Mathematical Foundations Accordion
        with st.expander("📐 Mathematical Foundations of Matrix Completion (SVD & Low-Rank Projection)"):
            st.markdown(
                """
                **Singular Value Decomposition (SVD):**
                Any matrix $A \\in \\mathbb{R}^{m \\times n}$ can be factorized as:
                $$A = U \\Sigma V^T = \\sum_{i=1}^r \\sigma_i u_i v_i^T$$
                where $U \\in \\mathbb{R}^{m \\times m}$ and $V \\in \\mathbb{R}^{n \\times n}$ are orthogonal matrices containing the left and right singular vectors, and $\\Sigma = \\text{diag}(\\sigma_1, \\sigma_2, \\dots)$ contains the non-negative singular values arranged in descending order.

                **Eckart-Young-Mirsky Theorem:**
                The optimal rank-$k$ approximation of $A$ under Frobenius norm is given by truncating the SVD:
                $$A_k = \\sum_{i=1}^k \\sigma_i u_i v_i^T$$
                When features are correlated, the energy $\\sum_{i=1}^k \\sigma_i^2 / \\sum \\sigma_j^2$ reaches $\\ge 90\\%$ at a low rank $k \\ll \\min(m, n)$.
                Iterative Truncated SVD (Hard-Impute) iteratively projects unobserved entries onto this low-rank subspace until convergence.
                """
            )

        # Distribution Summary Table
        st.markdown("#### Feature Distribution & Missingness Metrics")
        stats_df = PatternAnalyzer.compute_summary_statistics(dataset.numeric_df)
        st.dataframe(stats_df, use_container_width=True)

        if st.button("Proceed to Reconstruction & Error Correction ➔", type="primary"):
            st.session_state.current_tab = "4. Reconstruct & Correct"
            st.rerun()


# ==============================================================================
# TAB 4: RECONSTRUCTION & ERROR CORRECTION
# ==============================================================================
elif selected_step == "4. Reconstruct & Correct":
    st.markdown("### 🔄 Step 4: Matrix Reconstruction & Error Correction")
    st.write(
        "Fill unobserved entries and correct confirmed erroneous cells using low-rank subspace recovery or baseline imputation. "
        "Original observed values are never overwritten silently; users can accept or reject individual corrections."
    )

    if st.session_state.matrix_dataset is None:
        st.warning("Please load a dataset in Step 1 first.")
    elif st.session_state.status_grid is None:
        st.warning("Please run Step 2 (Detect Errors) to establish target cells for reconstruction.")
    else:
        dataset = st.session_state.matrix_dataset
        matrix = dataset.matrix_array
        col_names = dataset.numeric_columns
        row_labels = [str(idx) for idx in dataset.numeric_df.index]
        min_dim = min(matrix.shape)

        col_cfg1, col_cfg2 = st.columns([1, 2])

        with col_cfg1:
            st.markdown("#### Algorithm & Target Settings")
            method = st.selectbox(
                "Select Reconstruction Method:",
                [
                    ("iterative_svd", "Iterative Truncated SVD (Hard-Impute Matrix Completion)"),
                    ("soft_impute", "SoftImpute (Singular Value Soft-Thresholding)"),
                    ("knn", "K-Nearest Neighbors Imputer (Distance Weighted)"),
                    ("baseline", "Baseline Imputer (Column Median/Mean)")
                ],
                format_func=lambda x: x[1]
            )[0]

            svd_rank = 2
            soft_lambda = None
            knn_k = 5
            base_strat = "median"

            if method == "iterative_svd":
                default_rank = st.session_state.svd_result.suggested_rank_90 if st.session_state.svd_result else 2
                svd_rank = st.slider(
                    "Truncation Subspace Rank (k):",
                    min_value=1,
                    max_value=max(1, min_dim - 1),
                    value=min(default_rank, max(1, min_dim - 1)),
                    help="Subspace rank for projection: A ≈ U_k Σ_k V_k^T."
                )
            elif method == "soft_impute":
                auto_lambda = st.checkbox("Auto-tune Shrinkage Lambda (λ)", value=True)
                if not auto_lambda:
                    soft_lambda = st.number_input("Shrinkage Parameter (λ):", value=5.0, min_value=0.1, step=0.5)
            elif method == "knn":
                knn_k = st.slider("Number of Neighbors (k):", min_value=1, max_value=min(15, max(1, matrix.shape[0]-1)), value=5)
            elif method == "baseline":
                base_strat = st.selectbox("Baseline Strategy:", ["median", "mean"])

            st.markdown("#### Target Selection")
            recon_missing = st.checkbox("Reconstruct Missing Values (NaN)", value=True)
            correct_outliers = st.checkbox("Correct Statistical Outliers", value=True)
            correct_user = st.checkbox("Correct User-Marked Corrupt Cells", value=True)

            if st.button("Execute Reconstruction", type="primary", use_container_width=True):
                recon_res = run_matrix_reconstruction(
                    matrix=matrix,
                    status_grid=st.session_state.status_grid,
                    col_names=col_names,
                    row_labels=row_labels,
                    method=method,
                    reconstruct_missing=recon_missing,
                    correct_outliers=correct_outliers,
                    correct_user_corrupted=correct_user,
                    svd_rank=svd_rank,
                    soft_lambda=soft_lambda,
                    knn_neighbors=knn_k,
                    baseline_strategy=base_strat
                )
                st.session_state.reconstruction_result = recon_res
                st.session_state.confidence_report = None
                st.success(
                    f"Reconstruction finished! Corrected {len(recon_res.cell_records)} cells in {recon_res.iterations_run} iteration(s) "
                    f"(Relative Frobenius Δ: {recon_res.final_frobenius_change:.2e})."
                )

        with col_cfg2:
            if st.session_state.reconstruction_result is not None:
                res = st.session_state.reconstruction_result
                st.markdown("#### Reconstruction Execution Summary")
                r1, r2, r3 = st.columns(3)
                with r1:
                    render_kpi_card("Cells Reconstructed", f"{len(res.cell_records)}", f"Out of {matrix.size} total cells", "success", "🔄")
                with r2:
                    render_kpi_card("Iterations Run", f"{res.iterations_run}", "Convergence: " + ("Yes" if res.converged else "Max reached"), "default", "⏱️")
                with r3:
                    render_kpi_card("Relative Frobenius Δ", f"{res.final_frobenius_change:.1e}", "Norm convergence", "purple", "📉")

                st.info(f"**Method Selected:** `{res.method_name}` with parameters: `{res.parameters}`")

        # Interactive Acceptance Table
        if st.session_state.reconstruction_result is not None:
            res = st.session_state.reconstruction_result
            st.markdown("---")
            st.subheader("Interactive Changed Cells Review & Acceptance")
            st.caption(
                "Review each proposed cell replacement. You may uncheck individual cells to reject the reconstruction and retain original data."
            )

            review_rows = []
            for rec in res.cell_records:
                review_rows.append({
                    "Row": rec.row_label,
                    "Column": rec.col_name,
                    "Original Value": rec.original_val if rec.original_val is not None else np.nan,
                    "Reconstructed Value": round(rec.reconstructed_val, 3),
                    "Absolute Delta": round(rec.delta, 3) if rec.delta is not None else np.nan,
                    "Reason": rec.reason,
                    "Accept Correction": rec.accepted
                })

            df_review = pd.DataFrame(review_rows)
            edited_df = st.data_editor(
                df_review,
                column_config={
                    "Accept Correction": st.column_config.CheckboxColumn(
                        "Accept?",
                        help="Check to accept reconstructed value; uncheck to reject and retain original.",
                        default=True
                    )
                },
                disabled=["Row", "Column", "Original Value", "Reconstructed Value", "Absolute Delta", "Reason"],
                hide_index=True,
                use_container_width=True
            )

            # Synchronize accepted state
            for idx, row in edited_df.iterrows():
                res.cell_records[idx].accepted = bool(row["Accept Correction"])

            if st.button("Proceed to Confidence Estimation ➔", type="primary"):
                st.session_state.current_tab = "5. Estimate Confidence"
                st.rerun()


# ==============================================================================
# TAB 5: ESTIMATE CONFIDENCE
# ==============================================================================
elif selected_step == "5. Estimate Confidence":
    st.markdown("### 🛡️ Step 5: Uncertainty & Confidence Estimation")
    st.write(
        "Estimate uncertainty for each reconstructed entry using **Monte Carlo Jackknife Resampling** "
        "(measuring reconstruction variance across random secondary maskings) combined with **Cross-Method Ensemble Agreement**. "
        "*Labelled as an empirical stability heuristic, not a calibrated frequentist or Bayesian probability.*"
    )

    if st.session_state.matrix_dataset is None:
        st.warning("Please load a dataset in Step 1 first.")
    elif st.session_state.reconstruction_result is None:
        st.warning("Please run Step 4 (Reconstruction) first.")
    else:
        dataset = st.session_state.matrix_dataset
        matrix = dataset.matrix_array
        col_names = dataset.numeric_columns
        row_labels = [str(idx) for idx in dataset.numeric_df.index]
        recon_res = st.session_state.reconstruction_result

        col_cf1, col_cf2 = st.columns([1, 2])
        with col_cf1:
            st.markdown("#### Confidence Settings")
            review_thresh = st.slider(
                "Human Review Threshold (%):",
                min_value=40, max_value=90, value=70, step=5,
                help="Reconstructed cells with estimated confidence below this threshold will be flagged for manual review."
            ) / 100.0

            n_trials = st.slider(
                "Monte Carlo Resampling Trials:",
                min_value=3, max_value=12, value=5,
                help="Number of perturbed masking passes to measure variance."
            )

            if st.button("Calculate Confidence Scores", type="primary", use_container_width=True):
                with st.spinner("Computing Monte Carlo jackknife stability and ensemble consensus..."):
                    conf_rep = ConfidenceEstimator.estimate_confidence(
                        matrix=matrix,
                        target_mask=recon_res.target_mask,
                        reconstruction_result=recon_res,
                        col_names=col_names,
                        row_labels=row_labels,
                        review_threshold=review_thresh,
                        n_bootstrap_trials=n_trials
                    )
                    st.session_state.confidence_report = conf_rep
                    st.success("Confidence scores calculated successfully!")

        with col_cf2:
            if st.session_state.confidence_report is not None:
                cr = st.session_state.confidence_report
                st.markdown("#### Confidence Score Breakdown")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    render_kpi_card("Mean Confidence", f"{cr.mean_confidence * 100:.1f}%", "Average score", "success", "🌟")
                with c2:
                    render_kpi_card("High Conf (>=80%)", f"{cr.n_high_confidence}", "Reliable completions", "cyan", "🛡️")
                with c3:
                    render_kpi_card("Moderate Conf", f"{cr.n_moderate_confidence}", f"{int(review_thresh*100)}%-79%", "warning", "⚠️")
                with c4:
                    render_kpi_card("Needs Review", f"{cr.n_low_confidence}", f"Confidence < {int(review_thresh*100)}%", "danger", "🚨")

        # Distribution Chart and Limitations
        if st.session_state.confidence_report is not None:
            cr = st.session_state.confidence_report
            fig_conf = plot_confidence_distribution(cr, review_threshold=review_thresh)
            st.plotly_chart(fig_conf, use_container_width=True)

            with st.expander("⚠️ Important Methodology & Scientific Limitations", expanded=True):
                st.markdown(f"**Methodology Description:** {cr.method_description}")
                st.markdown("**Assumptions and Boundaries:**")
                for lim in cr.limitations:
                    st.markdown(f"- *{lim}*")

            if st.button("Proceed to Final Dashboard & Downloads ➔", type="primary"):
                st.session_state.current_tab = "6. Results Dashboard & Export"
                st.rerun()


# ==============================================================================
# TAB 6: RESULTS DASHBOARD & EXPORT
# ==============================================================================
elif selected_step == "6. Results Dashboard & Export":
    st.markdown("### 📈 Step 6: Output & Results Dashboard")
    st.write(
        "Interactive dashboard displaying original vs corrected matrices, comprehensive audit logs with confidence scores, "
        "distributional comparisons, and export options."
    )

    if (
        st.session_state.matrix_dataset is None or
        st.session_state.reconstruction_result is None or
        st.session_state.confidence_report is None
    ):
        st.warning("Please complete Steps 1 through 5 to generate the final dashboard.")
    else:
        dataset = st.session_state.matrix_dataset
        recon_res = st.session_state.reconstruction_result
        conf_rep = st.session_state.confidence_report
        counts = st.session_state.detection_counts

        # Top Executive KPI Grid
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            render_kpi_card("Total Cells", f"{dataset.matrix_array.size}", f"{dataset.matrix_array.shape[0]}×{dataset.matrix_array.shape[1]} matrix", "default", "📊")
        with k2:
            render_kpi_card("Missing Replaced", f"{counts.get('missing', 0)}", "Unobserved entries", "warning", "⚠️")
        with k3:
            render_kpi_card("Outliers Fixed", f"{counts.get('outliers', 0)}", "Anomalous entries", "danger", "🚨")
        with k4:
            render_kpi_card("Reconstructed", f"{len(conf_rep.records)}", "Total repaired cells", "purple", "🔄")
        with k5:
            render_kpi_card("Mean Confidence", f"{conf_rep.mean_confidence*100:.1f}%", "Overall stability", "success", "🌟")
        with k6:
            render_kpi_card("Needs Review", f"{conf_rep.n_low_confidence}", f"Confidence < {int(conf_rep.review_threshold*100)}%", "danger" if conf_rep.n_low_confidence > 0 else "success", "🛡️")

        # Matrix Previews (Original vs Corrected)
        st.markdown("---")
        st.subheader("Matrix Comparison Previews")

        corrected_df = ReportGenerator.generate_corrected_dataframe(
            raw_df=dataset.raw_df,
            reconstructed_numeric_matrix=recon_res.reconstructed_matrix,
            numeric_columns=dataset.numeric_columns,
            cell_records=conf_rep.records
        )

        tab_orig, tab_recon, tab_diff = st.tabs(["Original Matrix (With Missing/Outliers)", "Corrected & Completed Matrix", "Visual Diff Scatter Plot"])

        with tab_orig:
            st.dataframe(dataset.raw_df, use_container_width=True)

        with tab_recon:
            st.dataframe(corrected_df, use_container_width=True)

        with tab_diff:
            fig_scatter = plot_corrections_scatter(conf_rep.records)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Feature Distribution Comparison
        st.markdown("---")
        st.subheader("Feature Distribution Analysis (Before vs After)")
        selected_feat = st.selectbox("Select feature column to compare distribution:", dataset.numeric_columns)
        fig_feat = plot_feature_distribution_comparison(dataset.numeric_df, corrected_df[dataset.numeric_columns], selected_feat)
        st.plotly_chart(fig_feat, use_container_width=True)

        # Detailed Audit Table
        st.markdown("---")
        st.subheader("Reconstruction & Confidence Audit Table")
        st.caption("Complete breakdown of every changed cell, including original value, correction, estimated confidence, and review flag.")

        audit_df = ReportGenerator.generate_audit_dataframe(conf_rep.records)
        st.dataframe(audit_df, use_container_width=True)

        # High priority review items
        low_conf_records = [r for r in conf_rep.records if r.confidence_score < conf_rep.review_threshold]
        if low_conf_records:
            st.markdown(
                f"""
                <div class="callout-box callout-warning">
                    <div>⚠️</div>
                    <div>
                        <b>Human Review Alert:</b> {len(low_conf_records)} cell(s) have estimated confidence below
                        the {int(conf_rep.review_threshold*100)}% threshold. We recommend manual domain expert verification for these values.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Download Section
        st.markdown("---")
        st.subheader("📥 Export & Download Results")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        csv_bytes = ReportGenerator.to_csv_bytes(corrected_df)
        with col_d1:
            st.download_button(
                label="Download Corrected CSV",
                data=csv_bytes,
                file_name=f"{dataset.dataset_name}_corrected.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        excel_bytes = ReportGenerator.to_excel_bytes(corrected_df)
        with col_d2:
            st.download_button(
                label="Download Corrected Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"{dataset.dataset_name}_corrected.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        audit_csv = ReportGenerator.to_csv_bytes(audit_df)
        with col_d3:
            st.download_button(
                label="Download Audit Trail (.csv)",
                data=audit_csv,
                file_name=f"{dataset.dataset_name}_audit_log.csv",
                mime="text/csv",
                use_container_width=True
            )

        md_summary = ReportGenerator.generate_markdown_summary(
            dataset_name=dataset.dataset_name,
            n_rows=dataset.matrix_array.shape[0],
            n_cols=dataset.matrix_array.shape[1],
            detection_method=f"Multi-Strategy ({len(st.session_state.detected_entries)} flagged)",
            reconstruction_method=recon_res.method_name,
            params=recon_res.parameters,
            confidence_report=conf_rep,
            counts=counts
        )
        with col_d4:
            st.download_button(
                label="Download Audit Report (.md)",
                data=md_summary.encode("utf-8"),
                file_name=f"{dataset.dataset_name}_summary_report.md",
                mime="text/markdown",
                use_container_width=True
            )
