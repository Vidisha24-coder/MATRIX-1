# Matrix Reconstruction and Error Correction

An interactive, production-ready data science web application built with **Streamlit**, **NumPy**, **Pandas**, **Scikit-learn**, and **Plotly**. The application inspects matrix datasets, detects missing values and corrupted statistical outliers, analyzes low-rank patterns via Singular Value Decomposition (SVD), reconstructs entries using iterative matrix completion, quantifies cell-level confidence via Monte Carlo jackknife resampling and ensemble consensus, and flags low-confidence results for human review.

---

## 🏛️ System Architecture Workflow

The application directly implements the 6-stage architecture:

```
[1. Data Input] ➔ [2. Detect Errors] ➔ [3. Pattern Analysis] ➔ [4. Reconstruction] ➔ [5. Confidence] ➔ [6. Output & Dashboard]
```

1. **Data Input (`core/data_loader.py`)**:
   - Ingests CSV and Excel (`.xlsx`, `.xls`) files with encoding and delimiter auto-detection.
   - Extracts numeric matrix sub-blocks while preserving row labels and non-numeric columns.
   - Guardrails against empty files, duplicate column headers, infinite values, and sub-minimum dimensions ($< 2 \times 2$).
   - Built-in synthetic benchmark matrices (Environmental Sensor Network, Collaborative Star Ratings, Financial Asset Metrics) and a Synthetic Error Injector for controlled benchmarking.

2. **Detect Errors (`core/error_detection.py`)**:
   - Differentiates **Missing Values (NaN/Inf)**, **Statistical Outliers**, and **User-Marked Corruptions**.
   - Outlier detection algorithms:
     - **Robust Z-Score (MAD)**: $M_i = 0.6745 \cdot |x_i - \text{median}| / \text{MAD}$ with configurable threshold.
     - **Isolation Forest**: Multi-dimensional tree isolation with contamination rate tuning.
     - **IQR Fence**: Tukey's boxplot fence ($Q_1 - k\cdot\text{IQR}, Q_3 + k\cdot\text{IQR}$).
     - **Domain Value Range**: Enforces physical boundaries (e.g. $[1.0, 5.0]$ ratings).
   - Manual override tool: Users can explicitly flag/unflag any $(row, column)$ cell.
   - *Guardrail: Outliers are reported as candidates for inspection—not silently discarded.*

3. **Pattern Analysis (`core/pattern_analysis.py`)**:
   - Computes full Singular Value Decomposition: $A \approx U \Sigma V^T$.
   - Interactive Scree plot and Cumulative Energy curve ($E_k = \sum_{i=1}^k \sigma_i^2 / \sum \sigma_j^2$).
   - Algorithmic rank recommendations: 80%, 90%, 95% energy ranks, and Scree elbow detection.
   - Pairwise Pearson & Spearman correlation heatmaps and effective rank (Roy & Vetterli entropy).
   - Dynamic scientific justification explaining whether low-rank matrix completion is suitable for the data.

4. **Reconstruction & Error Correction (`core/reconstruction.py`)**:
   - **Iterative Truncated SVD (Hard-Impute Matrix Completion)**:
     - Alternating projection onto the rank-$k$ subspace while keeping observed cells fixed.
   - **SoftImpute**:
     - Nuclear norm regularized singular value soft-thresholding: $\mathcal{S}_\lambda(\sigma) = \max(0, \sigma - \lambda)$.
   - **K-Nearest Neighbors (KNN Imputer)**: Distance-weighted local neighborhood imputation.
   - **Baseline Imputer**: Column median or column mean.
   - Selective targeting: Choose whether to reconstruct missing cells, correct outliers, or correct user-flagged cells.
   - Interactive changed cells table: Users can accept or reject individual cell replacements.

5. **Confidence Estimation (`core/confidence.py`)**:
   - Dual-evidence confidence calculation:
     - **Monte Carlo Jackknife Resampling**: Masks random subsets of observed data across multiple trials to measure empirical reconstruction variance $\sigma_{\text{recon}}$.
     - **Cross-Method Ensemble Consensus**: Normalized agreement between SVD projection and KNN baseline.
   - Configurable human review threshold (default: 70%). Reconstructions below threshold are tagged with **"Low (Needs Review)"**.
   - *Explicit label: Empirical stability heuristic—not a calibrated frequentist or Bayesian probability.*

6. **Output & Dashboard (`app.py`, `components/`)**:
   - Modern Bootstrap 5-inspired UI with responsive KPI cards.
   - Side-by-side comparison tabs (Original vs Corrected vs Scatter diff).
   - Feature distribution comparison boxplots before and after correction.
   - Full audit trail table with row, column, original, replacement, delta, reason, confidence score, and status.
   - Instant export:
     - Corrected Dataset (`.csv` and `.xlsx`)
     - Complete Audit Trail (`.csv`)
     - Executive Summary Report (`.md`)

---

## 🚀 Installation & Launch Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd "C:\Users\Vishal Sanjay Shivam\Desktop\MATRIX"
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\pip install -r requirements.txt
   ```

3. **Launch the application:**
   - **Using the Windows Batch file:**
     Double click `run_app.bat` or run:
     ```cmd
     run_app.bat
     ```
   - **Using PowerShell:**
     ```powershell
     .\run_app.ps1
     ```
   - **Using direct command:**
     ```bash
     .\.venv\Scripts\streamlit.exe run app.py
     ```

4. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

The app uses the project's light theme configured in `.streamlit/config.toml`.

---

## 🧪 Running Unit Tests

Run the automated pytest test suite:
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v -p no:anyio
```
**Test Coverage:**
- `tests/test_data_loader.py`: CSV/Excel parsing, duplicate columns, guardrails, synthetic generators.
- `tests/test_detection.py`: Missing values, robust Z-scores, IQR fences, user overrides.
- `tests/test_reconstruction.py`: Rank-1 exact recovery, Truncated SVD, SoftImpute, KNN.
- `tests/test_confidence.py`: SVD energy spectrum, Monte Carlo stability, confidence score bounds.

---

## 📂 Project Structure

```
MATRIX/
├── app.py                      # Main Streamlit dashboard application
├── core/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, validation, synthetic data generators
│   ├── error_detection.py      # Statistical and ML outlier detection
│   ├── pattern_analysis.py     # SVD spectrum, energy ratios, correlation analysis
│   ├── reconstruction.py       # Iterative Truncated SVD, SoftImpute, KNN, Baseline
│   ├── confidence.py           # Monte Carlo jackknife uncertainty & ensemble confidence
│   └── report_generator.py     # Excel/CSV dataset export and audit report generation
├── components/
│   ├── __init__.py
│   ├── ui_helpers.py           # Bootstrap CSS injection, KPI cards, step flow bar
│   └── plots.py                # Plotly heatmaps, scree plots, scatter, boxplots
├── data/
│   ├── sample_sensor_network.csv       # Environmental sensor array benchmark
│   ├── sample_customer_ratings.csv     # Collaborative 1-5 star rating benchmark
│   └── sample_financial_metrics.csv    # Multi-asset financial indicator benchmark
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_detection.py
│   ├── test_reconstruction.py
│   └── test_confidence.py
├── generate_samples.py         # Utility script to re-generate sample datasets
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── run_app.bat                 # One-click Windows CMD launcher
├── run_app.ps1                 # PowerShell launcher
└── README.md                   # Complete documentation & methodology
```

---

## 🔬 Methodology, Assumptions & Limitations

1. **Low-Rank Subspace Assumption**:
   - Iterative SVD and SoftImpute assume the data matrix is generated from a low-rank linear subspace plus noise ($A = L + E$). If features are entirely uncorrelated (spherical white noise), SVD reconstruction acts primarily as an uninformative rank-truncated smoother.
2. **Missingness Mechanism**:
   - Assumes missing entries are Missing at Random (MAR) or Missing Completely at Random (MCAR). If large contiguous blocks of rows/columns are missing simultaneously, the subspace projection may suffer from identifiability issues.
3. **Outliers vs Genuine Extremes**:
   - Statistical detection flags extreme deviations from the median, but domain knowledge is required to confirm whether an outlier represents a sensor glitch or a genuine rare event. Users should review the Flagged Unreliable Cells table before triggering correction.
4. **Confidence Score Semantics**:
   - Reconstructed confidence scores represent **empirical consistency** (resampling stability and ensemble agreement). They are normalized indices between 0% and 100%, and should not be interpreted as frequentist coverage probabilities or Bayesian posterior probabilities.
