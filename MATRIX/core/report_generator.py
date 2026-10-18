"""Report and Dataset Export Generator."""

import io
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from core.reconstruction import ReconstructedCellRecord
from core.confidence import ConfidenceReport


class ReportGenerator:
    """Generates downloadable datasets and audit reports."""

    @staticmethod
    def generate_corrected_dataframe(
        raw_df: pd.DataFrame,
        reconstructed_numeric_matrix: np.ndarray,
        numeric_columns: List[str],
        cell_records: Optional[List[ReconstructedCellRecord]] = None
    ) -> pd.DataFrame:
        """Merge corrected numeric values back into the full dataframe preserving non-numeric columns.
        
        Respects individual user acceptance/rejection if provided.
        """
        corrected_df = raw_df.copy()

        # If individual cell rejections occurred, adjust matrix accordingly
        adjusted_numeric = reconstructed_numeric_matrix.copy()
        if cell_records:
            for rec in cell_records:
                if not rec.accepted:
                    # Revert to original value (or NaN)
                    if rec.original_val is not None:
                        adjusted_numeric[rec.row_idx, rec.col_idx] = rec.original_val
                    else:
                        adjusted_numeric[rec.row_idx, rec.col_idx] = np.nan

        # Replace numeric columns
        for idx, col in enumerate(numeric_columns):
            corrected_df[col] = adjusted_numeric[:, idx]

        return corrected_df

    @staticmethod
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to CSV bytes for download."""
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def to_excel_bytes(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to Excel bytes for download."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Corrected_Matrix")
        return buffer.getvalue()

    @staticmethod
    def generate_audit_dataframe(cell_records: List[ReconstructedCellRecord]) -> pd.DataFrame:
        """Convert list of reconstructed cell records to a structured audit DataFrame."""
        if not cell_records:
            return pd.DataFrame(columns=[
                "Row", "Column", "Original", "Reconstructed", "Delta",
                "Reason", "Confidence", "Confidence_Raw", "Uncertainty_Std", "Level", "Accepted"
            ])
        return pd.DataFrame([rec.to_dict() for rec in cell_records])

    @staticmethod
    def generate_markdown_summary(
        dataset_name: str,
        n_rows: int,
        n_cols: int,
        detection_method: str,
        reconstruction_method: str,
        params: Dict[str, any],
        confidence_report: ConfidenceReport,
        counts: Dict[str, int]
    ) -> str:
        """Generate markdown summary for export and inspection."""
        review_cells = [
            r for r in confidence_report.records
            if r.confidence_score < confidence_report.review_threshold
        ]

        md = f"""# Matrix Reconstruction and Error Correction Audit Report
**Dataset:** {dataset_name}  
**Matrix Dimensions:** {n_rows} rows × {n_cols} columns ({n_rows * n_cols} total cells)  
**Detection Strategy:** {detection_method}  
**Reconstruction Algorithm:** {reconstruction_method}  
**Algorithm Parameters:** `{params}`  

---

## 1. Summary of Matrix Changes
- **Total Matrix Cells:** {counts.get('total_cells', n_rows * n_cols):,}
- **Missing Cells Replaced:** {counts.get('missing', 0):,}
- **Statistical Outliers Corrected:** {counts.get('outliers', 0):,}
- **User-Flagged Cells Corrected:** {counts.get('user_corrupted', 0):,}
- **Total Reconstructed Cells:** {len(confidence_report.records):,}
- **Flagged for Human Review (Confidence < {int(confidence_report.review_threshold * 100)}%):** {len(review_cells):,}

---

## 2. Confidence & Uncertainty Overview
- **Mean Confidence Score:** {confidence_report.mean_confidence * 100:.1f}%
- **Median Confidence Score:** {confidence_report.median_confidence * 100:.1f}%
- **High Confidence Cells (>=80%):** {confidence_report.n_high_confidence}
- **Moderate Confidence Cells ({int(confidence_report.review_threshold * 100)}% - 79%):** {confidence_report.n_moderate_confidence}
- **Low Confidence Cells (Needs Review):** {confidence_report.n_low_confidence}

### Methodology
{confidence_report.method_description}

### Limitations & Assumptions
"""
        for lim in confidence_report.limitations:
            md += f"- {lim}\n"

        if review_cells:
            md += "\n---\n\n## 3. High-Priority Cells Flagged for Manual Review\n"
            md += "| Row | Column | Original | Reconstructed | Reason | Confidence | Uncertainty (Std) |\n"
            md += "|---|---|---|---|---|---|---|\n"
            for r in review_cells[:20]:
                orig_str = f"{r.original_val:.3f}" if r.original_val is not None else "NaN"
                md += (
                    f"| {r.row_label} | {r.col_name} | {orig_str} | "
                    f"{r.reconstructed_val:.3f} | {r.reason} | "
                    f"{int(r.confidence_score * 100)}% | ±{r.uncertainty_std:.3f} |\n"
                )
            if len(review_cells) > 20:
                md += f"\n*(Showing top 20 of {len(review_cells)} cells needing review)*\n"

        return md
