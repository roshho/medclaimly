#!/usr/bin/env python3
"""
Generate a compact performance report for Stage 1 + Stage 2 outputs.

Inputs:
- output/sampleXX_stage1_results.parquet
- output/sampleXX_stage1_rule_summary.csv
- output/sampleXX_stage2_metrics.csv

Output:
- output/sampleXX_performance_report.md
- output/sampleXX_stage1_validation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output"


def build_report(sample_id: int) -> tuple[Path, Path]:
    stage1_path = OUT_DIR / f"sample{sample_id:02d}_stage1_results.parquet"
    stage1_summary_path = OUT_DIR / f"sample{sample_id:02d}_stage1_rule_summary.csv"
    stage2_metrics_path = OUT_DIR / f"sample{sample_id:02d}_stage2_metrics.csv"
    stage2_labels_path = OUT_DIR / f"sample{sample_id:02d}_stage2_train_labels.parquet"
    pr_table_path = OUT_DIR / f"sample{sample_id:02d}_pr_operating_table.csv"

    if not stage1_path.exists():
        raise FileNotFoundError(f"Missing: {stage1_path}")
    if not stage2_metrics_path.exists():
        raise FileNotFoundError(f"Missing: {stage2_metrics_path}")

    report_path = OUT_DIR / f"sample{sample_id:02d}_performance_report.md"
    stage1_validation_csv = OUT_DIR / f"sample{sample_id:02d}_stage1_validation.csv"

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW stage1 AS SELECT * FROM read_parquet('{stage1_path}')")
    if stage2_labels_path.exists():
        con.execute(f"CREATE OR REPLACE VIEW stage2_labels AS SELECT * FROM read_parquet('{stage2_labels_path}')")

    stage1_totals = con.execute(
        """
        SELECT
            COUNT(*) AS total_claims,
            SUM(CASE WHEN stage1_decision='HARD_DENY' THEN 1 ELSE 0 END) AS hard_deny_claims,
            SUM(CASE WHEN stage1_decision='PASS_TO_ML' THEN 1 ELSE 0 END) AS pass_to_ml_claims,
            SUM(CASE WHEN observed_has_b THEN 1 ELSE 0 END) AS observed_b_claims,
            SUM(CASE WHEN observed_has_c THEN 1 ELSE 0 END) AS observed_c_claims,
            SUM(CASE WHEN stage1_decision='HARD_DENY' AND observed_has_b THEN 1 ELSE 0 END) AS hard_deny_with_b,
            SUM(CASE WHEN stage1_decision='HARD_DENY' AND observed_has_c THEN 1 ELSE 0 END) AS hard_deny_with_c,
            SUM(CASE WHEN stage1_decision='HARD_DENY' AND NOT observed_has_b AND NOT observed_has_c THEN 1 ELSE 0 END) AS hard_deny_without_bc
        FROM stage1
        """
    ).fetchone()

    contradiction_count = con.execute(
        """
        SELECT COUNT(*)
        FROM stage1
        WHERE stage1_decision='HARD_DENY'
          AND NOT observed_has_b
          AND NOT observed_has_c
        """
    ).fetchone()[0]

    con.execute(
        f"""
        COPY (
            SELECT
                stage1_reason_code,
                stage1_decision,
                COUNT(*) AS claim_count,
                AVG(CASE WHEN observed_has_b THEN 1 ELSE 0 END) AS pct_observed_b,
                AVG(CASE WHEN observed_has_c THEN 1 ELSE 0 END) AS pct_observed_c
            FROM stage1
            GROUP BY 1,2
            ORDER BY claim_count DESC
        ) TO '{stage1_validation_csv}' (HEADER, DELIMITER ',')
        """
    )
    # Get denial source breakdown if available
    denial_source_breakdown = ""
    if stage2_labels_path.exists():
        source_data = con.execute(
            """
            SELECT denial_sources, COUNT(*) AS count
            FROM stage2_labels
            WHERE is_denied = 1
            GROUP BY denial_sources
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()
        
        if source_data:
            total_denied = sum(row[1] for row in source_data)
            denial_source_breakdown = "\n### Denial Source Breakdown (Test Set Positives)\n\n"
            denial_source_breakdown += "| Source(s) | Count | Percentage |\n"
            denial_source_breakdown += "|-----------|-------|------------|\n"
            for source, count in source_data:
                pct = 100.0 * count / total_denied if total_denied > 0 else 0.0
                denial_source_breakdown += f"| {source or 'none'} | {count:,} | {pct:.2f}% |\n"


    con.close()

    stage2_metrics = pd.read_csv(stage2_metrics_path).iloc[0].to_dict()
    
    # Load PR operating table if available
    pr_table_md = ""
    if pr_table_path.exists():
        pr_table = pd.read_csv(pr_table_path)
        pr_table_md = "\n### PR Operating Table\n\n"
        pr_table_md += "| Threshold | Precision | Recall | F1 |\n"
        pr_table_md += "|-----------|-----------|--------|----|\n"
        for _, row in pr_table.iterrows():
            pr_table_md += f"| {row['threshold']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n"

    stage1_hard_deny_rate = stage1_totals[1] / stage1_totals[0] if stage1_totals[0] else 0.0
    stage1_pass_rate = stage1_totals[2] / stage1_totals[0] if stage1_totals[0] else 0.0
    best_threshold = float(stage2_metrics.get('best_f1_threshold', 0.5))
    best_f1 = float(stage2_metrics.get('best_f1', 0.0))
    best_precision = float(stage2_metrics.get('best_precision', 0.0))
    best_recall = float(stage2_metrics.get('best_recall', 0.0))
    

    report = f"""# Sample {sample_id:02d} Performance Report

## Stage 1 (Deterministic Rules)

- Total claims: {stage1_totals[0]:,}
- Hard deny: {stage1_totals[1]:,} ({stage1_hard_deny_rate:.2%})
- Pass to ML: {stage1_totals[2]:,} ({stage1_pass_rate:.2%})
- Observed B (guardrail): {stage1_totals[3]:,}
- Observed C (guardrail): {stage1_totals[4]:,}
- Hard deny ∩ observed B: {stage1_totals[5]:,}
- Hard deny ∩ observed C: {stage1_totals[6]:,}
- Hard deny without B/C (potential contradiction bucket): {stage1_totals[7]:,}

## Stage 2 (XGBoost)
### Primary Metrics


- Rows total: {int(stage2_metrics.get('rows_total', 0)):,}
- Positive rate: {float(stage2_metrics.get('positive_rate', 0.0)):.4f}
- PR-AUC: {float(stage2_metrics.get('pr_auc', 0.0)):.4f}

### Best Threshold Performance (Patient-level split)

- Best F1 threshold: {best_threshold:.4f}
- F1 @ best threshold: {best_f1:.4f}
- Precision @ best threshold: {best_precision:.4f}
- Recall @ best threshold: {best_recall:.4f}

### High-Precision Operating Point

- Recall @ Precision>=0.80: {float(stage2_metrics.get('recall_at_precision_0.80', 0.0)):.4f}
{pr_table_md}

{denial_source_breakdown}


## Notes

- Stage 1 B/C values are used for calibration only and excluded from Stage 2 target labels.
- MUE is treated as a soft feature in SynPUF context (no service unit fields).
- Current contradiction bucket size: {contradiction_count:,} claims.
- Patient-level split prevents same beneficiary appearing in both train and test sets.
- Best threshold optimized for F1 score on test set; may differ from fixed 0.50 threshold.
"""

    report_path.write_text(report)

    print("\nPerformance report summary")
    print(f"  sample_id              : {sample_id}")
    print(f"  stage1_hard_deny_rate  : {stage1_hard_deny_rate:.4f}")
    print(f"  stage2_pr_auc          : {float(stage2_metrics.get('pr_auc', 0.0)):.4f}")
    print(f"  stage2_best_f1         : {best_f1:.4f}")
    print(f"\nSaved: {report_path}")
    print(f"Saved: {stage1_validation_csv}")

    return report_path, stage1_validation_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 1/2 performance report")
    parser.add_argument("--sample-id", type=int, default=1, help="Sample ID to process (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(sample_id=args.sample_id)
