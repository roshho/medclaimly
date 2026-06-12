#!/usr/bin/env python3
"""
Full pipeline orchestrator for Stage 1 → Stage 2 execution.

Executes all pipeline stages sequentially:
1. load_reference_tables (one-time setup)
2. stage1_rules_engine
3. stage2_prepare_training
4. stage2_feature_engineering
5. stage2_train_xgboost
6. generate_performance_report

On successful completion, joins Stage 1 and Stage 2 outputs into a unified result file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src" / "src_CMS"
OUT_DIR = PROJECT_ROOT / "output"


def run_stage(stage_name: str, script_path: Path, args: list[str]) -> None:
    """Execute a pipeline stage and halt on failure."""
    print(f"\n{'='*80}")
    print(f"Running: {stage_name}")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"\n✗ PIPELINE FAILED at stage: {stage_name}")
        print(f"  Exit code: {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✓ {stage_name} completed successfully")


def join_stage1_stage2_outputs(sample_id: int) -> Path:
    """Join Stage 1 hard deny decisions with Stage 2 ML probability scores."""
    stage1_path = OUT_DIR / f"sample{sample_id:02d}_stage1_results.parquet"
    stage2_model_path = OUT_DIR / f"sample{sample_id:02d}_stage2_model.json"
    stage2_features_path = OUT_DIR / f"sample{sample_id:02d}_stage2_features.parquet"
    output_path = OUT_DIR / f"sample{sample_id:02d}_full_pipeline_results.parquet"
    
    if not stage1_path.exists():
        raise FileNotFoundError(f"Stage 1 output missing: {stage1_path}")
    if not stage2_model_path.exists():
        raise FileNotFoundError(f"Stage 2 model missing: {stage2_model_path}")
    if not stage2_features_path.exists():
        print(f"Warning: Stage 2 features missing: {stage2_features_path}")
        print("Skipping Stage 2 probability scoring.")
        return stage1_path
    
    print(f"\n{'='*80}")
    print("Joining Stage 1 and Stage 2 outputs")
    print(f"{'='*80}\n")
    
    try:
        from xgboost import XGBClassifier
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Warning: Cannot join outputs - {e}")
        return stage1_path
    
    # Load Stage 2 model and features
    model = XGBClassifier()
    model.load_model(str(stage2_model_path))
    
    df_features = pd.read_parquet(stage2_features_path)
    
    # Prepare feature matrix (same exclusion logic as training)
    drop_cols = {
        "CLM_ID", "DESYNPUF_ID", "sample_id",
        "CLM_FROM_DT", "CLM_THRU_DT", "CLM_DRG_CD",
        "is_denied",
    }
    
    X = df_features[[c for c in df_features.columns if c not in drop_cols]].copy()
    
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
        elif not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    
    X = X.fillna(0.0)
    
    # Predict Stage 2 denial probabilities
    y_prob = model.predict_proba(X)[:, 1]
    
    df_features["stage2_denial_probability"] = y_prob
    score_keys = ["sample_id", "CLM_ID", "DESYNPUF_ID", "CLM_FROM_DT", "CLM_THRU_DT", "CLM_DRG_CD"]
    df_stage2_scored = (
        df_features[score_keys + ["stage2_denial_probability"]]
        # Keep rows where any join key is NULL; default groupby(dropna=True) silently drops them.
        .groupby(score_keys, as_index=False, dropna=False)["stage2_denial_probability"]
        .max()
    )
    
    # Join with Stage 1 results using DuckDB
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW stage1 AS SELECT * FROM read_parquet('{stage1_path}')")
    con.register("stage2_scored", df_stage2_scored)
    
    con.execute(
        f"""
        CREATE OR REPLACE TABLE full_pipeline_results AS
        SELECT
            s1.*,
            CASE
                WHEN s1.stage1_decision = 'HARD_DENY' THEN NULL
                ELSE s2.stage2_denial_probability
            END AS stage2_denial_probability
        FROM stage1 s1
        LEFT JOIN stage2_scored s2
                    ON s1.sample_id IS NOT DISTINCT FROM s2.sample_id
                 AND s1.CLM_ID IS NOT DISTINCT FROM s2.CLM_ID
         AND s1.DESYNPUF_ID IS NOT DISTINCT FROM s2.DESYNPUF_ID
         AND s1.CLM_FROM_DT IS NOT DISTINCT FROM s2.CLM_FROM_DT
         AND s1.CLM_THRU_DT IS NOT DISTINCT FROM s2.CLM_THRU_DT
         AND s1.CLM_DRG_CD IS NOT DISTINCT FROM s2.CLM_DRG_CD
        """
    )
    
    con.execute(f"COPY full_pipeline_results TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    
    print(f"✓ Joined pipeline results saved: {output_path}")
    return output_path


def run_full_pipeline(sample_id: int, skip_reference_load: bool = False) -> None:
    """Execute full pipeline for a given sample ID."""
    print(f"\n{'#'*80}")
    print(f"# FULL PIPELINE EXECUTION: Sample {sample_id:02d}")
    print(f"{'#'*80}\n")
    
    # Stage 0: Load reference tables (optional, can be skipped if already loaded)
    if not skip_reference_load:
        run_stage(
            "Stage 0: Load Reference Tables",
            SRC_DIR / "load_reference_tables.py",
            []
        )
    
    # Stage 1: Deterministic rules engine
    run_stage(
        "Stage 1: Rules Engine",
        SRC_DIR / "stage1_rules_engine.py",
        ["--sample-id", str(sample_id)]
    )
    
    # Stage 2a: Prepare training labels
    run_stage(
        "Stage 2a: Prepare Training Labels",
        SRC_DIR / "stage2_prepare_training.py",
        ["--sample-id", str(sample_id)]
    )
    
    # Stage 2b: Feature engineering
    run_stage(
        "Stage 2b: Feature Engineering",
        SRC_DIR / "stage2_feature_engineering.py",
        ["--sample-id", str(sample_id)]
    )
    
    # Stage 2c: Train XGBoost
    run_stage(
        "Stage 2c: Train XGBoost",
        SRC_DIR / "stage2_train_xgboost.py",
        ["--sample-id", str(sample_id)]
    )
    
    # Stage 3: Generate performance report
    run_stage(
        "Stage 3: Generate Report",
        SRC_DIR / "generate_performance_report.py",
        ["--sample-id", str(sample_id)]
    )
    
    # Final: Join Stage 1 + Stage 2 outputs
    try:
        output_path = join_stage1_stage2_outputs(sample_id)
    except Exception as e:
        print(f"Warning: Failed to join outputs - {e}")
        output_path = None
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE: Sample {sample_id:02d}")
    print(f"{'#'*80}\n")
    
    if output_path:
        print(f"✓ All stages completed successfully")
        print(f"✓ Final output: {output_path}")
    else:
        print(f"✓ All stages completed (join skipped)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Stage 1 + Stage 2 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for Sample 1
  python run_pipeline.py --sample-id 1
  
  # Skip reference table reload
  python run_pipeline.py --sample-id 1 --skip-reference-load
        """
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=1,
        help="Sample ID to process (default: 1)"
    )
    parser.add_argument(
        "--skip-reference-load",
        action="store_true",
        help="Skip loading reference tables (use if already loaded)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        sample_id=args.sample_id,
        skip_reference_load=args.skip_reference_load
    )
