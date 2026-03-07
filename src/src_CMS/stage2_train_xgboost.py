#!/usr/bin/env python3
"""
Train Stage 2 XGBoost classifier on engineered features.

Input:
- output/sampleXX_stage2_features.parquet

Output:
- output/sampleXX_stage2_model.json
- output/sampleXX_stage2_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output"


def run_training(sample_id: int) -> tuple[Path, Path]:
    features_path = OUT_DIR / f"sample{sample_id:02d}_stage2_features.parquet"
    model_path = OUT_DIR / f"sample{sample_id:02d}_stage2_model.json"
    metrics_path = OUT_DIR / f"sample{sample_id:02d}_stage2_metrics.csv"
    pr_table_path = OUT_DIR / f"sample{sample_id:02d}_pr_operating_table.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path)
    if df.empty:
        raise ValueError("Feature dataset is empty")
    # Step 4: Pre-train leakage assertion
    # These columns must NEVER appear in the training feature matrix
    forbidden_cols = {
        "hcpcs_icd_mismatch",
        "lcd_condition_unmet",
        "unbundling_flag",
        "ptp_conflict",
        "symptom_primary_flag",
        "drg_short_stay_flag",
        "denial_sources",
        "denial_confidence",
        "soft_ptp_risk",
        "soft_mue_frequency_risk",
        "soft_mcd_no_diag_risk",
        "LINE_PRCSG_IND_CD_1",
        "LINE_PRCSG_IND_CD",
        "observed_has_a",
        "observed_has_b",
        "observed_has_c",
    }
    
    violations = forbidden_cols.intersection(df.columns)
    if violations:
        raise ValueError(
            f"LEAKAGE DETECTED: The following label-source columns are present in the "
            f"feature matrix and must be removed: {sorted(violations)}"
        )


    # Keep leakage-free numeric/boolean features
    drop_cols = {
        "CLM_ID",
        "DESYNPUF_ID",
        "sample_id",
        "CLM_FROM_DT",
        "CLM_THRU_DT",
        "CLM_DRG_CD",
        "is_denied",
    }
    train_df = df[df["is_denied"].notna()].copy()
    if train_df.empty:
        raise ValueError("No labeled rows available for training (is_denied is all NULL)")

    y = train_df["is_denied"].astype(int).values
    desynpuf_ids = train_df["DESYNPUF_ID"].values

    X = train_df[[c for c in train_df.columns if c not in drop_cols]].copy()

    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
        elif not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0.0)

    # Step 6: Patient-level train/test split
    # Split on unique DESYNPUF_ID to prevent same beneficiary in train and test
    unique_ids = np.unique(desynpuf_ids)
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=0.2, random_state=42
    )
    
    train_mask = np.isin(desynpuf_ids, train_ids)
    test_mask = np.isin(desynpuf_ids, test_ids)
    
    X_trainval = X[train_mask]
    y_trainval = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Further split train into train/val (80/20 of the 80% training set)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    neg = max((y_train == 0).sum(), 1)
    pos = max((y_train == 1).sum(), 1)
    scale_pos_weight = neg / pos

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Install with: pip install xgboost") from exc
    # Step 5: Upgraded XGBoost configuration for class imbalance

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        min_child_weight=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=40,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)

    # Step 7: Threshold optimization and PR operating table
    precision, recall, thresholds_curve = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds_curve[best_idx]) if best_idx < len(thresholds_curve) else 0.5
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])
    
    # Compute fixed operating points
    operating_thresholds = [0.20, 0.30, 0.40, 0.50, best_threshold]
    pr_table = []
    
    metrics = {
        "sample_id": sample_id,
        "rows_total": int(len(train_df)),
        "rows_all_features": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
        "rows_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "pr_auc": float(pr_auc),
        "best_f1_threshold": best_threshold,
        "best_f1": best_f1,
        "best_precision": best_precision,
        "best_recall": best_recall,
    }

    for t in operating_thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_val = float(f1_score(y_test, y_pred, zero_division=0))
        prec_val = float((y_pred & y_test).sum() / max(y_pred.sum(), 1))
        rec_val = float((y_pred & y_test).sum() / max(y_test.sum(), 1))
        
        pr_table.append({
            "threshold": t,
            "precision": prec_val,
            "recall": rec_val,
            "f1": f1_val,
        })
        
        if t != best_threshold:
            metrics[f"f1_at_{t:.2f}"] = f1_val

    valid = recall[precision >= 0.80]
    metrics["recall_at_precision_0.80"] = float(valid.max()) if len(valid) else 0.0

    model.save_model(str(model_path))
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(pr_table).to_csv(pr_table_path, index=False)

    print("\nStage 2 model training summary")
    print(f"  sample_id               : {sample_id}")
    print(f"  rows_total              : {metrics['rows_total']:,}")
    print(f"  positive_rate           : {metrics['positive_rate']:.4f}")
    print(f"  pr_auc                  : {metrics['pr_auc']:.4f}")
    print(f"  best_f1_threshold       : {metrics['best_f1_threshold']:.4f}")
    print(f"  best_f1                 : {metrics['best_f1']:.4f}")
    print(f"  best_precision          : {metrics['best_precision']:.4f}")
    print(f"  best_recall             : {metrics['best_recall']:.4f}")
    print(f"  recall@precision>=0.80  : {metrics['recall_at_precision_0.80']:.4f}")
    print(f"\nSaved: {model_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {pr_table_path}")

    return model_path, metrics_path, pr_table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 XGBoost model")
    parser.add_argument("--sample-id", type=int, default=1, help="Sample ID to process (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(sample_id=args.sample_id)
