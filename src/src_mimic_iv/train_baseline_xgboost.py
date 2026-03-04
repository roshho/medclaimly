import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost baseline model for denial prediction."
    )
    parser.add_argument(
        "--train",
        default=None,
        help="Path to training CSV (default: dataset/mimic-iv-3.1/master_admissions_train.csv).",
    )
    parser.add_argument(
        "--val",
        default=None,
        help="Path to validation CSV (default: dataset/mimic-iv-3.1/master_admissions_val.csv).",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Path to test CSV (default: dataset/mimic-iv-3.1/master_admissions_test.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for model and metrics (default: output/).",
    )
    return parser.parse_args()


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select and preprocess features for XGBoost."""
    categorical_cols = [
        "admission_type",
        "insurance",
        "race",
        "gender",
        "has_death_time",
        "has_ed_times",
        "hospital_expire_flag",
    ]
    
    numerical_cols = [
        "anchor_age",
        "los_minutes",
        "los_hours",
        "los_days",
        "ed_minutes",
        "ed_hours",
        "missing_primary_dx",
        "missing_primary_px",
        "dx_count",
        "dx_secondary_count",
        "px_count",
        "px_secondary_count",
        "hcpcs_count",
        "hcpcs_list_count",
        "drg_count",
        "drg_list_count",
        "service_changes",
        "service_path_count",
        "transfer_count",
        "transfer_path_count",
        "medication_events",
        "medications_count",
        "lab_test_count",
        "lab_abnormal_count",
        "micro_test_count",
        "micro_test_list_count",
        "procedures_per_day",
        "labs_per_day",
        "meds_per_day",
        "low_evidence_high_intensity_flag",
        "high_intensity_short_los_flag",
        "sparse_documentation_flag",
    ]
    
    feature_cols = categorical_cols + numerical_cols
    X = df[feature_cols].copy()
    
    # Encode categorical features
    le_dict = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna("MISSING").astype(str))
            le_dict[col] = le
    
    # Fill missing numerical
    for col in numerical_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    
    return X, feature_cols


def train_model(train_path: Path, val_path: Path, test_path: Path, output_dir: Path) -> None:
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Extract labels
    y_train = train_df["synthetic_denial_label"]
    y_val = val_df["synthetic_denial_label"]
    y_test = test_df["synthetic_denial_label"]
    
    # Preprocess features
    X_train, feature_cols = preprocess_features(train_df)
    X_val, _ = preprocess_features(val_df)
    X_test, _ = preprocess_features(test_df)
    
    # Align columns
    for col in feature_cols:
        if col not in X_val.columns:
            X_val[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    X_val = X_val[feature_cols]
    X_test = X_test[feature_cols]
    
    print(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "val_accuracy": accuracy_score(y_val, y_pred_val),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_roc_auc": roc_auc_score(y_train, y_proba_train),
        "val_roc_auc": roc_auc_score(y_val, y_proba_val),
        "test_roc_auc": roc_auc_score(y_test, y_proba_test),
    }
    
    # PR curve
    prec_test, rec_test, _ = precision_recall_curve(y_test, y_proba_test)
    metrics["test_pr_auc"] = auc(rec_test, prec_test)
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_denial_classifier.model"
    model.save_model(str(model_path))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    # Print results
    print("\n" + "=" * 60)
    print("BASELINE XGBOOST MODEL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"Val Accuracy:    {metrics['val_accuracy']:.4f}")
    print(f"Test Accuracy:   {metrics['test_accuracy']:.4f}")
    print(f"\nTrain ROC-AUC:   {metrics['train_roc_auc']:.4f}")
    print(f"Val ROC-AUC:     {metrics['val_roc_auc']:.4f}")
    print(f"Test ROC-AUC:    {metrics['test_roc_auc']:.4f}")
    print(f"Test PR-AUC:     {metrics['test_pr_auc']:.4f}")
    print(f"\nModel saved:     {model_path}")
    print("\nTop 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save metrics
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved:   {metrics_path}")


def main() -> None:
    args = parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    default_train = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_train.csv"
    default_val = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_val.csv"
    default_test = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_test.csv"
    default_output_dir = base_dir / "output"
    
    train_path = Path(args.train) if args.train else default_train
    val_path = Path(args.val) if args.val else default_val
    test_path = Path(args.test) if args.test else default_test
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    
    train_model(train_path, val_path, test_path, output_dir)


if __name__ == "__main__":
    main()
