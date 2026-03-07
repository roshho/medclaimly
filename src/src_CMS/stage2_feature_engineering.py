#!/usr/bin/env python3
"""
Stage 2 feature engineering for XGBoost.

Inputs:
- output/sampleXX_stage2_train_labels.parquet
- dataset/CMS/cms_master.parquet
- dataset/CMS/merged-beneficary.csv
- output/reference_tables/drg_reference.parquet

Output:
- output/sampleXX_stage2_features.parquet
- output/sampleXX_stage2_feature_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "CMS" / "cms_master.parquet"
BENE_PATH = PROJECT_ROOT / "dataset" / "CMS" / "merged-beneficary.csv"
DRG_PATH = PROJECT_ROOT / "output" / "reference_tables" / "drg_reference.parquet"
OUT_DIR = PROJECT_ROOT / "output"

CHRONIC_FLAGS = [
    "SP_DIABETES",
    "SP_CHF",
    "SP_CNCR",
    "SP_COPD",
    "SP_CHRNKIDN",
    "SP_OSTEOPRS",
    "SP_RA_OA",
    "SP_STRKETIA",
]


def _count_present(prefix: str, start: int, end: int) -> str:
    return " + ".join(
        [f"CASE WHEN COALESCE(TRIM({prefix}_{idx}), '') <> '' THEN 1 ELSE 0 END" for idx in range(start, end + 1)]
    )


def _age_expr() -> str:
    return (
        "CASE "
        "WHEN TRY_STRPTIME(CAST(c.CLM_FROM_DT AS VARCHAR), '%Y%m%d') IS NOT NULL "
        "AND TRY_STRPTIME(CAST(b.BENE_BIRTH_DT AS VARCHAR), '%Y%m%d') IS NOT NULL "
        "THEN DATE_DIFF('year', TRY_STRPTIME(CAST(b.BENE_BIRTH_DT AS VARCHAR), '%Y%m%d'), "
        "TRY_STRPTIME(CAST(c.CLM_FROM_DT AS VARCHAR), '%Y%m%d')) "
        "ELSE NULL END"
    )


def run_feature_engineering(sample_id: int) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = OUT_DIR / f"sample{sample_id:02d}_stage2_train_labels.parquet"
    out_parquet = OUT_DIR / f"sample{sample_id:02d}_stage2_features.parquet"
    out_summary = OUT_DIR / f"sample{sample_id:02d}_stage2_feature_summary.csv"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Stage 2 labels file not found: {labels_path}. Run stage2_prepare_training.py first."
        )

    con = duckdb.connect()

    con.execute(f"CREATE OR REPLACE VIEW cms_master AS SELECT * FROM read_parquet('{DATA_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW stage2_labels AS SELECT * FROM read_parquet('{labels_path}')")
    con.execute(f"CREATE OR REPLACE VIEW drg_ref AS SELECT * FROM read_parquet('{DRG_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW bene AS SELECT * FROM read_csv_auto('{BENE_PATH}', header=true)")

    drg_cols = {
        row[1] for row in con.execute("PRAGMA table_info('drg_ref')").fetchall()
    }
    drg_weight_expr = "COALESCE(CAST(d.weight AS DOUBLE), 0)" if "weight" in drg_cols else "0"
    drg_geo_expr = "CAST(d.geometric_mean_los AS DOUBLE)" if "geometric_mean_los" in drg_cols else "NULL"
    drg_alos_expr = "CAST(d.alos_hist AS DOUBLE)" if "alos_hist" in drg_cols else "NULL"
    drg_los_ref_expr = f"COALESCE({drg_geo_expr}, {drg_alos_expr}, 0)"

    icd_count_expr = _count_present("c.ICD9_DGNS_CD", 1, 10)
    line_icd_count_expr = _count_present("c.LINE_ICD9_DGNS_CD", 1, 13)
    hcpcs_count_expr = _count_present("c.HCPCS_CD", 1, 45)

    chronic_expr = " + ".join(
        [f"CASE WHEN COALESCE(CAST(b.{flag} AS VARCHAR), '0') IN ('1','Y','y') THEN 1 ELSE 0 END" for flag in CHRONIC_FLAGS]
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stage2_label_keys AS
        SELECT
            l.CLM_ID,
            l.DESYNPUF_ID,
            l.sample_id,
            l.CLM_FROM_DT,
            l.CLM_THRU_DT,
            l.CLM_DRG_CD,
            CASE
                WHEN SUM(CASE WHEN l.is_denied = 1 THEN 1 ELSE 0 END) > 0 THEN 1
                WHEN SUM(CASE WHEN l.is_denied = 0 THEN 1 ELSE 0 END) > 0 THEN 0
                ELSE NULL
            END AS is_denied
        FROM stage2_labels l
        WHERE l.sample_id = {sample_id}
        GROUP BY 1,2,3,4,5,6
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE stage2_features AS
        SELECT
            l.CLM_ID,
            l.DESYNPUF_ID,
            l.sample_id,
            l.is_denied,

            -- Claim complexity
            ({icd_count_expr}) AS icd9_diag_count,
            ({line_icd_count_expr}) AS line_icd9_diag_count,
            ({hcpcs_count_expr}) AS hcpcs_count,
            ({icd_count_expr}) + ({line_icd_count_expr}) AS total_diag_count,

            -- Temporal and utilization
            EXTRACT(MONTH FROM TRY_STRPTIME(CAST(c.CLM_FROM_DT AS VARCHAR), '%Y%m%d')) AS claim_month,
            COALESCE(CAST(c.CLM_UTLZTN_DAY_CNT AS DOUBLE), 0) AS utilization_days,

            -- DRG context
              {drg_weight_expr} AS drg_weight,
              {drg_los_ref_expr} AS drg_los_reference,
            CASE
                 WHEN {drg_los_ref_expr} > 0
                     AND COALESCE(CAST(c.CLM_UTLZTN_DAY_CNT AS DOUBLE), 0) > 1.5 * {drg_los_ref_expr}
                THEN 1 ELSE 0
            END AS los_outlier_flag,
            CASE
                 WHEN {drg_los_ref_expr} > 0
                     AND COALESCE(CAST(c.CLM_UTLZTN_DAY_CNT AS DOUBLE), 0) < 0.5 * {drg_los_ref_expr}
                THEN 1 ELSE 0
            END AS short_stay_flag,

            -- Beneficiary context (2008 cross-section proxy)
            COALESCE(CAST(b.BENE_HI_CVRAGE_TOT_MONS AS INTEGER), 0) AS bene_hi_coverage_months,
            COALESCE(CAST(b.BENE_SMI_CVRAGE_TOT_MONS AS INTEGER), 0) AS bene_smi_coverage_months,
            COALESCE(CAST(b.MEDREIMB_IP AS DOUBLE), 0) AS bene_medreimb_ip,
            COALESCE(CAST(b.MEDREIMB_CAR AS DOUBLE), 0) AS bene_medreimb_car,
            COALESCE(CAST(b.BENRES_CAR AS DOUBLE), 0) AS bene_benres_car,
            { _age_expr() } AS bene_age_at_claim,
            COALESCE(CAST(b.BENE_SEX_IDENT_CD AS INTEGER), 0) AS bene_sex,
            ({chronic_expr}) AS chronic_condition_count,

            -- Keep for audit / troubleshooting
            c.CLM_FROM_DT,
            c.CLM_THRU_DT,
            c.CLM_DRG_CD
                FROM stage2_label_keys l
                JOIN cms_master c
                    ON l.sample_id = c.sample_id
                 AND l.CLM_ID IS NOT DISTINCT FROM c.CLM_ID
                 AND l.DESYNPUF_ID IS NOT DISTINCT FROM c.DESYNPUF_ID
                 AND l.CLM_FROM_DT IS NOT DISTINCT FROM c.CLM_FROM_DT
                 AND l.CLM_THRU_DT IS NOT DISTINCT FROM c.CLM_THRU_DT
                 AND l.CLM_DRG_CD IS NOT DISTINCT FROM c.CLM_DRG_CD
        LEFT JOIN drg_ref d
          ON TRIM(CAST(c.CLM_DRG_CD AS VARCHAR)) = TRIM(CAST(d.drg_code AS VARCHAR))
        LEFT JOIN bene b
          ON l.DESYNPUF_ID = b.DESYNPUF_ID
        """
    )

    con.execute(f"COPY stage2_features TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    con.execute(
        f"""
        COPY (
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN is_denied = 1 THEN 1 ELSE 0 END) AS denied_rows,
                SUM(CASE WHEN is_denied = 0 THEN 1 ELSE 0 END) AS approved_rows,
                AVG(icd9_diag_count) AS avg_icd9_diag_count,
                AVG(hcpcs_count) AS avg_hcpcs_count,
                AVG(chronic_condition_count) AS avg_chronic_conditions,
                AVG(drg_weight) AS avg_drg_weight
            FROM stage2_features
        ) TO '{out_summary}' (HEADER, DELIMITER ',')
        """
    )

    totals = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_denied = 1 THEN 1 ELSE 0 END) AS denied_rows,
            SUM(CASE WHEN is_denied = 0 THEN 1 ELSE 0 END) AS approved_rows
        FROM stage2_features
        """
    ).fetchone()

    print("\nStage 2 feature summary")
    print(f"  sample_id          : {sample_id}")
    print(f"  total_rows         : {totals[0]:,}")
    print(f"  denied_rows (1)    : {totals[1]:,}")
    print(f"  approved_rows (0)  : {totals[2]:,}")
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_summary}")

    con.close()
    return out_parquet, out_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer Stage 2 features")
    parser.add_argument("--sample-id", type=int, default=1, help="Sample ID to process (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_engineering(sample_id=args.sample_id)
