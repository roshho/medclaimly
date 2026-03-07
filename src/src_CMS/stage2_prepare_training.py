#!/usr/bin/env python3
"""
Stage 2 training set preparation (gray-zone only).

Input:
- output/sampleXX_stage1_results.parquet
- dataset/CMS/cms_master.parquet

Logic:
- Keep only Stage 1 pass-through claims (stage1_decision='PASS_TO_ML')
- Exclude observed B/C claims from ML target construction
- Synthetic positive (is_denied=1) if any soft risk trigger fires:
  1) soft_ptp_risk: claim has any PTP pair (modifier in 0/1/9)
  2) mue_frequency_risk: same HCPCS appears >1 on same claim (proxy for same-day repeat)
  3) mcd_hcpcs_no_diag: claim has MCD-linked HCPCS but no diagnosis documented
- Synthetic negative (is_denied=0) if no risk trigger and observed A only

Output:
- output/sampleXX_stage2_train_labels.parquet
- output/sampleXX_stage2_label_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "CMS" / "cms_master.parquet"
REF_DIR = PROJECT_ROOT / "output" / "reference_tables"
OUT_DIR = PROJECT_ROOT / "output"


# Synthetic rule assets for label generation
# ICD-9 symptom-only primary diagnosis prefixes (780-789 range subsets)
SYMPTOM_ICD9_PREFIXES = [
    "780.0", "780.3", "780.6", "780.7",  # Consciousness, convulsions, fever, fatigue
    "781.", "782.", "783.", "784.",      # Nervous, skin, nutrition, head/neck symptoms
    "785.", "786.", "787.", "789.",      # Cardiovascular, respiratory, digestive, abdominal
]

# Curated HCPCS/CPT groups requiring confirmed diagnosis (not symptom-only primary)
# Group 1: Surgical procedures requiring confirmed diagnosis (weight 1.0)
SURGICAL_PROCEDURES = {
    "27447": 1.0, "27130": 1.0, "22551": 1.0, "22612": 1.0, "43239": 1.0,
    "45385": 1.0, "33533": 1.0, "33510": 1.0, "61510": 1.0, "58150": 1.0,
}

# Group 2: High-cost imaging requiring clinical indication (weight 0.7)
IMAGING_PROCEDURES = {
    "70553": 0.7, "71250": 0.7, "74177": 0.7, "78816": 0.7,
    "78452": 0.7, "93306": 0.7, "70450": 0.7, "72148": 0.7,
}

# Group 3: Chemotherapy and oncology procedures (weight 1.0)
ONCOLOGY_PROCEDURES = {
    "96413": 1.0, "96415": 1.0, "96401": 1.0,
    "J9035": 1.0, "J9055": 1.0, "J9305": 1.0,
}

# Group 4: DME and supplies requiring confirmed chronic condition (weight 0.8)
DME_PROCEDURES = {
    "E0601": 0.8, "E0470": 0.8, "K0001": 0.8,
    "A4253": 0.8, "A9150": 0.8, "L5100": 0.8,
}

# Combined mapping: HCPCS code -> confidence weight
DEFINITIVE_PROCEDURE_WEIGHTS = {
    **SURGICAL_PROCEDURES,
    **IMAGING_PROCEDURES,
    **ONCOLOGY_PROCEDURES,
    **DME_PROCEDURES,
}


def _coalesce_or_trim(prefix: str, start: int, end: int) -> str:
    return " OR ".join(
        [f"COALESCE(TRIM({prefix}_{idx}), '') <> ''" for idx in range(start, end + 1)]
    )


def _line_flag(prefix: str, target: str, start: int, end: int) -> str:
    return " OR ".join(
        [f"COALESCE(TRIM({prefix}_{idx}), '') = '{target}'" for idx in range(start, end + 1)]
    )


def _hcpcs_long_select(table_alias: str = "claims") -> str:
    parts = []
    for idx in range(1, 46):
        parts.append(
            f"SELECT CLM_ID, DESYNPUF_ID, CLM_FROM_DT, TRIM(HCPCS_CD_{idx}) AS hcpcs_code "
            f"FROM {table_alias} WHERE COALESCE(TRIM(HCPCS_CD_{idx}), '') <> ''"
        )
    return "\nUNION ALL\n".join(parts)


def run_prepare(sample_id: int) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_path = OUT_DIR / f"sample{sample_id:02d}_stage1_results.parquet"
    out_parquet = OUT_DIR / f"sample{sample_id:02d}_stage2_train_labels.parquet"
    out_summary = OUT_DIR / f"sample{sample_id:02d}_stage2_label_summary.csv"

    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {stage1_path}. Run stage1_rules_engine.py first."
        )

    con = duckdb.connect()

    con.execute(f"CREATE OR REPLACE VIEW cms_master AS SELECT * FROM read_parquet('{DATA_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW stage1 AS SELECT * FROM read_parquet('{stage1_path}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_ptp AS SELECT * FROM read_parquet('{REF_DIR / 'ptp_reference.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_mcd_hcpcs AS SELECT * FROM read_parquet('{REF_DIR / 'mcd_hcpcs_articles.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_drg AS SELECT * FROM read_parquet('{REF_DIR / 'drg_reference.parquet'}')")

    has_diag_expr = _coalesce_or_trim("ICD9_DGNS_CD", 1, 10)
    has_line_diag_expr = _coalesce_or_trim("LINE_ICD9_DGNS_CD", 1, 13)
    observed_a_expr = _line_flag("LINE_PRCSG_IND_CD", "A", 1, 13)
    observed_b_expr = _line_flag("LINE_PRCSG_IND_CD", "B", 1, 13)
    observed_c_expr = _line_flag("LINE_PRCSG_IND_CD", "C", 1, 13)
    # Build symptom ICD9 prefix match expression
    symptom_conditions = " OR ".join(
        [f"COALESCE(TRIM(c.LINE_ICD9_DGNS_CD_1), '') LIKE '{prefix}%'" for prefix in SYMPTOM_ICD9_PREFIXES]
    )

    # Build definitive procedure HCPCS match with weights
    definitive_hcpcs_cases = []
    for code, weight in DEFINITIVE_PROCEDURE_WEIGHTS.items():
        definitive_hcpcs_cases.append(f"WHEN COALESCE(TRIM(ch.hcpcs_code), '') = '{code}' THEN {weight}")
    
    definitive_weight_expr = "CASE " + " ".join(definitive_hcpcs_cases) + " ELSE 0 END"


    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE stage1_pass_keys AS
        SELECT DISTINCT
            CLM_ID,
            DESYNPUF_ID,
            sample_id,
            CLM_FROM_DT,
            CLM_THRU_DT,
            CLM_DRG_CD
        FROM stage1
        WHERE stage1_decision = 'PASS_TO_ML'
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE claims AS
        SELECT c.*
        FROM cms_master c
        JOIN stage1_pass_keys s
          ON c.sample_id = s.sample_id
         AND c.CLM_ID IS NOT DISTINCT FROM s.CLM_ID
         AND c.DESYNPUF_ID IS NOT DISTINCT FROM s.DESYNPUF_ID
         AND c.CLM_FROM_DT IS NOT DISTINCT FROM s.CLM_FROM_DT
         AND c.CLM_THRU_DT IS NOT DISTINCT FROM s.CLM_THRU_DT
         AND c.CLM_DRG_CD IS NOT DISTINCT FROM s.CLM_DRG_CD
        WHERE c.sample_id = {sample_id}
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE claim_hcpcs AS
        {_hcpcs_long_select('claims')}
        """
    )

    # Soft trigger 1: PTP pair conflict (weight 0.6)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE claim_hcpcs_pairs AS
        SELECT DISTINCT
            a.CLM_ID,
            LEAST(a.hcpcs_code, b.hcpcs_code) AS code_lo,
            GREATEST(a.hcpcs_code, b.hcpcs_code) AS code_hi
        FROM claim_hcpcs a
        JOIN claim_hcpcs b
          ON a.CLM_ID = b.CLM_ID
         AND a.hcpcs_code < b.hcpcs_code
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE ref_ptp_pairs AS
        SELECT DISTINCT
            LEAST(TRIM(code1), TRIM(code2)) AS code_lo,
            GREATEST(TRIM(code1), TRIM(code2)) AS code_hi
        FROM ref_ptp
        WHERE COALESCE(TRIM(code1), '') <> ''
          AND COALESCE(TRIM(code2), '') <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE soft_ptp_risk AS
        SELECT DISTINCT p.CLM_ID, 0.6 AS weight, 'ptp_conflict' AS source_name
        FROM claim_hcpcs_pairs p
        JOIN ref_ptp_pairs r
          ON p.code_lo = r.code_lo
         AND p.code_hi = r.code_hi
        """
    )

    # Soft trigger 2: MUE frequency proxy (weight 0.5)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE soft_mue_frequency_risk AS
        SELECT CLM_ID, 0.5 AS weight, 'mue_frequency' AS source_name
        FROM claim_hcpcs
        GROUP BY CLM_ID, hcpcs_code
        HAVING COUNT(*) > 1
        """
    )

    # Soft trigger 3: MCD-linked HCPCS with no diagnosis (weight 0.7)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE mcd_hcpcs AS
        SELECT DISTINCT TRIM(hcpcs_code) AS hcpcs_code
        FROM ref_mcd_hcpcs
        WHERE COALESCE(TRIM(hcpcs_code), '') <> ''
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE soft_mcd_no_diag_risk AS
        SELECT DISTINCT c.CLM_ID, 0.7 AS weight, 'lcd_no_diagnosis' AS source_name
        FROM claims c
        JOIN claim_hcpcs ch ON c.CLM_ID = ch.CLM_ID
        JOIN mcd_hcpcs m ON ch.hcpcs_code = m.hcpcs_code
        WHERE NOT ({has_diag_expr})
          AND NOT ({has_line_diag_expr})
        """
    )

    # Soft trigger 4: DRG short-stay upcoding proxy (weight 0.85)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE soft_drg_shortstay_risk AS
        SELECT DISTINCT
            c.CLM_ID,
            0.85 AS weight,
            'drg_shortstay_upcoding' AS source_name
        FROM claims c
        JOIN ref_drg d
          ON TRIM(CAST(c.CLM_DRG_CD AS VARCHAR)) = TRIM(CAST(d.drg_code AS VARCHAR))
        WHERE COALESCE(CAST(c.CLM_UTLZTN_DAY_CNT AS DOUBLE), 0) >= 1
          AND COALESCE(CAST(d.alos_hist AS DOUBLE), CAST(d.weight AS DOUBLE), 0) > 0
          AND COALESCE(CAST(c.CLM_UTLZTN_DAY_CNT AS DOUBLE), 0) < 0.5 * COALESCE(CAST(d.alos_hist AS DOUBLE), CAST(d.weight AS DOUBLE), 0)
        """
    )

    # Soft trigger 5: Symptom primary diagnosis + definitive procedure (weighted by group)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE soft_symptom_definitive_risk AS
        SELECT DISTINCT
            c.CLM_ID,
            MAX({definitive_weight_expr}) AS weight,
            'symptom_primary_definitive_proc' AS source_name
        FROM claims c
        JOIN claim_hcpcs ch ON c.CLM_ID = ch.CLM_ID
        WHERE ({symptom_conditions})
          AND ({definitive_weight_expr}) > 0
        GROUP BY c.CLM_ID
        """
    )

    # Aggregate all risk sources
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE all_risk_sources AS
        SELECT CLM_ID, weight, source_name FROM soft_ptp_risk
        UNION ALL
        SELECT CLM_ID, weight, source_name FROM soft_mue_frequency_risk
        UNION ALL
        SELECT CLM_ID, weight, source_name FROM soft_mcd_no_diag_risk
        UNION ALL
        SELECT CLM_ID, weight, source_name FROM soft_drg_shortstay_risk
        UNION ALL
        SELECT CLM_ID, weight, source_name FROM soft_symptom_definitive_risk
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE claim_risk_summary AS
        SELECT
            CLM_ID,
            MAX(weight) AS denial_confidence,
            STRING_AGG(DISTINCT source_name, '|' ORDER BY source_name) AS denial_sources
        FROM all_risk_sources
        GROUP BY CLM_ID
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE stage2_train_labels AS
        SELECT
            c.CLM_ID,
            c.DESYNPUF_ID,
            c.sample_id,
            c.CLM_FROM_DT,
            c.CLM_THRU_DT,
            c.CLM_DRG_CD,
            COALESCE(r.denial_confidence, 0.0) AS denial_confidence,
            COALESCE(r.denial_sources, '') AS denial_sources,
            CASE
                WHEN ({observed_b_expr}) OR ({observed_c_expr}) THEN NULL
                WHEN r.CLM_ID IS NOT NULL THEN 1
                WHEN ({observed_a_expr}) THEN 0
                ELSE NULL
            END AS is_denied
        FROM claims c
        LEFT JOIN claim_risk_summary r ON c.CLM_ID = r.CLM_ID
        """
    )

    con.execute(f"COPY stage2_train_labels TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    con.execute(
        f"""
        COPY (
            SELECT is_denied, denial_sources, COUNT(*) AS claim_count
            FROM stage2_train_labels
            WHERE is_denied IS NOT NULL
            GROUP BY 1, 2
            ORDER BY claim_count DESC
        ) TO '{out_summary}' (HEADER, DELIMITER ',')
        """
    )

    totals = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_denied = 1 THEN 1 ELSE 0 END) AS denied_rows,
            SUM(CASE WHEN is_denied = 0 THEN 1 ELSE 0 END) AS approved_rows,
            SUM(CASE WHEN is_denied IS NULL THEN 1 ELSE 0 END) AS excluded_or_unlabeled_rows
        FROM stage2_train_labels
        """
    ).fetchone()

    print("\nStage 2 label prep summary")
    print(f"  sample_id                  : {sample_id}")
    print(f"  total_rows                 : {totals[0]:,}")
    print(f"  synthetic denied (1)       : {totals[1]:,}")
    print(f"  synthetic approved (0)     : {totals[2]:,}")
    print(f"  excluded/unlabeled (NULL)  : {totals[3]:,}")
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_summary}")

    con.close()
    return out_parquet, out_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 synthetic training labels")
    parser.add_argument("--sample-id", type=int, default=1, help="Sample ID to process (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prepare(sample_id=args.sample_id)
