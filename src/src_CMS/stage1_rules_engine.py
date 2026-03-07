#!/usr/bin/env python3
"""
Stage 1 Deterministic Rules Engine (Sample-first implementation)

Hard deny rules:
1) HCPCS_NONCOVERED  : claim contains HCPCS flagged non-covered in reference
2) PTP_UNBUNDLING    : claim contains PTP pair with modifier=0 (not allowed)
3) LCD_CONDITION_UNMET: claim has MCD-linked HCPCS but no diagnosis documented (proxy)

Calibration-only guardrails:
- observed_has_b : any LINE_PRCSG_IND_CD_* == 'B'
- observed_has_c : any LINE_PRCSG_IND_CD_* == 'C'

Outputs:
- output/sampleXX_stage1_results.parquet
- output/sampleXX_stage1_rule_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "CMS" / "cms_master.parquet"
REF_DIR = PROJECT_ROOT / "output" / "reference_tables"
OUT_DIR = PROJECT_ROOT / "output"


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


def run_stage1(sample_id: int, max_claims: int | None = None) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_parquet = OUT_DIR / f"sample{sample_id:02d}_stage1_results.parquet"
    out_summary_csv = OUT_DIR / f"sample{sample_id:02d}_stage1_rule_summary.csv"

    con = duckdb.connect()

    # Register source and reference tables
    con.execute(f"CREATE OR REPLACE VIEW cms_master AS SELECT * FROM read_parquet('{DATA_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_hcpcs AS SELECT * FROM read_parquet('{REF_DIR / 'hcpcs_reference.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_ptp AS SELECT * FROM read_parquet('{REF_DIR / 'ptp_reference.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW ref_mcd_hcpcs AS SELECT * FROM read_parquet('{REF_DIR / 'mcd_hcpcs_articles.parquet'}')")

    limit_clause = f"LIMIT {max_claims}" if max_claims else ""

    has_diag_expr = _coalesce_or_trim("ICD9_DGNS_CD", 1, 10)
    has_line_diag_expr = _coalesce_or_trim("LINE_ICD9_DGNS_CD", 1, 13)
    observed_b_expr = _line_flag("LINE_PRCSG_IND_CD", "B", 1, 13)
    observed_c_expr = _line_flag("LINE_PRCSG_IND_CD", "C", 1, 13)

    claim_hcpcs_sql = _hcpcs_long_select("claims")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE claims AS
        SELECT *
        FROM cms_master
        WHERE sample_id = {sample_id}
        {limit_clause}
        """
    )

    # Long HCPCS table (claim, hcpcs_code)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE claim_hcpcs AS
        {claim_hcpcs_sql}
        """
    )

    # Rule 1: HCPCS non-covered
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE rule_hcpcs_noncovered AS
        SELECT DISTINCT ch.CLM_ID
        FROM claim_hcpcs ch
        JOIN ref_hcpcs h
          ON ch.hcpcs_code = h.hcpcs_code
        WHERE COALESCE(h.is_noncovered, FALSE) = TRUE
        """
    )

    # Rule 2: PTP unbundling (modifier 0)
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
            GREATEST(TRIM(code1), TRIM(code2)) AS code_hi,
            TRIM(modifier) AS modifier
        FROM ref_ptp
        WHERE COALESCE(TRIM(code1), '') <> ''
          AND COALESCE(TRIM(code2), '') <> ''
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE rule_ptp_unbundling AS
        SELECT DISTINCT p.CLM_ID
        FROM claim_hcpcs_pairs p
        JOIN ref_ptp_pairs r
          ON p.code_lo = r.code_lo
         AND p.code_hi = r.code_hi
        WHERE r.modifier = '0'
        """
    )

    # Rule 3: LCD condition unmet (proxy for SynPUF ICD9/ICD10 mismatch)
    # Proxy used: claim has HCPCS with MCD article linkage, but no diagnosis populated.
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
        CREATE OR REPLACE TEMP TABLE rule_lcd_condition_unmet AS
        SELECT DISTINCT c.CLM_ID
        FROM claims c
        JOIN claim_hcpcs ch
          ON c.CLM_ID = ch.CLM_ID
        JOIN mcd_hcpcs m
          ON ch.hcpcs_code = m.hcpcs_code
        WHERE NOT ({has_diag_expr})
          AND NOT ({has_line_diag_expr})
        """
    )

    # Assemble Stage 1 decision
    con.execute(
        f"""
        CREATE OR REPLACE TABLE stage1_results AS
        SELECT
            c.CLM_ID,
            c.DESYNPUF_ID,
            c.sample_id,
            c.CLM_FROM_DT,
            c.CLM_THRU_DT,
            c.CLM_DRG_CD,
            CASE WHEN n.CLM_ID IS NOT NULL THEN TRUE ELSE FALSE END AS rule_hcpcs_noncovered,
            CASE WHEN p.CLM_ID IS NOT NULL THEN TRUE ELSE FALSE END AS rule_ptp_unbundling,
            CASE WHEN l.CLM_ID IS NOT NULL THEN TRUE ELSE FALSE END AS rule_lcd_condition_unmet,
            CASE WHEN ({observed_b_expr}) THEN TRUE ELSE FALSE END AS observed_has_b,
            CASE WHEN ({observed_c_expr}) THEN TRUE ELSE FALSE END AS observed_has_c,
            CASE
                WHEN n.CLM_ID IS NOT NULL THEN 'HCPCS_NONCOVERED'
                WHEN p.CLM_ID IS NOT NULL THEN 'PTP_UNBUNDLING'
                WHEN l.CLM_ID IS NOT NULL THEN 'LCD_CONDITION_UNMET'
                ELSE 'NONE'
            END AS stage1_reason_code,
            CASE
                WHEN n.CLM_ID IS NOT NULL OR p.CLM_ID IS NOT NULL OR l.CLM_ID IS NOT NULL
                THEN 'HARD_DENY'
                ELSE 'PASS_TO_ML'
            END AS stage1_decision,
            CASE
                WHEN n.CLM_ID IS NOT NULL OR p.CLM_ID IS NOT NULL OR l.CLM_ID IS NOT NULL
                THEN 1.0
                ELSE 0.0
            END AS stage1_confidence
        FROM claims c
        LEFT JOIN rule_hcpcs_noncovered n ON c.CLM_ID = n.CLM_ID
        LEFT JOIN rule_ptp_unbundling p ON c.CLM_ID = p.CLM_ID
        LEFT JOIN rule_lcd_condition_unmet l ON c.CLM_ID = l.CLM_ID
        """
    )

    con.execute(f"COPY stage1_results TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    con.execute(
        f"""
        COPY (
            SELECT stage1_reason_code, stage1_decision, COUNT(*) AS claim_count
            FROM stage1_results
            GROUP BY 1,2
            ORDER BY claim_count DESC
        ) TO '{out_summary_csv}' (HEADER, DELIMITER ',')
        """
    )

    totals = con.execute(
        """
        SELECT
            COUNT(*) AS total_claims,
            SUM(CASE WHEN stage1_decision='HARD_DENY' THEN 1 ELSE 0 END) AS hard_deny_claims,
            SUM(CASE WHEN stage1_decision='PASS_TO_ML' THEN 1 ELSE 0 END) AS pass_to_ml_claims,
            SUM(CASE WHEN observed_has_b THEN 1 ELSE 0 END) AS observed_b_claims,
            SUM(CASE WHEN observed_has_c THEN 1 ELSE 0 END) AS observed_c_claims
        FROM stage1_results
        """
    ).fetchone()

    print("\nStage 1 summary")
    print(f"  sample_id         : {sample_id}")
    print(f"  total_claims      : {totals[0]:,}")
    print(f"  hard_deny_claims  : {totals[1]:,}")
    print(f"  pass_to_ml_claims : {totals[2]:,}")
    print(f"  observed_b_claims : {totals[3]:,} (guardrail only)")
    print(f"  observed_c_claims : {totals[4]:,} (guardrail only)")
    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_summary_csv}")

    con.close()
    return out_parquet, out_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 deterministic rules engine")
    parser.add_argument("--sample-id", type=int, default=1, help="Sample ID to process (default: 1)")
    parser.add_argument(
        "--max-claims",
        type=int,
        default=None,
        help="Optional row cap for smoke tests (default: full sample)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage1(sample_id=args.sample_id, max_claims=args.max_claims)
