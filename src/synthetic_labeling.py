import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign synthetic denial labels using weighted denial probability."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to master_admissions_features.csv (default: dataset/mimic-iv-3.1/master_admissions_features.csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: dataset/mimic-iv-3.1/master_admissions_labeled.csv).",
    )
    parser.add_argument(
        "--denial-rate",
        type=float,
        default=0.15,
        help="Target denial rate (0.0-1.0, default 0.15).",
    )
    return parser.parse_args()


def assign_labels(input_path: Path, output_path: Path, denial_rate: float) -> None:
    conn = duckdb.connect(database=":memory:")

    features_path = str(input_path)
    conn.execute(f"CREATE VIEW features AS SELECT * FROM read_csv_auto('{features_path}')")

    query = f"""
    WITH domain_rules AS (
        SELECT
            *,
            -- 1. DIRECT DOCUMENTATION (high priority)
            CAST(COALESCE(missing_primary_dx, 0) AS FLOAT) AS doc_missing_primary_dx,
            CAST(COALESCE(missing_primary_px, 0) AS FLOAT) AS doc_missing_primary_px,
            
            -- 2. EVIDENCE FLAGS (high priority)
            CAST(COALESCE(sparse_documentation_flag, 0) AS FLOAT) AS ev_sparse_doc,
            COALESCE(lab_abnormal_count, 0) AS ev_abnormal_labs,
            COALESCE(micro_test_count, 0) AS ev_micro_tests,
            
            -- 3. CODE VOLUME & STRUCTURE (medium priority)
            CAST(COALESCE(low_evidence_high_intensity_flag, 0) AS FLOAT) AS code_low_evidence_intensity,
            COALESCE(dx_count, 0) AS code_dx_count,
            COALESCE(px_count, 0) AS code_px_count,
            COALESCE(hcpcs_count, 0) AS code_hcpcs_count,
            -- Imbalance: many procedures but few diagnoses (weak justification)
            CASE
                WHEN COALESCE(px_count, 0) >= 2 AND COALESCE(dx_count, 0) <= 1 THEN 1
                ELSE 0
            END AS code_px_dx_imbalance,
            
            -- 4. INTENSITY & LOS ALIGNMENT (medium priority)
            CAST(COALESCE(high_intensity_short_los_flag, 0) AS FLOAT) AS intensity_high_short_los,
            COALESCE(procedures_per_day, 0) AS intensity_px_per_day,
            COALESCE(los_days, 0) AS intensity_los_days,
            
            -- 5. CARE TRAJECTORY (medium priority)
            COALESCE(service_changes, 0) AS traj_service_changes,
            COALESCE(transfer_count, 0) AS traj_transfers,
            
            -- 6. DOMAIN-SPECIFIC RULES (medium-high priority)
            -- a) Insurance type impact (Medicaid often stricter)
            CASE WHEN insurance = 'Medicaid' THEN 1 ELSE 0 END AS domain_medicaid,
            
            -- b) Elective procedure with weak/missing diagnosis
            CASE
                WHEN admission_type LIKE 'EW%' AND COALESCE(missing_primary_dx, 0) = 1 THEN 1
                ELSE 0
            END AS domain_elective_weak_dx,
            
            -- c) Age extremes with few comorbidities (unusual)
            CASE
                WHEN (anchor_age < 18 OR anchor_age > 85) AND COALESCE(dx_count, 0) < 2 THEN 1
                ELSE 0
            END AS domain_age_extremes_few_comorbidities,
            
            -- d) High procedure volume with low evidence (unbundling risk proxy)
            CASE
                WHEN COALESCE(px_count, 0) >= 3 AND COALESCE(lab_test_count, 0) = 0 THEN 1
                ELSE 0
            END AS domain_high_px_no_labs,
            
            -- e) Multiple HCPCS codes with short LOS (billing intensity)
            CASE
                WHEN COALESCE(hcpcs_count, 0) >= 3 AND COALESCE(los_days, 0) <= 1.0 THEN 1
                ELSE 0
            END AS domain_high_hcpcs_short_los
        FROM features
    ),
    scored AS (
        SELECT
            *,
            -- DIRECT DOCUMENTATION score (high weight: 0.20)
            (
                (doc_missing_primary_dx + doc_missing_primary_px) / 2.0
            ) * 0.20 AS score_documentation,
            
            -- EVIDENCE score (high weight: 0.20)
            -- Abnormal labs/micro get extra emphasis
            (
                ev_sparse_doc
                + CASE WHEN ev_abnormal_labs > 0 THEN 0.5 ELSE 0 END
                + CASE WHEN ev_micro_tests = 0 THEN 0.5 ELSE 0 END
            ) / 2.0 * 0.20 AS score_evidence,
            
            -- CODE VOLUME & STRUCTURE score (medium weight: 0.15)
            (
                code_low_evidence_intensity
                + code_px_dx_imbalance
                + CASE WHEN code_hcpcs_count >= 2 AND code_dx_count <= 1 THEN 1 ELSE 0 END
            ) / 3.0 * 0.15 AS score_code_structure,
            
            -- INTENSITY & LOS ALIGNMENT score (medium weight: 0.15)
            (
                intensity_high_short_los
                + CASE WHEN intensity_px_per_day > 2 THEN 1 ELSE 0 END
            ) / 2.0 * 0.15 AS score_intensity_los,
            
            -- CARE TRAJECTORY score (low-medium weight: 0.10)
            (
                CASE WHEN traj_service_changes >= 3 THEN 1 ELSE 0 END
                + CASE WHEN traj_transfers >= 3 THEN 1 ELSE 0 END
            ) / 2.0 * 0.10 AS score_trajectory,
            
            -- DOMAIN-SPECIFIC RULES (medium-high weight: 0.20)
            (
                domain_medicaid
                + domain_elective_weak_dx
                + domain_age_extremes_few_comorbidities
                + domain_high_px_no_labs
                + domain_high_hcpcs_short_los
            ) / 5.0 * 0.20 AS score_domain_rules
        FROM domain_rules
    ),
    normalized AS (
        SELECT
            *,
            COALESCE(score_documentation, 0)
            + COALESCE(score_evidence, 0)
            + COALESCE(score_code_structure, 0)
            + COALESCE(score_intensity_los, 0)
            + COALESCE(score_trajectory, 0)
            + COALESCE(score_domain_rules, 0) AS denial_probability
        FROM scored
    ),
    ranked AS (
        SELECT
            *,
            PERCENT_RANK() OVER (ORDER BY denial_probability ASC) AS percentile_rank
        FROM normalized
    )
    SELECT
        *,
        CASE
            WHEN percentile_rank <= {denial_rate} THEN 1
            ELSE 0
        END AS synthetic_denial_label
    FROM ranked
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"COPY ({query}) TO '{str(output_path)}' WITH (HEADER, DELIMITER ',')"
    )

    conn.execute(f"CREATE VIEW labeled AS ({query})")
    stats = conn.execute(
        "SELECT COUNT(*) as total, SUM(synthetic_denial_label) as denied_count, "
        "ROUND(100.0 * SUM(synthetic_denial_label) / COUNT(*), 2) as denial_rate "
        "FROM labeled"
    ).fetchall()
    
    total, denied_count, actual_rate = stats[0]
    print(f"✓ Labeled dataset created: {total} total admissions, {denied_count} denied ({actual_rate}%)")
    print(f"✓ Output: {output_path}")


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_features.csv"
    default_output = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_labeled.csv"

    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output

    assign_labels(input_path, output_path, args.denial_rate)


if __name__ == "__main__":
    main()
