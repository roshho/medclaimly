import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer denial-risk features from master_admissions.csv."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to master_admissions.csv (default: dataset/mimic-iv-3.1/master_admissions.csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: dataset/mimic-iv-3.1/master_admissions_features.csv).",
    )
    return parser.parse_args()


def build_features(input_path: Path, output_path: Path) -> None:
    conn = duckdb.connect(database=":memory:")

    master_path = str(input_path)
    conn.execute(f"CREATE VIEW master AS SELECT * FROM read_csv_auto('{master_path}')")

    query = """
    WITH base AS (
        SELECT
            *,
            CASE
                WHEN admittime IS NULL OR dischtime IS NULL THEN NULL
                ELSE date_diff('minute', try_cast(admittime AS TIMESTAMP), try_cast(dischtime AS TIMESTAMP))
            END AS los_minutes,
            CASE
                WHEN edregtime IS NULL OR edouttime IS NULL THEN NULL
                ELSE date_diff('minute', try_cast(edregtime AS TIMESTAMP), try_cast(edouttime AS TIMESTAMP))
            END AS ed_minutes
        FROM master
    ),
    counts AS (
        SELECT
            *,
            CASE WHEN dx_secondary_list IS NULL OR trim(dx_secondary_list) = ''
                THEN 0 ELSE len(str_split(dx_secondary_list, ';')) END AS dx_secondary_count,
            CASE WHEN px_secondary_list IS NULL OR trim(px_secondary_list) = ''
                THEN 0 ELSE len(str_split(px_secondary_list, ';')) END AS px_secondary_count,
            CASE WHEN hcpcs_list IS NULL OR trim(hcpcs_list) = ''
                THEN 0 ELSE len(str_split(hcpcs_list, ';')) END AS hcpcs_list_count,
            CASE WHEN drg_list IS NULL OR trim(drg_list) = ''
                THEN 0 ELSE len(str_split(drg_list, ';')) END AS drg_list_count,
            CASE WHEN service_path IS NULL OR trim(service_path) = ''
                THEN 0 ELSE len(str_split(service_path, ';')) END AS service_path_count,
            CASE WHEN transfer_path IS NULL OR trim(transfer_path) = ''
                THEN 0 ELSE len(str_split(transfer_path, ';')) END AS transfer_path_count,
            CASE WHEN medications IS NULL OR trim(medications) = ''
                THEN 0 ELSE len(str_split(medications, ';')) END AS medications_count,
            CASE WHEN micro_test_list IS NULL OR trim(micro_test_list) = ''
                THEN 0 ELSE len(str_split(micro_test_list, ';')) END AS micro_test_list_count
        FROM base
    )
    SELECT
        subject_id,
        hadm_id,
        admission_type,
        insurance,
        race,
        gender,
        anchor_age,
        hospital_expire_flag,
        CASE WHEN deathtime IS NULL THEN 0 ELSE 1 END AS has_death_time,
        CASE WHEN edregtime IS NULL OR edouttime IS NULL THEN 0 ELSE 1 END AS has_ed_times,
        los_minutes,
        CASE WHEN los_minutes IS NULL THEN NULL ELSE los_minutes / 60.0 END AS los_hours,
        CASE WHEN los_minutes IS NULL THEN NULL ELSE los_minutes / 1440.0 END AS los_days,
        ed_minutes,
        CASE WHEN ed_minutes IS NULL THEN NULL ELSE ed_minutes / 60.0 END AS ed_hours,
        CASE WHEN dx_primary_icd_code IS NULL OR trim(dx_primary_icd_code) = '' THEN 1 ELSE 0 END AS missing_primary_dx,
        CASE WHEN px_primary_icd_code IS NULL OR trim(px_primary_icd_code) = '' THEN 1 ELSE 0 END AS missing_primary_px,
        coalesce(dx_count, 0) AS dx_count,
        dx_secondary_count,
        coalesce(px_count, 0) AS px_count,
        px_secondary_count,
        coalesce(hcpcs_count, 0) AS hcpcs_count,
        hcpcs_list_count,
        coalesce(drg_count, 0) AS drg_count,
        drg_list_count,
        coalesce(service_changes, 0) AS service_changes,
        service_path_count,
        coalesce(transfer_count, 0) AS transfer_count,
        transfer_path_count,
        coalesce(medication_events, 0) AS medication_events,
        medications_count,
        coalesce(lab_test_count, 0) AS lab_test_count,
        coalesce(lab_abnormal_count, 0) AS lab_abnormal_count,
        coalesce(micro_test_count, 0) AS micro_test_count,
        micro_test_list_count,
        CASE
            WHEN los_minutes IS NULL OR los_minutes = 0 THEN NULL
            ELSE coalesce(px_count, 0) / (los_minutes / 1440.0)
        END AS procedures_per_day,
        CASE
            WHEN los_minutes IS NULL OR los_minutes = 0 THEN NULL
            ELSE coalesce(lab_test_count, 0) / (los_minutes / 1440.0)
        END AS labs_per_day,
        CASE
            WHEN los_minutes IS NULL OR los_minutes = 0 THEN NULL
            ELSE coalesce(medication_events, 0) / (los_minutes / 1440.0)
        END AS meds_per_day,
        CASE
            WHEN (coalesce(px_count, 0) >= 2 OR coalesce(hcpcs_count, 0) >= 2)
                AND (coalesce(lab_test_count, 0) < 2)
                AND (coalesce(medication_events, 0) < 2)
            THEN 1 ELSE 0
        END AS low_evidence_high_intensity_flag,
        CASE
            WHEN los_minutes IS NOT NULL AND los_minutes <= 1440
                AND (coalesce(px_count, 0) >= 2 OR coalesce(hcpcs_count, 0) >= 2)
            THEN 1 ELSE 0
        END AS high_intensity_short_los_flag,
        CASE
            WHEN coalesce(lab_test_count, 0) = 0 AND coalesce(medication_events, 0) = 0
            THEN 1 ELSE 0
        END AS sparse_documentation_flag
    FROM counts
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"COPY ({query}) TO '{str(output_path)}' WITH (HEADER, DELIMITER ',')"
    )


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions.csv"
    default_output = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_features.csv"

    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output

    build_features(input_path, output_path)


if __name__ == "__main__":
    main()
