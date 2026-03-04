import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build admission-level master dataset from MIMIC-IV hosp tables."
    )
    parser.add_argument(
        "--use-full",
        action="store_true",
        default=False,
        help="Use full CSVs instead of samples.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: dataset/mimic-iv-3.1/master_admissions_sample.csv).",
    )
    return parser.parse_args()


def read_table(
    hosp_dir: Path,
    name: str,
    use_samples: bool,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    subdir = "sample" if use_samples else "OG"
    suffix = "-sample.csv" if use_samples else ".csv"
    path = hosp_dir / subdir / f"{name}{suffix}"
    return pd.read_csv(path, dtype=dtype)


def agg_unique(series: pd.Series, sep: str = ";") -> str:
    values = []
    seen = set()
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return sep.join(values)


def merge_primary_codes(df: pd.DataFrame, key: str, code_cols: list[str], prefix: str) -> pd.DataFrame:
    sorted_df = df.sort_values([key, "seq_num"])
    primary = sorted_df.groupby(key, as_index=False).first()
    rename_map = {col: f"{prefix}_{col}" for col in code_cols}
    primary = primary[[key] + code_cols].rename(columns=rename_map)
    return primary


def list_by_admission(df: pd.DataFrame, key: str, column: str, out_col: str) -> pd.DataFrame:
    grouped = df.groupby(key)[column].apply(agg_unique).reset_index()
    grouped = grouped.rename(columns={column: out_col})
    return grouped


def count_by_admission(df: pd.DataFrame, key: str, out_col: str) -> pd.DataFrame:
    grouped = df.groupby(key).size().reset_index(name=out_col)
    return grouped


def build_master_duckdb(hosp_dir: Path) -> None:
    """
    Build master admissions dataset using DuckDB for memory-efficient processing of large CSVs.
    Writes directly to output file without loading everything into memory.
    """
    conn = duckdb.connect(database=":memory:")
    
    # Register CSV paths
    def csv_path(name: str) -> str:
        return str(hosp_dir / "OG" / f"{name}.csv")
    
    print("Registering CSV files...")
    
    # Create views for each table by reading CSVs directly
    conn.execute(f"CREATE VIEW admissions AS SELECT * FROM read_csv_auto('{csv_path('admissions')}')")
    conn.execute(f"CREATE VIEW patients AS SELECT * FROM read_csv_auto('{csv_path('patients')}')")
    conn.execute(f"CREATE VIEW diagnoses_icd AS SELECT * FROM read_csv_auto('{csv_path('diagnoses_icd')}')")
    conn.execute(f"CREATE VIEW procedures_icd AS SELECT * FROM read_csv_auto('{csv_path('procedures_icd')}')")
    conn.execute(f"CREATE VIEW hcpcsevents AS SELECT * FROM read_csv_auto('{csv_path('hcpcsevents')}')")
    conn.execute(f"CREATE VIEW drgcodes AS SELECT * FROM read_csv_auto('{csv_path('drgcodes')}')")
    conn.execute(f"CREATE VIEW services AS SELECT * FROM read_csv_auto('{csv_path('services')}')")
    conn.execute(f"CREATE VIEW transfers AS SELECT * FROM read_csv_auto('{csv_path('transfers')}')")
    conn.execute(f"CREATE VIEW prescriptions AS SELECT * FROM read_csv_auto('{csv_path('prescriptions')}')")
    conn.execute(f"CREATE VIEW pharmacy AS SELECT * FROM read_csv_auto('{csv_path('pharmacy')}')")
    conn.execute(f"CREATE VIEW emar AS SELECT * FROM read_csv_auto('{csv_path('emar')}')")
    conn.execute(f"CREATE VIEW labevents AS SELECT * FROM read_csv_auto('{csv_path('labevents')}')")
    conn.execute(f"CREATE VIEW microbiologyevents AS SELECT * FROM read_csv_auto('{csv_path('microbiologyevents')}')")
    conn.execute(f"CREATE VIEW d_icd_diagnoses AS SELECT * FROM read_csv_auto('{csv_path('d_icd_diagnoses')}')")
    conn.execute(f"CREATE VIEW d_icd_procedures AS SELECT * FROM read_csv_auto('{csv_path('d_icd_procedures')}')")
    conn.execute(f"CREATE VIEW d_hcpcs AS SELECT * FROM read_csv_auto('{csv_path('d_hcpcs')}')")
    
    print("Building master dataset with SQL aggregations...")
    
    # Create master query that builds the full result
    query = """
    WITH 
    -- Base: admissions + patients
    base AS (
        SELECT a.*, p.* EXCLUDE (subject_id)
        FROM admissions a
        LEFT JOIN patients p ON a.subject_id = p.subject_id
    ),
    
    -- Diagnoses enriched with descriptions
    dx_enriched AS (
        SELECT 
            d.hadm_id,
            d.seq_num,
            d.icd_code,
            d.icd_version,
            dd.long_title,
            CAST(d.icd_code AS VARCHAR) || '|' || CAST(d.icd_version AS VARCHAR) || '|' || COALESCE(dd.long_title, '') AS dx_key,
            ROW_NUMBER() OVER (PARTITION BY d.hadm_id ORDER BY TRY_CAST(d.seq_num AS INTEGER)) - 1 AS dx_rank
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses dd ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
    ),
    
    -- Primary diagnosis
    dx_primary AS (
        SELECT 
            hadm_id,
            icd_code AS dx_primary_icd_code,
            icd_version AS dx_primary_icd_version,
            long_title AS dx_primary_long_title
        FROM dx_enriched
        WHERE dx_rank = 0
    ),
    
    -- Secondary diagnoses aggregated
    dx_secondary AS (
        SELECT 
            hadm_id,
            STRING_AGG(DISTINCT dx_key, ';' ORDER BY dx_key) AS dx_secondary_list
        FROM dx_enriched
        WHERE dx_rank > 0
        GROUP BY hadm_id
    ),
    
    -- Diagnosis counts
    dx_counts AS (
        SELECT hadm_id, COUNT(*) AS dx_count
        FROM dx_enriched
        GROUP BY hadm_id
    ),
    
    -- Procedures enriched with descriptions
    px_enriched AS (
        SELECT 
            p.hadm_id,
            p.seq_num,
            p.icd_code,
            p.icd_version,
            dp.long_title,
            CAST(p.icd_code AS VARCHAR) || '|' || CAST(p.icd_version AS VARCHAR) || '|' || COALESCE(dp.long_title, '') AS px_key,
            ROW_NUMBER() OVER (PARTITION BY p.hadm_id ORDER BY TRY_CAST(p.seq_num AS INTEGER)) - 1 AS px_rank
        FROM procedures_icd p
        LEFT JOIN d_icd_procedures dp ON p.icd_code = dp.icd_code AND p.icd_version = dp.icd_version
    ),
    
    -- Primary procedure
    px_primary AS (
        SELECT 
            hadm_id,
            icd_code AS px_primary_icd_code,
            icd_version AS px_primary_icd_version,
            long_title AS px_primary_long_title
        FROM px_enriched
        WHERE px_rank = 0
    ),
    
    -- Secondary procedures aggregated
    px_secondary AS (
        SELECT 
            hadm_id,
            STRING_AGG(DISTINCT px_key, ';' ORDER BY px_key) AS px_secondary_list
        FROM px_enriched
        WHERE px_rank > 0
        GROUP BY hadm_id
    ),
    
    -- Procedure counts
    px_counts AS (
        SELECT hadm_id, COUNT(*) AS px_count
        FROM px_enriched
        GROUP BY hadm_id
    ),
    
    -- HCPCS events
    hcpcs_agg AS (
        SELECT 
            h.hadm_id,
            STRING_AGG(DISTINCT 
                CAST(h.hcpcs_cd AS VARCHAR) || '|' || COALESCE(dh.short_description, dh.long_description, ''),
                ';' ORDER BY CAST(h.hcpcs_cd AS VARCHAR) || '|' || COALESCE(dh.short_description, dh.long_description, '')
            ) AS hcpcs_list,
            COUNT(*) AS hcpcs_count
        FROM hcpcsevents h
        LEFT JOIN d_hcpcs dh ON h.hcpcs_cd = dh.code
        GROUP BY h.hadm_id
    ),
    
    -- DRG codes
    drg_agg AS (
        SELECT 
            hadm_id,
            STRING_AGG(DISTINCT 
                CAST(drg_type AS VARCHAR) || ':' || CAST(drg_code AS VARCHAR) || ':' || COALESCE(description, ''),
                ';' ORDER BY CAST(drg_type AS VARCHAR) || ':' || CAST(drg_code AS VARCHAR) || ':' || COALESCE(description, '')
            ) AS drg_list,
            COUNT(*) AS drg_count
        FROM drgcodes
        GROUP BY hadm_id
    ),
    
    -- Services
    services_agg AS (
        SELECT 
            hadm_id,
            STRING_AGG(COALESCE(curr_service, ''), ';' ORDER BY transfertime) AS service_path,
            COUNT(*) AS service_changes
        FROM services
        GROUP BY hadm_id
    ),
    
    -- Transfers
    transfers_agg AS (
        SELECT 
            hadm_id,
            STRING_AGG(COALESCE(careunit, ''), ';' ORDER BY intime) AS transfer_path,
            COUNT(*) AS transfer_count
        FROM transfers
        GROUP BY hadm_id
    ),
    
    -- Medications (union of prescriptions, pharmacy, emar)
    meds_union AS (
        SELECT hadm_id, COALESCE(drug, '') AS med_key FROM prescriptions WHERE hadm_id IS NOT NULL
        UNION ALL
        SELECT hadm_id, COALESCE(medication, '') AS med_key FROM pharmacy WHERE hadm_id IS NOT NULL
        UNION ALL
        SELECT hadm_id, COALESCE(medication, '') AS med_key FROM emar WHERE hadm_id IS NOT NULL
    ),
    meds_agg AS (
        SELECT 
            hadm_id,
            STRING_AGG(DISTINCT med_key, ';' ORDER BY med_key) AS medications,
            COUNT(*) AS medication_events
        FROM meds_union
        GROUP BY hadm_id
    ),
    
    -- Lab events
    lab_agg AS (
        SELECT 
            hadm_id,
            SUM(CASE WHEN flag IS NOT NULL AND TRIM(CAST(flag AS VARCHAR)) != '' THEN 1 ELSE 0 END) AS lab_abnormal_count,
            COUNT(DISTINCT itemid) AS lab_test_count
        FROM labevents
        WHERE hadm_id IS NOT NULL
        GROUP BY hadm_id
    ),
    
    -- Microbiology
    micro_agg AS (
        SELECT 
            hadm_id,
            STRING_AGG(DISTINCT test_name, ';' ORDER BY test_name) AS micro_test_list,
            COUNT(*) AS micro_test_count
        FROM microbiologyevents
        WHERE hadm_id IS NOT NULL
        GROUP BY hadm_id
    )
    
    -- Final join
    SELECT 
        base.*,
        dx_primary.* EXCLUDE (hadm_id),
        dx_secondary.dx_secondary_list,
        dx_counts.dx_count,
        px_primary.* EXCLUDE (hadm_id),
        px_secondary.px_secondary_list,
        px_counts.px_count,
        hcpcs_agg.* EXCLUDE (hadm_id),
        drg_agg.* EXCLUDE (hadm_id),
        services_agg.* EXCLUDE (hadm_id),
        transfers_agg.* EXCLUDE (hadm_id),
        meds_agg.* EXCLUDE (hadm_id),
        lab_agg.* EXCLUDE (hadm_id),
        micro_agg.* EXCLUDE (hadm_id)
    FROM base
    LEFT JOIN dx_primary ON base.hadm_id = dx_primary.hadm_id
    LEFT JOIN dx_secondary ON base.hadm_id = dx_secondary.hadm_id
    LEFT JOIN dx_counts ON base.hadm_id = dx_counts.hadm_id
    LEFT JOIN px_primary ON base.hadm_id = px_primary.hadm_id
    LEFT JOIN px_secondary ON base.hadm_id = px_secondary.hadm_id
    LEFT JOIN px_counts ON base.hadm_id = px_counts.hadm_id
    LEFT JOIN hcpcs_agg ON base.hadm_id = hcpcs_agg.hadm_id
    LEFT JOIN drg_agg ON base.hadm_id = drg_agg.hadm_id
    LEFT JOIN services_agg ON base.hadm_id = services_agg.hadm_id
    LEFT JOIN transfers_agg ON base.hadm_id = transfers_agg.hadm_id
    LEFT JOIN meds_agg ON base.hadm_id = meds_agg.hadm_id
    LEFT JOIN lab_agg ON base.hadm_id = lab_agg.hadm_id
    LEFT JOIN micro_agg ON base.hadm_id = micro_agg.hadm_id
    """
    
    return conn.execute(query).df()


def build_master(hosp_dir: Path, use_samples: bool) -> pd.DataFrame:
    admissions = read_table(hosp_dir, "admissions", use_samples)
    patients = read_table(hosp_dir, "patients", use_samples)
    diagnoses = read_table(
        hosp_dir,
        "diagnoses_icd",
        use_samples,
        dtype={"icd_code": "string", "icd_version": "string"},
    )
    procedures = read_table(
        hosp_dir,
        "procedures_icd",
        use_samples,
        dtype={"icd_code": "string", "icd_version": "string"},
    )
    hcpcs = read_table(
        hosp_dir,
        "hcpcsevents",
        use_samples,
        dtype={"hcpcs_cd": "string"},
    )
    drgcodes = read_table(
        hosp_dir,
        "drgcodes",
        use_samples,
        dtype={"drg_code": "string"},
    )
    services = read_table(hosp_dir, "services", use_samples)
    transfers = read_table(hosp_dir, "transfers", use_samples)
    prescriptions = read_table(hosp_dir, "prescriptions", use_samples)
    pharmacy = read_table(hosp_dir, "pharmacy", use_samples)
    emar = read_table(hosp_dir, "emar", use_samples)
    labevents = read_table(hosp_dir, "labevents", use_samples)
    microbiology = read_table(hosp_dir, "microbiologyevents", use_samples)

    d_icd_dx = read_table(
        hosp_dir,
        "d_icd_diagnoses",
        use_samples,
        dtype={"icd_code": "string", "icd_version": "string"},
    )
    d_icd_px = read_table(
        hosp_dir,
        "d_icd_procedures",
        use_samples,
        dtype={"icd_code": "string", "icd_version": "string"},
    )
    d_hcpcs = read_table(
        hosp_dir,
        "d_hcpcs",
        use_samples,
        dtype={"code": "string"},
    )

    master = admissions.merge(patients, on="subject_id", how="left")

    diagnoses["seq_num"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce")
    diagnoses = diagnoses.merge(
        d_icd_dx,
        on=["icd_code", "icd_version"],
        how="left",
    )
    diagnoses["dx_key"] = (
        diagnoses["icd_code"].astype(str)
        + "|"
        + diagnoses["icd_version"].astype(str)
        + "|"
        + diagnoses["long_title"].fillna("")
    )
    dx_primary = merge_primary_codes(
        diagnoses, "hadm_id", ["icd_code", "icd_version", "long_title"], "dx_primary"
    )
    diagnoses = diagnoses.sort_values(["hadm_id", "seq_num"])
    diagnoses["dx_rank"] = diagnoses.groupby("hadm_id").cumcount()
    dx_secondary = diagnoses[diagnoses["dx_rank"] > 0]
    dx_secondary_list = list_by_admission(dx_secondary, "hadm_id", "dx_key", "dx_secondary_list")
    dx_counts = count_by_admission(diagnoses, "hadm_id", "dx_count")

    procedures["seq_num"] = pd.to_numeric(procedures["seq_num"], errors="coerce")
    procedures = procedures.merge(
        d_icd_px,
        on=["icd_code", "icd_version"],
        how="left",
    )
    procedures["px_key"] = (
        procedures["icd_code"].astype(str)
        + "|"
        + procedures["icd_version"].astype(str)
        + "|"
        + procedures["long_title"].fillna("")
    )
    px_primary = merge_primary_codes(
        procedures, "hadm_id", ["icd_code", "icd_version", "long_title"], "px_primary"
    )
    procedures = procedures.sort_values(["hadm_id", "seq_num"])
    procedures["px_rank"] = procedures.groupby("hadm_id").cumcount()
    px_secondary = procedures[procedures["px_rank"] > 0]
    px_secondary_list = list_by_admission(px_secondary, "hadm_id", "px_key", "px_secondary_list")
    px_counts = count_by_admission(procedures, "hadm_id", "px_count")

    hcpcs = hcpcs.merge(
        d_hcpcs,
        left_on="hcpcs_cd",
        right_on="code",
        how="left",
    )
    # Handle column names flexibly based on what's available
    if "short_description" in hcpcs.columns:
        hcpcs["hcpcs_desc"] = hcpcs["short_description"].fillna(hcpcs.get("long_description", ""))
    elif "long_description" in hcpcs.columns:
        hcpcs["hcpcs_desc"] = hcpcs["long_description"]
    else:
        hcpcs["hcpcs_desc"] = ""
    
    hcpcs["hcpcs_key"] = (
        hcpcs["hcpcs_cd"].astype(str)
        + "|"
        + hcpcs["hcpcs_desc"].fillna("").astype(str)
    )
    hcpcs_list = list_by_admission(hcpcs, "hadm_id", "hcpcs_key", "hcpcs_list")
    hcpcs_counts = count_by_admission(hcpcs, "hadm_id", "hcpcs_count")

    drgcodes["drg_key"] = (
        drgcodes["drg_type"].astype(str)
        + ":"
        + drgcodes["drg_code"].astype(str)
        + ":"
        + drgcodes["description"].fillna("")
    )
    drg_list = list_by_admission(drgcodes, "hadm_id", "drg_key", "drg_list")
    drg_counts = count_by_admission(drgcodes, "hadm_id", "drg_count")

    services = services.sort_values(["hadm_id", "transfertime"])
    services["service_key"] = services["curr_service"].fillna("")
    service_path = list_by_admission(services, "hadm_id", "service_key", "service_path")
    service_counts = count_by_admission(services, "hadm_id", "service_changes")

    transfers = transfers.sort_values(["hadm_id", "intime"])
    transfers["transfer_key"] = transfers["careunit"].fillna("")
    transfer_path = list_by_admission(transfers, "hadm_id", "transfer_key", "transfer_path")
    transfer_counts = count_by_admission(transfers, "hadm_id", "transfer_count")

    prescriptions["med_key"] = prescriptions["drug"].fillna("")
    pharmacy["med_key"] = pharmacy["medication"].fillna("")
    emar["med_key"] = emar["medication"].fillna("")
    meds = pd.concat(
        [
            prescriptions[["hadm_id", "med_key"]],
            pharmacy[["hadm_id", "med_key"]],
            emar[["hadm_id", "med_key"]],
        ],
        ignore_index=True,
    )
    meds = meds[meds["hadm_id"].notna()]
    meds_list = list_by_admission(meds, "hadm_id", "med_key", "medications")
    meds_counts = count_by_admission(meds, "hadm_id", "medication_events")

    labevents = labevents[labevents["hadm_id"].notna()]
    labevents["abnormal_flag"] = labevents["flag"].fillna("").astype(str).str.strip()
    labevents["is_abnormal"] = labevents["abnormal_flag"].ne("")
    lab_abnormal = (
        labevents.groupby("hadm_id")["is_abnormal"].sum().reset_index(name="lab_abnormal_count")
    )
    lab_tests = labevents.groupby("hadm_id")["itemid"].nunique().reset_index(name="lab_test_count")

    microbiology = microbiology[microbiology["hadm_id"].notna()]
    micro_tests = microbiology.groupby("hadm_id")["test_name"].apply(agg_unique).reset_index()
    micro_tests = micro_tests.rename(columns={"test_name": "micro_test_list"})
    micro_counts = count_by_admission(microbiology, "hadm_id", "micro_test_count")

    aggregates = [
        dx_primary,
        dx_secondary_list,
        dx_counts,
        px_primary,
        px_secondary_list,
        px_counts,
        hcpcs_list,
        hcpcs_counts,
        drg_list,
        drg_counts,
        service_path,
        service_counts,
        transfer_path,
        transfer_counts,
        meds_list,
        meds_counts,
        lab_abnormal,
        lab_tests,
        micro_tests,
        micro_counts,
    ]

    for agg in aggregates:
        master = master.merge(agg, on="hadm_id", how="left")

    return master


def main() -> None:
    args = parse_args()
    use_samples = not args.use_full

    base_dir = Path(__file__).resolve().parents[1]
    hosp_dir = base_dir / "dataset" / "mimic-iv-3.1" / "hosp"

    output_path = args.output
    if output_path is None:
        output_name = "master_admissions_sample.csv" if use_samples else "master_admissions.csv"
        output_path = str(base_dir / "dataset" / "mimic-iv-3.1" / "output" / output_name)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if use_samples:
        # Use pandas for sample CSVs (fast enough)
        print("Processing sample CSVs with pandas...")
        master = build_master(hosp_dir, use_samples)
        master.to_csv(output_file, index=False)
    else:
        # Use DuckDB for full CSVs (memory-efficient)
        print("Processing full CSVs with DuckDB (memory-efficient)...")
        master = build_master_duckdb(hosp_dir)
        master.to_csv(output_file, index=False)
    
    print(f"Output written to: {output_file}")
    print(f"Shape: {master.shape if use_samples else len(master)} rows")


if __name__ == "__main__":
    main()
