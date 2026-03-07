#!/usr/bin/env python3
"""
Reference Table Ingestion Module
Parses and normalizes all Medicare reference rule files for Stage 1 and Stage 2:
- HCPCS: Annual Release fixed-width format (coverage status, action codes)
- MUE: Medically Unlikely Edits CSV (unit limits per code)
- PTP: Correct Coding Initiative code pair edits TSV (unbundling rules)
- DRG: Diagnosis Related Group weights and LOS thresholds
- MCD: Local/National Coverage Determination CSVs (diagnosis-HCPCS linkages)

All output normalized to parquet for fast DuckDB joins in Stage 1/2.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import re
from datetime import datetime

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
RULES_DIR = PROJECT_ROOT / "rules"
OUTPUT_DIR = PROJECT_ROOT / "output" / "reference_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_hcpcs_fixed_width(hcpcs_file: Path) -> pd.DataFrame:
    """
    Parse HCPCS Annual Release fixed-width format.
    
    Key columns (1-indexed in spec):
    - Positions 1-5: HCPCS code
    - Positions 7-126: Long description
    - Positions 127-180: Short description
    - Positions 181-186: Pricing indicator
    - Position 293: Action code (A=add, C=change, D=delete, T=terminate)
    - Positions 269-276: Add date (YYYYMMDD)
    - Positions 277-284: Act Eff date (YYYYMMDD)
    - Positions 285-292: Term date (YYYYMMDD)
    
    Coverage status inferred from action code and termination date.
    """
    print(f"[HCPCS] Parsing {hcpcs_file.name}...")
    
    # Fixed-width column specs (0-indexed for Python)
    colspecs = [
        (0, 5),      # HCPCS code
        (6, 126),    # Long desc
        (126, 180),  # Short desc
        (180, 186),  # Pricing indicator
        (292, 293),  # Action code
        (268, 276),  # Add date
        (276, 284),  # Act Eff date
        (284, 292),  # Term date
    ]
    
    names = [
        'hcpcs_code',
        'long_desc',
        'short_desc',
        'pricing_indicator',
        'action_code',
        'add_date',
        'effective_date',
        'termination_date'
    ]
    
    df = pd.read_fwf(
        hcpcs_file,
        colspecs=colspecs,
        names=names,
        dtype=str
    )
    
    # Clean whitespace
    df['hcpcs_code'] = df['hcpcs_code'].str.strip()
    df['action_code'] = df['action_code'].str.strip()
    
    # Parse dates
    for date_col in ['add_date', 'effective_date', 'termination_date']:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
    
    # Infer coverage status
    # Noncovered if: action_code in ['D', 'T'] (deleted/terminated) or termination_date <= 2010
    df['is_noncovered'] = (
        df['action_code'].isin(['D', 'T']) |
        (df['termination_date'] <= pd.Timestamp('2010-12-31'))
    )
    
    # Keep only codes with effective date <= 2010 (relevant for SynPUF 2008-2010)
    df = df[
        (df['effective_date'].isna()) |
        (df['effective_date'] <= pd.Timestamp('2010-12-31'))
    ].copy()
    
    print(f"[HCPCS] Parsed {len(df):,} codes ({df['is_noncovered'].sum():,} noncovered)")
    return df[['hcpcs_code', 'long_desc', 'short_desc', 'action_code', 
               'effective_date', 'termination_date', 'is_noncovered']]


def parse_mue_csv(mue_dir: Path) -> pd.DataFrame:
    """
    Parse MUE (Medically Unlikely Edits) CSV files.
    
    Three variants: Practitioner, Outpatient, DME
    Columns: "HCPCS/CPT Code", "MUE Values", "MUE Adjudication Indicator" (1/2/3)
    
    Adjudication indicators:
    - 1: Claim line edit (per line item)
    - 2: Date of service edit (per DOS)
    - 3: DME/clinical edit
    
    Note: SynPUF lacks LINE_SRVC_CNT, so MUE used as SOFT feature in Stage 2.
    """
    print(f"[MUE] Parsing CSV files from {mue_dir}...")
    
    dfs = []
    for csv_file in mue_dir.glob("*.csv"):
        if csv_file.name.startswith('.'):
            continue
        
        variant = csv_file.stem  # Extract variant name from filename
        print(f"[MUE]   - {csv_file.name} ({variant})")
        
        # MUE files have copyright header (first 9 rows) and header spans 2 lines
        # Skip to data and manually set column names
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(
                    csv_file, 
                    dtype=str, 
                    encoding=encoding, 
                    skiprows=9,  # Skip copyright + split header
                    header=None,  # No header row
                    names=['hcpcs_code', 'mue_value', 'adjudication_indicator', 'rationale']
                )
                break
            except UnicodeDecodeError:
                if encoding == 'cp1252':
                    raise
        
        # Keep only first 3 columns
        df = df[['hcpcs_code', 'mue_value', 'adjudication_indicator']].copy()
        df['hcpcs_code'] = df['hcpcs_code'].str.strip()
        df['variant'] = variant
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['hcpcs_code'] = combined['hcpcs_code'].str.strip()
    combined['mue_value'] = pd.to_numeric(combined['mue_value'], errors='coerce')
    
    print(f"[MUE] Parsed {len(combined):,} MUE rules across {len(dfs)} variants")
    return combined


def parse_ptp_tsv(ptp_dir: Path) -> pd.DataFrame:
    """
    Parse PTP (Correct Coding Initiative) code pair edits.
    
    Files are .txt (tab-separated) in subdirectories.
    Columns: Column1, Column2, Effective Date, Deletion Date, Modifier (0/1/9), Rationale
    
    Modifier values:
    - 0: Not allowed (unbundling, always deny pair)
    - 1: Allowed with modifier (appropriate modifier can override)
    - 9: Not applicable
    
    Returns pairs active during SynPUF period (2008-2010).
    """
    print(f"[PTP] Parsing TXT files from {ptp_dir}...")
    
    dfs = []
    # Recursively find all .txt files in subdirectories
    for txt_file in ptp_dir.rglob("*.txt"):
        if txt_file.name.startswith('.'):
            continue
        
        print(f"[PTP]   - {txt_file.relative_to(ptp_dir)}")
        
        # Skip copyright header (first 6 rows)
        df = pd.read_csv(
            txt_file, 
            sep='\t', 
            dtype=str, 
            skiprows=6,
            header=None,
            names=['code1', 'code2', 'unknown', 'effective_date', 'deletion_date', 'modifier', 'rationale']
        )
        
        # Drop unknown column
        df = df.drop(columns=['unknown'])
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Parse dates
    combined['effective_date'] = pd.to_datetime(combined['effective_date'], errors='coerce')
    combined['deletion_date'] = pd.to_datetime(combined['deletion_date'], errors='coerce')
    
    # Filter to active pairs during SynPUF period (2008-2010)
    synpuf_start = pd.Timestamp('2008-01-01')
    synpuf_end = pd.Timestamp('2010-12-31')
    
    combined = combined[
        (combined['effective_date'] <= synpuf_end) &
        ((combined['deletion_date'].isna()) | (combined['deletion_date'] >= synpuf_start))
    ].copy()
    
    # Clean code columns
    combined['code1'] = combined['code1'].str.strip()
    combined['code2'] = combined['code2'].str.strip()
    combined['modifier'] = combined['modifier'].str.strip()
    
    print(f"[PTP] Parsed {len(combined):,} active code pairs for 2008-2010")
    return combined[['code1', 'code2', 'modifier', 'effective_date', 'deletion_date', 'rationale']]


def parse_drg_files(drg_dir: Path) -> pd.DataFrame:
    """
    Parse DRG weight and LOS files.
    
    Two sources:
    1. Historical_Weight_File_FR_2010.TXT (fixed-width):
       - Positions 1-3: DRG code
       - Positions 4-9: Weight (5 decimals, e.g., 1.23456)
       - Positions 10-12: ALOS (arithmetic mean)
       - Positions 13-15: Trim point
    
    2. FY 2010 FR Table 5 TSV:
       - Columns: "MS-DRG", "Weights", "Geometric mean LOS", "Arithmetic mean LOS"
    
    Merge both, prefer Table 5 TSV for geometric mean LOS.
    """
    print(f"[DRG] Parsing files from {drg_dir}...")
    
    # Parse fixed-width Historical file
    historical_file = drg_dir / "Historical_Weight_File_FR_2010.TXT"
    if historical_file.exists():
        print(f"[DRG]   - {historical_file.name} (fixed-width)")
        colspecs = [
            (0, 3),    # DRG code
            (3, 9),    # Weight
            (9, 12),   # ALOS
            (12, 15),  # Trim point
        ]
        df_hist = pd.read_fwf(
            historical_file,
            colspecs=colspecs,
            names=['drg_code', 'weight_hist', 'alos_hist', 'trim_point'],
            dtype=str
        )
        df_hist['drg_code'] = df_hist['drg_code'].str.strip()
        df_hist['weight_hist'] = pd.to_numeric(df_hist['weight_hist'], errors='coerce')
        df_hist['alos_hist'] = pd.to_numeric(df_hist['alos_hist'], errors='coerce')
    else:
        df_hist = pd.DataFrame(columns=['drg_code', 'weight_hist', 'alos_hist', 'trim_point'])
    
    # Parse TSV Table 5
    table5_files = list(drg_dir.glob("*Table*5*.tsv")) + list(drg_dir.glob("*table*5*.tsv"))
    if table5_files:
        table5_file = table5_files[0]
        print(f"[DRG]   - {table5_file.name} (TSV)")
        df_table5 = pd.read_csv(table5_file, sep='\t', dtype=str)
        
        # Normalize column names
        df_table5.columns = df_table5.columns.str.strip()
        col_map = {}
        for col in df_table5.columns:
            if 'ms-drg' in col.lower() or 'drg' in col.lower():
                col_map[col] = 'drg_code'
            elif 'weight' in col.lower():
                col_map[col] = 'weight'
            elif 'geometric' in col.lower():
                col_map[col] = 'geometric_mean_los'
            elif 'arithmetic' in col.lower():
                col_map[col] = 'arithmetic_mean_los'
        
        df_table5 = df_table5.rename(columns=col_map)
        df_table5['drg_code'] = df_table5['drg_code'].str.strip()
        df_table5['weight'] = pd.to_numeric(df_table5['weight'], errors='coerce')
        df_table5['geometric_mean_los'] = pd.to_numeric(df_table5['geometric_mean_los'], errors='coerce')
    else:
        df_table5 = pd.DataFrame(columns=['drg_code', 'weight', 'geometric_mean_los', 'arithmetic_mean_los'])
    
    # Merge both sources
    if not df_hist.empty and not df_table5.empty:
        df_drg = pd.merge(df_hist, df_table5, on='drg_code', how='outer')
        df_drg['weight'] = df_drg['weight'].fillna(df_drg['weight_hist'])
        df_drg = df_drg.drop(columns=['weight_hist'])
    elif not df_hist.empty:
        df_drg = df_hist.rename(columns={'weight_hist': 'weight'})
    elif not df_table5.empty:
        df_drg = df_table5
    else:
        raise FileNotFoundError(f"No DRG files found in {drg_dir}")
    
    print(f"[DRG] Parsed {len(df_drg):,} DRG codes with weights and LOS")
    
    # Return available columns only
    base_cols = ['drg_code', 'weight']
    optional_cols = ['geometric_mean_los', 'arithmetic_mean_los', 'alos_hist', 'trim_point']
    return_cols = base_cols + [c for c in optional_cols if c in df_drg.columns]
    return df_drg[return_cols]


def parse_mcd_articles(mcd_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse MCD (Local/National Coverage Determination) article CSVs.
    
    Three key files:
    1. article_x_hcpc_code.csv: Links articles to HCPCS codes
    2. article_x_icd10_covered.csv: Diagnosis coverage conditions (ICD10)
    3. article_x_icd10_noncovered.csv: Diagnosis noncoverage conditions
    
    Returns three normalized dataframes for Stage 1 LCD condition checks.
    """
    print(f"[MCD] Parsing article CSVs from {mcd_dir}...")
    
    def pick_col(columns, candidates):
        lower_map = {c.lower().strip(): c for c in columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    # Parse HCPCS-article linkage
    hcpc_file = mcd_dir / "article_x_hcpc_code.csv"
    if hcpc_file.exists():
        print(f"[MCD]   - {hcpc_file.name}")
        df_hcpc = pd.read_csv(hcpc_file, dtype=str)
        article_col = pick_col(df_hcpc.columns, ['article_id'])
        hcpcs_col = pick_col(df_hcpc.columns, ['hcpcs_code', 'hcpc_code', 'hcpc_code_id'])
        if article_col and hcpcs_col:
            df_hcpc = df_hcpc[[article_col, hcpcs_col]].copy()
            df_hcpc.columns = ['article_id', 'hcpcs_code']
        else:
            df_hcpc = pd.DataFrame(columns=['article_id', 'hcpcs_code'])
    else:
        df_hcpc = pd.DataFrame(columns=['article_id', 'hcpcs_code'])
    
    # Parse ICD10 covered conditions
    covered_file = mcd_dir / "article_x_icd10_covered.csv"
    if covered_file.exists():
        print(f"[MCD]   - {covered_file.name}")
        df_covered = pd.read_csv(covered_file, dtype=str)
        article_col = pick_col(df_covered.columns, ['article_id'])
        icd_col = pick_col(df_covered.columns, ['icd10_code', 'icd10_code_id'])
        if article_col and icd_col:
            df_covered = df_covered[[article_col, icd_col]].copy()
            df_covered.columns = ['article_id', 'icd10_code']
        else:
            df_covered = pd.DataFrame(columns=['article_id', 'icd10_code'])
    else:
        df_covered = pd.DataFrame(columns=['article_id', 'icd10_code'])
    
    # Parse ICD10 noncovered conditions
    noncovered_file = mcd_dir / "article_x_icd10_noncovered.csv"
    if noncovered_file.exists():
        print(f"[MCD]   - {noncovered_file.name}")
        df_noncovered = pd.read_csv(noncovered_file, dtype=str)
        article_col = pick_col(df_noncovered.columns, ['article_id'])
        icd_col = pick_col(df_noncovered.columns, ['icd10_code', 'icd10_code_id'])
        if article_col and icd_col:
            df_noncovered = df_noncovered[[article_col, icd_col]].copy()
            df_noncovered.columns = ['article_id', 'icd10_code']
        else:
            df_noncovered = pd.DataFrame(columns=['article_id', 'icd10_code'])
    else:
        df_noncovered = pd.DataFrame(columns=['article_id', 'icd10_code'])
    
    print(f"[MCD] Parsed {len(df_hcpc):,} HCPCS-article links, "
          f"{len(df_covered):,} covered conditions, {len(df_noncovered):,} noncovered conditions")
    
    return df_hcpc, df_covered, df_noncovered


def load_all_reference_tables() -> Dict[str, Path]:
    """
    Master function: Parse all reference tables and save as normalized parquet files.
    
    Returns dict mapping table name to output parquet path.
    """
    print(f"\n{'='*80}")
    print("REFERENCE TABLE INGESTION MODULE")
    print(f"{'='*80}\n")
    
    output_paths = {}
    
    # 1. HCPCS
    hcpcs_dir = RULES_DIR / "HCPCS-alpha-num"
    hcpcs_files = list(hcpcs_dir.glob("HCPC*.txt"))
    if hcpcs_files:
        df_hcpcs = parse_hcpcs_fixed_width(hcpcs_files[0])  # Use first file (typically most recent)
        output_path = OUTPUT_DIR / "hcpcs_reference.parquet"
        df_hcpcs.to_parquet(output_path, index=False, compression='zstd')
        output_paths['hcpcs'] = output_path
        print(f"[HCPCS] ✓ Saved to {output_path.relative_to(PROJECT_ROOT)}\n")
    else:
        print("[HCPCS] ✗ No HCPCS files found\n")
    
    # 2. MUE
    mue_dir = RULES_DIR / "MUE"
    if mue_dir.exists():
        df_mue = parse_mue_csv(mue_dir)
        output_path = OUTPUT_DIR / "mue_reference.parquet"
        df_mue.to_parquet(output_path, index=False, compression='zstd')
        output_paths['mue'] = output_path
        print(f"[MUE] ✓ Saved to {output_path.relative_to(PROJECT_ROOT)}\n")
    else:
        print("[MUE] ✗ MUE directory not found\n")
    
    # 3. PTP
    ptp_dir = RULES_DIR / "PTP"
    if ptp_dir.exists():
        df_ptp = parse_ptp_tsv(ptp_dir)
        output_path = OUTPUT_DIR / "ptp_reference.parquet"
        df_ptp.to_parquet(output_path, index=False, compression='zstd')
        output_paths['ptp'] = output_path
        print(f"[PTP] ✓ Saved to {output_path.relative_to(PROJECT_ROOT)}\n")
    else:
        print("[PTP] ✗ PTP directory not found\n")
    
    # 4. DRG
    drg_dir = RULES_DIR / "DRG"
    if drg_dir.exists():
        df_drg = parse_drg_files(drg_dir)
        output_path = OUTPUT_DIR / "drg_reference.parquet"
        df_drg.to_parquet(output_path, index=False, compression='zstd')
        output_paths['drg'] = output_path
        print(f"[DRG] ✓ Saved to {output_path.relative_to(PROJECT_ROOT)}\n")
    else:
        print("[DRG] ✗ DRG directory not found\n")
    
    # 5. MCD
    mcd_dir = RULES_DIR / "MCD" / "past-and-present-articles" / "all_article_csv"
    if mcd_dir.exists():
        df_hcpc, df_covered, df_noncovered = parse_mcd_articles(mcd_dir)
        
        output_path_hcpc = OUTPUT_DIR / "mcd_hcpcs_articles.parquet"
        df_hcpc.to_parquet(output_path_hcpc, index=False, compression='zstd')
        output_paths['mcd_hcpcs'] = output_path_hcpc
        
        output_path_covered = OUTPUT_DIR / "mcd_icd10_covered.parquet"
        df_covered.to_parquet(output_path_covered, index=False, compression='zstd')
        output_paths['mcd_covered'] = output_path_covered
        
        output_path_noncovered = OUTPUT_DIR / "mcd_icd10_noncovered.parquet"
        df_noncovered.to_parquet(output_path_noncovered, index=False, compression='zstd')
        output_paths['mcd_noncovered'] = output_path_noncovered
        
        print(f"[MCD] ✓ Saved 3 tables to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/mcd_*.parquet\n")
    else:
        print("[MCD] ✗ MCD directory not found\n")
    
    print(f"{'='*80}")
    print(f"✓ Reference table ingestion complete: {len(output_paths)} tables saved")
    print(f"{'='*80}\n")
    
    return output_paths


if __name__ == "__main__":
    output_paths = load_all_reference_tables()
    
    # Print summary
    print("\nOutput file summary:")
    for table_name, path in output_paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {table_name:20s} → {path.name:30s} ({size_mb:>6.2f} MB)")
