#!/usr/bin/env python3
"""
Process CMS data: extract zips, merge CSVs per folder, combine into master file.
Uses DuckDB for memory-efficient processing of large CSV files.
"""

import os
import zipfile
import duckdb
from pathlib import Path
import glob

def extract_zips_in_folder(folder_path):
    """Extract all zip files in a folder."""
    zip_files = list(Path(folder_path).glob("*.zip"))
    
    if not zip_files:
        return 0
    
    print(f"  Extracting {len(zip_files)} zip files in {folder_path}...")
    extracted_count = 0
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
            extracted_count += 1
        except Exception as e:
            print(f"    Error extracting {zip_path.name}: {e}")
    
    return extracted_count

def merge_csvs_in_folder_with_duckdb(folder_path):
    """
    Merge all CSVs in a folder using DuckDB (memory-efficient).
    Saves as combined.csv in the same folder.
    """
    csv_files = [f for f in Path(folder_path).glob("*.csv") 
                 if f.name != "combined.csv"]
    
    if not csv_files:
        print(f"  No CSV files found in {folder_path}")
        return False
    
    print(f"  Merging {len(csv_files)} CSV files in {folder_path}...")
    output_path = Path(folder_path) / "combined.csv"
    
    # Skip if already exists
    if output_path.exists():
        print(f"  combined.csv already exists in {folder_path}, skipping...")
        return True
    
    try:
        # Create DuckDB connection (in-memory)
        con = duckdb.connect(database=':memory:')
        
        # Read all CSVs into a single query using UNION ALL
        # DuckDB will handle this efficiently without loading everything into memory
        csv_pattern = str(Path(folder_path) / "DE1_*.csv")
        
        # Create a combined view using DuckDB's read_csv_auto with glob pattern
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto('{csv_pattern}', 
                    union_by_name=true,
                    ignore_errors=true,
                    parallel=true)
            ) TO '{output_path}' (HEADER, DELIMITER ',');
        """)
        
        con.close()
        print(f"  ✓ Created combined.csv in {folder_path}")
        return True
        
    except Exception as e:
        print(f"  Error merging CSVs in {folder_path}: {e}")
        return False

def combine_all_to_master(cms_folder, output_format='parquet'):
    """
    Combine all combined.csv files from subfolders into a master file.
    Uses DuckDB for memory-efficient processing.
    
    Args:
        cms_folder: Path to CMS folder
        output_format: 'parquet' or 'csv'
    """
    combined_files = list(Path(cms_folder).glob("sample*/combined.csv"))
    
    if not combined_files:
        print("No combined.csv files found in sample folders!")
        return False
    
    print(f"\nCombining {len(combined_files)} combined.csv files into master file...")
    
    if output_format == 'parquet':
        output_path = Path(cms_folder) / "cms_master.parquet"
    else:
        output_path = Path(cms_folder) / "cms_master.csv"
    
    try:
        # Create DuckDB connection
        con = duckdb.connect(database=':memory:')
        
        # Read all combined CSVs using glob pattern
        combined_pattern = str(Path(cms_folder) / "sample*/combined.csv")
        
        if output_format == 'parquet':
            # Export to parquet with compression
            con.execute(f"""
                COPY (
                    SELECT *, 
                           regexp_extract(filename, 'sample([0-9]+)', 1)::INTEGER as sample_id
                    FROM read_csv_auto('{combined_pattern}', 
                        union_by_name=true,
                        ignore_errors=true,
                        parallel=true,
                        filename=true)
                ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """)
        else:
            # Export to CSV
            con.execute(f"""
                COPY (
                    SELECT *,
                           regexp_extract(filename, 'sample([0-9]+)', 1)::INTEGER as sample_id
                    FROM read_csv_auto('{combined_pattern}', 
                        union_by_name=true,
                        ignore_errors=true,
                        parallel=true,
                        filename=true)
                ) TO '{output_path}' (HEADER, DELIMITER ',');
            """)
        
        # Get row count
        result = con.execute(f"""
            SELECT COUNT(*) as total_rows
            FROM read_csv_auto('{combined_pattern}', 
                union_by_name=true,
                ignore_errors=true,
                parallel=true)
        """).fetchone()
        
        total_rows = result[0] if result else 0
        
        con.close()
        
        print(f"\n✓ Master file created: {output_path}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"Error creating master file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Set up paths
    dataset_folder = Path(__file__).parent.parent / "dataset"
    cms_folder = dataset_folder / "CMS"
    
    print("=" * 60)
    print("CMS Data Processing Pipeline (Memory-Efficient)")
    print("=" * 60)
    
    # Get all sample folders
    sample_folders = sorted([f for f in cms_folder.glob("sample*") if f.is_dir()])
    
    print(f"\nFound {len(sample_folders)} sample folders")
    
    # Step 1: Extract zips
    print("\n" + "=" * 60)
    print("STEP 1: Extracting ZIP files")
    print("=" * 60)
    
    total_extracted = 0
    for folder in sample_folders:
        extracted = extract_zips_in_folder(folder)
        total_extracted += extracted
    
    print(f"\n✓ Extracted {total_extracted} zip files")
    
    # Step 2: Merge CSVs in each folder
    print("\n" + "=" * 60)
    print("STEP 2: Merging CSVs in each subfolder")
    print("=" * 60)
    
    successful_merges = 0
    for folder in sample_folders:
        if merge_csvs_in_folder_with_duckdb(folder):
            successful_merges += 1
    
    print(f"\n✓ Successfully merged CSVs in {successful_merges}/{len(sample_folders)} folders")
    
    # Step 3: Combine all into master file
    print("\n" + "=" * 60)
    print("STEP 3: Creating master combined file")
    print("=" * 60)
    
    # Create both parquet (compressed) and CSV versions
    combine_all_to_master(cms_folder, output_format='parquet')
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
