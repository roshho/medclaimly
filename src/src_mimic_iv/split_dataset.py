import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split labeled dataset into train/val/test by subject_id with stratification."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to master_admissions_labeled.csv (default: dataset/mimic-iv-3.1/master_admissions_labeled.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: dataset/mimic-iv-3.1/).",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
        help="Proportion for train (default 0.70).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion for val (default 0.15); test = 1 - train - val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_size: float,
    val_size: float,
    seed: int,
) -> None:
    # Read labeled data
    df = pd.read_csv(input_path)
    
    # Group by subject_id to avoid patient leakage
    subjects = df[["subject_id"]].drop_duplicates()
    
    # Split by subject_id with stratification on denial label
    # First, compute denial rate per subject
    subject_denial_rate = (
        df.groupby("subject_id")["synthetic_denial_label"].mean()
        .reset_index()
        .rename(columns={"synthetic_denial_label": "denial_rate"})
    )
    subjects = subjects.merge(subject_denial_rate, on="subject_id")
    
    # Divide into high/low denial groups for stratification
    subjects["stratum"] = (subjects["denial_rate"] > subjects["denial_rate"].median()).astype(int)
    
    # Train/test split (leaving val aside)
    test_size = 1.0 - train_size - val_size
    train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=train_size,
        test_size=test_size + val_size,
        stratify=subjects["stratum"],
        random_state=seed,
    )
    
    # Val/test split from remaining
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=val_size / (val_size + test_size),
        test_size=test_size / (val_size + test_size),
        stratify=temp_subjects["stratum"],
        random_state=seed,
    )
    
    # Extract subject IDs
    train_subject_ids = set(train_subjects["subject_id"])
    val_subject_ids = set(val_subjects["subject_id"])
    test_subject_ids = set(test_subjects["subject_id"])
    
    # Split data by subject_id
    train = df[df["subject_id"].isin(train_subject_ids)]
    val = df[df["subject_id"].isin(val_subject_ids)]
    test = df[df["subject_id"].isin(test_subject_ids)]
    
    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "master_admissions_train.csv"
    val_path = output_dir / "master_admissions_val.csv"
    test_path = output_dir / "master_admissions_test.csv"
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    # Print statistics
    train_pos = train["synthetic_denial_label"].sum()
    val_pos = val["synthetic_denial_label"].sum()
    test_pos = test["synthetic_denial_label"].sum()
    
    train_rate = 100.0 * train_pos / len(train)
    val_rate = 100.0 * val_pos / len(val)
    test_rate = 100.0 * test_pos / len(test)
    
    print("✓ Dataset split by subject_id with stratification")
    print(f"  Train: {len(train):,} admissions | {train_pos:,} denied ({train_rate:.2f}%)")
    print(f"  Val:   {len(val):,} admissions | {val_pos:,} denied ({val_rate:.2f}%)")
    print(f"  Test:  {len(test):,} admissions | {test_pos:,} denied ({test_rate:.2f}%)")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")


def main() -> None:
    args = parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_labeled.csv"
    default_output_dir = base_dir / "dataset" / "mimic-iv-3.1"
    
    input_path = Path(args.input) if args.input else default_input
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    
    split_dataset(input_path, output_dir, args.train_size, args.val_size, args.seed)


if __name__ == "__main__":
    main()
