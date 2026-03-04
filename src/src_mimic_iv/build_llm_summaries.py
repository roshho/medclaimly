import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM-ready text summaries from claim data for reason generation."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to labeled dataset (default: dataset/mimic-iv-3.1/master_admissions_labeled.csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: dataset/mimic-iv-3.1/master_admissions_llm_ready.csv).",
    )
    return parser.parse_args()


def build_llm_summaries(input_path: Path, output_path: Path) -> None:
    """Generate textual claim summaries for LLM reasoning."""
    df = pd.read_csv(input_path)
    
    def build_claim_summary(row: pd.Series) -> str:
        """Generate a narrative summary of the admission/claim."""
        parts = []
        
        # Demographics & admission
        age = row.get("anchor_age")
        gender = row.get("gender")
        race = row.get("race")
        admission_type = row.get("admission_type", "")
        insurance = row.get("insurance", "")
        
        parts.append(f"Patient: {age}-year-old {gender} ({race})")
        parts.append(f"Admission: {admission_type} via {insurance}")
        
        # Primary diagnosis
        dx_primary = row.get("dx_primary_long_title")
        if pd.notna(dx_primary) and dx_primary.strip():
            parts.append(f"Primary diagnosis: {dx_primary}")
        else:
            parts.append("Primary diagnosis: [NOT DOCUMENTED]")
        
        # Secondary diagnoses
        dx_secondary = row.get("dx_secondary_list")
        dx_count = row.get("dx_count", 0)
        if pd.notna(dx_secondary) and dx_secondary.strip():
            diags = dx_secondary.split(";")[:3]  # Top 3
            parts.append(f"Comorbidities ({int(dx_count)} total): {'; '.join(diags)}")
        elif dx_count > 0:
            parts.append(f"Comorbidities: {int(dx_count)} documented, details unavailable")
        else:
            parts.append("Comorbidities: None documented")
        
        # Procedures
        px_primary = row.get("px_primary_long_title")
        px_count = row.get("px_count", 0)
        if pd.notna(px_primary) and px_primary.strip():
            parts.append(f"Primary procedure: {px_primary}")
        if px_count > 1:
            parts.append(f"Additional procedures: {int(px_count) - 1} other procedures")
        elif px_count == 0:
            parts.append("Procedures: None documented")
        
        # HCPCS/CPT codes
        hcpcs_count = row.get("hcpcs_count", 0)
        if hcpcs_count > 0:
            parts.append(f"Billable services (HCPCS): {int(hcpcs_count)} codes")
        
        # LOS
        los_days = row.get("los_days")
        if pd.notna(los_days) and los_days > 0:
            parts.append(f"Length of stay: {los_days:.1f} days")
        
        # Labs & evidence
        lab_count = row.get("lab_test_count", 0)
        lab_abnormal = row.get("lab_abnormal_count", 0)
        micro_count = row.get("micro_test_count", 0)
        
        evidence_parts = []
        if lab_count > 0:
            evidence_parts.append(f"{int(lab_count)} lab tests")
        if lab_abnormal > 0:
            evidence_parts.append(f"{int(lab_abnormal)} abnormal")
        if micro_count > 0:
            evidence_parts.append(f"{int(micro_count)} microbiology")
        
        if evidence_parts:
            parts.append(f"Diagnostic evidence: {', '.join(evidence_parts)}")
        else:
            parts.append("Diagnostic evidence: Minimal or none documented")
        
        # Medications
        med_count = row.get("medication_events", 0)
        if med_count > 0:
            parts.append(f"Medications: {int(med_count)} administered")
        else:
            parts.append("Medications: None documented")
        
        # Risk flags
        risk_flags = []
        if row.get("missing_primary_dx") == 1:
            risk_flags.append("missing primary diagnosis")
        if row.get("low_evidence_high_intensity_flag") == 1:
            risk_flags.append("high-intensity services with sparse documentation")
        if row.get("high_intensity_short_los_flag") == 1:
            risk_flags.append("multiple procedures in very short stay")
        if row.get("sparse_documentation_flag") == 1:
            risk_flags.append("no labs or medications recorded")
        
        if risk_flags:
            parts.append(f"Denial risk signals: {'; '.join(risk_flags)}")
        
        return " | ".join(parts)
    
    # Generate summaries
    print("Generating LLM-ready claim summaries...")
    df["claim_summary"] = df.apply(build_claim_summary, axis=1)
    
    # Select key columns for LLM pipeline
    llm_cols = [
        "subject_id",
        "hadm_id",
        "synthetic_denial_label",
        "denial_probability",
        "claim_summary",
        "dx_primary_icd_code",
        "dx_primary_long_title",
        "px_primary_icd_code",
        "px_primary_long_title",
        "hcpcs_list",
        "dx_secondary_list",
        "px_secondary_list",
        "los_days",
        "lab_test_count",
        "medication_events",
        "missing_primary_dx",
        "low_evidence_high_intensity_flag",
        "high_intensity_short_los_flag",
        "sparse_documentation_flag",
    ]
    
    # Include only columns that exist
    llm_cols = [col for col in llm_cols if col in df.columns]
    df_llm = df[llm_cols].copy()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_llm.to_csv(output_path, index=False)
    
    print(f"✓ LLM-ready dataset: {len(df_llm)} admissions")
    print(f"✓ Columns: {len(llm_cols)}")
    print(f"✓ Output: {output_path}")
    print("\nSample claim summary:")
    print(df_llm["claim_summary"].iloc[0][:200] + "...")


def main() -> None:
    args = parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    default_input = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_labeled.csv"
    default_output = base_dir / "dataset" / "mimic-iv-3.1" / "master_admissions_llm_ready.csv"
    
    input_path = Path(args.input) if args.input else default_input
    output_path = Path(args.output) if args.output else default_output
    
    build_llm_summaries(input_path, output_path)


if __name__ == "__main__":
    main()
