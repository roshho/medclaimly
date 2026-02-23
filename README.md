# Virtual venv activation
``source venv/bin/activate``

# Setup
- Insert MIMIC-IV-3.1 uncompressed CSVs under ``dataset/mimic-iv-3.1/hosp/OG``
    - All files will refer to this (pending fix)

# Step-by-step Notes
1) Concatted important information and ommited unecessary such as emar details (medication issuing in detail and dosage) or POE_detail or provider (when it's not available detail in MIMIC-IV)
    - merge admissions + diagnoses_icd + procedures_icd + hcpcsevents into a single training dataset based on admission number (not claim or ID due to potential duplicates or multiple claims or bundled claims) using ``/src/etl_master_admissions.py``
        - ``python src/etl_master_admissions.py``
        - OR ``python src/etl_master_admissions.py --use-full`` to not just use sample but the OG data sets 
        - Input/output: ``dataset/mimic-iv-3.1/hosp/OG/[CSVs]`` -> ``dataset/mimic-iv-3.1/master_admissions.csv``
    - Notes:
        - CSVs are too big, thus using duckDB to chunk

2) Engineer denial features - e.g., "diagnosis-procedure mismatch," "unusually high procedure count," "weak primary diagnosis". 
    - Identify instances in existing dataset that are high likelihood for denial for clean dataset. THis is seperate from synthetic, as those are additional cases ot be injected should there not be enough of a ratio of accepted : denied cases (i.e. 10-15%)
    - Aggregate file should be much smaller as details are typically omitted and unnecessary
    - ``python src/feature_engineering.py``
    - ``python src/feature_engineering.py --input /path/to/master_admissions.csv --output /path/to/master_admissions_features.csv``
    - Input/output: ``dataset/mimic-iv-3.1/master_admissions.csv`` -> ``dataset/mimic-iv-3.1/master_admissions_features.csv`` 

    - What is done here?
        - Calculates length of stay (LOS) metrics:
            - Converts admittime and dischtime to minutes, then to hours/days.
            - Does the same for ED times (edregtime, edouttime).
        - Counts list fields (parses semicolon‑delimited codes):
            - Splits dx_secondary_list, hcpcs_list, medications, etc.
                - px: procedure; dx: diagnosis
            - Counts how many items in each list.
        - Flags missing critical data:
            - missing_primary_dx: 1 if no primary diagnosis code.
            - missing_primary_px: 1 if no primary procedure code.
        - Derives intensity and documentation ratios:
            - procedures_per_day: procedures ÷ LOS days.
            - labs_per_day: lab tests ÷ LOS days.
            - meds_per_day: medication events ÷ LOS days.
        - Creates denial‑risk flags (heuristics):
            - low_evidence_high_intensity_flag: ≥2 procedures/HCPCS but <2 labs AND <2 meds → sparse documentation for complex case.
            - high_intensity_short_los_flag: ≥2 procedures/HCPCS in ≤1 day → medically unusual.- sparse_documentation_flag: 0 labs AND 0 medications → missing evidence.

3) Create synthetic denied cases - e.g., remove supporting diagnoses, add contradictory procedures, etc.
    - ``python src/synthetic_labeling.py``
    - Input/output: ``dataset/mimic-iv-3.1/master_admissions.csv`` -> ``dataset/mimic-iv-3.1/master_admissions_labeled.csv`` 
    - Due to lack of denial claim labels, we are to synthetically alter existing data to fit as denied cases. Cases that are current ambigious on possible denials will be alterted to be confirmed denied cases as a ground truth, but will definitiyl cause distribution skew and poor generalization. 
    - New approach: instead of mutating cases, will use risk-score based weak labeling, where I'll assign a denial risk probability, where cases < 15% (national, annual average for case rejetcions: 15-16%, Medicare denial rate: 17%) and sort into binary "denied or not" 
        - Calculation will be done via:
            1) Domain-specific rules:
                - Insurance type (Medicaid stricter)
                - Elective admission with weak diagnosis
                - Age extremes with few comorbidities
                - High procedure volume with no labs
                - Multiple HCPCS with short LOS
            2) Priority-weighted scoring:
                - Direct documentation: 20%
                - Evidence (abnormal labs/micro emphasized): 20%
                - Code volume & structure (imbalance detection): 15%
                - Intensity & LOS alignment: 15%
                - Care trajectory: 10%
                - Domain rules: 20%
            3) Normalized denial_probability (0–1 scale) combines all scores.
            4) Label bottom 15% by percentile rank.


4) Splitting data
    - Train/val/test: 70/15/15
    - ``python3 src/split_dataset.py``
    - Input/output: ``master_admissions_labeled.csv`` -> ``master_admissions_train.csv`` + ``master_admissions_val.csv`` + ``master_admissions_test.csv``

5) MedGemmaVLM for multi-modal data -> tabular model -> small reasoning model (e.g. Phi 3.5, Llmama 3, Mistral 7B, Meditron) for appeal letter drafting
    - ``python3 src/train_baseline_xgboost.py``
    - input/output: ``xgboost_denial_classifier.model``
    - metrics output: ``output/metrics.csv``

    - ``python3 src/build_llm_summaries.py``: generates claim summaries
    - input/output: ``dataset/mimic-iv-3.1/master_admissions_llm_ready.csv``

