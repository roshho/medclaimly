# Sample 01 Performance Report

## Stage 1 (Deterministic Rules)

- Total claims: 11,151,319
- Hard deny: 176,994 (1.59%)
- Pass to ML: 10,974,325 (98.41%)
- Observed B (guardrail): 11,336
- Observed C (guardrail): 289,300
- Hard deny ∩ observed B: 510
- Hard deny ∩ observed C: 11,340
- Hard deny without B/C (potential contradiction bucket): 165,249

## Stage 2 (XGBoost)
### Primary Metrics


- Rows total: 5,473,188
- Positive rate: 0.3146
- PR-AUC: 0.9754

### Best Threshold Performance (Patient-level split)

- Best F1 threshold: 0.6526
- F1 @ best threshold: 0.9090
- Precision @ best threshold: 0.9500
- Recall @ best threshold: 0.8713

### High-Precision Operating Point

- Recall @ Precision>=0.80: 0.9555

### PR Operating Table

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.2000 | 0.7453 | 0.9665 | 0.8416 |
| 0.3000 | 0.8129 | 0.9526 | 0.8773 |
| 0.4000 | 0.8213 | 0.9498 | 0.8809 |
| 0.5000 | 0.8957 | 0.9107 | 0.9032 |
| 0.6526 | 0.9500 | 0.8713 | 0.9090 |



### Denial Source Breakdown (Test Set Positives)

| Source(s) | Count | Percentage |
|-----------|-------|------------|
| mue_frequency | 332,552 | 54.48% |
| ptp_conflict | 133,093 | 21.80% |
| mue_frequency|ptp_conflict | 80,723 | 13.22% |
| drg_shortstay_upcoding | 64,038 | 10.49% |



## Notes

- Stage 1 B/C values are used for calibration only and excluded from Stage 2 target labels.
- MUE is treated as a soft feature in SynPUF context (no service unit fields).
- Current contradiction bucket size: 165,249 claims.
- Patient-level split prevents same beneficiary appearing in both train and test sets.
- Best threshold optimized for F1 score on test set; may differ from fixed 0.50 threshold.
