# Scenario 2: Provider-to-Payer IVR — Claim Status and Denial Details

## SynPUF Case Used

| Field | Value |
|-------|-------|
| Claim ID | 196471176998116 |
| Beneficiary ID | 819303389B2AC8B3 |
| Service Window | 2008-02-16 to 2008-02-23 |
| DRG | 207 — Respiratory system diagnosis with ventilator support 96+ hours |
| Diagnoses | 49322 (chronic obstructive asthma w/ acute exacerbation), 1124 (candidiasis of lung), 42731 (atrial fibrillation) |
| Length of Stay | 7 days |
| Denial Risk Score | 0.999998 (99.9998%) |
| Denial Source | drg_shortstay_upcoding |
| CARC | CO-4 — Procedure code inconsistent with modifier or required modifier missing |
| RARC | N386 — Decision based on Local Coverage Determination (LCD) |

### Caveat

DRG 207 geometric mean LOS is 9–12 days. A 7-day stay is shorter than average but clinically complex (ventilator ≥96 hrs, pulmonary candidiasis, atrial fibrillation). The synthetic labeling rule flagged this as short-stay upcoding, but a real-world clinical review would likely uphold medical necessity. This is a known limitation of the rule-based labeling approach.

---

## Simulated Conversation

### Phase 1 — Automated IVR Menu

**IVR:** Thank you for calling Medicare claims status. For English, press 1.

**Provider:** *(presses 1)*

**IVR:** Please enter your 10-digit National Provider Identifier followed by the pound sign.

**Provider:** *(enters 1234567890#)*

**IVR:** NPI verified. For claim status, press 1. For eligibility, press 2. For prior authorization, press 3.

**Provider:** *(presses 1)*

**IVR:** Please enter the 15-digit claim control number followed by the pound sign.

**Provider:** *(enters 196471176998116#)*

**IVR:** Claim 196471176998116. Date of service February 16, 2008 through February 23, 2008. Claim status: denied. Claim Adjustment Reason Code CO-4. Remittance Advice Remark Code N386. For more information, press 1. To speak with a representative, press 0.

**Provider:** *(presses 0)*

### Phase 2 — Live Representative

**Representative:** Claims department, this is Sarah. How can I help you?

**Provider:** I'm calling about claim 196471176998116 for beneficiary 819303389B2AC8B3. The IVR gave me denial codes CO-4 and N386 but I need more detail on which specific service lines were affected and what documentation would support a reconsideration.

**Representative:** Let me pull that up. I see an inpatient claim, DRG 207, admitted February 16 through February 23. The denial was applied to the full claim under CO-4 with remark N386 referencing a Local Coverage Determination review. The review indicated the length of stay was inconsistent with the DRG assignment.

**Provider:** The patient was on ventilator support for over 96 hours with concurrent pulmonary candidiasis and atrial fibrillation. A 7-day stay is clinically appropriate given those comorbidities.

**Representative:** I understand. To pursue reconsideration, you'll need to submit a written appeal to the address on the remittance advice. Include the admission H&P, daily progress notes, ventilator weaning logs, the discharge summary, and a medical necessity letter from the attending physician addressing the LOS relative to the patient's clinical complexity.

**Provider:** What's the appeal filing deadline?

**Representative:** You have 120 days from the date on the remittance advice. I'm generating a call reference number for this inquiry: RN-48291057.

**Provider:** Got it. RN-48291057. Thank you, Sarah.

**Representative:** You're welcome. Is there anything else?

**Provider:** No, that's all.

**Representative:** Thank you for calling. Goodbye.

---

## Key Workflow Points

1. **IVR delivers structured codes only** — CARC CO-4 and RARC N386. It does not explain clinical rationale or denial logic.
2. **Authentication is NPI-based** — providers authenticate via NPI on payer IVR systems, not TIN.
3. **Detailed information requires a live representative** — IVR cannot explain which service lines were affected or what documentation to submit.
4. **Appeal is a written process** — the representative directs the provider to submit written documentation, not resolve it over the phone.
5. **Reference number format** — realistic payer-generated format (RN-XXXXXXXX), not a concatenation of claim fields.

---

## Source Artifact

Populated from `src/TTS/synpuf_high_rejection_case.csv`