# Scenario 3: Provider-to-Payer — Appeal Submission and Confirmation

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
| CARC | CO-4 |
| RARC | N386 |

### Caveat

DRG 207 geometric mean LOS is 9–12 days. A 7-day stay is shorter than average but clinically complex (ventilator ≥96 hrs, pulmonary candidiasis, atrial fibrillation). The synthetic labeling rule flagged this as short-stay upcoding, but a real-world clinical review would likely uphold medical necessity. This is a known limitation of the rule-based labeling approach.

---

## Context

This scenario follows Scenario 2. The provider has already obtained the denial codes (CO-4, N386) and call reference number (RN-48291057) from the payer. The provider is now calling back to confirm receipt of a written appeal that was submitted by fax.

**Appeals in Medicare are not submitted through IVR systems.** The provider submits a written appeal package (fax or mail), then calls the payer to confirm receipt and obtain an appeal tracking number.

---

## Simulated Conversation

### Phase 1 — Automated IVR Menu

**IVR:** Thank you for calling Medicare appeals and grievances. For English, press 1.

**Provider:** *(presses 1)*

**IVR:** Please enter your 10-digit National Provider Identifier followed by the pound sign.

**Provider:** *(enters 1234567890#)*

**IVR:** NPI verified. To check the status of an existing appeal, press 1. For general appeals information, press 2. To speak with a representative, press 0.

**Provider:** *(presses 0)*

### Phase 2 — Live Representative

**Representative:** Appeals department, this is Mark. How can I help you?

**Provider:** I'm calling to confirm receipt of a written appeal we faxed yesterday for claim 196471176998116, beneficiary 819303389B2AC8B3. The original denial was under CO-4 and N386. Our prior call reference was RN-48291057.

**Representative:** Let me check. I see claim 196471176998116, DRG 207, denied under CO-4. And yes, I do show a fax received on file dated today. Let me confirm what was included.

**Provider:** We sent the admission history and physical, daily progress notes for the full 7-day stay, ventilator weaning protocol and logs, the discharge summary, and a medical necessity letter from the attending pulmonologist.

**Representative:** I see five documents on file. That matches. The attending's letter — does it specifically address the length of stay relative to clinical complexity?

**Provider:** Yes. It explains that the patient required ventilator support exceeding 96 hours due to acute exacerbation of chronic obstructive asthma complicated by pulmonary candidiasis. The concurrent atrial fibrillation required cardiac monitoring that extended the stay. The 7-day LOS was clinically necessary despite being shorter than the DRG geometric mean.

**Representative:** Good. That addresses the LOS concern directly. I'm assigning this appeal tracking number RN-73058412. The clinical review team will evaluate within 30 calendar days. You'll receive a written determination by mail. If additional records are needed, we'll fax a request to the number on file.

**Provider:** RN-73058412, 30-day review window. Is there a way to check status before the determination letter?

**Representative:** You can call this same number and use option 1 to check appeal status with your NPI and claim number. The system will show whether it's in review, if additional information was requested, or if a determination has been made.

**Provider:** Understood. Thank you, Mark.

**Representative:** You're welcome. Is there anything else?

**Provider:** No, that covers it.

**Representative:** Thank you for calling. Goodbye.

---

## Key Workflow Points

1. **Appeals are submitted in writing** — fax or mail, never through IVR voice prompts. This call is to *confirm receipt*, not to *submit* the appeal.
2. **Authentication is NPI-based** — same as Scenario 2.
3. **Clinical justification references patient evidence** — ventilator duration, specific diagnoses, comorbidity impact on LOS. Never references model predictions or denial probability scores.
4. **Appeal tracking number is payer-generated** — realistic format (RN-XXXXXXXX), separate from the inquiry reference number in Scenario 2.
5. **Review timeline is explicit** — 30 calendar days for Medicare redetermination, with written determination by mail.
6. **Supporting documents are clinical** — admission H&P, progress notes, ventilator logs, discharge summary, attending physician medical necessity letter.
7. **Status checking available** — representative explains how to check appeal status via the IVR system after submission.

---

## Source Artifact

Populated from `src/TTS/synpuf_high_rejection_case.csv`
