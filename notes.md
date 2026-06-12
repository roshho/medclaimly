# Notes
- utlimate goal: replace billing team, eliminate provider resources for appeals (via email and phone) by providing pre-auth deinal verification and post-auth appeals
- potentially also use indepednent reviewer to tackle erronous insurance denials

- failed attempt by another person & their notes: https://www.kaggle.com/code/davidcsullivan/claim-denial-prediction-with-synthetic-data#Predictive-Modeling-for-Healthcare-Claim-Denials
- Fraud Detection in Health Insurance using GNNs - discusses how to inject anomolies:
	- https://www.kaggle.com/code/alirezaebrahimi/fraud-detection-in-health-insurance-using-gnns
- CMS data visualized breakdown: https://www.kaggle.com/code/smagh777/claims-eda#3.-Beneficiary-Demographics-Analysis
- pier to pier evluation: insruance hired doctors trying to dispute claims
- consider medical records request from insurance that request from clinic (dataVant - is one such contractors)

## Required data for rules
- ICD <-> HCPCS Validity matrix
	- Based on ICD code, what procedure/supplies can be issued
	- Issue: varies across every insurance based on coverage
	- However most follow: NCCI PTP/MUE edits (HCPCS/CPT pairs) + LCD/NCD policies (explicit ICD links)
		- NCCI: National Correct Coding Initiative 
		- LCD/NCD = context coverage
			- LCD: Local Coverage Determination
				- Cna skip, same as NCDs except rare DME (Durable Medical Eqiuipment like wheelchairs, orthotics....)
			- NCD: National Coverage Determination
			- MCD: Database/search tool housing all LCDs, NCDs, Articles (billing/coding details with code lists). CPT/HCPCS/ICD now mostly in Articles (not LCDs except DME). I.e. this is where you download
		- PTP: Procedure to Procedure
		- MUE: Medically Unlikely Edits
			- Max # of claims per day  by provider for same beneficiary
		- NCCI: National Correct Coding Initiative,
		- Change Request: CR
		- AOC Edits: Add on code edits - added onto procedure, rarely approved. Has 3 types:
			- Type 1: Used when limited primary proecdure codes. Payable only when primary proceure also paid to same practitioner for the same patient on the same date
			- Type 2: Used when no specfic primary procedure code in Change Request. Contractors encouraged to create their own acceptable primary codes list.
			- Type 3: Used when primary proecdure code avilable, but listed codes are limited - can expand list


## Meeting with Amir 
- autonomus AI billing from auto transcribing
- EMR is trying to come up with billing solution, hospitals have tried to use claim denial appeal softwares. Waystar, claude, rivet, cofactor
- infusion center and dialysis centers are money makers - bought out by VCs
	- Look at which departments generate the most money, maybe some surgical specialty
	- hospice care, skilled nursing facility
- automated translation for medical purpose?
- social case worker AI wrapper


## Meeting with Dr Joe cca Feb 25
- Dentistry typically appealand denial
- Dentistry different from denistry 
	- Many of common denials in hospital rejection are not as common in denistry
	- Medical field, unless out of control, seems like most things are denied
- In denitistry, downgrade occurs when 
	- Might be more helpful specficially for dental, because downgrading is more blatant
	- It's all manual labour with dedicated department to do call and appeal downgrade
	- Dental can do pre-auth and post-op.
- Insurance in general, have yearly $500 deductable before insurance starts covering things. 
	- Will have list of things that are partially covered, e.g. exam is 100% coverages or filling is 75% - those count towards deductable
	- Dental is different, passing deductable, others are covered out of pocket
	- Any treatment can pre-auth everything, office and patient will know is covered or not
	- Generate treatment plans after pre-auth. 
	- Post-op - is determined subjectivly 
- Dataset: 
	- Best dataset is from insurance company
	- Check out umbrella insurance, will group different services together and might be willing to provide data to us
- AI insurance company
	- Early 2025, but wasn't very good for insurance dental claim
	- Company is called "verifying insurance"
	- Possible to automate verifying insurance 

	
## Christopher Foley, MPA [Linkedin post](https://www.linkedin.com/posts/christophermfoley_start-preventing-denials-christopher-foley-activity-7425267779852910592-j3Qm/)
- The four prevention points that actually move the needle:
	- 🔴 Prior authorization - AI flags high-risk procedures needing auth before scheduling 
	- 🔴 Clinical documentation - Real-time CDI prompts before the claim is filed
	- 🔴 Medical necessity - Predictive scoring catches LCD/NCD gaps at registration 
	- 🔴 Coding accuracy - Ambient AI suggests correct codes during the encounter

## Muhammad Afzal's denial [linkedin claim denial project](https://www.linkedin.com/posts/muhammad-afzal-7594581a9_github-92-afzalclaimsdenialanalyzer-activity-7424417626711318529-49y0/)
- Assigned cases randomly to be denied, thus not credible way to get denied cases

# Models

|Model|Description|Use case|
|---|---|---|
|Medtok|AI advanced dictionray for explaining IPT/CPT code| |
|GatorTron / ClinicalBERT|EHR Code prediction, however GPT O3 and DeepSeek R1 finetuned performs better in accuracy - just use multi-agent query using medgemma instead|
|LLMCoder|ICD-10|Similar to EHR Code prediction|

# Keywords

| Keyword    | Defintion                                                                                       | Summary                                 |
| ---------- | ----------------------------------------------------------------------------------------------- | --------------------------------------- |
| CARC codes | Claim Adjustment Reason Codes                                                                   | Rejection reason                        |
| LCD/NCD    | Local Coverage Determination/National Coverage Determination                                    |                                         |
| ICD        | identifies patient diagnoses and conditions                                                     | Diagnosis/justifies necessities - "why" |
| CPT        | describes procedures/services performed                                                         | Procedure - "what"                      |
| EHR Codes  | Collective: All codes (ICD/CPT/HCPCS/RxNorm)                                                    | Same as ICD/CPT                         |
| DOS        | Date of Service                                                                                 | Dates of charges                        |
| LOS        | Length of Stay. If LOS > DRG geometric length of stay, means staying longer than avg benchmarks |                                         |
| DRG        | Diagnosis-Related Group - categorize hospital stay based on diagnoses and treatment             | In-patient stay labels                  |
| IVR        | Interactive Voice Response - automated phone/portal system to check claim status.               |                                         |

# Datasets:
|Dataset|Function|Remarks|Link|
|---|---|---|---|
|Synthea|Has approved codes |  | |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/abuthahir1998/synthetic-healthcare-claims-dataset?resource=download |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/abuthahir1998/synthetic-ar-medical-dataset-with-realistic-denial |
|Medicaid CMS dataset| Denial case samples|Not enough detail (I think?)|https://data.cms.gov/sites/default/files/2023-05/d51e1218-68c3-4c7c-9598-0b81f22fe903/User%20Guide%20-%20CMS%20Synthetic%20RIF%20Files%20May%202023_AM508_v2.pdf |
|CMS BSA Hospice Beneficary | 5% de-identified real medicare data, but no explicit denial data  | | https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/bsa-hospice-beneficiary-puf |
|CMS ResDAC/T-MSIS|Real data with real denial cases, 6-12 months of request time, >$3.5k annunal data rental fee |Overkill for prototype, but a good first step to acquire limited medicare RIF sample as test dataset| |
|MIMIC-IV| EHR hospital care info including ICD, CPT | No billing information, not worth
|CMS SynPUF| Large synthetic medicare and medicaid dat, but no explicit denial data | Worth using payer information as proxy| |
|CMS Limited Data Set (LDS)|Limited dataset with medicare claims, but requires LDS request page| | https://www.cms.gov/data-research/cms-data/data-available-researchers/limited-data-set-lds-files|
|HCUP-US|Healthcare cost and utilization project|Similar to CNS SynPUF, but real cases|https://hcup-us.ahrq.gov/ |
|APCD|All-Payer Claims Data from states - prices vary, and data quality varies. | Worth while dataset from, e.g. Colorado APCD, but would be more beneficial if doing that specfic state expansion| |

# Rules
|Data|Remarks|Link|
|---|---|---|
|CMS MUE 2026| |https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-medically-unlikely-edits-mues|
|CMS PTP 2026 Hospital & Practitioner| | https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits/medicare-ncci-procedure-procedure-ptp-edits|
|CMS NCD, MCD| Medicare Coverage Database - central CMS repo for LCD & NCD | https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx |
|CMS HCPCS alpha numeric coverage labeling| |https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/quarterly-update|
|DRG Files| Provides averages, e.g. length of stays | https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Acute-Inpatient-Files-for-Download-Items/CMS1247873 |
|CMS Beneficary context year | Part of SynPUF - shows deductables left - useful for classifying via rules | | 

# Voice synthesis notes
- Things that can't be solved by email:
	1. Provider v Insurance Payer
		- IVR (automated voice services) typically lack complete information andonly provide status (e.g. claim pending), not exact issue (e.g. need medical records for CPT 99214)
		- Authorizations and medical necessity appeals:often providers need authoarization from payer and need to obtain payer received confirmation and reference number
	2. Peer-to-peer review
		- Payer medical provider will dispute provider medical necessity
			- Most important effective appeals
	3. Patient to payer
		- Least common and vague - not applicable in our case

# MVP
- Models to consider: isolation forest?, SL-GAD?, medgemma?

# Takeaway:
- How are trainings done for other defective:
	- image clasifiers medical dataset how to classify for defects
	- manufacturing dataset how to classify for defects
- Contact ansaf
- Check CMS data:


# Pipeline results & status (dev log, preserved from old README)

## Approach history
- **v1 (failed):** trained on MIMIC-IV. No billing/denial labels, so denials had to be
  synthesized by mutating admissions — caused distribution skew and poor generalization.
  Scripts removed (`src/src_mimic_iv/`).
- **v2:** moved to CMS DE-SynPUF. Engineered denial proxies (zero-payment, non-covered,
  benefits-exhausted). Proxy denial rate came out ~39.8%, far above the real ~15-17%
  Medicare rate — driven mostly by missing codes/dates/IDs (99.4% of claims had missing
  documentation). Symptom (`zero_payment`) vs root-cause layering was added.
- **v3 (current):** two-stage pipeline with soft/weak labels + confidence threshold. The
  XGBoost classifier is used as a **pre-emptive denial-risk proxy**, not a verdict.

## Stage results (CMS DE-SynPUF Sample 1, 11.15M claims)
- Stage 1 (deterministic rules): 176,994 hard denies (1.59%), 10.97M passed to ML.
- Stage 2 (XGBoost): trained on 5.47M labeled rows, positive rate 31.46%.
  - PR-AUC: **0.9754**
  - Best F1: **0.9090** @ threshold 0.6526
  - Recall @ P≥0.80: **0.9555**
  - Patient-level split on `DESYNPUF_ID` (no beneficiary leakage); pre-train leakage assertion.

## Known limitations
- SynPUF "denial" labels are derived from the same reference rules used as features —
  the model partially learns its own rules back.
- HCPCS/MUE/PTP reference files are 2026 vintage applied to 2008-2010 data — some code
  coverage statuses have changed.
- No real D-type denial ground truth exists in SynPUF; Stage 2 is a validated prototype
  pending ResDAC access.
- Final joined artifact scores only a subset of pass-through rows (grouped inference on
  labeled composite keys); full coverage needs an inference-persistence pass.

## TODO / roadmap
- [ ] ICD ↔ HCPCS validity matrix per payer
- [ ] Prior-authorization + pre-claim review features (post-2020 CMS PA list)
- [ ] Update DRG weight files to 2026
- [ ] Shift rules references from soft signals to hard-deny rules for modern claims
- [ ] ICD-9 → ICD-10
- [ ] Retrain Stage 2 on real adjudication labels (incl. D-type denials)
- [ ] RAG on latest NCD/LCD policy for appeal drafting / pre-emptive submission
- [ ] Local RAG vector store (currently relies on scikit-learn embeddings in-process)
- [ ] Multi-claim support — query the pipeline parquet to load any flagged claim dynamically


