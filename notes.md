# Notes
- utlimate goal: replace billing team, eliminate provider resources for appeals (via email and phone) by providing pre-auth deinal verification and post-auth appeals
- potentially also use indepednent reviewer to tackle erronous insurance denials

- failed attempt by another person & their notes: https://www.kaggle.com/code/davidcsullivan/claim-denial-prediction-with-synthetic-data#Predictive-Modeling-for-Healthcare-Claim-Denials
- Fraud Detection in Health Insurance using GNNs - discusses how to inject anomolies:
	- https://www.kaggle.com/code/alirezaebrahimi/fraud-detection-in-health-insurance-using-gnns
- CMS data visualized breakdown: https://www.kaggle.com/code/smagh777/claims-eda#3.-Beneficiary-Demographics-Analysis
- pier to pier evluation: insruance hired doctors trying to dispute claims
- consider medical records request from insurance that request from clinic (dataVant - is one such contractors)

- autonomus AI billing from auto transcribing

- EMR is trying to come up with billing solution, hospitals have tried to use claim denial appeal softwares. Waystar, claude, rivet, cofactor
- infusion center and dialysis centers are money makers - bought out by VCs
	- Look at which departments generate the most money, maybe some surgical specialty
	- hospice care, skilled nursing facility
- automated translation for medical purpose?
- social case worker AI wrapper

## Meeting with Joe cca Feb 25
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

# Models:
|Model|Description|Use case|
|---|---|---|
|Medtok|AI advanced dictionray for explaining IPT/CPT code| |
|GatorTron / ClinicalBERT|EHR Code prediction, however GPT O3 and DeepSeek R1 finetuned performs better in accuracy - just use multi-agent query using medgemma instead|
|LLMCoder|ICD-10|Similar to EHR Code prediction|

# Keywords:
|Keyword|Defintion|Summary|
|---|---|---|
|CARC codes|Claim Adjustment Reason Codes| Rejection reason|
|LCD/NCD |Local Coverage Determination/National Coverage Determination||
|ICD |identifies patient diagnoses and conditions |Diagnosis/justifies necessities - "why"|
|CPT|describes procedures/services performed|Procedure - "what"|
|EHR Codes|Collective: All codes (ICD/CPT/HCPCS/RxNorm)|Same as ICD/CPT|
|DOS|Date of Service|Dates of charges|
|LOS|Length of Stay. If LOS > DRG geometric length of stay, means staying longer than avg benchmarks||

# Datasets:
|Dataset|Function|Remarks|Link|
|---|---|---|---|
|Synthea|Has approved codes |  | |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/abuthahir1998/synthetic-healthcare-claims-dataset?resource=download |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset |
|Kaggle - SynthetiC Healthcare Claims Dataset|Denial case samples|Not enough detail|https://www.kaggle.com/datasets/abuthahir1998/synthetic-ar-medical-dataset-with-realistic-denial |
|Medicaid CMS dataset| Denial case samples|Not enough detail (I think?)|https://data.cms.gov/sites/default/files/2023-05/d51e1218-68c3-4c7c-9598-0b81f22fe903/User%20Guide%20-%20CMS%20Synthetic%20RIF%20Files%20May%202023_AM508_v2.pdf |
|CMS BSA Hospice Beneficary | 5% de-identified real medicare data, but no explicit denial data  | | https://www.cms.gov/data-research/statistics-trends-and-reports/basic-stand-alone-medicare-claims-public-use-files/bsa-hospice-beneficiary-puf |
|CMS T-MSIS|Real data with real denial cases, 6-12 months of request time, >$3.5k annunal data rental fee |Overkill for prototype | |
|MIMIC-IV| EHR hospital care info including ICD, CPT | No billing information, not worth
|CMS SynPUF| Large synthetic medicare and medicaid dat, but no explicit denial data | Worth using payer information as proxy| |
|CMS Limited Data Set (LDS)|Limited dataset with medicare claims, but requires LDS request page| | https://www.cms.gov/data-research/cms-data/data-available-researchers/limited-data-set-lds-files|


# MVP

- v1 process:
	- Due to limited information from CMS, will primarily base data on Synthea
	- Will follow process in "Fraud Detection in Health Insurance Using GNNs", which shows higher performance than isolation forrest (already as seen in other papers)
		- https://www.kaggle.com/code/alirezaebrahimi/fraud-detection-in-health-insurance-using-gnns. 
	1) Generate data
	2) Drop empty columns
	3) Handle missing values - either fill in data "missing" for physicians or drop rows where essential details like dates/identifiers are not available
	4) Inject anomolie - duplication submission, services not covered, lack pre-approval, missing ICD/CPT, insufficient documents for medicial necessities, timely filing limit exceeded, incorrect patient/policy information, referral expired, authorization mismatch, bundling multiple procedures, maximum benefit exceeded, frequency over plan limit
	5) determine not qulitateive data and qualitative data
	6) evaluation - ROC-AU 

- v2 process
	- 
- Models to consider: isolation forest?, SL-GAD?, medgemma?

# Takeaway:
- How are trainings done for other defective:
	- image clasifiers medical dataset how to classify for defects
	- manufacturing dataset how to classify for defects
- Contact ansaf
- Check CMS data:



T-MSIS data, including denied claims via CLAIM-DENIED-INDICATOR ("0" for full denial) and CLAIM-LINE-STATUS (e.g., "542", "585", "654" for denied lines), is accessible through CMS's T-MSIS Analytic Files (TAF)—research-ready versions of state-submitted Medicaid/CHIP data.

No public downloads; requires a Data Use Agreement (DUA) via CMS or ResDAC/CCW for researchers.
Access Process

Submit via CMS Research Data Assistance Center (ResDAC) or Chronic Conditions Data Warehouse (CCW).

    ResDAC: Apply for TAF files (claims/enrollment); training required. Details at resdac.org.​

    CCW: TAF Research Identifiable Files (RIFs); register at ccwdata.org.​

    CMS Enterprise Portal: States view dashboard; researchers request via DataConnect/IDR.​

Key Documentation

    Data Guide: tmsis.medicaid.gov/dataguide (dictionary, validation rules).

    Denied Claims Guidance: medicaid.gov/tmsis/dataguide/t-msis-coding-blog/cms-guidance-reporting-denied-claims... (full specs).​

    TAF Tech Docs: Claims files cover inpatient/OT/long-term care/Prescription; includes adjustment codes.​

Data covers 2010–present (full states), with denial reasons in ADJUSTMENT-REASON-CODE.
