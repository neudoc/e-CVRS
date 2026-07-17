# Journal of Alzheimer's Disease — Submission Package

> Prepared for submission to the *Journal of Alzheimer's Disease* (SAGE). This file contains the JAD-formatted **title page**, **structured abstract**, **keywords**, **running title**, and the **required ADNI statements**. A separate cover letter is provided in `JAD_cover_letter.docx`.
>
> **Items to insert before submission (highlighted):** ORCID iDs, e-mail addresses, and any funding, where marked. (Author affiliations are confirmed.)

---

## TITLE PAGE

**Title**
An Explainable Automated Atrophy Visual Rating Scale (e-CVRS) Based on Center-of-Mass Alignment: Development and Multi-Domain Clinical Validation in ADNI T1-Weighted MRI

**Running title** (≤ 45 characters)
Explainable automated atrophy rating (e-CVRS)

**Authors**
Ho Tae Jeong¹, Seolah Lee¹, Jaeyoung Youn⁴, Jae-Won Jang², Young Chul Youn¹,³,\*, for the Alzheimer's Disease Neuroimaging Initiative†

**Affiliations**
1. Department of Neurology, Chung-Ang University Hospital, Seoul, Republic of Korea
2. Department of Neurology, Konkuk University Medical Center, Seoul, Republic of Korea
3. Department of Neurology, Chung-Ang University College of Medicine, Seoul, Republic of Korea
4. Jaseng Hospital of Korean Medicine, Seoul, Republic of Korea

**\*Correspondence to:** Young Chul Youn, MD, PhD, Department of Neurology, Chung-Ang University Hospital, Chung-Ang University College of Medicine, 102 Heukseok-ro, Dongjak-gu, Seoul 06973, Republic of Korea. Tel.: +82-2-6299-1501; E-mail: neudoc@cau.ac.kr. ORCID: 0000-0002-2742-1759.

**Author e-mail addresses**
Ho Tae Jeong (hotaejeong@cau.ac.kr); Seolah Lee (seolah920831@gmail.com); Jaeyoung Youn (sdjjdb129@gmail.com); Jae-Won Jang (jaewon26@gmail.com); Young Chul Youn (neudoc@cau.ac.kr).

**†** Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf.

**Author ORCID iDs**
Ho Tae Jeong 0000-0001-9228-6912; Seolah Lee 0000-0002-1285-1334; Jaeyoung Youn *(insert — not found in wiki)*; Jae-Won Jang *(insert — not found in wiki)*; Young Chul Youn 0000-0002-2742-1759.

---

## STRUCTURED ABSTRACT

*(JAD format: five headings — Background, Objective, Methods, Results, Conclusions; ≤ 250 words. Current count ≈ 235.)*

**Background:** Visual rating scales for brain atrophy on MRI are fast and intuitive but reader-dependent, whereas automated volumetry such as FreeSurfer is slow and operates as a black box.

**Objective:** To develop and validate e-CVRS, an explainable automated atrophy rating requiring no external registration or segmentation software, against blinded expert ratings, multi-domain cognition, and cerebrospinal fluid (CSF) biomarkers.

**Methods:** We analyzed baseline T1-weighted MRI from 339 ADNI participants with mild cognitive impairment. Using the physical head center-of-mass as origin, hippocampal, frontal, parietal, and temporal regions of interest were automatically placed; for each region the local CSF ratio, proxy volume, global ventricular fraction, and (for the hippocampi) contralateral ratio were combined into a transparent linear atrophy score and mapped to ordinal grades. Combination weights and thresholds were learned only within training folds of a 5-fold cross-validation (leakage-controlled); 95% confidence intervals used 2,000 bootstrap resamples. Incremental value over covariates across six cognitive outcomes was benchmarked against manual CVRS and UCSF FreeSurfer hippocampal volume (Benjamini–Hochberg correction). CSF Aβ42/p-tau associations were tested (n = 321).

**Results:** Agreement ranged from fair to moderate-to-substantial (temporal κw = 0.620; total ICC = 0.661, 95% CI 0.597–0.718). e-CVRS was non-significant on MMSE (FDR p = 0.21) but significant on ADAS-Cog-13 (p = 0.001), memory (p = 0.012), executive function (p < 0.001), and language (p = 0.008), and correlated with CSF Aβ42 (ρ = −0.227, p < 0.001), matching manual CVRS. FreeSurfer was most sensitive throughout.

**Conclusions:** e-CVRS quantifies atrophy in seconds on a standard CPU while remaining explainable; its MMSE null reflects range restriction, not absence of signal.

**Keywords:** Alzheimer's disease; cerebral atrophy; explainable artificial intelligence; magnetic resonance imaging; mild cognitive impairment; neuropsychological tests; visual rating scale

---

## REQUIRED ADNI STATEMENTS (place as indicated in the main text)

**In the Methods (data source paragraph), add:**
Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public–private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial MRI, PET, other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment and early Alzheimer's disease. For up-to-date information, see www.adni-info.org.

**In the Acknowledgments, add the full ADNI funding statement:**
Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI; National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following: [list of ADNI industry partners per the standard ADNI acknowledgment]. The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer's Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California.

*(The full, current partner list and wording are available on the ADNI site; paste the up-to-date block before submission.)*

---

## SUBMISSION CHECKLIST (JAD)

- [ ] Title page (this file) — title, running title (≤45 char), full byline with ADNI group author, affiliations, corresponding author with e-mail/ORCID.
- [ ] Structured abstract ≤250 words, five headings; 5–10 keywords (alphabetical).
- [ ] Main text: Introduction, Materials and Methods, Results, Discussion, Conclusions (use `draft_paper_en_final.docx`).
- [ ] Tables 1–4 and Figures 1–4 with legends (embedded or separate per journal preference; figures ≥300 dpi).
- [ ] ADNI data-use statement in Methods + full ADNI funding block in Acknowledgments.
- [ ] Declarations: Conflict of Interest; Funding; Data Availability; Ethics (ADNI IRB-approved, written informed consent; secondary de-identified data).
- [ ] Author contributions (CRediT).
- [ ] ORCID for corresponding author (required by SAGE).
- [ ] Cover letter (`JAD_cover_letter.docx`).
- [ ] Reporting: STARD/TRIPOD checklist as supplementary if requested.
