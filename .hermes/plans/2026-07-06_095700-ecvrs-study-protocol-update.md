# e-CVRS 자동화 연구 프로토콜 업데이트

작성일: 2026-07-06 09:57
대상 폴더: `/mnt/e/CVRS_MCI_APET_ADNI`
자료 성격: 사용자가 공개 자료라고 명시함. 단, 외부 모델/API 전송은 별도 승인 전 금지.

## 1. 현재 확인된 자료와 즉시 해석 가능한 범위

### 1.1 폴더 인벤토리

확인된 파일 수:

- `.hdr`: 341개
- `.img`: 341개
- `.xlsx`: 1개 (`ADNI_MRI_rating.xlsx`)
- `.pdf`: 2개 (`CVRS_article.pdf`, `CVRS_Table.pdf`)
- `.py`: 2개 (`src/e_cvrs_pipeline.py`, `src/evaluate_pipeline.py`)

즉, 현재 폴더는 Analyze 7.5 형식의 T1 MRI 341건과 사람이 평가한 CVRS 계열 rating 파일, CVRS 원 논문/측정표, 그리고 Gemini 또는 이전 작업에서 만든 초기 자동화 스크립트로 구성되어 있다.

### 1.2 CVRS 문헌/테이블에서 확인한 핵심 구조

`CVRS_article.pdf`에서 확인한 내용:

- CVRS는 hippocampal atrophy, cortical atrophy, subcortical atrophy/ventricular enlargement, small vessel disease를 통합한다.
- 총점은 0–30점이며 점수가 높을수록 구조적 손상이 많다는 의미다.
- ADNI prodromal AD 189명 연구에서 3년 추적 중 61명(32.3%)이 AD dementia로 진행했고, baseline CVRS는 진행 예측과 관련 있었다.
- 보고된 예: stable vs progressive 평균 CVRS 9.9±5.1 vs 12.4±4.9, p=0.002; CVRS HR=1.110, 95% CI 1.043–1.182.

`CVRS_Table.pdf`에서 확인한 구조:

- Brain atrophy score: 0–23
  - Hippocampus: Rt/Lt, 각 0–4, 합 0–8
  - Cortical atrophy: frontal, temporal, parietal, 각 0–3, 합 0–9
  - Subcortical atrophy/ventricular enlargement: anterior/posterior horn, 합 0–6
- Small vessel disease score: 0–7
  - WMH: 0–3
  - Lacune: 0–2
  - Microbleeds: 0–2
- Total CVRS: brain atrophy 0–23 + SVD 0–7 = 0–30

현재 사용자가 언급한 `ADNI_MRI_rating.xlsx`의 사람이 평가한 항목은 `Hippo_Lt`, `Hippo_Rt`, `Front`, `Parietal`, `Temporal`이다. 따라서 현재 즉시 가능한 1차 연구 범위는 “T1 기반 atrophy e-CVRS 0–17점 자동화”이다.

[추정] 현재 폴더명에 `APET`가 포함되어 있어 amyloid PET 관련 임상/바이오마커 변수가 rating 파일 또는 외부 ADNI table과 연결될 가능성이 있으나, 아직 xlsx 내부를 완전히 확인하지 못했다. WSL 기본 Python에 `openpyxl`이 없어 xlsx 세부 열 확인은 다음 단계에서 환경 준비 후 수행해야 한다.

## 2. Gemini 의견을 반영한 업데이트된 핵심 연구 가설

### 2.1 핵심 컨셉

의사가 임상 현장에서 사용하는 CVRS를 그대로 디지털화하는 것이 아니라, 다음 세 층으로 분해해 검증한다.

1. Manual CVRS: 사람이 실제로 매긴 기준 점수
2. e-CVRS: CVRS 지침을 영상 처리/기하학적 규칙으로 자동화한 점수
3. Volumetry: hippocampus, cortical lobe, ventricle 등 표준 또는 준표준 ROI volume

핵심 질문은 “e-CVRS가 manual CVRS와 충분히 일치하면서, volumetry 대비 더 빠르고 설명 가능하며 임상 예측력에서 비열등한가?”이다.

### 2.2 논문성 있는 주 가설

Primary hypothesis:

- T1 MRI에서 자동 산출한 atrophy e-CVRS 0–17점은 사람이 평가한 atrophy CVRS 0–17점과 좋은 수준의 일치도를 보인다.

Secondary hypotheses:

- e-CVRS 각 영역 점수는 해당 ROI volumetry와 유의한 상관을 보인다.
- e-CVRS 총점은 MMSE/CDR-SB/ADAS 등 임상 지표와 manual CVRS 또는 volumetry에 비해 비열등한 관련성을 보인다.
- e-CVRS는 scanner/site 차이와 영상 품질 변화에 대해 volumetry보다 더 안정적일 수 있다.
- [추정] amyloid PET 양성/음성 또는 MCI progression 정보가 연결 가능하다면, e-CVRS는 amyloid-positive MCI의 임상 진행 예측에 유용하다.

## 3. 권장 연구 디자인

### 3.1 1차 논문: T1-only atrophy e-CVRS 자동화 및 volumetry 비교

권장 제목:

- Automated T1-Based Atrophy Visual Rating Scale on ADNI MRI: Agreement with Human CVRS Ratings and Comparison with Volumetry

분석 대상:

- 현재 폴더의 T1 MRI 341건 중 `ADNI_MRI_rating.xlsx`와 RID가 매칭되는 사례
- Manual labels: `Hippo_Lt`, `Hippo_Rt`, `Front`, `Parietal`, `Temporal`
- 총점: `Atrophy17 = Hippo_Lt + Hippo_Rt + Front + Parietal + Temporal`

주요 endpoint:

1. 영역별 ordinal agreement
   - Hippo_Lt, Hippo_Rt: 0–4
   - Front, Parietal, Temporal: 0–3
   - metric: quadratic weighted kappa, exact agreement, ±1 agreement
2. 총점 일치도
   - Atrophy17 manual vs e-CVRS
   - metric: ICC(2,1), Bland-Altman plot, MAE/RMSE
3. Volumetry와의 관계
   - hippocampal volume vs Hippo score/e-Hippo score
   - frontal/parietal/temporal cortical volume or thickness vs e-CVRS lobe score
   - ICV-normalized volume 사용
4. 임상 관련성
   - MMSE, CDR-SB, ADAS11/ADAS13 등과의 상관/회귀
   - 가능하면 diagnosis 또는 amyloid PET status와의 분류 성능

### 3.2 2차 확장 논문: full CVRS 0–30 또는 progression prediction

현재 폴더에는 FLAIR/T2*/SWI가 보이지 않으므로 full CVRS 0–30 자동화는 즉시 수행 범위가 아니다.

확장 조건:

- FLAIR 영상 확보: WMH 자동화
- T2*/SWI 확보: microbleeds 자동화
- T1/FLAIR에서 lacune 후보 탐지 가능성 검토
- longitudinal diagnosis/progression table 확보

확장 연구 제목 예:

- Fully Automated Comprehensive Visual Rating Scale Including Atrophy and Small Vessel Disease for Predicting MCI-to-AD Progression

## 4. 알고리즘 구현 전략 업데이트

### 4.1 현재 Gemini/기존 스크립트 접근의 장점

`src/e_cvrs_pipeline.py`는 다음 방향성이 좋다.

- MRI 전체 segmentation 없이 특정 ROI와 CSF ratio를 계산하려 한다.
- hippocampus, frontal, parietal, temporal의 국소 지표를 추출한다.
- volumetry proxy도 함께 계산해 e-CVRS와 비교할 수 있게 한다.

이 접근은 “가볍고 빠른 CVRS 자동화”라는 연구 컨셉과 잘 맞는다.

### 4.2 반드시 수정해야 할 문제점

현재 스크립트는 pilot 수준이며 논문용으로는 다음 문제가 있다.

1. 경로가 Windows `e:\CVRS_MCI_APET_ADNI`로 hard-coded 되어 있어 WSL/재현성에 취약하다.
2. MRI orientation/affine/voxel spacing 처리가 불완전하다.
3. center-of-mass와 고정 offset만으로 slice/ROI를 잡아 개인차, head tilt, FOV 차이에 취약하다.
4. skull stripping/bias correction/registration이 없다.
5. intensity threshold 기반 CSF/tissue 판정은 scanner/site에 취약할 수 있다.
6. `evaluate_pipeline.py`의 rank-based mapping은 전체 manual score 분포를 사용해 자동 점수를 맞추므로 leakage 성격이 강하다. 논문에서는 train/test split 또는 cross-validation 내부에서 threshold를 학습해야 한다.
7. 현재 volumetry proxy는 “전통적 volumetry”가 아니라 단순 박스 ROI volume이다. 논문 제목에서 volumetry 비교를 주장하려면 FreeSurfer/FastSurfer/SynthSeg/ADNI UCSF FreeSurfer table 중 하나를 comparator로 써야 한다.

### 4.3 권장 알고리즘: 2-track 설계

Track A: Explainable rule-based e-CVRS, 논문 주력

- 목표: CVRS 지침에 대응되는 단면/랜드마크/거리/면적비로 점수를 산출
- 장점: 설명 가능성, SaMD 확장성, 임상 수용성
- 출력: score + 근거 단면 + 측정선/ROI overlay + 수치 feature

Track B: 딥러닝 보조 모델, 보조 분석

- 목표: rule-based feature가 놓치는 복잡한 패턴을 보완
- 입력: 표준화된 ROI crop 또는 2.5D key slices
- 출력: ordinal class probability
- 제한: black-box가 되지 않도록 Grad-CAM/attention map 또는 rule feature와 결합

권장 순서: Track A를 먼저 완성하고, Track B는 reviewer 대응 또는 성능 향상용 보조 분석으로 둔다.

## 5. 구체적 알고리즘 첫 단계: slice 선택과 landmark 추출

### 5.1 공통 전처리

1. DICOM/Analyze/NIfTI 입력 정리
   - 현재 `.hdr/.img` pair를 nibabel로 로드
   - 모든 이미지를 RAS orientation으로 통일
   - voxel spacing과 affine 검증
2. N4 bias field correction 또는 robust intensity normalization
3. skull stripping
   - 빠른 방법: SynthStrip/SynthSeg 계열 로컬 실행
   - fallback: BET/HD-BET/간단 brain mask
4. MNI 또는 ADNI template affine registration
   - 최소 affine, 가능하면 nonlinear optional
   - 목적: ROI와 slice 좌표를 개인별 anatomy에 안정적으로 매핑
5. QC 지표 생성
   - brain mask coverage
   - registration cost
   - voxel spacing/FOV
   - motion/contrast outlier score

### 5.2 Hippocampal atrophy: Hippo_Lt/Rt 0–4

CVRS/Scheltens MTA 취지에 맞춰 다음 feature를 추천한다.

- 표준 coronal plane에서 hippocampal body 수준 slice 자동 선택
- 좌우 hippocampal ROI 주변에서:
  - temporal horn CSF area
  - choroid fissure CSF area
  - hippocampal height 또는 tissue area
  - perihippocampal CSF ratio
  - inferior lateral ventricle volume/area
- 자동 점수화:
  - 방법 1: ordinal threshold model, score 0–4
  - 방법 2: proportional odds ordinal regression
  - 방법 3: monotonic gradient boosting with constraints [선택]

설명 가능한 출력:

- 선택 coronal slice
- hippocampal ROI box
- CSF mask overlay
- `CSF_ratio`, `hippocampal_height_proxy`, `temporal_horn_area` 표시

### 5.3 Frontal cortical atrophy: Front 0–3

추천 feature:

- AC-PC 또는 MNI 기준 frontal convexity axial/coronal slices 선택
- interhemispheric fissure width
- frontal sulcal CSF fraction
- frontal cortical ribbon/tissue fraction
- frontal lobe convexity sulcal widening index

주의:

- 단일 slice보다는 3–5개 slice 평균이 안정적이다.
- 나이/성별/ICV 보정 여부를 sensitivity analysis로 비교한다.

### 5.4 Parietal cortical atrophy: Parietal 0–3

추천 feature:

- Koedam parietal atrophy scale 취지 반영
- precuneus/posterior cingulate/parietal convexity 영역에서:
  - parietal sulcal CSF fraction
  - posterior cingulate sulcus/precuneus fissure widening proxy
  - parietal cortical tissue fraction

출력:

- parietal ROI overlay
- score 근거가 된 sulcal widening 지표

### 5.5 Temporal cortical atrophy: Temporal 0–3

추천 feature:

- lateral temporal cortex 및 Sylvian fissure 주변 CSF fraction
- temporal horn/insula 주변 CSF expansion과 분리해서 평가
- hippocampal MTA feature와 과도하게 중복되지 않도록 lateral temporal ROI 별도 정의

### 5.6 점수화 모델

Rule-based thresholding만으로 시작하되, threshold는 다음처럼 공정하게 산출한다.

- train/test split 또는 5-fold CV
- training fold에서만 manual rating과 feature의 cut-point를 추정
- test fold에서 score 예측
- cut-point 방법:
  - ordinal logistic regression
  - decision tree depth 1–2
  - isotonic regression 후 score threshold
  - Youden 또는 MAE 최소화 threshold

금지할 것:

- 전체 데이터의 manual 분포를 이용해 rank mapping으로 자동 score 분포를 강제로 맞추는 방식. 이는 성능을 과대평가할 위험이 있다.

## 6. Volumetry comparator 설계

### 6.1 권장 comparator 우선순위

1. ADNI 공식 FreeSurfer/UCSF ROI table이 RID/scan date와 매칭 가능하면 최우선
2. 로컬 FastSurfer 또는 SynthSeg로 hippocampus/ventricle/lobar volume 산출
3. FreeSurfer recon-all은 표준성이 높지만 시간이 오래 걸리므로 subset 또는 기존 ADNI table 권장
4. 현재 스크립트의 box-based `Vol_*`는 exploratory proxy로만 사용

### 6.2 비교 항목

- Hippocampus: left/right hippocampal volume, ICV-normalized
- Cortical lobes: frontal/parietal/temporal cortical GM volume 또는 mean thickness
- Ventricles: lateral ventricle 또는 inferior lateral ventricle volume
- Optional: whole brain volume, cortical thickness summary

### 6.3 통계 분석

- Spearman correlation: ordinal score vs volume
- Partial correlation/regression: age, sex, education, ICV, scanner/site 보정
- e-CVRS vs volumetry 임상 예측력 비교:
  - nested regression ΔR²
  - AUC/MAE for diagnosis/cognition
  - likelihood ratio test 또는 bootstrap CI
  - non-inferiority margin 사전 정의

## 7. 통계 프로토콜

### 7.1 Dataset split

권장:

- 70/15/15 train/validation/test 또는 5-fold cross-validation
- subject-level split
- 동일 RID 중복 scan이 있다면 같은 fold에 묶기
- site/scanner stratification 가능하면 적용

### 7.2 평가 지표

영역별 점수:

- Quadratic weighted kappa
- Exact agreement
- ±1 agreement
- Macro-F1 for ordinal class
- Confusion matrix

총점:

- ICC(2,1)
- Pearson/Spearman correlation
- MAE/RMSE
- Bland-Altman bias/limits of agreement

임상 예측:

- 연속 cognition: β, partial R², MAE
- 진단/amyloid positivity: AUC, balanced accuracy
- progression 가능 시: Cox PH HR, C-index, Kaplan-Meier by tertile/quartile

### 7.3 주요 모델 비교

Model 1: covariates only

- age, sex, education, APOE4 [가능 시], site/scanner

Model 2: manual CVRS

- manual Atrophy17 또는 영역별 manual score

Model 3: e-CVRS

- e-Atrophy17 또는 영역별 e-score/features

Model 4: volumetry

- hippocampal/lobar/ventricle volumes normalized by ICV

Model 5: combined

- e-CVRS + volumetry

핵심 해석:

- e-CVRS가 manual CVRS와 얼마나 가까운가?
- e-CVRS가 volumetry만큼 임상 지표를 설명하는가?
- e-CVRS + volumetry가 상승효과를 내는가, 아니면 중복 정보인가?

## 8. SaMD/상용화 관점의 산출물 정의

논문용 결과와 별도로 초기 제품 정의를 함께 설계한다.

Minimum viable software output:

1. 입력 MRI 파일 목록과 자동 QC 결과
2. 영역별 e-CVRS 점수
3. 총 atrophy e-CVRS 점수
4. 근거 단면 PNG
5. ROI/측정선 overlay
6. 수치 feature table
7. PDF 리포트 또는 PACS-friendly JSON/DICOM-SR prototype

임상 메시지 예:

- “좌측 MTA score 2로 추정됨: hippocampal ROI 내 CSF fraction 0.xx, temporal horn area 0.xx mm²”
- “Parietal atrophy score 1로 추정됨: parietal sulcal CSF fraction이 training reference의 x percentile”

주의:

- 연구 단계에서는 “진단”이 아니라 “시각 평가 보조/정량화”로 포지셔닝해야 한다.

## 9. Hermes 오케스트레이션 실행 계획

사용자 선호에 맞춰 다음 역할 분담을 권장한다.

### 9.1 Hermes

- 전체 계획/프로토콜 관리
- 데이터 인벤토리와 개인정보/외부전송 안전성 확인
- Claude/Gemini 결과 통합
- 최종 통계 및 논문 스토리라인 검증

### 9.2 Claude Code

- 로컬 구현/리팩터링
- `.hdr/.img` 로딩, orientation 표준화, preprocessing pipeline 작성
- e-CVRS feature extractor 작성
- cross-validation evaluation 작성
- 결과 테이블/그림 생성

### 9.3 Gemini 또는 독립 리뷰 에이전트

- 알고리즘/통계 누수 검토
- CVRS 측정표와 구현의 불일치 검토
- 논문 novelty와 reviewer 관점 비판
- SaMD/임상 workflow 관점 검토

중요: 현재 자료가 공개 자료라고 하더라도, 실제 MRI 파일/엑셀 전체를 외부 모델에 전송하기 전에는 사용자에게 별도 확인을 받아야 한다.

## 10. 다음 실행 단계 제안

### Step 1. Rating xlsx 세부 확인

목표:

- 열 이름, 행 수, RID unique 수, missingness, score 분포 확인
- MMSE/CDRSB/ADAS/APET/progression 관련 열 존재 확인

필요 조치:

- WSL venv에 `openpyxl` 설치 또는 Windows/PowerShell 기반 xlsx 변환
- 결과를 `analysis/data_inventory.md`로 저장

### Step 2. MRI-RID 매칭 테이블 생성

목표:

- 341개 `.hdr/.img` 파일에서 RID, site, date, image ID 추출
- rating xlsx와 매칭
- 중복 scan 처리 정책 결정: baseline 우선, 또는 rating과 가장 가까운 scan date

산출물:

- `analysis/mri_manifest.csv`
- `analysis/matched_subjects.csv`

### Step 3. 기존 script를 pilot용으로 정리

수정 방향:

- hard-coded path 제거
- CLI argument 지원
- output directory 명시
- RAS orientation 통일
- logging/QC 추가
- current box-volume은 `proxy_volume`으로 명명

### Step 4. 논문용 e-CVRS v1 feature extractor 구현

최소 구현:

- N4/normalization optional
- skull/brain mask
- template affine registration 또는 robust landmark-based alignment
- hippocampus/frontal/parietal/temporal ROI feature extraction
- overlay PNG 생성

### Step 5. 공정한 점수화/검증

- fold 내부 threshold 학습
- held-out prediction 생성
- kappa/ICC/MAE/Bland-Altman 산출
- 기존 rank mapping 결과와 공정한 CV 결과 비교

### Step 6. Volumetry comparator 확보

- ADNI FreeSurfer table 매칭 가능 여부 확인
- 불가능하면 SynthSeg/FastSurfer 로컬 실행 검토
- 현재 box proxy는 supplementary exploratory로 제한

### Step 7. 논문용 결과 패키지 생성

산출물:

- `RESULTS_REPORT.md`
- Table 1: cohort characteristics
- Table 2: agreement metrics
- Table 3: volumetry correlation/regression
- Table 4: clinical prediction/non-inferiority
- Figure 1: pipeline schematic
- Figure 2: example overlay for each CVRS region
- Figure 3: confusion matrices
- Figure 4: Bland-Altman/ICC
- Figure 5: e-CVRS vs volumetry clinical utility

## 11. 리스크와 대응

### 리스크 1. Manual rating이 한 명 평가자일 수 있음

대응:

- inter-rater reliability는 주장하지 않고 human-reference agreement라고 표현
- 가능하면 subset 재평가 또는 second rater 추가

### 리스크 2. 341명 표본이 딥러닝에 작음

대응:

- rule-based/ordinal model을 주력으로 사용
- 딥러닝은 ROI localization 또는 supplementary analysis로 제한

### 리스크 3. Full CVRS 0–30이 아님

대응:

- 1차 논문은 명확히 T1-based atrophy CVRS 0–17로 제한
- full CVRS는 FLAIR/T2* 확장 연구로 분리

### 리스크 4. Volumetry comparator가 부정확할 수 있음

대응:

- box proxy를 “volumetry”라고 부르지 않기
- ADNI 공식 FreeSurfer 또는 로컬 validated segmentation 사용

### 리스크 5. Data leakage

대응:

- rank-based distribution matching 금지 또는 pilot로만 표기
- threshold/model은 train fold에서만 추정
- 최종 test set은 한 번만 평가

## 12. 의사결정이 필요한 항목

1. 1차 목표를 “T1 atrophy e-CVRS 0–17”로 확정할지?
2. Volumetry comparator를 ADNI 공식 FreeSurfer table로 할지, 로컬 SynthSeg/FastSurfer로 할지?
3. progression/amyloid PET 분석까지 1차 논문에 포함할지, 2차 논문으로 분리할지?
4. 외부 에이전트 리뷰에 MRI/엑셀의 실제 내용을 공유해도 되는지, 아니면 요약 통계와 코드만 공유할지?

## 13. 추천 결론

가장 현실적이고 논문성이 높은 1차 계획은 다음이다.

“ADNI T1 MRI 341건에서 사람이 평가한 Hippo_Lt, Hippo_Rt, Front, Parietal, Temporal 점수를 기준으로, 설명 가능한 rule-based/ordinal e-CVRS 0–17점을 자동 산출하고, manual CVRS agreement 및 표준 volumetry 대비 임상 관련성을 비교한다.”

이 설계는 Gemini가 강조한 장점인 빠른 연산, 설명 가능성, 임상 친화성, SaMD 확장 가능성을 살리면서도, 현재 폴더의 실제 자료 구조와 맞는다. Full CVRS 0–30 및 SVD 자동화는 후속 확장으로 두는 것이 안전하다.
