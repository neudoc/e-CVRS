# 구현 계획: 자동 e-CVRS T1 위축 점수화(0–17) — 프로토타입에서 애플리케이션까지

> 문서 버전 3.0 · 최종 개정 2026-07-07 · 상태: 원자료 재검증 및 논문 초안 v3 방법론과 정합화
>
> 본 계획은 현재 파이썬 프로토타입(`src/e_cvrs_pipeline.py`, `src/evaluate_pipeline.py`)의 **실제 실행 결과·소스코드를 재검증**하여 대상자 수와 현행 성능을 정정하고, **논문 초안 v3가 기술한 방법론(Nelder–Mead 카파 최대화·부트스트랩 CI·위계적 회귀·FDR)을 재현하기 위해 커밋된 코드에 반영해야 할 격차**를 명시한다.
>
> **⚠ 재현성 최우선 이슈:** 논문 v3의 §2.4는 임계값을 **Nelder–Mead 카파 최대화**로 학습하고 부트스트랩 CI·위계적 회귀·FDR을 보고한다고 기술하나, **현재 커밋된 `evaluate_pipeline.py`는 quantile matching(백분위 임계값)과 Pearson 상관만 수행**한다. 따라서 두 산출물의 카파 수치가 서로 다르다(§0.1). 논문을 재현하려면 평가 스크립트를 v3 방법론으로 상향해야 한다.

---

## 0. 현재 구현 상태 (Ground Truth)

### 0.1 대상자 수 (원자료 재검증)

| 단계 | N | 비고 |
| :--- | :--- | :--- |
| 원본 T1 스캔(.hdr/.img) | 341 | |
| `e-CVRS_automated_scores.csv` 행 수 | 339 | 특징추출 실패 2건: RID 4212(파일 절단·손상 9.5 MB), RID 4764(로딩 실패) |
| 수기 판독(`ADNI_MRI_rating.xlsx`) | 340 | |
| **병합 분석 대상** | **338** | 특징은 있으나 수기 판독 결측 1건(RID 4917) 제외 |

> 이전 판의 "340명 처리 → CSV 340행 → 339명 병합"은 부정확하다. 실제 CSV는 339행, 병합 평가는 **338명**이다. 논문 v3의 "339명" 문구도 338로 정정 필요.

### 0.2 현행 커밋 코드의 실행 결과 (quantile matching 기준선)

`evaluate_pipeline.py`(현행)를 338명에 실행한 결과:

- **일치도(5-fold CV, quantile matching):** 가중 카파 — 좌해마 0.4729 / 우해마 0.3832 / 전두 0.3399 / 두정 0.4349 / 측두 0.4928, 총점 ICC(2,1) 0.5686.
- **논문 v3 보고값(Nelder–Mead 카파 최대화):** 좌해마 0.4669 / 우해마 0.3877 / 전두 0.3419 / 두정 0.3986 / 측두 0.4953, ICC 0.5746. → **두 방법의 수치가 다르며, 커밋 코드로는 논문을 재현할 수 없다.**
- **탐색 프록시 용적 상관:** 좌해마 r=+0.10, 우해마 r=+0.23, 전두 r=−0.12, **두정·측두 r=NaN**(전 z-범위 박스 산출이 퇴화). → 프록시 용적 신뢰 불가, FreeSurfer로 대체.
- **임상 상관(단순 Pearson):** e-CVRS–MMSE r=−0.11, 수기 −0.20. 단, 논문 v3의 **위계적 회귀·FDR은 커밋 코드에 미구현**.

**구현 관점의 진단(무엇을 고쳐야 하는가):**

1. **[최우선] 보정 방식이 논문과 불일치.** 커밋 코드는 quantile matching인데 논문은 Nelder–Mead 카파 최대화. `calibration/`에 카파 최대화 학습을 구현하고, quantile matching은 초기값(θ⁰)으로만 사용하도록 재구성 → 논문 재현의 전제.
2. **위계적 회귀·FDR 미구현.** 논문 v3 Table 3(Model 1–4, 증분 F, FDR)을 코드는 단순 상관으로만 수행 → 회귀 모듈 신규 구현 필요.
3. **부트스트랩 CI 미구현.** 논문은 2,000회 부트스트랩 95% CI를 보고하나 코드에 없음 → `stats/agreement.py`에 추가.
4. **목 제거 단위 버그.** `e_cvrs_pipeline.py`가 `z_limit = int(z_top − 130)`으로 130을 **복셀 수**로 처리 → 물리 130 mm와 어긋남. `int(z_top − round(130/spacing[2]))`로 정정.
5. **자체 통계 구현의 검증 부재.** ICC·가중 카파를 손수 구현 → `pingouin`/`scikit-learn` 대조로 정확성 확인.
6. **하드코딩·재현성 취약점.** 평가 스크립트에 절대경로 기본값(`e:\\...`) 존재, 랜덤시드·파라미터가 코드에 산재, 로깅·테스트·설정 파일 부재.
7. **QC 부재.** Analyze(.hdr/.img) 전용, 실패 케이스(RID 4212/4764/4917 등) 플래그·리포트 없음.
8. **설명가능성 산출물 부재.** XAI를 표방하지만 ROI 오버레이 이미지 등 시각 근거를 저장하지 않음 — 논문 Figure의 핵심 자산 누락.

---

## 1. 설계 원칙 (Design Principles)

- **순수 파이썬·무외부의존 유지:** FSL/ANTs/FreeSurfer 없이 COM 정렬로 구동하는 강점 보존(선택적 FreeSurfer 연동은 비교 검증용으로만).
- **설정 주도(config-driven):** ROI 오프셋·CSF factor·슬라이스 위치를 코드에서 분리해 YAML로 관리 → 실험·재현·앱 배포 용이.
- **누출 없는 학습:** 모든 임계값·보정 파라미터는 훈련 fold 내부에서만 학습.
- **잠금 가능한 아티팩트:** 최종 학습된 임계값/보정기를 JSON/픽클로 직렬화하여, 앱은 재학습 없이 즉시 추론.
- **설명가능 우선:** 모든 점수에 대해 근거 오버레이(측정 슬라이스·ROI·비율)를 함께 생성.
- **크로스플랫폼:** 모든 스크립트는 CLI 인자(`--input_dir`, `--output_csv` 등)와 `pathlib` 사용, 하드코딩 경로 제거.

---

## 2. 목표 아키텍처 (Target Package Architecture)

프로토타입 2개 스크립트를 아래 모듈형 패키지로 리팩터링한다.

```
ecvrs/
├── config/
│   └── default.yaml          # ROI 오프셋, CSF factor, 슬라이스 위치, 시드
├── io/
│   ├── readers.py            # Analyze/NIfTI/DICOM 로더 (nibabel + pydicom)
│   └── writers.py            # CSV/JSON/오버레이 PNG 저장
├── preprocess/
│   ├── reorient.py           # RAS 재정렬
│   ├── neck_strip.py         # z_top-130mm 목 제거
│   ├── normalize.py          # (신규) 백질 피크/z-score 강도 정규화
│   └── com.py                # 질량중심 계산 + QC 플래그
├── features/
│   ├── roi.py                # COM 상대 ROI 정의 (config에서 로드)
│   ├── csf_ratio.py          # 국소 CSF 비율 (MTA/전두/두정/측두)
│   └── proxy_volume.py       # 박스 프록시 용적(내부 점검용, 강등)
├── calibration/
│   ├── quantile.py           # 현행 quantile matching (기준선)
│   └── ordinal.py            # (신규) 순서형 로지스틱/등온회귀 매핑
├── stats/
│   ├── agreement.py          # 가중 카파, ICC(2,1) + 부트스트랩 CI
│   ├── regression.py         # (신규) 위계적 Model 1–4, FDR 보정
│   └── validate.py           # pingouin/sklearn 대조 검증
├── viz/
│   └── overlays.py           # (신규) ROI·측정선·비율 주석 슬라이스 PNG
├── qc/
│   └── checks.py             # COM 실패, ROI 범위 이탈, 이상 강도 플래그
├── cli.py                    # extract / evaluate / render 서브커맨드
└── service/
    └── api.py                # (신규) FastAPI 단일 스캔 추론 엔드포인트
tests/                        # 단위·회귀 테스트
```

> 기존 `src/e_cvrs_pipeline.py`·`src/evaluate_pipeline.py`는 리팩터링 완료 전까지 **호환 유지**하되, 신규 기능은 모듈에 구현하고 스크립트는 얇은 래퍼로 전환한다.

---

## 3. 단계별 작업 계획 (Phased Work Plan)

### Phase A — 견고성·재현성 (즉시)
- **목 제거 단위 버그 수정:** `z_limit = int(z_top − 130)` → `int(z_top − round(130/spacing[2]))` (물리 130 mm). 수정 전후 특징값 변화를 회귀 테스트로 정량화.
- 평가 스크립트의 하드코딩 절대경로 제거, 전 파라미터를 `config/default.yaml`로 이전(CSF factor: MTA 0.25·전두 0.30·두정 0.22·측두 0.30 포함).
- `logging` 도입(진행·경고·QC), `--seed` 인자화(현재 42 고정).
- `requirements.txt`/`pyproject.toml`와 버전 고정, 실행 로그에 라이브러리 버전 기록.
- **QC 모듈:** COM 실패, ROI 경계 이탈, I98 이상치, 빈 마스크를 감지해 subject별 QC 플래그 CSV 생성. **RID 4212(손상)·4764(로딩 실패)·4917(판독 결측)의 제외 사유를 자동 기록**.

### Phase B — 통계 정합화·검증 (논문 재현 필수)
- **[재현 핵심] 카파 최대화 보정 구현:** 훈련 fold에서 **Nelder–Mead simplex로 −κw를 최소화**하는 임계 경계벡터를 학습(quantile matching은 초기값 θ⁰). 이를 검증 fold에 적용해 논문 v3 §2.4를 재현. 현행 quantile-only 결과와 나란히 보고.
- **부트스트랩 95% CI:** 카파·ICC·상관에 2,000회 복원추출 CI 추가.
- **위계적 회귀 모듈 신규 구현:** Model 1–4(공변량 / +수기 CVRS / +e-CVRS / +FreeSurfer 용적) 위계적 OLS, 증분 F검정·조정 R², **Benjamini–Hochberg FDR** 보정. 이분 결과 확장 시 우도비·ΔAUC(DeLong).
- **자체 통계 검증:** 가중 카파·ICC·회귀를 `pingouin`, `sklearn.metrics.cohen_kappa_score(weights='quadratic')`, `statsmodels`와 소수점 4자리까지 대조. 불일치 시 자체 구현 수정.
- **Bland–Altman·혼동행렬** 플롯 산출.

### Phase C — 성능 개선 (연구 프로토콜 §5 연동)
- **보정 고도화:** `calibration/ordinal.py`에 순서형 로지스틱/등온회귀 매핑 구현, quantile 기준선과 nested CV로 비교.
- **강도 정규화·다중 슬라이스·좌우 결합** 특징 실험을 config 토글로 제공.
- **하이퍼파라미터 공동 최적화:** CSF factor·오프셋·ROI 크기를 훈련 fold 내부 nested CV로 탐색(누출 없이).
- **층화 K-fold**와 희소 등급 병합(MTA 3–4) 민감도 실행.
- 각 실험 결과를 표준 리포트(§0 요약표 형식)로 자동 축적.

### Phase D — 설명가능성 산출물 (논문·앱 공통 자산)
- `viz/overlays.py`: 각 부위 측정 슬라이스에 ROI 경계·CSF 마스크·측정 비율·부여 등급을 주석한 PNG를 subject별 저장.
- 판독 보조용 1페이지 요약 이미지(5개 부위 + 총점) 생성.
- 대표 사례(등급별) 갤러리 자동 생성 → 논문 Figure·앱 UI 재사용.

### Phase E — 애플리케이션화 (임상 적용)
- **추론 아티팩트 저장:** 최종 fold 재학습 대신 전체 데이터로 잠금한 임계값/보정기를 `model_artifact.json`으로 직렬화.
- **FastAPI 서비스:** 단일 스캔 업로드 → 점수 + 오버레이 반환 엔드포인트. 배치 모드 CLI 병행.
- **DICOM 입력 지원:** `pydicom`으로 임상 워크플로 대응(Analyze/NIfTI/DICOM 자동 판별).
- **패키징·배포:** Docker 이미지, 헬스체크, 구조적 로깅, 처리시간 측정(초 단위 강점 실증).
- **규제·안전 표기:** "연구용 한정, 비의료기기" 명시. 자동 QC 실패 시 결과 보류·경고.

---

## 4. 인터페이스 사양 (I/O Contracts)

### CLI
```bash
# 특징 추출 (config 주도, 경로 인자화)
python -m ecvrs.cli extract --input_dir <DIR> --config config/default.yaml \
    --output_csv e-CVRS_automated_scores.csv --qc_csv qc_report.csv

# 평가 (일치도 + 위계적 회귀 + 오버레이)
python -m ecvrs.cli evaluate --scores_csv e-CVRS_automated_scores.csv \
    --ratings_excel ADNI_MRI_rating.xlsx --freesurfer_csv <UCSF_FS.csv> \
    --calibration ordinal --seed 42 --output_dir results/

# 단일 스캔 앱 추론 (잠금 아티팩트 사용)
python -m ecvrs.cli render --scan <FILE> --artifact model_artifact.json \
    --output_dir case_out/
```

### 산출물
- `e-CVRS_automated_scores.csv`: RID별 CSF 비율·(강등된) 프록시 용적.
- `qc_report.csv`: subject별 QC 플래그·사유.
- `results/agreement.json`: 카파·ICC + 95% CI, 혼동행렬.
- `results/regression.json`: Model 1–4 지표·FDR 보정 p값.
- `overlays/<RID>_*.png`: 부위별 근거 오버레이, `<RID>_summary.png`.

---

## 5. 검증 계획 (Verification Plan)

### 자동 테스트 (`tests/`)
- **단위:** COM 계산이 알려진 합성 볼륨에서 정확한 중심을 반환. mm→복셀 변환이 spacing에 무관하게 일관.
- **회귀:** 리팩터링 전후 `e-CVRS_automated_scores.csv` 수치가 허용오차 내 동일(현행 339행 기준; 단 목 제거 버그 수정은 의도된 변화로 별도 기록).
- **통계 대조:** 자체 카파/ICC가 `pingouin`/`sklearn`과 소수점 4자리 이내 일치.
- **누출 방지:** 임계값 학습이 훈련 인덱스만 참조함을 단위 테스트로 강제.

### 실행 검증
```bash
python -m ecvrs.cli extract --input_dir . --output_csv e-CVRS_automated_scores.csv --qc_csv qc_report.csv
python -m ecvrs.cli evaluate --scores_csv e-CVRS_automated_scores.csv \
    --ratings_excel ADNI_MRI_rating.xlsx --calibration ordinal --output_dir results/
```

### 수기 확인
- 교차검증 카파가 출력되고 데이터 누출로 부풀지 않았는지 확인(기준선 대비 상승 폭이 nested CV에서도 유지되는지).
- 오버레이 PNG가 실제 해부 표적을 포착하는지 표본 육안 검수.
- QC 리포트에서 특징추출 실패 2건(RID 4212·4764) 및 판독 결측 1건(RID 4917)의 341→339→338 계보를 확인·문서화.
- 개선(Phase C) 전후 §0 요약표 갱신 및 목표(ICC ≥ 0.80, κw ≥ 0.60) 대비 진척 기록.

---

## 6. 우선순위 요약 (What to do first)

1. **Phase A + B**: 경로·설정·로깅 정리 및 통계 정합화·검증 → 논문 신뢰성 확보의 최소 요건.
2. **Phase D 오버레이**: 저비용·고효과. XAI 주장을 실증하는 즉시 산출물.
3. **Phase C 보정 고도화**: 성능(카파·ICC) 실질 개선의 핵심.
4. **Phase E**: 성능이 임상 유용 수준에 근접하면 서비스화.
