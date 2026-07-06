# 구현 계획: 자동 e-CVRS T1 위축 점수화(0–17) — 프로토타입에서 애플리케이션까지

> 문서 버전 2.0 · 최종 개정 2026-07-06
>
> 본 계획은 현재 파이썬 프로토타입(`src/e_cvrs_pipeline.py`, `src/evaluate_pipeline.py`)의 **실제 실행 결과를 반영**하고, 이를 (a) 재현 가능한 연구 코드베이스와 (b) 임상 스크리닝·판독 보조 **애플리케이션**으로 발전시키기 위한 로드맵을 제시한다.

---

## 0. 현재 구현 상태 (Ground Truth)

파이프라인은 실제로 340명 스캔을 처리해 `e-CVRS_automated_scores.csv`(340행)를 생성하고, 평가 스크립트는 339명 병합 후 다음을 산출한다.

- **일치도(5-fold CV):** 가중 카파 — 좌해마 0.47 / 우해마 0.37 / 전두 0.33 / 두정 0.41 / 측두 0.51, 총점 ICC(2,1) 0.56.
- **탐색 프록시 용적 상관:** 약하거나 부호 역전(전두 −0.19, 측두 −0.33) — 프록시 용적 신뢰 불가.
- **임상 상관:** e-CVRS–MMSE −0.11, 수기 −0.20 — 현재 e-CVRS가 수기보다 약함(범위 제한 코호트 영향).

**구현 관점의 진단(무엇을 고쳐야 하는가):**

1. **평가 스크립트가 프로토콜과 불일치.** 프로토콜은 위계적 회귀 Model 1–4를 명시했으나 코드는 단순 상관만 수행 → 회귀 모듈 신규 구현 필요.
2. **자체 통계 구현의 검증 부재.** ICC·가중 카파를 손수 구현했으므로 `pingouin`/`scikit-learn` 대조로 정확성 확인 필요.
3. **보정 방식이 성능 상한을 제약.** quantile matching은 분포만 맞추고 순위 정확도를 최적화하지 못함(연구 프로토콜 §5 참조).
4. **하드코딩·재현성 취약점.** 평가 스크립트에 절대경로 기본값(`e:\\...`) 존재, 랜덤시드·파라미터가 코드에 산재, 로깅·테스트·설정 파일 부재.
5. **단일 포맷·QC 부재.** Analyze(.hdr/.img) 전용, 실패 케이스(COM 실패·ROI 범위 이탈) 플래그·리포트 없음.
6. **설명가능성 산출물 부재.** XAI를 표방하지만 ROI 오버레이 이미지 등 시각 근거를 저장하지 않음 — 논문·앱 양쪽의 핵심 자산 누락.

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
- 평가 스크립트의 하드코딩 절대경로 제거, 전 파라미터를 `config/default.yaml`로 이전.
- `logging` 도입(진행·경고·QC), `--seed` 인자화(현재 42 고정).
- `requirements.txt`/`pyproject.toml`와 버전 고정, 실행 로그에 라이브러리 버전 기록.
- **QC 모듈:** COM 계산 실패, ROI가 영상 경계를 벗어남, I98 이상치, 빈 마스크를 감지해 subject별 QC 플래그 CSV 생성. (현재 340/341만 처리된 원인 추적 포함.)

### Phase B — 통계 정합화·검증 (논문 필수)
- **회귀 모듈 신규 구현:** 프로토콜 Model 1–4(공변량 / +수기 CVRS / +e-CVRS / +FreeSurfer 용적) 위계적 회귀, 조정 R²·우도비·ΔAUC(DeLong), **Benjamini–Hochberg FDR** 보정.
- **자체 통계 검증:** 가중 카파·ICC를 `pingouin.intraclass_corr`, `sklearn.metrics.cohen_kappa_score(weights='quadratic')`로 대조. 불일치 시 자체 구현 수정.
- **부트스트랩 95% CI**를 카파·ICC·상관에 추가.
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
- **회귀:** 리팩터링 전후 `e-CVRS_automated_scores.csv` 수치가 허용오차 내 동일(현행 340행 기준).
- **통계 대조:** 자체 카파/ICC가 `pingouin`/`sklearn`과 소수점 3자리 이내 일치.
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
- QC 리포트에서 누락 1명(341→340) 원인 확인·문서화.
- 개선(Phase C) 전후 §0 요약표 갱신 및 목표(ICC ≥ 0.80, κw ≥ 0.60) 대비 진척 기록.

---

## 6. 우선순위 요약 (What to do first)

1. **Phase A + B**: 경로·설정·로깅 정리 및 통계 정합화·검증 → 논문 신뢰성 확보의 최소 요건.
2. **Phase D 오버레이**: 저비용·고효과. XAI 주장을 실증하는 즉시 산출물.
3. **Phase C 보정 고도화**: 성능(카파·ICC) 실질 개선의 핵심.
4. **Phase E**: 성능이 임상 유용 수준에 근접하면 서비스화.
