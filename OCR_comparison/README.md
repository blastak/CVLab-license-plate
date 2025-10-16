# OCR Comparison

번호판 검출기(VIN_LPD, IWPOD-tf)별 OCR 성능 비교 파이프라인

## 개요

이 프로젝트는 서로 다른 번호판 검출기의 검출 결과에 대해 OCR을 수행하여 성능을 비교합니다.

### 전체 파이프라인

```
이미지 → 검출(VIN_LPD/IWPOD-tf) → CSV 저장 → Frontalization → OCR → 성능 비교
```

### 주요 단계

1. **Step 1: CSV 생성** - 검출기로 번호판 위치 검출 및 CSV 저장
2. **Step 2: Frontalization** - Homography를 이용한 번호판 정면화
3. **Step 3: OCR 평가** - VIN_OCR 실행 및 GT와 비교

## 사용 방법

### 방법 1: 통합 스크립트 실행 (권장)

```bash
cd OCR_comparison

python run_all.py \
    --image_folder /workspace/repo/ultralytics/ultralytics/assets/good_all_obb1944/images/test \
    --json_folder /workspace/repo/ultralytics/ultralytics/assets/good_all_obb1944/labels/test_original \
    --output_dir ./output
```

#### 옵션

- `--image_folder`: 입력 이미지 폴더 경로 (필수)
- `--json_folder`: GT JSON 폴더 경로 (필수)
- `--output_dir`: 출력 디렉토리 (기본값: ./output)
- `--detectors`: 실행할 검출기 목록 (기본값: VIN_LPD IWPOD_tf)
- `--vinlpd_model`: VIN_LPD 모델 경로 (기본값: ../LP_Detection/VIN_LPD/weight)
- `--iwpod_model`: IWPOD-tf 모델 경로 (기본값: ../LP_Detection/IWPOD_tf/weights/iwpod_net)
- `--ocr_model`: VIN_OCR 모델 경로 (기본값: ../LP_Recognition/VIN_OCR/weight)
- `--skip_csv`: CSV 생성 단계 스킵
- `--skip_frontalization`: Frontalization 단계 스킵

### 방법 2: 단계별 실행

#### Step 1: CSV 생성

```bash
# VIN_LPD
python step1_generate_csv.py \
    --image_folder /path/to/images \
    --output_csv_folder ./output/csv/VIN_LPD \
    --detector VIN_LPD \
    --model_path ../LP_Detection/VIN_LPD/weight

# IWPOD-tf
python step1_generate_csv.py \
    --image_folder /path/to/images \
    --output_csv_folder ./output/csv/IWPOD_tf \
    --detector IWPOD_tf \
    --model_path ../LP_Detection/IWPOD_tf/weights/iwpod_net
```

#### Step 2: Frontalization

```bash
# VIN_LPD
python step2_frontalization.py \
    --image_folder /path/to/images \
    --csv_folder ./output/csv/VIN_LPD \
    --output_folder ./output/frontalized \
    --detector_name VIN_LPD

# IWPOD-tf
python step2_frontalization.py \
    --image_folder /path/to/images \
    --csv_folder ./output/csv/IWPOD_tf \
    --output_folder ./output/frontalized \
    --detector_name IWPOD_tf
```

#### Step 3: OCR 평가

```bash
python step3_ocr_evaluation.py \
    --frontalized_folder ./output/frontalized \
    --json_folder /path/to/json \
    --ocr_model_path ../LP_Recognition/VIN_OCR/weight \
    --output_csv ./output/ocr_results.csv \
    --detectors VIN_LPD IWPOD_tf
```

## 출력 구조

```
output/
├── csv/
│   ├── VIN_LPD/           # VIN_LPD 검출 결과 CSV
│   │   └── *.csv
│   └── IWPOD_tf/          # IWPOD-tf 검출 결과 CSV
│       └── *.csv
├── frontalized/
│   ├── VIN_LPD/           # VIN_LPD frontalized 이미지
│   │   ├── P1-1/
│   │   ├── P1-2/
│   │   └── ...
│   └── IWPOD_tf/          # IWPOD-tf frontalized 이미지
│       └── P0/
└── ocr_results.csv        # 최종 OCR 평가 결과
```

## CSV 형식

### 검출 결과 CSV (step1_generate_csv.py 출력)

```csv
class,x1,y1,x2,y2,x3,y3,x4,y4,conf
P1-1,100.5,200.3,250.7,202.1,248.9,280.5,98.3,278.7,0.95
```

- 좌표: 좌상단부터 시계방향으로 4개 꼭지점
- class: 번호판 클래스 (VIN_LPD) 또는 'P0' (IWPOD-tf)
- conf: 검출 신뢰도

### OCR 평가 결과 CSV (step3_ocr_evaluation.py 출력)

```csv
detector,filename,det_idx,gt_class,gt_text,pred_text,match
VIN_LPD,image001,0,P1-1,12가3456,12가3456,True
IWPOD_tf,image001,0,P1-1,12가3456,12가3457,False
```

## Frontalization 크기

클래스별 정면화된 번호판 크기:

- **P1 계열** (P1-1, P1-2, P1-3, P1-4): 520x110
- **P2**: 440x200
- **P3**: 440x220
- **P4**: 520x110
- **P5**: 335x170
- **P6**: 335x170
- **P0** (IWPOD): 520x110

## GT JSON 형식

```json
{
  "shapes": [
    {
      "label": "P1-1_12가3456",
      "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

- `label`: `{클래스}_{번호판번호}` 형식
- `points`: 번호판 4개 꼭지점 좌표

## 외부 데이터 사용

다른 프로젝트에서 생성한 이미지와 CSV를 사용하려면:

1. 이미지를 준비
2. Step 2부터 실행 (CSV가 이미 있는 경우)
3. 또는 Step 3부터 실행 (frontalized 이미지가 이미 있는 경우)

```bash
# CSV가 이미 있는 경우
python run_all.py \
    --image_folder /path/to/external/images \
    --json_folder /path/to/external/json \
    --output_dir ./output \
    --skip_csv

# Frontalized 이미지가 이미 있는 경우
python step3_ocr_evaluation.py \
    --frontalized_folder /path/to/frontalized \
    --json_folder /path/to/json \
    --ocr_model_path ../LP_Recognition/VIN_OCR/weight \
    --output_csv ./results.csv \
    --detectors VIN_LPD IWPOD_tf
```

## 주의사항

1. **환경 설정**: CLAUDE.md의 개발 환경 설정 참조
2. **GPU 메모리**: 대량 이미지 처리 시 메모리 부족 주의
3. **경로 설정**: 상대 경로 사용 시 작업 디렉토리 확인
4. **CSV 형식**: 좌표가 좌상단부터 시계방향 순서인지 확인
