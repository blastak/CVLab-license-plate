# CCPD2019 번호판 각도 분포 히스토그램

CCPD2019 데이터셋의 모든 CSV 파일을 통합하여 번호판 각도 분포를 분석하고 히스토그램으로 시각화합니다.

## 생성된 파일

### 히스토그램 PDF 파일
- `sqrt_method_histogram.pdf` - sqrt(x² + y² + z²) 방식
- `arccos_method_histogram.pdf` - arccos(cos(x) * cos(y) * cos(z)) 방식
- `solvepnp_normal_method_histogram.pdf` - 번호판 법선과 카메라 광축 사이 각도
- `all_methods_comparison.pdf` - 3가지 방법 통합 비교

**주요 특징:**
- 모든 히스토그램의 X축 범위: 0-90도로 통일
- 히스토그램 bin 범위: 0-90도
- 평균(빨강 점선), 중앙값(보라 점선) 표시

## 통계 요약 (전체 305,963개 데이터)

### sqrt_method
- 평균: 36.73°
- 중앙값: 35.45°
- 표준편차: 17.17°
- 범위: 0.17° ~ 130.18°

### arccos_method
- 평균: 35.66°
- 중앙값: 35.06°
- 표준편차: 15.82°
- 범위: 0.17° ~ 90.05°

### solvepnp_normal_method
- 평균: 34.72°
- 중앙값: 34.50°
- 표준편차: 15.12°
- 범위: 0.07° ~ 87.22°

## 사용 방법

### 기본 실행 (PDF 생성)
```bash
python plot_angle_histograms.py
```

### 옵션 지정
```bash
# Bin 개수 변경
python plot_angle_histograms.py --bins 50

# PNG 형식으로 저장
python plot_angle_histograms.py --format png

# 출력 디렉토리 변경
python plot_angle_histograms.py --output_dir ./output

# 데이터 디렉토리 변경
python plot_angle_histograms.py --data_dir /path/to/csv/files
```

### 모든 옵션
```bash
python plot_angle_histograms.py \
    --data_dir ../ \
    --bins 30 \
    --output_dir . \
    --format pdf
```

## 옵션 설명 (plot_angle_histograms.py)

- `--data_dir`: CSV 파일이 있는 디렉토리 경로 (기본값: `../`)
- `--bins`: 히스토그램 bin 개수 (기본값: 30)
- `--output_dir`: 히스토그램 저장 디렉토리 (기본값: `.` 현재 디렉토리)
- `--format`: 출력 파일 형식 - pdf, png, svg 중 선택 (기본값: pdf)

---

## Bin별 샘플 이미지 추출

각 히스토그램 bin에 해당하는 샘플 이미지를 추출하여 폴더별로 정리할 수 있습니다.

### 사용 방법

```bash
# 기본 실행 (arccos, 30 bins, bin당 10개 샘플)
python export_bin_samples.py

# 옵션 지정
python export_bin_samples.py --method arccos --bins 30 --samples 5

# 다른 메소드로 실행
python export_bin_samples.py --method solvepnp --samples 10
```

### 옵션 설명 (export_bin_samples.py)

- `--data_dir`: CSV 파일이 있는 디렉토리 (기본값: `../`)
- `--base_dir`: CCPD2019 이미지 기본 디렉토리 (기본값: `/workspace/DB/01_LicensePlate/CCPD2019`)
- `--method`: 각도 계산 방법 - sqrt, arccos, solvepnp 중 선택 (기본값: arccos)
- `--bins`: bin 개수 (기본값: 30)
- `--samples`: bin당 샘플 개수 (기본값: 10)
- `--output_dir`: 출력 디렉토리 (기본값: `bin_samples`)

### 출력 폴더 구조

```
bin_samples/
└── arccos_bins/
    ├── bin_00_0.0-3.0deg/
    │   ├── 1.95deg_ccpd_weather_0402-0_1-198&461_509&569-...jpg
    │   ├── 2.42deg_ccpd_weather_025-1_1-309&440_549&527-...jpg
    │   └── 2.55deg_ccpd_base_0110201149425-90_86-245&465_437&537-...jpg
    ├── bin_01_3.0-6.0deg/
    │   └── ...
    ├── bin_02_6.0-9.0deg/
    │   └── ...
    └── ...
```

**파일명 형식:** `{각도}deg_{폴더명}_{원본파일명}.jpg`
- 예: `87.93deg_ccpd_fn_0656-19_30-182&471_436&687-...jpg`
- 각도 정보가 파일명에 포함되어 있어 쉽게 확인 가능

## 데이터 소스

다음 8개 CSV 파일의 데이터를 통합:
- ccpd_base_GoodMatches.csv (198,478개)
- ccpd_blur_GoodMatches.csv (8,904개)
- ccpd_challenge_GoodMatches.csv (31,548개)
- ccpd_db_GoodMatches.csv (6,470개)
- ccpd_fn_GoodMatches.csv (16,026개)
- ccpd_rotate_GoodMatches.csv (9,277개)
- ccpd_tilt_GoodMatches.csv (25,558개)
- ccpd_weather_GoodMatches.csv (9,702개)

**총합: 305,963개 데이터**
