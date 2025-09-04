# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 레포지토리에서 작업할 때 참고할 지침을 제공합니다.

## 프로젝트 개요

CVLab 차량 번호판 인식 프로젝트 - 한국, 중국 번호판 검출, 인식, 추적 및 합성을 위한 컴퓨터 비전 파이프라인

## 개발 환경 설정

### 필수 환경
- Python 3.10 (conda 환경)
- CUDA 11.2 + cuDNN 8.1.0
- TensorFlow < 2.11 (IWPOD-tf용)
- PyTorch 2.1.1 with CUDA 11.8

### 환경 구축 명령어

```bash
# Conda 환경 생성 및 활성화
conda create -n CVLab-license-plate python=3.10
conda activate CVLab-license-plate

# CUDA 설정
conda update -n base -c defaults conda
conda config --add channels defaults
conda config --add channels conda-forge
conda install cudatoolkit=11.2 cudnn=8.1.0

# 주요 패키지 설치
pip install --upgrade pip
pip install "tensorflow<2.11" numpy==1.22
pip install numpy==1.22 opencv-python==4.10.0.84
pip install numpy==1.22 torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install selenium webdriver_manager pyautogui pyperclip pytubefix
pip install matplotlib scipy natsort pyqt5 pyqt5-tools tqdm gradio pyffx
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
pip install albumentations numpy==1.22.0
conda install -c conda-forge ffmpeg
```

### 환경 검증

```bash
# TensorFlow GPU 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# PyTorch GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 프로젝트 구조

### 주요 모듈
- **LP_Detection**: 번호판 검출 (IWPOD-tf, IWPOD-torch, VIN_LPD, VIN_LPD_ONNX)
- **LP_Recognition**: 번호판 문자 인식 (VIN_OCR, VIN_OCR_ONNX)
- **LP_Swapping**: GAN 기반 번호판 합성/교체
- **LP_Tracking**: 다중 객체 추적 (Kalman Filter 기반)
- **Data_Collection**: 웹 크롤링, 유튜브 다운로드, 비디오 샘플링
- **Data_Labeling**: 라벨링 및 그래픽 모델 생성
- **Demo**: Gradio 웹 데모 및 비디오 플레이어

### 핵심 유틸리티 (Utils/__init__.py)
- `imread_uni`: 유니코드 경로 이미지 읽기
- `bd_eng2kor_v1p3`: 영문-한글 번호판 문자 변환 테이블
- `trans_eng2kor_v1p3/trans_kor2eng_v1p3`: 번호판 문자 변환
- `add_text_with_background`: 한글 텍스트 오버레이

## 실행 명령어

### 웹 데모 실행
```bash
cd Demo
export PYTHONPATH=$PWD:$PWD/..
python web_demo.py
```

### 번호판 검출 (VIN_LPD)
```python
from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from Utils import imread_uni

d_net = load_model_VinLPD('./weight')
img = imread_uni('image_path.jpg')
detections = d_net.forward(img)[0]
```

### 번호판 인식 (VIN_OCR)
```python
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import trans_eng2kor_v1p3

o_net = load_model_VinOCR('./weight')
# 검출된 번호판 영역에 대해 OCR 수행
ocr_results = o_net.forward(plate_img)
korean_text = trans_eng2kor_v1p3(ocr_results)
```

### 번호판 합성/교체
```python
from LP_Swapping.swap import Swapper

swapper = Swapper('checkpoint_path')
# A: 원본 이미지, B: 타겟 이미지, M: 마스크
result = swapper.swap(A, B, M)
```

## 데이터 형식

### 라벨 파일 형식
- **JSON**: LabelMe 형식 (polygon 좌표, 번호판 타입 및 번호)
- **XML**: Pascal VOC 형식 (bounding box)
- **CSV**: 검출 결과 (class, 좌표, confidence)

### 번호판 타입 분류
- **P1-1**: 일반 승용차 (2006년 이후, 흰색)
- **P1-2**: 전기차 (파란색)
- **P1-3**: 영업용 (노란색)
- **P1-4**: 렌터카 (허/하/호)
- **P2**: 구형 승용차 (2006년 이전, 초록색)
- **P3**: 대형 화물차 (지역명 포함)
- **P4**: 대형 버스 (지역명 포함)
- **P5**: 이륜차 (2024년 이전)
- **P6**: 이륜차 (2024년 이후)

## 주의사항

1. **numpy 버전**: 반드시 numpy==1.22 유지 (TensorFlow 호환성)
2. **Git 파일 권한**: `git config core.filemode false` 설정으로 권한 변경 무시
3. **PYTHONPATH**: Demo 실행 시 반드시 프로젝트 루트 포함
4. **GPU 메모리**: 대량 처리 시 배치 크기 조절 필요
5. **한글 처리**: PIL 폰트 사용, UTF-8 인코딩 필수

## 테스트 데이터

- 검출 테스트: `LP_Detection/sample_image/`
- 인식 테스트: `LP_Recognition/sample_image/`
- 라벨 샘플: `Data_Labeling/Dataset_Loader/sample_image_label/`