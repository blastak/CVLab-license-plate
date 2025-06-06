# CVLab-license-plate


## 개발 환경 구축

### conda 세팅

```bash
conda create -n CVLab-license-plate python=3.10
```
```bash
conda activate CVLab-license-plate
```

### cuda 세팅 (by conda)

![image](https://github.com/user-attachments/assets/7531b017-16f9-472d-800c-c1ef55f94a99)


~~`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`~~   <-- 이 명령이 잘 안먹힘. 아래 4줄로 해결

```bash
conda update -n base -c defaults conda
conda config --add channels defaults
conda config --add channels conda-forge
conda install cudatoolkit=11.2 cudnn=8.1.0
```

### tensorflow 세팅 (by pip) for IWPOD-tf

```bash
pip install --upgrade pip
pip install "tensorflow<2.11" numpy==1.22
```

* 2.11 부터는 윈도우 native에서 gpu 지원 안함

#### CPU 확인

```bash
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

#### GPU 확인

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 웹크롤링 세팅

```bash
pip install selenium webdriver_manager pyautogui pyperclip
```

### 유튜브 4K 비디오 다운로더 세팅

```bash
pip install pytubefix
```

### 한영변환표

```bash
pip install bidict
```

### OpenCV

```bash
pip install numpy==1.22 opencv-python==4.10.0.84
```

### PyTorch with CUDA 11.8

```bash
pip install numpy==1.22 torch==2.1.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

### MatPlotLib & SciPy

```bash
pip install numpy==1.22 matplotlib scipy
```

### Natsort

```bash
pip install natsort
```

### PyQt5

```bash
pip install pyqt5 pyqt5-tools
```

### tqdm

```bash
pip install tqdm
```

### gradio

```bash
pip install gradio
```

### ffmpeg 설치
```bash
conda install -c conda-forge ffmpeg
```

### onnxruntime-gpu for cuda11.x

```bash
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

### pyffx 설치
```bash
pip install pyffx
```

### albumentations 설치
```bash
pip install albumentations numpy==1.22.0
```

---
---


# 귀찮으면 

환경 만든 후에 한줄씩 복붙

* ~~conda install cudatoolkit=11.2 cudnn=8.1.0~~

* ~~pip install --upgrade pip~~

* ~~pip install "tensorflow<2.11" numpy==1.22 opencv-python==4.10.0.84 selenium webdriver_manager pyautogui pyperclip pytubefix bidict matplotlib scipy natsort pyqt5 pyqt5-tools tqdm gradio pyffx~~

* ~~pip install numpy==1.22 torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118~~

* ~~pip install numpy==1.22 onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/~~

* ~~conda install -c conda-forge ffmpeg~~


