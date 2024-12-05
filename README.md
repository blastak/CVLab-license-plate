# CVLab-license-plate


## 개발 환경 구축

### conda 세팅

```bash
conda create -n CVLab-license-plate python=3.9
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
pip install "tensorflow<2.11"
```

* 2.11 부터는 윈도우 native에서 gpu 지원 안함

#### CPU 확인

```bash
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

##### 에러해결

```bash
pip install numpy==1.22
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
pip install opencv-python==4.10.0.84
```

### PyTorch with CUDA 11.8

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
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
pip install PyQt5
```
