# CVLab-license-plate


## 개발 환경 구축

### conda 세팅

`conda create -n CVLab-license-plate python=3.9`

`conda activate CVLab-license-plate`

### cuda 세팅 (by conda)

![image](https://github.com/user-attachments/assets/7531b017-16f9-472d-800c-c1ef55f94a99)


~~`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`~~


위에 명령이 잘 안먹힘. 아래걸로 conda update해서 사용

`conda update -n base -c defaults conda`

`conda config --add channels defaults`

`conda config --add channels conda-forge`

`conda install cudatoolkit=11.2 cudnn=8.1.0`

    
### tensorflow 세팅 (by pip) for IWPOD-tf

`pip install --upgrade pip`

`pip install "tensorflow<2.11"`

* 2.11 부터는 윈도우 native에서 gpu 지원 안함

#### CPU 확인

`python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`

##### 에러해결

`pip install numpy==1.22`

#### GPU 확인

`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### 웹크롤링 세팅

`pip install selenium webdriver_manager pyautogui pyperclip`

### 유튜브 4K 비디오 다운로더 세팅

`pip install pytubefix`

### 한영변환표

`pip install bidict`


