import os
from datetime import datetime, timedelta
from pathlib import Path

from natsort import natsorted

st = datetime(2018, 8, 2, 18, 17, 20, 0)  # 기준 시간 (user edit) <---- 여기 직접 입력해야함
fps = 40  # 프레임 변화율 (user edit) <---- 여기 직접 입력해야함

### test 구문 - 시작
frame_num = 33916
dt = st + timedelta(milliseconds=int(frame_num * fps))
print(datetime.strftime(dt, "%Y%m%d_%H%M%S_%f")[:-3])
assert input() == 'y'
### test 구문 - 끝

p_in = Path(r"E:\Dataset\01_LicensePlate\52_Namyangju_823\52")  # <------ 여기 직접 입력해야함
img_paths = natsorted(p.absolute() for p in p_in.glob('**/*.jpg'))

for img_path in img_paths:
    filename = img_path.stem  # 순수 파일명 (확장자 제외)

    frame_num = int(filename)  # 프레임 넘버로 되어있는 파일명을
    dt = st + timedelta(milliseconds=int(frame_num * fps))  # 기준 시간에서 delta 초 만큼 변화시키고
    new_filename = datetime.strftime(dt, "%Y%m%d_%H%M%S_%f")[:-3]  # YYYYMMDD_hhmmss_zzz 형태로 반환

    img_path_new = p_in / (new_filename + '.jpg')  # 새로운 이미지 파일명
    os.rename(img_path, img_path_new)

    xml_path = p_in / (filename + '.xml')  # 기존 xml 파일명
    xml_path_new = p_in / (new_filename + '.xml')  # 새로운 xml 파일명
    os.rename(xml_path, xml_path_new)
