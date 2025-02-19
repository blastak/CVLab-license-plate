import os
from datetime import datetime, timedelta
from pathlib import Path

from natsort import natsorted

p_in = Path(r"E:\Dataset\01_LicensePlate\53_Suwon_\20200806")  # <------ 여기 직접 입력해야함
st = datetime(2020,8,6,9,4,5)  # 기준 시간 (user edit) <---- 여기 직접 입력해야함
fps = 33.3  # 프레임 변화율 (user edit) <---- 여기 직접 입력해야함

### test 구문 - 시작
# frame_num = 5567
# dt = st + timedelta(milliseconds=int(frame_num * fps))
# print(datetime.strftime(dt, "%Y%m%d_%H%M%S_%f")[:-3])
# assert input() == 'y'
### test 구문 - 끝

img_paths = natsorted(p.absolute() for p in p_in.glob('**/*.jpg'))

for img_path in img_paths:
    filename = img_path.stem  # 순수 파일명 (확장자 제외)

    hh = int(filename[-10:-8])
    mm = int(filename[-8:-6])
    ss = int(filename[-6:-4])
    zzz = int(filename[-3:])
    dt = st + timedelta(hours=hh,minutes=mm, seconds=ss, milliseconds=zzz)

    # frame_num = int(filename)  # 프레임 넘버로 되어있는 파일명을
    # dt = st + timedelta(milliseconds=int(frame_num * fps))  # 기준 시간에서 delta 초 만큼 변화시키고

    new_filename = datetime.strftime(dt, "%Y%m%d_%H%M%S_%f")[:-3]  # YYYYMMDD_hhmmss_zzz 형태로 반환

    # xml_path = p_in / (filename + '.xml')  # 기존 xml 파일명
    # xml_path_new = p_in / (new_filename + '.xml')  # 새로운 xml 파일명
    # os.rename(xml_path, xml_path_new)

    img_path_new = p_in / (new_filename + '.jpg')  # 새로운 이미지 파일명
    os.rename(img_path, img_path_new)
