import os
import shutil

plate = 'P3'
source_folder = rf'C:\Users\amc85\Workspace\LPR\synthetic_generation\BetaType\korean_character_cropping\03_patch\{plate}'
destination_folder = f'./{plate}'


if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for filename in os.listdir(source_folder):
    source_file = os.path.join(source_folder, filename)

    # 파일인 경우에만 처리
    if os.path.isfile(source_file):
        # 파일 이름으로 폴더 생성
        folder_name = os.path.join(destination_folder, os.path.splitext(filename)[0])
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 파일을 새 폴더로 복사
        shutil.copy(source_file, folder_name)
