import os
import hashlib
from Data_Collection.Duplicate_Checker import calc_file_hash

def remove_duplicate_images(path):
    """폴더 내에서 해시값이 같은 중복 이미지 파일 삭제"""
    hash_dict = {}
    duplicates = []

    # 디렉토리 내 모든 파일 순회
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # 파일의 해시값 계산
            file_hash = calc_file_hash(file_path)

            # 해시값이 이미 존재하는 경우, 중복 파일로 처리
            if file_hash in hash_dict:
                duplicates.append(file_path)  # 중복 파일 리스트에 추가
            else:
                hash_dict[file_hash] = file_path  # 해시값을 키로 파일 경로 저장

    # 중복된 파일을 삭제하거나 다른 처리를 하려면 아래에서 작업을 추가
    for duplicate in duplicates:
        print(f"중복 파일 발견: {duplicate}")
        os.remove(duplicate)

if __name__ == "__main__":
    folder_path = r'D:\Dataset\LicensePlate\for_p5p6\extract\HR'
    remove_duplicate_images(folder_path)