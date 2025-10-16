"""
Step 2: Frontalization 스크립트
CSV 파일을 읽어서 클래스별 크기로 Homography warping 수행
저장 구조: frontalized/{detector_name}/{class}/filename_idx.jpg
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from Utils import imread_uni

# 클래스별 출력 크기 정의 (Graphical Model 기준)
PLATE_SIZES = {
    'P1-1': (520, 110), 'P1-2': (520, 110), 'P1-3': (520, 110), 'P1-4': (520, 110),
    'P1': (520, 110),  # P1 병합된 경우
    'P2': (440, 200),
    'P3': (440, 220),
    'P4': (520, 110),
    'P5': (335, 170),
    'P6': (335, 170),
    'P0': (520, 110),  # IWPOD의 경우 기본 크기
}


def get_plate_size(plate_class):
    """클래스에 따른 번호판 크기 반환"""
    # P1 계열은 모두 같은 크기
    if plate_class.startswith('P1'):
        return PLATE_SIZES['P1-1']
    return PLATE_SIZES.get(plate_class, (520, 110))


def apply_homography(img, src_points, dst_size):
    """
    Homography를 이용한 perspective warping 수행

    Args:
        img: 원본 이미지
        src_points: 원본 이미지의 4개 꼭지점 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dst_size: 목표 크기 (width, height)

    Returns:
        warped: frontalized된 이미지
    """
    src_points = np.array(src_points, dtype=np.float32)

    # 목표 사각형 좌표 (좌상단부터 시계방향)
    width, height = dst_size
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Homography 행렬 계산
    H = cv2.getPerspectiveTransform(src_points, dst_points)

    # Perspective warping 적용
    warped = cv2.warpPerspective(img, H, dst_size)

    return warped


def frontalize_from_csv(image_folder, csv_folder, output_folder, detector_name):
    """
    CSV 파일을 읽어서 frontalization 수행

    Args:
        image_folder: 원본 이미지 폴더 경로
        csv_folder: CSV 폴더 경로
        output_folder: 출력 폴더 경로
        detector_name: 검출기 이름 (VIN_LPD, IWPOD_tf 등)
    """
    image_folder = Path(image_folder)
    csv_folder = Path(csv_folder)
    output_folder = Path(output_folder)

    # CSV 파일 목록 가져오기
    csv_files = sorted([f for f in csv_folder.iterdir() if f.suffix == '.csv'])

    print(f"총 {len(csv_files)}개 CSV 파일 처리 시작...")

    total_plates = 0

    for csv_idx, csv_file in enumerate(csv_files):
        # 해당하는 이미지 파일 찾기
        img_name = csv_file.stem
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_folder / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            print(f"경고: {img_name}에 해당하는 이미지 파일을 찾을 수 없습니다.")
            continue

        # 이미지 로드
        img = imread_uni(str(img_file))

        # CSV 파일 읽기
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            detections = list(reader)

        # 각 검출 결과에 대해 frontalization 수행
        for det_idx, row in enumerate(detections):
            if len(row) != 10:
                continue

            plate_class = row[0]
            x1, y1 = float(row[1]), float(row[2])
            x2, y2 = float(row[3]), float(row[4])
            x3, y3 = float(row[5]), float(row[6])
            x4, y4 = float(row[7]), float(row[8])
            conf = float(row[9])

            src_points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

            # 클래스별 목표 크기
            dst_size = get_plate_size(plate_class)

            # Homography warping 적용
            try:
                warped = apply_homography(img, src_points, dst_size)

                # 출력 경로 생성: frontalized/{detector_name}/{class}/
                output_class_folder = output_folder / detector_name / plate_class
                output_class_folder.mkdir(parents=True, exist_ok=True)

                # 파일명: {원본파일명}_{검출인덱스}.jpg
                output_filename = output_class_folder / f"{img_name}_{det_idx}.jpg"
                cv2.imwrite(str(output_filename), warped)

                total_plates += 1

            except Exception as e:
                print(f"경고: {img_name}의 {det_idx}번째 검출 결과 처리 실패: {e}")
                continue

        if (csv_idx + 1) % 100 == 0:
            print(f"  진행률: {csv_idx + 1}/{len(csv_files)}, 총 {total_plates}개 번호판 처리됨")

    print(f"\nFrontalization 완료!")
    print(f"  처리된 CSV 파일: {len(csv_files)}개")
    print(f"  생성된 번호판 이미지: {total_plates}개")
    print(f"  출력 폴더: {output_folder / detector_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSV 파일로부터 번호판 frontalization 수행')
    parser.add_argument('--image_folder', type=str, required=True, help='원본 이미지 폴더 경로')
    parser.add_argument('--csv_folder', type=str, required=True, help='CSV 폴더 경로')
    parser.add_argument('--output_folder', type=str, required=True, help='출력 폴더 경로')
    parser.add_argument('--detector_name', type=str, required=True, help='검출기 이름 (예: VIN_LPD, IWPOD_tf)')

    args = parser.parse_args()

    frontalize_from_csv(args.image_folder, args.csv_folder, args.output_folder, args.detector_name)
