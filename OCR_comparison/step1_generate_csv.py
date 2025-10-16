"""
Step 1: CSV 생성 스크립트
이미지 폴더 경로를 입력받아 검출기(VIN_LPD 또는 IWPOD-tf)를 돌려서 CSV 파일 생성
CSV 형식: [class, x1, y1, x2, y2, x3, y3, x4, y4, conf] (좌상단부터 시계방향)
"""

import argparse
import csv
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import load_model_tf, find_lp_corner
from Utils import imread_uni


def ensure_clockwise_from_topleft(points):
    """
    4개의 꼭지점을 좌상단부터 시계방향으로 정렬
    points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    번호판은 가로로 긴 직사각형이므로:
    1. y 좌표로 상단/하단 구분 (y가 작을수록 상단)
    2. 상단 2개 중 x가 작은 것이 좌상단, 큰 것이 우상단
    3. 하단 2개 중 x가 작은 것이 좌하단, 큰 것이 우하단
    """
    import numpy as np

    points = np.array(points, dtype=np.float32)

    # y 좌표로 상단/하단 구분 (y가 작을수록 상단)
    y_coords = points[:, 1]

    # 상단 2개 점 (y가 작은 2개)
    top_indices = np.argsort(y_coords)[:2]
    top_points = points[top_indices]

    # 하단 2개 점 (y가 큰 2개)
    bottom_indices = np.argsort(y_coords)[2:]
    bottom_points = points[bottom_indices]

    # 상단 점들 중 x가 작은 것이 좌상단, 큰 것이 우상단
    if top_points[0, 0] < top_points[1, 0]:
        top_left = top_points[0]
        top_right = top_points[1]
    else:
        top_left = top_points[1]
        top_right = top_points[0]

    # 하단 점들 중 x가 큰 것이 우하단, 작은 것이 좌하단
    if bottom_points[0, 0] > bottom_points[1, 0]:
        bottom_right = bottom_points[0]
        bottom_left = bottom_points[1]
    else:
        bottom_right = bottom_points[1]
        bottom_left = bottom_points[0]

    # 좌상단 → 우상단 → 우하단 → 좌하단 순서로 정렬
    ordered_points = np.array([top_left, top_right, bottom_right, bottom_left])

    return ordered_points.tolist()


def generate_csv_vinlpd(image_folder, output_csv_folder, model_path):
    """
    VIN_LPD 검출기를 사용하여 CSV 생성
    """
    image_folder = Path(image_folder)
    output_csv_folder = Path(output_csv_folder)
    output_csv_folder.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print(f"VIN_LPD 모델 로딩 중: {model_path}")
    d_net = load_model_VinLPD(model_path)

    # 이미지 파일 목록 가져오기
    img_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    print(f"총 {len(img_paths)}개 이미지 처리 시작...")

    for idx, img_path in enumerate(img_paths):
        img = imread_uni(str(img_path))
        detections = d_net.forward(img)[0]

        # CSV 데이터 준비
        csv_data = []
        for det in detections:
            # Bounding Box를 4개 꼭지점으로 변환 (좌상단부터 시계방향)
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.w, det.y
            x3, y3 = det.x + det.w, det.y + det.h
            x4, y4 = det.x, det.y + det.h

            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            points = ensure_clockwise_from_topleft(points)

            row = [det.class_str,
                   points[0][0], points[0][1],
                   points[1][0], points[1][1],
                   points[2][0], points[2][1],
                   points[3][0], points[3][1],
                   det.conf]
            csv_data.append(row)

        # CSV 파일 저장
        csv_filename = output_csv_folder / f"{img_path.stem}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in csv_data:
                writer.writerow(row)

        if (idx + 1) % 100 == 0:
            print(f"  진행률: {idx + 1}/{len(img_paths)}")

    print(f"VIN_LPD CSV 생성 완료: {output_csv_folder}")


def generate_csv_iwpod_tf(image_folder, output_csv_folder, model_path):
    """
    IWPOD-tf 검출기를 사용하여 CSV 생성
    """
    image_folder = Path(image_folder)
    output_csv_folder = Path(output_csv_folder)
    output_csv_folder.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print(f"IWPOD-tf 모델 로딩 중: {model_path}")
    iwpod_net = load_model_tf(model_path)

    # 이미지 파일 목록 가져오기
    img_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    print(f"총 {len(img_paths)}개 이미지 처리 시작...")

    for idx, img_path in enumerate(img_paths):
        img = imread_uni(str(img_path))
        xys_list, prob_list = find_lp_corner(img, iwpod_net)

        # CSV 데이터 준비
        csv_data = []
        for xys, prob in zip(xys_list, prob_list):
            # xys는 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태
            points = ensure_clockwise_from_topleft(xys)

            row = ['P0',  # IWPOD는 클래스를 구분하지 않음
                   points[0][0], points[0][1],
                   points[1][0], points[1][1],
                   points[2][0], points[2][1],
                   points[3][0], points[3][1],
                   prob]
            csv_data.append(row)

        # CSV 파일 저장
        csv_filename = output_csv_folder / f"{img_path.stem}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in csv_data:
                writer.writerow(row)

        if (idx + 1) % 100 == 0:
            print(f"  진행률: {idx + 1}/{len(img_paths)}")

    print(f"IWPOD-tf CSV 생성 완료: {output_csv_folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='번호판 검출 결과를 CSV로 생성')
    parser.add_argument('--image_folder', type=str, required=True, help='입력 이미지 폴더 경로')
    parser.add_argument('--output_csv_folder', type=str, required=True, help='출력 CSV 폴더 경로')
    parser.add_argument('--detector', type=str, choices=['VIN_LPD', 'IWPOD_tf'], required=True,
                        help='검출기 종류 (VIN_LPD 또는 IWPOD_tf)')
    parser.add_argument('--model_path', type=str, required=True, help='검출기 모델 가중치 경로')

    args = parser.parse_args()

    if args.detector == 'VIN_LPD':
        generate_csv_vinlpd(args.image_folder, args.output_csv_folder, args.model_path)
    elif args.detector == 'IWPOD_tf':
        generate_csv_iwpod_tf(args.image_folder, args.output_csv_folder, args.model_path)
