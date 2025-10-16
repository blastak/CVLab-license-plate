"""
단일 이미지 테스트 스크립트
CSV 생성 → Frontalization을 한 이미지에 대해 빠르게 테스트
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import load_model_tf, find_lp_corner
from Utils import imread_uni

# 클래스별 출력 크기 정의
PLATE_SIZES = {
    'P0': (520, 110),
}


def ensure_clockwise_from_topleft(points):
    """
    4개의 꼭지점을 좌상단부터 시계방향으로 정렬 (테스트용)

    번호판은 가로로 긴 직사각형이므로:
    1. y 좌표로 상단/하단 구분 (y가 작을수록 상단)
    2. 상단 2개 중 x가 작은 것이 좌상단
    3. 하단 2개 중 x가 큰 것이 우하단
    """
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

    return ordered_points


def apply_homography(img, src_points, dst_size):
    """Homography를 이용한 perspective warping"""
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


def test_single_image(image_path, model_path):
    """
    단일 이미지에 대해 검출 → CSV 생성 → Frontalization 테스트
    """
    print(f"=== 테스트 시작 ===")
    print(f"이미지: {image_path}")

    # 이미지 로드
    img = imread_uni(image_path)
    print(f"이미지 크기: {img.shape}")

    # IWPOD-tf 모델 로드 및 검출
    print(f"\nIWPOD-tf 모델 로딩...")
    iwpod_net = load_model_tf(model_path)

    print("번호판 검출 중...")
    xys_list, prob_list = find_lp_corner(img, iwpod_net)

    if len(xys_list) == 0:
        print("검출된 번호판이 없습니다!")
        return

    print(f"검출된 번호판: {len(xys_list)}개")

    # 첫 번째 검출 결과만 테스트
    xys = xys_list[0]
    prob = prob_list[0]

    print(f"\n원본 좌표 (IWPOD 출력):")
    for i, pt in enumerate(xys):
        print(f"  점{i}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # 좌표 정렬
    ordered_points = ensure_clockwise_from_topleft(xys)

    print(f"\n정렬된 좌표 (좌상단부터 시계방향):")
    labels = ["좌상단", "우상단", "우하단", "좌하단"]
    for i, (pt, label) in enumerate(zip(ordered_points, labels)):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # 시각화: 원본 이미지에 좌표 표시
    img_vis = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 파랑, 초록, 빨강, 노랑

    for i, (pt, color, label) in enumerate(zip(ordered_points, colors, labels)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (x, y), 10, color, -1)
        cv2.putText(img_vis, f"{i}:{label}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Polygon 그리기
    pts = ordered_points.astype(np.int32)
    cv2.polylines(img_vis, [pts], True, (0, 255, 255), 3)

    # 시각화 이미지 저장
    vis_path = "./test_output_visualization.jpg"
    cv2.imwrite(vis_path, img_vis)
    print(f"\n시각화 이미지 저장: {vis_path}")

    # Frontalization 수행
    dst_size = PLATE_SIZES['P0']
    warped = apply_homography(img, ordered_points, dst_size)

    # Frontalized 이미지 저장
    front_path = "./test_output_frontalized.jpg"
    cv2.imwrite(front_path, warped)
    print(f"Frontalized 이미지 저장: {front_path}")

    print(f"\n=== 테스트 완료 ===")
    print(f"1. {vis_path} - 원본 이미지에 좌표 표시 (0=좌상단, 1=우상단, 2=우하단, 3=좌하단)")
    print(f"2. {front_path} - Frontalized 결과 (번호판이 정면으로 보여야 함)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='단일 이미지 테스트')
    parser.add_argument('--image', type=str, required=True, help='테스트할 이미지 경로')
    parser.add_argument('--model', type=str,
                        default='../LP_Detection/IWPOD_tf/weights/iwpod_net',
                        help='IWPOD-tf 모델 경로')

    args = parser.parse_args()

    test_single_image(args.image, args.model)
