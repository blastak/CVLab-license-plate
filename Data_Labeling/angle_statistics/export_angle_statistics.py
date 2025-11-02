"""
번호판 각도 계산 및 CSV 생성 스크립트

solvePnP 기반으로 번호판의 3D 회전 각도(X, Y, Z)를 계산하고,
세 가지 방식(sqrt, arccos, normvec_camaxis)으로 조합 각도를 계산하여 CSV로 저장합니다.

주요 기능:
- 번호판 타입별 실제 크기 적용 (한국: P1~P4, 중국: CHN)
- solvePnP를 통한 3D 회전 각도 추정
- 3가지 조합 각도 계산 방식 제공
  1. sqrt_method: sqrt(x² + y² + z²) - 3D 벡터 크기
  2. arccos_method: arccos(cos(x) * cos(y) * cos(z)) - 방향 코사인
  3. normvec_camaxis_method: 번호판 법선과 카메라 광축 사이 각도

사용 예시:
    python export_angle_statistics.py --dir /path/to/dataset --country CHN
    python export_angle_statistics.py --dir /path/to/dataset --country KOR --output result.csv
"""

import json
import numpy as np
import cv2
import os
import math
import csv
import re
import argparse
from glob import glob
from pathlib import Path
from typing import Tuple, Optional


def extract_plate_type(filename: str, country: str, json_data: Optional[dict] = None) -> str:
    """
    JSON label 필드 또는 파일명에서 번호판 타입 추출

    Args:
        filename: 파일명 (예: "14112898_P2_71거1377.jpg" 또는 "20250407_162943_327.json")
        country: 'KOR' 또는 'CHN'
        json_data: JSON 데이터 (선택, label 필드에서 타입 추출 시도)

    Returns:
        plate_type: 한국은 P1, P2, P3, P4 등 (예: "P1-1", "P2") / 중국은 'CHN'
    """
    if country == 'CHN':
        return 'CHN'

    # 1. JSON label 필드에서 타입 추출 시도 (우선순위)
    if json_data and 'shapes' in json_data and json_data['shapes']:
        label = json_data['shapes'][0].get('label', '')
        if label and '_' in label:
            # "P1-1_11거7111" 형식에서 "P1-1" 추출
            plate_type = label.split('_')[0]
            if plate_type.startswith('P'):
                return plate_type

    # 2. 파일명에서 타입 추출 시도 (폴백)
    # 한국 번호판: P숫자 또는 P숫자-숫자 패턴 찾기
    pattern = r'_P(\d+)(?:-\d+)?_'
    match = re.search(pattern, filename)

    if match:
        plate_number = match.group(1)
        return f"P{plate_number}"

    # 패턴을 찾지 못한 경우 기본값
    print(f"⚠️  파일명과 JSON label에서 번호판 타입을 찾을 수 없음: {filename}, 기본값 P2 사용")
    return "P2"


def get_plate_dimensions(plate_type: str) -> Tuple[float, float]:
    """
    번호판 타입에 따른 실제 크기 반환

    Args:
        plate_type: P1, P2, P3, P4, CHN 또는 P1-1, P1-2 등 상세 타입

    Returns:
        (width, height): 너비, 높이 (mm)
    """
    # 번호판 타입별 실제 크기 정의
    plate_dimensions = {
        'P1': (520.0, 110.0),  # P1 클래스 (신형 승용차)
        'P2': (440.0, 200.0),  # P2 클래스 (구형 승용차)
        'P3': (440.0, 200.0),  # P3 클래스 (대형 화물차)
        'P4': (520.0, 110.0),  # P4 클래스 (대형 버스)
        'CHN': (440.0, 140.0), # 중국 번호판
    }

    # 상세 타입 (P1-1, P1-2 등)을 기본 타입 (P1)으로 변환
    base_type = plate_type.split('-')[0] if '-' in plate_type else plate_type

    if base_type in plate_dimensions:
        return plate_dimensions[base_type]
    else:
        print(f"⚠️  알 수 없는 번호판 타입: {plate_type}, 기본값 P2 사용")
        return plate_dimensions['P2']


def load_json(json_path: str) -> dict:
    """JSON 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_corners(data: dict) -> np.ndarray:
    """JSON 데이터에서 번호판 코너 좌표 추출

    Returns:
        corners: (4, 2) numpy array [좌상, 우상, 우하, 좌하]
    """
    if not data['shapes']:
        return None

    points = np.array(data['shapes'][0]['points'])

    # 포인트 순서 정렬을 위한 중심점 계산
    center = points.mean(axis=0)

    # 각 점을 각도로 정렬
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    # 좌상단부터 시작하도록 조정
    top_left_idx = np.argmin(sorted_points[:, 0] + sorted_points[:, 1])
    sorted_points = np.roll(sorted_points, -top_left_idx, axis=0)

    # [좌상, 우상, 우하, 좌하] 순서로 반환
    return sorted_points


def calc_relative_angle_with_plate_type(xy1, xy2, xy3, xy4, plate_width, plate_height, image_width, image_height):
    """
    번호판 타입별 크기를 사용한 상대적인 각도 계산 (solvePnP 방식)

    Args:
        xy1~xy4: 번호판 네 꼭짓점 좌표
        plate_width, plate_height: 번호판 실제 크기 (mm)
        image_width, image_height: 이미지 크기

    Returns:
        list: 번호판의 상대적인 회전 각도 [x, y, z] (단위: 도)
    """

    # 정규화를 위해 최대값으로 나눔
    vmax = max(plate_height, plate_width)
    vh = plate_height / vmax
    vw = plate_width / vmax

    # 3D 상의 점 (정규화된 좌표)
    canonical_rect = [
        [[-vw / 2], [-vh / 2], [0]],  # 좌상
        [[vw / 2], [-vh / 2], [0]],   # 우상
        [[vw / 2], [vh / 2], [0]],    # 우하
        [[-vw / 2], [vh / 2], [0]],   # 좌하
    ]

    # virtual camera matrix
    focal_length = max(image_width, image_height)
    camera_matrix = np.float64([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # quad-box centering (중요!)
    # centering을 하지 않으면 translation 값 때문에 rotation의 해석이 어렵게 됨
    projected_points_f = np.float32([xy1, xy2, xy3, xy4])
    centering_offset = projected_points_f.mean(axis=0) - np.array([image_width, image_height]) / 2
    projected_points_f -= centering_offset

    # solvePnP
    pts3d = np.float64(canonical_rect).squeeze(2)
    pts2d = np.float64(projected_points_f)
    success, rot_vec, trans_vec = cv2.solvePnP(
        pts3d, pts2d, camera_matrix, distortion_matrix,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    angles_in_deg = [0, 0, 0]
    if success:
        # 회전 벡터를 회전 행렬로 변환
        rmat, jac = cv2.Rodrigues(rot_vec)

        # RQDecomp3x3를 사용하여 오일러 각도 추출
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # NaN 체크
        if math.isnan(any(angles)):
            angles_in_deg = [0, 0, 0]
        else:
            # 기존 스크립트와 동일한 부호 처리
            angles_in_deg = [-angles[0], -angles[1], angles[2]]

    return angles_in_deg


def calculate_combined_angles(angles_in_deg):
    """
    X, Y, Z 각도를 조합하여 단일 각도 계산 (세 가지 방식)

    Returns:
        tuple: (sqrt_method, arccos_method, normvec_camaxis_method) 조합된 각도들 (도 단위)
    """
    # 도를 라디안으로 변환
    x_rad = math.radians(angles_in_deg[0])
    y_rad = math.radians(angles_in_deg[1])
    z_rad = math.radians(angles_in_deg[2])

    # 방법 1: sqrt(x² + y² + z²) - 3D 벡터 크기
    sqrt_method = math.sqrt(angles_in_deg[0]**2 + angles_in_deg[1]**2 + angles_in_deg[2]**2)

    # 방법 2: arccos(cos(x) * cos(y) * cos(z)) - 방향 코사인 확장
    cos_product_xyz = math.cos(x_rad) * math.cos(y_rad) * math.cos(z_rad)
    cos_product_xyz = max(-1.0, min(1.0, cos_product_xyz))
    arccos_method = math.degrees(math.acos(cos_product_xyz))

    # 방법 3: 카메라 광축과 번호판 normal vector 사이의 실제 각도 (solvePnP 회전 행렬 기반)
    # ZYX 순서로 회전 행렬 구성 R = Rz(z) * Ry(y) * Rx(x)
    cx, sx = math.cos(x_rad), math.sin(x_rad)
    cy, sy = math.cos(y_rad), math.sin(y_rad)
    cz, sz = math.cos(z_rad), math.sin(z_rad)

    # 완전한 회전 행렬 계산
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx           ]
    ])

    # 번호판의 normal vector는 회전 행렬의 3번째 열 (Z축 방향)
    normal_vector = R[:, 2]

    # 카메라의 광축 벡터 (Z축)
    camera_axis = np.array([0, 0, 1])

    # 두 벡터 사이의 각도 계산
    dot_product = np.dot(normal_vector, camera_axis)
    dot_product = max(-1.0, min(1.0, dot_product))  # 수치 오류 방지

    # 각도 계산 (절댓값 사용하여 0-90도 범위로 제한)
    normvec_camaxis_method = math.degrees(math.acos(abs(dot_product)))

    return sqrt_method, arccos_method, normvec_camaxis_method


def process_image_pair(json_path: str, image_path: str, country: str):
    """이미지-JSON 쌍 처리

    Returns:
        tuple: (plate_type, plate_dims, angles_in_deg, sqrt_method, arccos_method,
                normvec_camaxis_method) 또는 None (실패 시)
    """
    # JSON 데이터 로드
    data = load_json(json_path)

    # JSON label 필드 또는 파일명에서 번호판 타입 추출
    filename = os.path.basename(json_path)
    plate_type = extract_plate_type(filename, country, data)
    plate_width, plate_height = get_plate_dimensions(plate_type)

    # 코너 좌표 추출
    corners = extract_corners(data)
    if corners is None:
        return None

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    # 각 코너 좌표 추출
    xy1 = tuple(corners[0])
    xy2 = tuple(corners[1])
    xy3 = tuple(corners[2])
    xy4 = tuple(corners[3])

    # 1. 상대 각도 계산 (solvePnP 방식, 번호판 타입별 크기 사용)
    angles_in_deg = calc_relative_angle_with_plate_type(
        xy1, xy2, xy3, xy4, plate_width, plate_height, image_width, image_height
    )

    # 2. 조합된 각도 계산 (3가지 방식)
    sqrt_method, arccos_method, normvec_camaxis_method = calculate_combined_angles(angles_in_deg)

    return (plate_type, (plate_width, plate_height), angles_in_deg,
            sqrt_method, arccos_method, normvec_camaxis_method)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='번호판 각도 계산 및 CSV 생성 (v5 - normal_vector_method 추가)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='JSON 파일이 있는 디렉토리 경로'
    )
    parser.add_argument(
        '--country',
        type=str,
        choices=['KOR', 'CHN'],
        default='KOR',
        help='번호판 국가 (KOR: 한국, CHN: 중국)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='angle_results_v5.csv',
        help='출력 CSV 파일명'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("번호판 각도 계산 스크립트 v5")
    print("solvePnP 기반 3가지 각도 계산 방식")
    print("=" * 80)
    print(f"📁 데이터 경로: {args.dir}")
    print(f"🌏 국가: {args.country}")
    print(f"💾 출력 파일: {args.output}")

    if args.country == 'KOR':
        print("   번호판 크기: P1,P4(520x110mm) | P2,P3(440x200mm)")
    else:
        print("   번호판 크기: CHN(440x140mm)")
    print("=" * 80)
    print()

    # JSON 파일 목록 가져오기
    base_path = Path(args.dir)
    json_files = sorted(base_path.rglob('*.json'))

    if not json_files:
        print("❌ JSON 파일을 찾을 수 없습니다.")
        return

    print(f"📁 총 {len(json_files)}개의 JSON 파일 발견\n")

    # CSV 파일 준비
    csv_data = []

    success_count = 0
    fail_count = 0

    # 번호판 타입별 통계
    plate_type_stats = {}

    for idx, json_path in enumerate(json_files, 1):
        filename = json_path.name
        image_name = filename.replace('.json', '.jpg')
        image_path = json_path.with_suffix('.jpg')

        # 진행 상황 표시 (매 100개마다)
        if idx % 100 == 1:
            print(f"📄 처리 중: {filename} ({idx}/{len(json_files)})")

        # 이미지 파일 존재 확인
        if not image_path.exists():
            if idx % 100 == 1:
                print(f"  ⚠️  이미지 파일 없음: {image_name}")
            # 이미지 없어도 JSON 처리는 시도
            image_path = str(json_path.with_suffix('.jpg'))

        # 처리
        result = process_image_pair(str(json_path), str(image_path), args.country)

        if result is None:
            fail_count += 1
            if idx % 100 == 1:
                print(f"  ❌ 처리 실패")
            continue

        (plate_type, plate_dims, angles_in_deg, sqrt_method,
         arccos_method, normvec_camaxis_method) = result
        success_count += 1

        # 번호판 타입별 통계 수집
        if plate_type not in plate_type_stats:
            plate_type_stats[plate_type] = 0
        plate_type_stats[plate_type] += 1

        # CSV 데이터 추가
        csv_data.append([
            image_name,
            plate_type,
            f"{plate_dims[0]}x{plate_dims[1]}",
            round(angles_in_deg[0], 2),
            round(angles_in_deg[1], 2),
            round(angles_in_deg[2], 2),
            round(sqrt_method, 2),
            round(arccos_method, 2),
            round(normvec_camaxis_method, 2)
        ])

        # 간략한 진행상황 출력 (매 100개마다)
        if success_count % 100 == 0:
            print(f"  ✅ {success_count}개 처리 완료...")

    # CSV 파일 작성
    from datetime import datetime
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        # 메타데이터 주석 작성
        csvfile.write(f"# 번호판 각도 계산 결과 (export_angle_statistics.py)\n")
        csvfile.write(f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        csvfile.write(f"# 입력 경로: {args.dir}\n")
        csvfile.write(f"# 국가: {args.country}\n")
        csvfile.write(f"# 성공: {success_count}개 / 실패: {fail_count}개 / 전체: {len(json_files)}개\n")
        csvfile.write(f"#\n")

        writer = csv.writer(csvfile)
        # 헤더 작성
        writer.writerow([
            'filename', 'plate_type', 'dimensions',
            'x_deg', 'y_deg', 'z_deg',
            'sqrt_method', 'arccos_method', 'normvec_camaxis_method'
        ])
        # 데이터 작성
        writer.writerows(csv_data)

    # 요약
    print("\n" + "=" * 80)
    print("📊 처리 요약")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"  📁 전체: {len(json_files)}개")
    print()
    print("📋 번호판 타입별 분포:")
    for plate_type, count in sorted(plate_type_stats.items()):
        plate_dims = get_plate_dimensions(plate_type)
        print(f"  {plate_type}: {count}개 ({plate_dims[0]}x{plate_dims[1]}mm)")
    print("=" * 80)
    print(f"\n💾 결과가 {args.output}에 저장되었습니다.")

    # CSV 내용 일부 출력
    if csv_data:
        print(f"\n📋 CSV 파일 미리보기 (처음 5개):")
        print("-" * 120)
        print(f"{'파일명':<25} | {'타입':<4} | {'크기':<9} | {'X(°)':<6} | {'Y(°)':<6} | {'Z(°)':<6} | {'sqrt':<6} | {'arccos':<7} | {'pnp_nv':<7}")
        print("-" * 120)
        for i, row in enumerate(csv_data[:5]):
            print(f"{row[0]:<25} | {row[1]:<4} | {row[2]:<9} | {row[3]:6.2f} | {row[4]:6.2f} | {row[5]:6.2f} | {row[6]:6.2f} | {row[7]:7.2f} | {row[8]:7.2f}")
        if len(csv_data) > 5:
            print(f"... 외 {len(csv_data)-5}개")


if __name__ == "__main__":
    main()
