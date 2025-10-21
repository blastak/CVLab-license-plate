import argparse
import json
import math
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from tqdm import tqdm

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR as GMG_KOR
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_CHN import Graphical_Model_Generator_CHN as GMG_CHN
from Utils import imread_uni, add_text_with_background


def calc_relative_angle(xy1, xy2, xy3, xy4, plate_type, image_width, image_height, GMG):
    """
    이미지 내 번호판의 상대적인 각도 계산

    이 함수는 이미지 내 번호판의 네 꼭짓점 좌표, 번호판 종류, 이미지 크기를 입력받아
    3차원 공간에서 번호판의 회전 각도를 추정합니다.

    Args:
        xy1 (tuple): 번호판 왼쪽 상단 꼭짓점 좌표
        xy2 (tuple): 번호판 오른쪽 상단 꼭짓점 좌표
        xy3 (tuple): 번호판 오른쪽 하단 꼭짓점 좌표
        xy4 (tuple): 번호판 왼쪽 하단 꼭짓점 좌표
        plate_type (str): 번호판 종류
        image_width (int): 이미지 너비
        image_height (int): 이미지 높이
        GMG: Graphical Model Generator (KOR 또는 CHN)

    Returns:
        list: 번호판의 상대적인 회전 각도 (x, y, z)를 요소로 하는 리스트 (단위: 도)
    """

    # 3D 상의 점 - 번호판 크기 가져오기
    if isinstance(GMG.plate_wh, dict):
        # 한국 번호판 (dict 형식)
        if plate_type not in GMG.plate_wh.keys():
            raise NotImplementedError
        vw, vh = GMG.plate_wh[plate_type]
    else:
        # 중국 번호판 (tuple 형식)
        vw, vh = GMG.plate_wh
    vmax = max(vh, vw)
    vh /= vmax
    vw /= vmax
    canonical_rect = [
        [[-vw / 2], [-vh / 2], [0]], [[vw / 2], [-vh / 2], [0]],
        [[vw / 2], [vh / 2], [0]], [[-vw / 2], [vh / 2], [0]],
    ]

    # virtual camera matrix
    focal_length = max(image_width, image_height)
    camera_matrix = np.float64([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # quad-box centering
    projected_points_f = np.float32([xy1, xy2, xy3, xy4])
    centering_offset = projected_points_f.mean(axis=0) - np.array([image_width, image_height]) / 2
    projected_points_f -= centering_offset  # centering을 하지 않으면 translation 값 때문에 rotation의 해석이 어렵게 되어버린다.

    # solvePnP
    pts3d = np.float64(canonical_rect).squeeze(2)
    pts2d = np.float64(projected_points_f)
    success, rot_vec, trans_vec = cv2.solvePnP(pts3d, pts2d, camera_matrix, distortion_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

    # # reprojection test
    # reproj_, jacobian = cv2.projectPoints(pts3d, rot_vec, trans_vec, camera_matrix, distortion_matrix)
    # reproj = np.int32(reproj_.squeeze(1) + centering_offset)

    angles_in_deg = [0, 0, 0]
    if success:
        rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles
        if math.isnan(any(angles)):
            angles_in_deg = [0, 0, 0]
        else:
            angles_in_deg = [-angles[0], -angles[1], angles[2]]
    return angles_in_deg


def extract_plate_info_from_json(json_path):
    """
    JSON 파일에서 번호판 정보 추출

    Args:
        json_path: JSON 파일 경로

    Returns:
        tuple: (plate_type, xy1, xy2, xy3, xy4, image_width, image_height) 또는 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data.get('shapes') or len(data['shapes']) == 0:
            return None

        # 첫 번째 shape에서 정보 추출
        shape = data['shapes'][0]
        label = shape.get('label', '')
        points = shape.get('points', [])

        if len(points) != 4:
            return None

        # 라벨에서 plate_type 추출 (예: "P1-1_12가3456" -> "P1-1")
        plate_type = label.split('_')[0] if '_' in label else label

        # 꼭지점 좌표 (LabelMe 형식: [좌상, 우상, 우하, 좌하])
        xy1, xy2, xy3, xy4 = [tuple(pt) for pt in points]

        image_width = data.get('imageWidth', 0)
        image_height = data.get('imageHeight', 0)

        return plate_type, xy1, xy2, xy3, xy4, image_width, image_height

    except Exception as e:
        print(f"❌ JSON 파싱 오류: {json_path} - {e}")
        return None


def process_directory(base_dir, plate_country):
    """
    디렉토리 내의 모든 JSON 파일에 각도 정보 추가

    Args:
        base_dir: 처리할 디렉토리 경로
        plate_country: 'KOR' 또는 'CHN'
    """
    # Graphical Model Generator 인스턴스 생성
    if plate_country == 'KOR':
        GMG = GMG_KOR()  # 기본 경로 사용
        print("📋 한국 번호판 모드")
    elif plate_country == 'CHN':
        # 중국 번호판 그래픽 모델 경로
        chn_model_path = str(project_root / 'Data_Labeling' / 'Graphical_Model_Generation' / 'BetaType' / 'chinese_LP')
        GMG = GMG_CHN(chn_model_path)
        print("📋 중국 번호판 모드")
        print(f"   그래픽 모델 경로: {chn_model_path}")
    else:
        raise ValueError(f"지원하지 않는 국가 코드: {plate_country}")

    # 모든 JSON 파일 찾기
    base_path = Path(base_dir)
    json_files = list(base_path.rglob('*.json'))

    if not json_files:
        print(f"❌ {base_dir}에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"📁 총 {len(json_files)}개의 JSON 파일 발견")
    print(f"📍 처리 경로: {base_dir}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0
    already_processed = 0

    for json_path in tqdm(json_files, desc="JSON 파일 처리 중"):
        # 이미 각도가 계산되어 있는지 확인
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'flags' in data and 'angle' in data['flags']:
                already_processed += 1
                continue
        except:
            pass

        # JSON에서 정보 추출
        result = extract_plate_info_from_json(json_path)

        if result is None:
            skip_count += 1
            continue

        plate_type, xy1, xy2, xy3, xy4, image_width, image_height = result

        # plate_type 검증 (한국 번호판만 해당)
        if plate_country == 'KOR':
            if plate_type not in GMG.plate_wh.keys():
                skip_count += 1
                continue
        # 중국 번호판은 단일 타입이므로 검증 불필요

        try:
            # 각도 계산
            angle_xyz = calc_relative_angle(
                xy1, xy2, xy3, xy4, plate_type,
                image_width, image_height, GMG
            )

            # JSON 파일 업데이트
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # flags 필드가 없으면 생성
            if 'flags' not in data:
                data['flags'] = {}

            data['flags']['angle'] = {
                'x': round(angle_xyz[0], 2),
                'y': round(angle_xyz[1], 2),
                'z': round(angle_xyz[2], 2)
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            success_count += 1

        except Exception as e:
            fail_count += 1
            tqdm.write(f"❌ 처리 실패: {json_path.name} - {e}")

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 처리 완료")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  🔄 이미 처리됨: {already_processed}개")
    print(f"  ⏭️  스킵: {skip_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"  📁 전체: {len(json_files)}개")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='JSON 라벨 파일에 번호판 각도 정보 추가'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='처리할 디렉토리 경로 (하위 폴더 포함 모든 JSON 파일 처리)'
    )
    parser.add_argument(
        '--country',
        type=str,
        choices=['KOR', 'CHN'],
        required=True,
        help='번호판 국가 (KOR: 한국, CHN: 중국)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("번호판 각도 계산 및 JSON 업데이트")
    print("=" * 60)

    process_directory(args.dir, args.country)
