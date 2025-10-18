"""
CCPD 데이터셋의 러프한 번호판 위치를 정밀한 위치로 라벨링하는 스크립트

CCPD 파일명에서 추출한 러프한 번호판 좌표를 기반으로
그래픽 모델과의 feature 매칭을 통해 정밀한 좌표를 계산하고
JSON 형태로 저장합니다.

작업 흐름:
1. CCPD 파일명 파싱 (DatasetLoader_CCPD)
2. 그래픽 모델 생성 (Graphical_Model_Generator_CHN)
3. Feature 추출 및 추적 (goodFeaturesToTrack + calcOpticalFlowPyrLK)
4. 반복적 RANSAC 기반 호모그래피 변환을 통한 정밀 좌표 계산
5. LabelMe JSON 형식으로 저장
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
    from LP_Detection.Bases import BBox
    VINOCR_AVAILABLE = True
except ImportError:
    VINOCR_AVAILABLE = False
    print("경고: VinOCR를 로드할 수 없습니다.")

from Data_Labeling.Dataset_Loader.DatasetLoader_CCPD import DatasetLoader_CCPD
from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_CHN import Graphical_Model_Generator_CHN
from Data_Labeling.labeling_utils import calculate_text_area_coordinates, frontalization
from LP_Detection.Bases import Quadrilateral
from Utils import imread_uni, imwrite_uni


def extract_N_track_features_CHN(img_gened, mask_text_area, img_front):
    """
    중국 번호판용 feature 추출 및 추적

    Args:
        img_gened: 그래픽 모델 이미지
        mask_text_area: 텍스트 영역 마스크
        img_front: 정면화된 원본 이미지

    Returns:
        tuple: (pt1, pt2) - 매칭된 feature 좌표들
    """
    img_gen_gray = cv2.cvtColor(img_gened, cv2.COLOR_BGR2GRAY)
    pt_gen = cv2.goodFeaturesToTrack(img_gen_gray, 500, 0.01, 5, mask=mask_text_area)  # feature extraction

    img_front_gray = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    img_front_gray_histeq = cv2.equalizeHist(img_front_gray)  # histogram equalization
    pt_tracked, status, err = cv2.calcOpticalFlowPyrLK(img_gen_gray, img_front_gray_histeq, pt_gen, None)  # feature tracking

    pt1 = pt_gen[status == 1].astype(np.int32)
    pt2 = pt_tracked[status == 1].astype(np.int32)
    return pt1, pt2


def calculate_text_area_coordinates_CHN(generator):
    """
    중국 번호판의 텍스트 영역 마스크 생성

    Args:
        generator: Graphical_Model_Generator_CHN 인스턴스

    Returns:
        numpy.ndarray: 텍스트 영역 마스크 (grayscale)
    """
    # 중국 번호판은 char_xywh가 리스트로 정의되어 있음
    char_xywh = generator.char_xywh

    # 모든 문자 영역을 포함하는 bounding box 계산
    min_x = min([xywh[0] for xywh in char_xywh])
    min_y = min([xywh[1] for xywh in char_xywh])
    max_x = max([xywh[0] + xywh[2] for xywh in char_xywh])
    max_y = max([xywh[1] + xywh[3] for xywh in char_xywh])

    # margin 추가
    margin = [-10, -10, 10, 10]
    min_x, min_y, max_x, max_y = map(int, [min_x + margin[0], min_y + margin[1],
                                            max_x + margin[2], max_y + margin[3]])

    mask_text_area = np.zeros(generator.plate_wh[::-1], dtype=np.uint8)
    mask_text_area[min_y:max_y, min_x:max_x] = 255
    return mask_text_area


def calculate_center(box):
    """
    Bounding box의 중심점 계산

    Args:
        box: (x, y, w, h) 형태의 bounding box

    Returns:
        numpy.ndarray: [center_x, center_y]
    """
    x_min, y_min, width, height = box
    center_x = x_min + (width / 2)
    center_y = y_min + (height / 2)
    return np.array([center_x, center_y])


def calculate_offset(reference_points, bbox_centers):
    """
    Hungarian Algorithm을 사용하여 참조점과 bbox 중심점 간 최소 거리 합 계산

    Args:
        reference_points: 그래픽 모델의 문자 중심점 리스트
        bbox_centers: OCR bbox 중심점 리스트

    Returns:
        float: 거리 합계
    """
    if len(reference_points) == 0 or len(bbox_centers) == 0:
        return float('inf')

    offset_sum = 0
    # Hungarian Algorithm
    cost_matrix = np.zeros((len(reference_points), len(bbox_centers)))
    for i, ref_point in enumerate(reference_points):
        for j, bbox_point in enumerate(bbox_centers):
            cost_matrix[i, j] = np.linalg.norm(ref_point - bbox_point)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for r, c in zip(row_ind, col_ind):
        offset_sum += cost_matrix[r, c]
    return offset_sum


def find_homography_with_minimum_error_CHN(img_gen, mask_text_area, img_front, pt1, pt2):
    """
    반복적 RANSAC을 통한 최소 에러 호모그래피 찾기

    Args:
        img_gen: 그래픽 모델 이미지
        mask_text_area: 텍스트 영역 마스크
        img_front: 정면화된 원본 이미지
        pt1: 그래픽 모델의 feature 좌표
        pt2: 원본 이미지의 feature 좌표

    Returns:
        numpy.ndarray: 최적 호모그래피 변환 행렬 (3x3)
    """
    if len(pt1) < 4:
        return None

    img_front_gray = cv2.cvtColor(img_front, cv2.COLOR_BGR2GRAY)
    img_front_adap_th = cv2.adaptiveThreshold(img_front_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)

    mat_H = None
    minval = float('inf')

    for ransac_th in range(1, 15):
        H, stat = cv2.findHomography(pt1, pt2, cv2.RANSAC, ransacReprojThreshold=ransac_th)
        if H is not None:
            img_warped = cv2.warpPerspective(img_gen, H, img_gen.shape[1::-1])
            img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
            _, img_warped_th = cv2.threshold(img_warped_gray, 0, 255, cv2.THRESH_OTSU)

            mask_warped = cv2.warpPerspective(mask_text_area, H, img_gen.shape[1::-1])
            img_front_adap_th_ = cv2.bitwise_and(img_front_adap_th, mask_warped)
            img_warped_th_ = cv2.bitwise_and(img_warped_th, mask_warped)
            img_subtract = cv2.absdiff(img_warped_th_, img_front_adap_th_)
            s = img_subtract.sum() / mask_warped.sum() if mask_warped.sum() > 0 else float('inf')

            if minval > s:
                minval = s
                mat_H = H.copy()

    return mat_H


def calculate_ocr_error_CHN(img_front, generator, ocr_reader, plate_number=''):
    """
    VinOCR character detection 결과와 그래픽 모델의 문자 위치를 비교하여 에러 계산

    VinOCR은 흰색 바탕/검은색 글자에 최적화되어 있으므로 색 반전 후 사용
    번호판 번호에서 숫자인 자리와 VinOCR 검출 결과를 순서대로 매칭

    Args:
        img_front: Frontalization된 이미지
        generator: Graphical_Model_Generator_CHN 인스턴스
        ocr_reader: VinOCR reader 인스턴스
        plate_number: 번호판 번호 (예: "皖A783R7")

    Returns:
        float: offset_sum (character bbox 중심점과 그래픽 모델 문자 위치 간 거리 합)
    """
    if not VINOCR_AVAILABLE or ocr_reader is None:
        return 0.0  # VinOCR이 없으면 에러 0으로 간주 (모두 통과)

    h, w = img_front.shape[:2]

    # 색 반전 (흰색 바탕/검은색 글자로 변환)
    img_inverted = 255 - img_front

    # VinOCR은 BBox를 입력으로 받으므로 전체 이미지를 bbox로 지정
    dummy_bbox = BBox(0, 0, w, h, '', 0, 1.0)

    # keep_ratio_padding으로 전처리 (256x224)
    crop_resized_img = ocr_reader.keep_ratio_padding(img_inverted, dummy_bbox)

    # VinOCR forward (character detection)
    char_boxes = ocr_reader.forward([crop_resized_img])[0]

    if len(char_boxes) == 0:
        return float('inf')

    # 숫자만 필터링하여 중심점 계산
    # VinOCR 출력 좌표계(256x224)를 원본 좌표계(w x h)로 변환
    inpWidth = 256
    inpHeight = 224

    # keep_ratio_padding 역변환 계산
    aspect_ratio_target = inpWidth / inpHeight
    aspect_ratio_img = w / h

    if aspect_ratio_img > aspect_ratio_target:
        # 폭 기준으로 조정됨
        scale = w / inpWidth
        offset_x = 0
        offset_y = (inpHeight - (h / scale)) / 2
    else:
        # 높이 기준으로 조정됨
        scale = h / inpHeight
        offset_x = (inpWidth - (w / scale)) / 2
        offset_y = 0

    # VinOCR 검출된 모든 문자의 bounding box 중심점 계산
    bbox_centers = []
    for cb in char_boxes:
        # 256x224 좌표계 -> 원본 좌표계
        center_x_scaled = (cb.x + cb.w / 2 - offset_x) * scale
        center_y_scaled = (cb.y + cb.h / 2 - offset_y) * scale
        bbox_centers.append(np.array([center_x_scaled, center_y_scaled]))

    if len(bbox_centers) < 3:
        return float('inf')

    # 번호판 번호에서 숫자인 자리의 인덱스 추출
    # 중국 번호판: 7자리 [중국문자, 알파벳, 숫자, 숫자, 숫자, 숫자/알파벳, 숫자]
    if len(plate_number) != 7:
        return float('inf')

    # 그래픽 모델의 참조점 (숫자 위치만)
    reference_points = []
    for i, char in enumerate(plate_number):
        if char.isdigit():
            ref_center = calculate_center(generator.char_xywh[i])
            reference_points.append(ref_center)

    if len(reference_points) < 3:
        return float('inf')

    # Hungarian Algorithm으로 최소 거리 합 계산
    offset_sum = calculate_offset(reference_points, bbox_centers)

    return offset_sum


def calculate_total_transformation(mat_A, mat_H):
    """
    Affine 변환과 Homography 변환을 결합한 전체 변환 행렬 계산

    Args:
        mat_A: Affine 변환 행렬 (2x3) 또는 (3x3)
        mat_H: Homography 변환 행렬 (3x3)

    Returns:
        numpy.ndarray: 전체 변환 행렬 (3x3)
    """
    # Transform Matrix 역변환
    if mat_A.shape[0] == 2:
        mat_A_homo = np.vstack((mat_A, [0, 0, 1]))
    else:
        mat_A_homo = mat_A

    mat_A_inv = np.linalg.inv(mat_A_homo)
    mat_T = mat_A_inv @ mat_H
    return mat_T


def refine_plate_coordinates(img, rough_quad, graphical_model, generator, ocr_reader=None, plate_number=''):
    """
    러프한 번호판 좌표를 그래픽 모델과의 feature 매칭을 통해 정밀화

    반복적 RANSAC + adaptive threshold + absdiff 기반 최소 에러 호모그래피 계산
    VinOCR을 사용하여 문자 위치 기반 에러 계산

    Args:
        img: 원본 이미지
        rough_quad: 러프한 번호판 좌표 (Quadrilateral)
        graphical_model: 생성된 그래픽 모델 이미지
        generator: Graphical_Model_Generator_CHN 인스턴스
        ocr_reader: VinOCR reader 인스턴스 (선택)
        plate_number: 번호판 번호 (예: "皖A783R7")

    Returns:
        tuple: (정밀화된 네 꼭지점 좌표 (xy1, xy2, xy3, xy4), frontalization 이미지, OCR 에러값) 또는 (None, img_front, error)
    """
    g_h, g_w = graphical_model.shape[:2]

    # 1. 정면화 (frontalization) - mode=3은 Affine 변환 (3점 기반)
    # rough_quad가 Quadrilateral인 경우 처리
    img_front, mat_A = frontalization(img, rough_quad, g_w, g_h, mode=3)

    # 2. 텍스트 영역 마스크 생성
    mask_text_area = calculate_text_area_coordinates_CHN(generator)

    # 3. Feature 추출 및 추적
    pt1, pt2 = extract_N_track_features_CHN(graphical_model, mask_text_area, img_front)

    if len(pt1) < 4:
        print(f"  Feature 매칭 실패: {len(pt1)}개만 매칭됨")
        return None, img_front, float('inf')

    # 4. 반복적 RANSAC + adaptive threshold + absdiff 기반 최소 에러 호모그래피 계산
    mat_H = find_homography_with_minimum_error_CHN(graphical_model, mask_text_area, img_front, pt1, pt2)

    if mat_H is None:
        print(f"  호모그래피 계산 실패")
        return None, img_front, float('inf')

    # 5. 전체 변환 행렬 계산 (Affine 역변환 @ Homography)
    mat_T = calculate_total_transformation(mat_A, mat_H)

    # 6. 그래픽 모델의 네 꼭지점을 원본 이미지 좌표계로 변환
    model_corners = np.float32([
        [[0, 0]],
        [[g_w, 0]],
        [[g_w, g_h]],
        [[0, g_h]]
    ])

    refined_corners = cv2.perspectiveTransform(model_corners, mat_T)
    refined_corners = refined_corners.reshape(-1, 2)

    xy1 = refined_corners[0].tolist()
    xy2 = refined_corners[1].tolist()
    xy3 = refined_corners[2].tolist()
    xy4 = refined_corners[3].tolist()

    # 7. OCR 기반 에러 계산 (frontalization된 이미지에서)
    ocr_error = calculate_ocr_error_CHN(img_front, generator, ocr_reader, plate_number)

    return (xy1, xy2, xy3, xy4), img_front, ocr_error


def create_labelme_json(img_filename, img_height, img_width, plate_type, plate_number, xy1, xy2, xy3, xy4):
    """
    LabelMe 형식의 JSON 데이터 생성

    Args:
        img_filename: 이미지 파일명
        img_height: 이미지 높이
        img_width: 이미지 너비
        plate_type: 번호판 타입 (CHN)
        plate_number: 번호판 번호
        xy1, xy2, xy3, xy4: 번호판 네 꼭지점 좌표

    Returns:
        dict: LabelMe JSON 데이터
    """
    json_data = {
        "version": "5.5.0",
        "flags": {
            "plate_type": plate_type,
            "plate_number": plate_number
        },
        "shapes": [
            {
                "label": "plate",
                "points": [
                    list(xy1),
                    list(xy2),
                    list(xy3),
                    list(xy4)
                ],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    return json_data


def process_ccpd_dataset(ccpd_dir, output_dir, graphical_model_base_path, error_threshold=22):
    """
    CCPD 데이터셋 전체를 처리하여 정밀 라벨링 수행

    Args:
        ccpd_dir: CCPD 데이터셋 경로
        output_dir: 출력 디렉토리 (GoodMatches_CHN 등)
        graphical_model_base_path: 중국 번호판 그래픽 모델 베이스 경로
        error_threshold: OCR 에러 임계값 (픽셀 단위, 이 값 이하면 Front 폴더, 초과하면 etc 폴더)
    """
    # 출력 디렉토리 생성
    output_goodmatches = Path(output_dir) / 'GoodMatches_CHN'
    output_front = Path(output_dir) / f'GoodMatches_CHN_Front_H{int(error_threshold):02d}'
    output_etc = Path(output_dir) / 'GoodMatches_CHN_etc'

    output_goodmatches.mkdir(parents=True, exist_ok=True)
    output_front.mkdir(parents=True, exist_ok=True)
    output_etc.mkdir(parents=True, exist_ok=True)

    # 데이터 로더 및 그래픽 모델 생성기 초기화
    loader = DatasetLoader_CCPD(ccpd_dir)
    generator = Graphical_Model_Generator_CHN(graphical_model_base_path)

    # VinOCR reader 초기화
    ocr_reader = None
    if VINOCR_AVAILABLE:
        print("VinOCR 초기화 중... (최초 1회만 수행)")
        ocr_reader = load_model_VinOCR('/workspace/repo/CVLab-license-plate/LP_Recognition/VIN_OCR/weight')
        print("VinOCR 초기화 완료")

    # CCPD 이미지 파일 리스트
    ccpd_path = Path(ccpd_dir)
    img_files = sorted([f for f in ccpd_path.glob('*.jpg')])

    print(f"총 {len(img_files)}개 파일 처리 시작")
    print(f"출력 경로: {output_goodmatches}")
    print(f"Front 경로: {output_front}")
    print(f"Etc 경로: {output_etc}")
    print(f"에러 임계값: {error_threshold}")

    success_count = 0
    fail_count = 0
    use_rough_count = 0
    front_count = 0
    etc_count = 0

    for img_file in tqdm(img_files, desc="CCPD 라벨링"):
        try:
            # 1. 파일명에서 정보 추출
            img_filename = img_file.name
            plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = \
                loader.parse_ccpd_filename(img_filename)

            if plate_number == '' or len(xy1) == 0:
                print(f"파싱 실패: {img_filename}")
                fail_count += 1
                continue

            # 2. 이미지 로드
            img = imread_uni(str(img_file))
            if img is None:
                print(f"이미지 로드 실패: {img_filename}")
                fail_count += 1
                continue

            img_height, img_width = img.shape[:2]

            # 3. 그래픽 모델 생성
            graphical_model_2x = generator.make_LP(plate_number)
            if graphical_model_2x is None:
                print(f"그래픽 모델 생성 실패: {plate_number}")
                fail_count += 1
                continue

            # 그래픽 모델을 0.5배로 축소 (2배 크기로 생성되므로)
            graphical_model = cv2.resize(graphical_model_2x, None, fx=0.5, fy=0.5)

            # 4. 러프한 좌표를 Quadrilateral로 변환
            rough_quad = Quadrilateral(xy1, xy2, xy3, xy4)

            # 5. 정밀 좌표 계산 및 frontalization (OCR 에러 계산 포함)
            coords_result, img_front, error_value = refine_plate_coordinates(
                img, rough_quad, graphical_model, generator, ocr_reader, plate_number
            )

            if coords_result is not None:
                refined_xy1, refined_xy2, refined_xy3, refined_xy4 = coords_result
            else:
                # 실패 시 러프한 좌표 사용
                print(f"  러프한 좌표 사용: {img_filename}")
                refined_xy1, refined_xy2, refined_xy3, refined_xy4 = xy1, xy2, xy3, xy4
                use_rough_count += 1

            # 6. JSON 생성 및 저장 (GoodMatches_CHN)
            json_data = create_labelme_json(
                img_filename, img_height, img_width,
                plate_type, plate_number,
                refined_xy1, refined_xy2, refined_xy3, refined_xy4
            )

            json_filename = img_file.stem + '.json'
            json_path = output_goodmatches / json_filename
            jpg_path = output_goodmatches / img_filename

            # JSON 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            # 이미지 복사
            imwrite_uni(str(jpg_path), img)

            # 7. Frontalization 이미지 저장 (threshold 기반 분류)
            front_filename = f'front_{img_filename}'

            if error_value <= error_threshold:
                # 에러가 작으면 Front 폴더에 저장
                front_path = output_front / front_filename
                imwrite_uni(str(front_path), img_front)
                front_count += 1
            else:
                # 에러가 크면 etc 폴더에 저장
                etc_path = output_etc / front_filename
                imwrite_uni(str(etc_path), img_front)
                etc_count += 1

            success_count += 1

        except Exception as e:
            print(f"처리 중 오류 ({img_filename}): {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

    print(f"\n처리 완료!")
    print(f"성공: {success_count}개")
    print(f"  - 정밀 좌표: {success_count - use_rough_count}개")
    print(f"  - 러프 좌표: {use_rough_count}개")
    print(f"  - Front (에러 ≤ {error_threshold}): {front_count}개")
    print(f"  - Etc (에러 > {error_threshold}): {etc_count}개")
    print(f"실패: {fail_count}개")
    print(f"출력 경로: {output_goodmatches}")
    print(f"Front 경로: {output_front}")
    print(f"Etc 경로: {output_etc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCPD 데이터셋 정밀 라벨링')
    parser.add_argument(
        '--ccpd_dir',
        type=str,
        default='/workspace/DB/01_LicensePlate/CCPD_sample',
        help='CCPD 데이터셋 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/workspace/repo/CVLab-license-plate/Data_Labeling',
        help='출력 디렉토리 경로 (GoodMatches_CHN 폴더가 생성됨)'
    )
    parser.add_argument(
        '--graphical_model_path',
        type=str,
        default='/workspace/repo/CVLab-license-plate/Data_Labeling/Graphical_Model_Generation/BetaType/chinese_LP/',
        help='중국 번호판 그래픽 모델 베이스 경로'
    )
    parser.add_argument(
        '--error_threshold',
        type=float,
        default=22,
        help='OCR 에러 임계값 (픽셀 단위, 이 값 이하면 Front 폴더, 초과하면 etc 폴더)'
    )

    args = parser.parse_args()

    process_ccpd_dataset(
        args.ccpd_dir,
        args.output_dir,
        args.graphical_model_path,
        args.error_threshold
    )
