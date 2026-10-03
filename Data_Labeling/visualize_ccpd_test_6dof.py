"""
CCPD test 데이터셋 시각화 스크립트
파일명에서 번호판 사각형 좌표를 파싱하여 녹색 선으로 그려서 저장
"""

import os
import sys
import cv2
from glob import glob
from tqdm import tqdm

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_Labeling.Dataset_Loader.DatasetLoader_CCPD import DatasetLoader_CCPD
from Utils import imread_uni


def visualize_ccpd_samples(input_dir, output_dir):
    """
    CCPD 샘플 이미지에 번호판 사각형을 시각화하여 저장

    Args:
        input_dir: 입력 이미지 디렉토리 경로
        output_dir: 출력 이미지 디렉토리 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # CCPD loader 초기화
    loader = DatasetLoader_CCPD(input_dir)

    # 입력 디렉토리에서 모든 jpg 파일 찾기
    jpg_files = glob(os.path.join(input_dir, '*.jpg'))

    print(f"총 {len(jpg_files)}개의 이미지를 처리합니다...")

    processed_count = 0
    error_count = 0

    for jpg_path in tqdm(jpg_files):
        try:
            # 이미지 읽기
            img = imread_uni(jpg_path)
            if img is None:
                print(f"이미지 읽기 실패: {jpg_path}")
                error_count += 1
                continue

            # 파일명에서 GT 정보 파싱
            filename = os.path.basename(jpg_path)
            plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = \
                loader.parse_ccpd_filename(filename)

            # 좌표가 유효한지 확인
            if len(xy1) == 2 and len(xy2) == 2 and len(xy3) == 2 and len(xy4) == 2:
                # 좌표를 정수로 변환
                pts = [
                    (int(xy1[0]), int(xy1[1])),
                    (int(xy2[0]), int(xy2[1])),
                    (int(xy3[0]), int(xy3[1])),
                    (int(xy4[0]), int(xy4[1]))
                ]

                # 녹색 선으로 사각형 그리기 (두께 2)
                cv2.line(img, pts[0], pts[1], (0, 255, 0), 2)
                cv2.line(img, pts[1], pts[2], (0, 255, 0), 2)
                cv2.line(img, pts[2], pts[3], (0, 255, 0), 2)
                cv2.line(img, pts[3], pts[0], (0, 255, 0), 2)

                # 결과 저장 (파일명에 "box_" prefix 추가)
                output_filename = "box_" + filename
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, img)
                processed_count += 1
            else:
                print(f"좌표 파싱 실패: {filename}")
                error_count += 1

        except Exception as e:
            print(f"처리 중 오류 발생 ({filename}): {str(e)}")
            error_count += 1

    print(f"\n처리 완료!")
    print(f"성공: {processed_count}개")
    print(f"실패: {error_count}개")
    print(f"결과 저장 위치: {output_dir}")


if __name__ == '__main__':
    # 입력/출력 디렉토리 설정
    INPUT_DIR = '/workspace/repo/ultralytics/ultralytics/assets/ccpd_over60_xyxyxyxy/images/test'
    OUTPUT_DIR = '/workspace/repo/ultralytics/ultralytics/assets/ccpd_over60_xyxyxyxy/images/test_vis6dof'

    visualize_ccpd_samples(INPUT_DIR, OUTPUT_DIR)
