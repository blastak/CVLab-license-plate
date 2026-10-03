#!/usr/bin/env python3
"""
IWPOD-tf를 사용하여 이미지 디렉토리의 모든 이미지를 추론하고 CSV 파일로 저장

사용 예시:
    python inference_iwpod_to_csv.py \
        --input_dir /workspace/repo/ultralytics/ultralytics/assets/ccpd_over60_xyxyxyxy/images/test \
        --output_dir /workspace/repo/ultralytics/runs/iwpod/inference_csv_ccpd_over60_test
"""

import sys
import csv
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import load_model_tf, find_lp_corner
from Utils import imread_uni


def inference_to_csv(input_dir, output_dir, iwpod_model):
    """
    디렉토리 내의 모든 이미지에 대해 IWPOD-tf 추론을 수행하고 CSV로 저장

    Args:
        input_dir: 입력 이미지 디렉토리 경로
        output_dir: 출력 CSV 디렉토리 경로
        iwpod_model: IWPOD-tf 모델
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 가져오기 (jpg, jpeg, png 지원)
    image_extensions = ['.jpg', '.jpeg', '.png']
    img_paths = []
    for ext in image_extensions:
        img_paths.extend(input_path.glob(f'*{ext}'))
        img_paths.extend(input_path.glob(f'*{ext.upper()}'))

    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"❌ {input_dir}에서 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"📁 총 {len(img_paths)}개의 이미지 발견")
    print(f"📍 입력 경로: {input_dir}")
    print(f"📍 출력 경로: {output_dir}\n")

    success_count = 0
    fail_count = 0
    total_detections = 0

    for idx, img_path in enumerate(img_paths, 1):
        try:
            # 이미지 로드
            img = imread_uni(str(img_path))

            if img is None:
                print(f"❌ [{idx}/{len(img_paths)}] 이미지 로드 실패: {img_path.name}")
                fail_count += 1
                continue

            # IWPOD 추론
            corners, probs = find_lp_corner(img, iwpod_model)

            # CSV 파일 생성
            csv_filename = output_path / f"{img_path.stem}.csv"

            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # 검출된 번호판이 있는 경우
                if len(corners) > 0:
                    for i, corner in enumerate(corners):
                        # corner는 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태
                        # 좌상단부터 시계방향으로 정렬되어 있음
                        row = [
                            'license_plate',  # cls
                            corner[0][0], corner[0][1],  # 좌상단 (x1, y1)
                            corner[1][0], corner[1][1],  # 우상단 (x2, y2)
                            corner[2][0], corner[2][1],  # 우하단 (x3, y3)
                            corner[3][0], corner[3][1],  # 좌하단 (x4, y4)
                            probs[i]  # confidence
                        ]
                        writer.writerow(row)

                    total_detections += len(corners)
                    success_count += 1

                    if idx % 10 == 0:
                        print(f"✅ [{idx}/{len(img_paths)}] {img_path.name} - {len(corners)}개 검출")
                else:
                    # 검출된 번호판이 없는 경우에도 빈 CSV 파일 생성
                    success_count += 1
                    if idx % 10 == 0:
                        print(f"⚠️  [{idx}/{len(img_paths)}] {img_path.name} - 검출 없음")

        except Exception as e:
            print(f"❌ [{idx}/{len(img_paths)}] 처리 실패: {img_path.name} - {e}")
            fail_count += 1

    # 결과 출력
    print("\n" + "=" * 80)
    print("📊 추론 완료")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"  📁 전체: {len(img_paths)}개")
    print(f"  🎯 총 검출 수: {total_detections}개")
    print(f"  📂 CSV 저장 위치: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='IWPOD-tf를 사용하여 이미지 추론 및 CSV 생성'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='입력 이미지 디렉토리 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='출력 CSV 디렉토리 경로'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='./weights/iwpod_net',
        help='IWPOD-tf 모델 가중치 경로 (기본값: ./weights/iwpod_net)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("IWPOD-tf Inference to CSV")
    print("=" * 80)
    print(f"📦 모델 로딩 중: {args.weights}")

    # IWPOD-tf 모델 로드
    try:
        iwpod_model = load_model_tf(args.weights)
        print("✅ 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        sys.exit(1)

    # 추론 수행
    inference_to_csv(args.input_dir, args.output_dir, iwpod_model)


if __name__ == '__main__':
    main()
