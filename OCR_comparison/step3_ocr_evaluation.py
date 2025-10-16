"""
Step 3: OCR 실행 및 성능 비교
Frontalized된 이미지를 읽어서 VIN_OCR을 실행하고, JSON GT와 비교하여 성능 평가
출력: CSV 파일 (GT, Pred 열거)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3


def load_ground_truth(json_folder):
    """
    JSON 파일들로부터 GT 정보를 로드
    Returns: {filename: [(class, license_plate_number), ...]}
    """
    json_folder = Path(json_folder)
    gt_dict = {}

    json_files = [f for f in json_folder.iterdir() if f.suffix == '.json']

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filename = json_file.stem
        gt_list = []

        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if '_' in label:
                plate_class, plate_number = label.split('_', 1)
                gt_list.append((plate_class, plate_number))

        gt_dict[filename] = gt_list

    return gt_dict


def run_ocr_on_frontalized(frontalized_folder, detector_name, ocr_model, gt_dict):
    """
    Frontalized된 이미지에 대해 OCR 실행

    Args:
        frontalized_folder: frontalized 폴더 경로
        detector_name: 검출기 이름
        ocr_model: VIN_OCR 모델
        gt_dict: GT 딕셔너리

    Returns:
        results: [(detector, filename, det_idx, gt_class, gt_text, pred_text, match)]
    """
    frontalized_folder = Path(frontalized_folder) / detector_name
    results = []

    if not frontalized_folder.exists():
        print(f"경고: {frontalized_folder} 폴더가 존재하지 않습니다.")
        return results

    # 모든 클래스 폴더를 순회
    class_folders = [f for f in frontalized_folder.iterdir() if f.is_dir()]

    total_processed = 0

    for class_folder in class_folders:
        plate_class = class_folder.name

        # 해당 클래스의 모든 이미지 파일 처리
        img_files = sorted([f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

        for img_file in img_files:
            # 파일명 파싱: {원본파일명}_{검출인덱스}.jpg
            filename_parts = img_file.stem.rsplit('_', 1)
            if len(filename_parts) != 2:
                continue

            orig_filename, det_idx = filename_parts

            # GT 찾기
            gt_list = gt_dict.get(orig_filename, [])

            # 이미지 로드 및 OCR 수행
            try:
                img = imread_uni(str(img_file))

                # OCR 실행 (forward는 배치 입력을 받으므로 리스트로 전달)
                ocr_results_batch = ocr_model.forward([img])

                # 결과는 리스트의 리스트이므로 첫 번째 요소 가져오기
                if len(ocr_results_batch) > 0:
                    ocr_results = ocr_results_batch[0]  # 첫 번째 이미지의 결과
                else:
                    ocr_results = []

                # GT와 매칭 (같은 인덱스 순서로)
                det_idx_int = int(det_idx)
                if det_idx_int < len(gt_list):
                    gt_class, gt_text = gt_list[det_idx_int]
                else:
                    # GT가 없는 경우 (False Positive)
                    gt_class = ""
                    gt_text = ""

                # 클래스 인덱스 추출 (P1-1 -> 1, P2 -> 2 등)
                if gt_class:
                    try:
                        # P1-1, P1-2 등의 경우 P1로 통일
                        if '-' in gt_class:
                            plate_type_idx = int(gt_class.split('-')[0][1:])
                        else:
                            plate_type_idx = int(gt_class[1:])
                    except:
                        plate_type_idx = 1  # 기본값
                else:
                    plate_type_idx = 1

                # check_align으로 문자열 정렬 및 조합
                if len(ocr_results) > 0:
                    list_char = ocr_model.check_align(ocr_results, plate_type_idx)
                    pred_text_list = trans_eng2kor_v1p3(list_char)
                    pred_text = ''.join(pred_text_list) if isinstance(pred_text_list, list) else pred_text_list
                else:
                    pred_text = ""

                # 정확도 판단 (완전 일치)
                match = (pred_text == gt_text) if gt_text else False

                results.append((detector_name, orig_filename, det_idx_int, gt_class, gt_text, pred_text, match))

                total_processed += 1

            except Exception as e:
                print(f"경고: {img_file} 처리 실패: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"  {detector_name}: {total_processed}개 번호판 OCR 완료")

    return results


def save_results_to_csv(results, output_csv_path):
    """
    결과를 CSV 파일로 저장
    형식: detector, filename, det_idx, gt_class, gt_text, pred_text, match
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 헤더 작성
        writer.writerow(['detector', 'filename', 'det_idx', 'gt_class', 'gt_text', 'pred_text', 'match'])

        # 데이터 작성
        for row in results:
            writer.writerow(row)

    print(f"\n결과 CSV 저장 완료: {output_csv_path}")


def calculate_accuracy(results):
    """
    검출기별 정확도 계산 및 출력
    """
    detector_stats = {}

    for row in results:
        detector, filename, det_idx, gt_class, gt_text, pred_text, match = row

        if detector not in detector_stats:
            detector_stats[detector] = {'total': 0, 'correct': 0}

        # GT가 있는 경우만 카운트 (FP 제외)
        if gt_text:
            detector_stats[detector]['total'] += 1
            if match:
                detector_stats[detector]['correct'] += 1

    print("\n=== OCR 성능 비교 ===")
    for detector, stats in detector_stats.items():
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"[{detector}] 정확도: {correct}/{total} = {accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frontalized 이미지에 대해 OCR 실행 및 성능 비교')
    parser.add_argument('--frontalized_folder', type=str, required=True, help='Frontalized 이미지 폴더 경로')
    parser.add_argument('--json_folder', type=str, required=True, help='GT JSON 폴더 경로')
    parser.add_argument('--ocr_model_path', type=str, required=True, help='VIN_OCR 모델 경로')
    parser.add_argument('--output_csv', type=str, required=True, help='출력 CSV 파일 경로')
    parser.add_argument('--detectors', type=str, nargs='+', required=True,
                        help='비교할 검출기 이름들 (예: VIN_LPD IWPOD_tf)')

    args = parser.parse_args()

    # GT 로드
    print("GT 데이터 로딩 중...")
    gt_dict = load_ground_truth(args.json_folder)
    print(f"  {len(gt_dict)}개 이미지의 GT 로드 완료")

    # OCR 모델 로드
    print(f"\nVIN_OCR 모델 로딩 중: {args.ocr_model_path}")
    ocr_model = load_model_VinOCR(args.ocr_model_path)

    # 각 검출기에 대해 OCR 실행
    all_results = []

    print("\nOCR 실행 중...")
    for detector_name in args.detectors:
        print(f"\n[{detector_name}] 처리 시작...")
        results = run_ocr_on_frontalized(args.frontalized_folder, detector_name, ocr_model, gt_dict)
        all_results.extend(results)

    # 결과 저장
    save_results_to_csv(all_results, args.output_csv)

    # 정확도 계산 및 출력
    calculate_accuracy(all_results)
