"""
OCR 평가 유닛 테스트 스크립트
각 단계별로 테스트하여 문제 진단
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from Utils import imread_uni, trans_eng2kor_v1p3


def test_1_load_gt():
    """테스트 1: JSON에서 GT 로드"""
    print("\n" + "="*80)
    print("테스트 1: JSON에서 GT 로드")
    print("="*80)

    json_path = Path("/workspace/repo/ultralytics/ultralytics/assets/good_all_obb1944/labels/test_original/20250407_162943_340.json")

    print(f"JSON 파일: {json_path}")

    if not json_path.exists():
        print(f"❌ JSON 파일이 존재하지 않습니다!")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n전체 JSON 구조:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"\nshapes 파싱:")
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])

        print(f"  label: {label}")

        if '_' in label:
            plate_class, plate_number = label.split('_', 1)
            print(f"    클래스: {plate_class}")
            print(f"    번호: {plate_number}")
        else:
            print(f"    ⚠️ underscore가 없어서 파싱 불가")

        print(f"    좌표 개수: {len(points)}")

    print("\n✅ 테스트 1 완료")
    return data


def test_2_load_ocr_model():
    """테스트 2: OCR 모델 로드"""
    print("\n" + "="*80)
    print("테스트 2: OCR 모델 로드")
    print("="*80)

    model_path = "../LP_Recognition/VIN_OCR/weight"
    print(f"모델 경로: {model_path}")

    try:
        ocr_model = load_model_VinOCR(model_path)
        print("✅ OCR 모델 로드 성공")
        return ocr_model
    except Exception as e:
        print(f"❌ OCR 모델 로드 실패: {e}")
        return None


def test_3_run_ocr_on_frontalized():
    """테스트 3: Frontalized 이미지에 OCR 실행"""
    print("\n" + "="*80)
    print("테스트 3: Frontalized 이미지에 OCR 실행")
    print("="*80)

    # 테스트용 frontalized 이미지
    front_img_path = "./test_output_frontalized.jpg"

    if not Path(front_img_path).exists():
        print(f"❌ Frontalized 이미지가 없습니다: {front_img_path}")
        print("먼저 test_single_image.py를 실행하세요.")
        return None

    print(f"이미지: {front_img_path}")

    # OCR 모델 로드
    ocr_model = test_2_load_ocr_model()
    if ocr_model is None:
        return None

    # 이미지 로드
    img = imread_uni(front_img_path)
    print(f"이미지 크기: {img.shape}")

    # OCR 실행
    print("\nOCR 실행 중...")
    try:
        # forward는 배치 입력을 받음
        ocr_results_batch = ocr_model.forward([img])

        print(f"배치 결과 개수: {len(ocr_results_batch)}")

        if len(ocr_results_batch) > 0:
            ocr_results = ocr_results_batch[0]  # 첫 번째 이미지의 결과
            print(f"검출된 문자 박스 개수: {len(ocr_results)}")

            if len(ocr_results) > 0:
                print("\n개별 문자 검출 결과:")
                for i, det in enumerate(ocr_results):
                    print(f"  문자 {i}: '{det.class_str}' (conf: {det.conf:.3f}, pos: x={det.x}, y={det.y})")

                # check_align으로 정렬 (P1-1 -> plate_type_idx=1)
                plate_type_idx = 1  # 테스트 이미지가 P1-1이므로
                list_char = ocr_model.check_align(ocr_results, plate_type_idx)
                pred_text_list = trans_eng2kor_v1p3(list_char)
                pred_text = ''.join(pred_text_list) if isinstance(pred_text_list, list) else pred_text_list

                print(f"\n정렬된 문자 리스트: {list_char}")
                print(f"한글 변환 리스트: {pred_text_list}")
                print(f"최종 결과: {pred_text}")

                print("\n✅ 테스트 3 완료")
                return pred_text
            else:
                print("⚠️ OCR 결과 없음")
                return ""
        else:
            print("⚠️ 배치 결과 없음")
            return ""

    except Exception as e:
        print(f"❌ OCR 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_4_match_gt_and_pred():
    """테스트 4: GT와 예측 매칭"""
    print("\n" + "="*80)
    print("테스트 4: GT와 예측 매칭")
    print("="*80)

    # GT 로드
    gt_data = test_1_load_gt()
    if gt_data is None:
        return

    gt_text = None
    gt_class = None

    for shape in gt_data.get('shapes', []):
        label = shape.get('label', '')
        if '_' in label:
            gt_class, gt_text = label.split('_', 1)
            break

    if gt_text is None:
        print("❌ GT 텍스트를 찾을 수 없습니다")
        return

    print(f"GT 클래스: {gt_class}")
    print(f"GT 텍스트: {gt_text}")

    # OCR 실행
    pred_text = test_3_run_ocr_on_frontalized()

    if pred_text is None:
        return

    print(f"\n예측 텍스트: {pred_text}")

    # 매칭
    match = (pred_text == gt_text)

    print(f"\n매칭 결과: {'✅ 일치' if match else '❌ 불일치'}")

    if not match:
        print(f"  GT:   '{gt_text}'")
        print(f"  Pred: '{pred_text}'")

        # 문자별 비교
        print("\n  문자별 비교:")
        max_len = max(len(gt_text), len(pred_text))
        for i in range(max_len):
            gt_char = gt_text[i] if i < len(gt_text) else '(없음)'
            pred_char = pred_text[i] if i < len(pred_text) else '(없음)'
            match_char = '✓' if gt_char == pred_char else '✗'
            print(f"    위치 {i}: GT='{gt_char}' vs Pred='{pred_char}' [{match_char}]")


def test_5_frontalized_folder_structure():
    """테스트 5: Frontalized 폴더 구조 확인"""
    print("\n" + "="*80)
    print("테스트 5: Frontalized 폴더 구조 확인")
    print("="*80)

    # 예상 폴더 구조
    frontalized_root = Path("./output/frontalized")

    if not frontalized_root.exists():
        print(f"❌ Frontalized 폴더가 없습니다: {frontalized_root}")
        return

    print(f"Frontalized 루트: {frontalized_root}")

    # 검출기별 폴더 확인
    for detector_folder in frontalized_root.iterdir():
        if detector_folder.is_dir():
            print(f"\n검출기: {detector_folder.name}")

            # 클래스별 폴더 확인
            class_folders = [f for f in detector_folder.iterdir() if f.is_dir()]
            print(f"  클래스 폴더 개수: {len(class_folders)}")

            total_images = 0
            for class_folder in class_folders:
                images = list(class_folder.glob("*.jpg"))
                total_images += len(images)
                print(f"    {class_folder.name}: {len(images)}개 이미지")

            print(f"  총 이미지: {total_images}개")

    print("\n✅ 테스트 5 완료")


def test_6_filename_parsing():
    """테스트 6: 파일명 파싱 테스트"""
    print("\n" + "="*80)
    print("테스트 6: 파일명 파싱 테스트")
    print("="*80)

    test_filenames = [
        "20250407_162943_340_0.jpg",
        "image001_1.jpg",
        "test_image_2.jpg",
    ]

    for filename in test_filenames:
        filename_parts = Path(filename).stem.rsplit('_', 1)

        print(f"\n파일명: {filename}")
        print(f"  파싱 결과: {filename_parts}")

        if len(filename_parts) == 2:
            orig_filename, det_idx = filename_parts
            print(f"    원본 파일명: {orig_filename}")
            print(f"    검출 인덱스: {det_idx}")
        else:
            print(f"    ⚠️ 파싱 실패 (underscore로 분리 불가)")

    print("\n✅ 테스트 6 완료")


def main():
    print("\n" + "="*80)
    print("OCR 평가 유닛 테스트")
    print("="*80)

    # 모든 테스트 실행
    test_1_load_gt()
    test_2_load_ocr_model()
    test_3_run_ocr_on_frontalized()
    test_4_match_gt_and_pred()
    test_5_frontalized_folder_structure()
    test_6_filename_parsing()

    print("\n" + "="*80)
    print("전체 테스트 완료")
    print("="*80)


if __name__ == '__main__':
    main()
