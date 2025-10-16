"""
검출 성능 평가 (mAP 계산)
Excel 저장 없이 콘솔 출력만
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from Evaluation.csv_loader import TestsetLoader
from Evaluation.eval import prepare_predictions, tp_fp, calculate_metrics_per_class, print_metrics

def eval_detection(csv_dir, json_dir, detector_name, mode='quad'):
    """검출 성능 평가"""
    print(f"\n{'='*80}")
    print(f"검출 성능 평가: {detector_name}")
    print(f"{'='*80}")

    loader = TestsetLoader(csv_path=csv_dir, json_path=json_dir)

    if not loader.valid:
        print("❌ CSV와 JSON 개수가 일치하지 않습니다!")
        return

    print(f"CSV 파일: {len(loader.list_csv)}개")
    print(f"JSON 파일: {len(loader.list_json)}개")
    print(f"모노 클래스: {loader.mono_cls} (mc={loader.mc})")
    print(f"P1 병합: {loader.merge_P1}")

    # 1. 데이터 준비
    predictions, count_classes = prepare_predictions(loader, mode)
    print(f"\n총 예측 개수: {len(predictions)}")
    print(f"GT 클래스별 개수: {count_classes}")

    # 2. TP/FP 계산
    results, iou_thresholds = tp_fp(predictions, count_classes)

    # 3. 메트릭 계산
    metrics_per_class, mAP50, mAP5095, precision, recall = calculate_metrics_per_class(
        results, count_classes, iou_thresholds
    )

    # 4. 출력
    print_metrics(metrics_per_class, mAP50, mAP5095, precision, recall)

    return {
        'detector': detector_name,
        'mAP50': mAP50,
        'mAP5095': mAP5095,
        'precision': precision,
        'recall': recall,
        'metrics_per_class': metrics_per_class
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='검출 성능 평가')
    parser.add_argument('--csv_dir', type=str, required=True, help='CSV 폴더 경로')
    parser.add_argument('--json_dir', type=str, required=True, help='JSON 폴더 경로')
    parser.add_argument('--detector', type=str, required=True, help='검출기 이름')
    parser.add_argument('--mode', type=str, default='quad', choices=['quad', 'BBox'],
                        help='IoU 계산 모드')

    args = parser.parse_args()

    eval_detection(args.csv_dir, args.json_dir, args.detector, args.mode)
