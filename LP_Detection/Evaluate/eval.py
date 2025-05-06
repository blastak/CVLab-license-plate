import argparse
import os

import numpy as np
from shapely.geometry import Polygon

from LP_Detection.Evaluate.csv_loader import DatasetLoader


def QQ_iou(poly1, poly2, mode=0):
    if not poly1 or not poly2:
        return 0.0
    if mode == 1:  # BBox로 변환
        poly1 = Polygon(convert_Quad_to_Box(poly1))
        poly2 = Polygon(convert_Quad_to_Box(poly2))
    else:
        poly1 = Polygon(poly1)
        poly2 = Polygon(poly2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return inter / union


def convert_Quad_to_Box(Quad):
    x_min = min(p[0] for p in Quad)
    y_min = min(p[1] for p in Quad)
    x_max = max(p[0] for p in Quad)
    y_max = max(p[1] for p in Quad)
    BBox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return BBox


def calculate_metrics_per_class(results, count_classes, iou_thresholds):
    # 특정 클래스와 IoU threshold
    metrics_per_class = {}
    total_tp = 0
    total_fp = 0

    for cls in list(count_classes.keys()):
        gt_count = count_classes[cls]
        ap_per_iou = []
        precisions_at_50 = None
        recalls_at_50 = None

        for target_iou in iou_thresholds:
            tp = np.array(results[cls]['tp'][target_iou])  # True Positive
            conf = np.array(results[cls]['conf'][target_iou])  # Confidence scores
            if len(tp) == 0:
                ap_per_iou.append(0)
                continue

            # Precision, Recall 계산
            sorted_indices = np.argsort(-conf)
            tp_sorted = tp[sorted_indices]

            cum_tp = np.cumsum(tp_sorted)
            cum_fp = np.cumsum(~tp_sorted)

            # Precision, Recall 계산
            precisions = cum_tp / (cum_tp + cum_fp)
            recalls = cum_tp / gt_count
            if target_iou == 0.5:
                precisions_at_50 = precisions[-1]
                recalls_at_50 = recalls[-1]
                total_tp += np.sum(tp_sorted)
                total_fp += np.sum(~tp_sorted)

            ap = compute_ap(precisions, recalls)
            ap_per_iou.append(ap)

        map5095 = np.mean(ap_per_iou)
        metrics_per_class[cls] = {
            "map50": ap_per_iou[0],  # AP at IoU 0.5
            "map5095": map5095,
            "precision": precisions_at_50,
            "recall": recalls_at_50
        }
    # 전체 Precision과 Recall 계산 (IoU 0.5 기준)
    overall_precision = total_tp / (total_tp + total_fp)
    overall_recall = total_tp / sum(count_classes.values())

    return metrics_per_class, overall_precision, overall_recall


def compute_ap(precision, recall):
    """
    Computes Average Precision (AP) using the Precision-Recall curve.

    Args:
        precision (list or np.ndarray): Precision values.
        recall (list or np.ndarray): Recall values.

    Returns:
        float: Average Precision (AP) value.
    """
    # Ensure recall starts with 0 and ends with 1
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute the area under the curve using trapezoidal rule
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap


def calculate_metrics(predictions, count_classes, iou_threshold=0.5):
    all_classes = set(count_classes.keys()) | {pred[1] for pred in predictions}
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    results = {cls: {"tp": {iou: [] for iou in iou_thresholds}, "conf": {iou: [] for iou in iou_thresholds}} for cls in all_classes}

    for pb, pc, pcf, gb, gc, i in predictions:
        for iou_threshold in iou_thresholds:
            detected = False
            if pc == gc:  # Only match boxes of the same class
                if i >= iou_threshold and not detected:
                    detected = True
                    results[gc]["tp"][iou_threshold].append(True)  # True Positive
                    results[gc]["conf"][iou_threshold].append(pcf)
                else:
                    results[gc]["tp"][iou_threshold].append(False)  # False Positive
                    results[gc]["conf"][iou_threshold].append(pcf)
            else:
                results[pc]["tp"][iou_threshold].append(False)  # False Positive
                results[pc]["conf"][iou_threshold].append(pcf)

    metrics_per_class, precision, recall = calculate_metrics_per_class(results, count_classes, iou_thresholds)

    mAP50 = sum(metrics_per_class[cls]['map50'] for cls in metrics_per_class) / len(metrics_per_class)
    mAP5095 = sum(metrics_per_class[cls]['map5095'] for cls in metrics_per_class) / len(metrics_per_class)

    return metrics_per_class, mAP50, mAP5095, precision, recall


def check_mono_cls(loader):  # P0 있으면 단일 클래스
    mono_cls = False

    for jpg_file in loader.list_jpg:
        base_name = os.path.splitext(jpg_file)[0]
        csv_file = f"{base_name}.csv"

        if csv_file in loader.list_csv:
            pred = loader.parse_detect(csv_file)
            if any(p[0] == 'P0' for p in pred):
                mono_cls = True
                break  # 하나만 확인되면 충분
    return mono_cls

def check_P1(loader):
    """예측 결과에 P1이 있으면 GT에서 P1-*을 모두 P1로 병합하도록 판단."""
    for jpg_file in loader.list_jpg:
        base_name = os.path.splitext(jpg_file)[0]
        csv_file = f"{base_name}.csv"

        if csv_file in loader.list_csv:
            pred = loader.parse_detect(csv_file)
            if any(p[0] == 'P1' for p in pred):  # 예측 클래스에 'P1'이 포함되어 있다면
                return True
    return False


def eval(prefix, mode='quad'):
    loader = DatasetLoader(base_path=prefix)
    if loader.valid:
        predictions = []
        count_classes = {}
        mono_cls = check_mono_cls(loader)
        merge_P1 = check_P1(loader)

        for jpg_file in loader.list_jpg:
            base_name = os.path.splitext(jpg_file)[0]
            csv_file = f"{base_name}.csv"
            json_file = f"{base_name}.json"

            if csv_file in loader.list_csv and json_file in loader.list_json:
                pred = loader.parse_detect(csv_file)
                gt = loader.parse_label(json_file)

                for plate_type_gt, gt_coords in gt:
                    if mono_cls:
                        gt_class = 'P0'
                    elif merge_P1 and plate_type_gt.startswith('P1'):
                        gt_class = 'P1'
                    else:
                        gt_class = plate_type_gt
                    count_classes[gt_class] = count_classes.get(gt_class, 0) + 1

                matched = set()  # 이미 매칭된 GT의 인덱스를 저장

                for plate_type_pred, pred_coords, conf in pred:
                    best_iou = 0
                    best_match = None
                    best_type = None
                    best_idx = -1

                    for idx, (plate_type_gt, gt_coords) in enumerate(gt):
                        if idx in matched:
                            continue  # 이미 매칭된 GT는 스킵

                        if mono_cls:
                            effective_gt_type = 'P0'
                        elif merge_P1 and plate_type_gt.startswith('P1'):
                            effective_gt_type = 'P1'
                        else:
                            effective_gt_type = plate_type_gt

                        if mode == 'quad':
                            iou_QQ = QQ_iou(pred_coords, gt_coords)  # quad
                        else:
                            iou_QQ = QQ_iou(pred_coords, gt_coords, 1)  # Box

                        if iou_QQ > best_iou:
                            best_iou = iou_QQ
                            best_match = gt_coords
                            best_type = effective_gt_type
                            best_idx = idx

                    if best_idx != -1:
                        matched.add(best_idx)  # 이 GT는 더 이상 사용되지 않도록 설정

                    predictions.append([pred_coords, plate_type_pred, conf, best_match, best_type, best_iou])
        if mono_cls:
            total_gt = sum(count_classes.values())
            count_classes = {'P0': total_gt}

        # 메트릭 계산
        metrics_per_class, mAP50, mAP5095, precision, recall = calculate_metrics(
            predictions,
            count_classes,
            iou_threshold=0.5
        )

        # 출력
        print("\n== Per-Class Metrics ==")
        for cls, metric in metrics_per_class.items():
            print(f"[{cls}] mAP@0.5: {metric['map50']:.4f}, mAP@0.5:0.95: {metric['map5095']:.4f}, "
                  f"Precision: {metric['precision']:.4f}, Recall: {metric['recall']:.4f}")

        print(f"\nOverall mAP@0.5: {mAP50:.4f}")
        print(f"Overall mAP@0.5:0.95: {mAP5095:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default=r'D:\Dataset\LicensePlate\test\test_IWPOD_\GoodMatches_P4', help='Input Image folder')
    opt = parser.parse_args()
    prefix = opt.data

    eval(prefix)
    eval(prefix, 'BBox')
