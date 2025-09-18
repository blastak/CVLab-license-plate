import argparse
import json
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from Evaluation.csv_loader import TestsetLoader


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


def save_results_json(metrics_per_class, mAP50, mAP5095, precision, recall, output_path):
    results = {
        "overall": {
            "mAP@0.5": mAP50,
            "mAP@0.5:0.95": mAP5095,
            "precision": precision,
            "recall": recall
        },
        "per_class": metrics_per_class
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def save_results_excel(metrics_per_class, mAP50, mAP5095, precision, recall, output_path):
    # per-class metrics
    df = pd.DataFrame.from_dict(metrics_per_class, orient='index')
    df.index.name = "Class"

    # add overall as a separate row
    overall = pd.DataFrame([{
        "map50": mAP50,
        "map5095": mAP5095,
        "precision": precision,
        "recall": recall
    }], index=["OVERALL"])

    df_combined = pd.concat([df, overall])
    df_combined.to_excel(output_path)


def calculate_metrics_per_class(results, count_classes, iou_thresholds):
    metrics_per_class = {}
    total_tp = 0
    total_fp = 0
    eps = 1e-9

    for cls in list(count_classes.keys()):
        gt_count = count_classes[cls]
        ap_per_iou = []
        precisions_at_50 = None
        recalls_at_50 = None

        for target_iou in iou_thresholds:
            tp = np.array(results[cls]['tp'][target_iou]).astype(int)
            conf = np.array(results[cls]['conf'][target_iou])

            if len(tp) == 0:
                ap_per_iou.append(0)
                continue

            # confidence 내림차순 정렬
            sorted_indices = np.argsort(-conf)
            tp_sorted = tp[sorted_indices]
            fp_sorted = 1 - tp_sorted  # FP 보정

            cum_tp = np.cumsum(tp_sorted)
            cum_fp = np.cumsum(fp_sorted)

            recalls = cum_tp / (gt_count + eps)
            precisions = cum_tp / (cum_tp + cum_fp + eps)

            # precision 보간 (단조 감소)
            precisions = np.maximum.accumulate(precisions[::-1])[::-1]

            # (0,0) 시작점 추가
            recalls = np.concatenate(([0.0], recalls))
            precisions = np.concatenate(([1.0], precisions))

            # ap 계산 (COCO-style 101-point interpolation)
            recall_levels = np.linspace(0, 1, 101)
            precisions_interp = []
            for t in recall_levels:
                precisions_interp.append(np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0)
            ap = np.mean(precisions_interp)

            ap_per_iou.append(ap)

            if target_iou == 0.5:
                precisions_at_50 = np.mean(precisions)
                recalls_at_50 = recalls[-1]
                total_tp += np.sum(tp_sorted)
                total_fp += np.sum(fp_sorted)

        map5095 = np.mean(ap_per_iou)  # mAP50-95 계산
        metrics_per_class[cls] = {
            "map50": ap_per_iou[0],
            "map5095": map5095,
            "precision": precisions_at_50,
            "recall": recalls_at_50
        }

    # 종합 precision, recall, mAP 계산
    overall_precision = total_tp / (total_tp + total_fp + eps)
    overall_recall = total_tp / (sum(count_classes.values()) + eps)

    mAP50 = sum(metrics_per_class[cls]['map50'] for cls in metrics_per_class) / len(metrics_per_class)
    mAP5095 = sum(metrics_per_class[cls]['map5095'] for cls in metrics_per_class) / len(metrics_per_class)

    return metrics_per_class, mAP50, mAP5095, overall_precision, overall_recall


def tp_fp(predictions, count_classes):
    all_classes = set(count_classes.keys()) | {pred[1] for pred in predictions}
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    results = {cls: {"tp": {iou: [] for iou in iou_thresholds}, "conf": {iou: [] for iou in iou_thresholds}} for cls in all_classes}

    for pb, pc, pcf, gb, gc, i in predictions:
        for iou_threshold in iou_thresholds:
            TP = False if gc is None or pc != gc or i < iou_threshold else True
            results[pc]["tp"][iou_threshold].append(TP)
            results[pc]["conf"][iou_threshold].append(pcf)
    return results, iou_thresholds


def normalize_class(loader, plate_type):
    """loader 상태에 따라 GT 클래스 이름을 보정"""
    if loader.mono_cls:
        return loader.mc
    elif loader.merge_P1 and plate_type.startswith("P1"):
        return "P1"
    return plate_type

def prepare_predictions(loader, mode="quad"):
    """
    GT와 예측을 매칭하여 predictions, count_classes 반환
    predictions: [pred_coords, pred_class, conf, matched_gt, gt_class, iou]
    """
    predictions = []
    count_classes = {}

    for csv_file in loader.list_csv:
        base_name = os.path.splitext(csv_file)[0]
        json_file = f"{base_name}.json"

        if json_file not in loader.list_json:
            continue

        pred = loader.parse_detect(csv_file)
        gt = loader.parse_label(json_file)

        # GT 클래스 카운트
        for plate_type_gt, gt_coords in gt:
            gt_class = normalize_class(loader, plate_type_gt)
            count_classes[gt_class] = count_classes.get(gt_class, 0) + 1

        matched = set()  # 이미 매칭된 GT 인덱스 기록

        for plate_type_pred, pred_coords, conf in pred:
            best_iou = 0
            best_match = None
            best_type = None
            best_idx = -1

            for idx, (plate_type_gt, gt_coords) in enumerate(gt):
                if idx in matched:
                    continue

                effective_gt_type = normalize_class(loader, plate_type_gt)

                if mode == "quad":
                    iou_val = QQ_iou(pred_coords, gt_coords)  # 사각형 IoU
                else:
                    iou_val = QQ_iou(pred_coords, gt_coords, 1)  # 박스 IoU

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_match = gt_coords
                    best_type = effective_gt_type
                    best_idx = idx

            if best_idx != -1:
                matched.add(best_idx)

            predictions.append([pred_coords, plate_type_pred, conf, best_match, best_type, best_iou])
    if loader.mono_cls:
        total_gt = sum(count_classes.values())
        count_classes = {loader.mc: total_gt}

    return predictions, count_classes


def save_predictions(predictions, folder_name, mode):
    """예측 결과를 텍스트 파일로 저장"""
    path = f"./results/predictions_{folder_name}_{mode}.txt"
    with open(path, "w") as f:
        for pred in predictions:
            line = " ".join(map(str, pred))
            f.write(line + "\n")


def save_tpfp(results, folder_name, mode):
    """TP/FP 결과 저장 (요약 + 상세)"""
    # 요약 저장
    summary_path = f"./results/tpfp_{folder_name}_{mode}.txt"
    with open(summary_path, "w") as f:
        for r in results:
            line = " ".join(map(str, r))  # 리스트를 문자열로 변환
            f.write(line + "\n")
    with open(summary_path, "w") as f:
        for image_id, data in results.items():
            f.write(f"Image: {image_id}\n")
            for iou_thresh, confs in data["conf"].items():
                tps = data["tp"][iou_thresh]
                for idx, (conf, tp) in enumerate(zip(confs, tps)):
                    f.write(f"  IoU {iou_thresh}, Instance {idx}, "
                            f"Conf: {conf:.4f}, TP: {tp}\n")
            f.write("\n")


def print_metrics(metrics_per_class, mAP50, mAP5095, precision, recall):
    """평가지표 콘솔 출력"""
    print("\n== Per-Class Metrics ==")
    for cls, metric in metrics_per_class.items():
        print(f"[{cls}] mAP@0.5: {metric['map50']:.4f}, "
              f"mAP@0.5:0.95: {metric['map5095']:.4f}, "
              f"Precision: {metric['precision']:.4f}, "
              f"Recall: {metric['recall']:.4f}")

    print(f"\nOverall mAP@0.5: {mAP50:.4f}")
    print(f"Overall mAP@0.5:0.95: {mAP5095:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")


def eval(csv_dir, json_dir, mode='quad'):
    loader = TestsetLoader(csv_path=csv_dir, json_path=json_dir)
    if loader.valid:
        folder_name = os.path.basename(os.path.normpath(csv_dir))
        folder_name = folder_name.removesuffix("_csv")
        os.makedirs('./results', exist_ok=True)

        # 1. 데이터 준비
        predictions, count_classes = prepare_predictions(loader, mode)

        # 2. 결과 저장
        save_predictions(predictions, folder_name, mode)

        # 3. 평가 지표 계산
        results, iou_thresholds = tp_fp(predictions, count_classes)
        save_tpfp(results, folder_name, mode)

        metrics_per_class, mAP50, mAP5095, precision, recall = calculate_metrics_per_class(results, count_classes, iou_thresholds)

        save_results_json(metrics_per_class, mAP50, mAP5095, precision, recall, f"./results/eval_result_{folder_name}_{mode}.json")
        save_results_excel(metrics_per_class, mAP50, mAP5095, precision, recall, f"./results/eval_result_{folder_name}_{mode}.xlsx")

        # 4. 출력
        print_metrics(metrics_per_class, mAP50, mAP5095, precision, recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--csv_path', type=str, default=r'VIN_csv', help='Input Image folder')
    parser.add_argument('-o', '--json_path', type=str, default=r'testset', help='Input Image folder')
    opt = parser.parse_args()
    csv_path = opt.csv_path
    json_path = opt.json_path

    eval(csv_path, json_path)  # 실행 전 detector 에서 VIN_to_csv() 먼저 실행. -> csv 파일 생성
    eval(csv_path, json_path, 'BBox')

# list = [
#     'VIN_LPD', 'YOLOv11', 'YOLOv11OBB', 'IWPOD_torch', 'IWPOD_tf'
# ]