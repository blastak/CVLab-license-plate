import argparse
import os
import time

import numpy as np

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from LP_Detection.VIN_LPD import load_model_VinLPD
from Utils import imread_uni, iou


def calculate_precision_recall(tp, fp, fn):
    # Calculate Precision, Recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def calculate_metrics(d_net, loader, img_paths, prefix, iou_threshold=0.5):
    """
    Calculate Precision, Recall, F1-score, mAP, and FPS for the VIN_LPD detection model on the test dataset.
    """
    tp, fp, fn = 0, 0, 0  # True Positive, False Positive, False Negative
    all_precisions = []
    all_recalls = []

    total_time = 0  # 총 처리 시간
    num_images = len(img_paths)  # 이미지 개수

    for img_path in img_paths:
        # Load image and predictions
        img = imread_uni(os.path.join(prefix, img_path))

        # 처리 시작 시간 기록
        start_time = time.time()

        d_out = d_net.forward(img)  # 예측 결과

        # 처리 종료 시간 기록 및 시간 누적
        total_time += time.time() - start_time

        # Load ground truth (GT) bounding box
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(img_path[:-4] + '.json')
        gt_box = [left, top, right, bottom]

        # Parse predicted boxes
        predicted_boxes = []
        confidences = []
        if d_out:  # If there are predictions
            for det in d_out:
                x_min, y_min, x_max, y_max = det.x, det.y, det.x + det.w, det.y + det.h
                predicted_boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(det.conf)  # Detection confidence

            # Sort predictions by confidence
            predictions_sorted = sorted(zip(predicted_boxes, confidences), key=lambda x: x[1], reverse=True)
            predicted_boxes = [x[0] for x in predictions_sorted]

        # Match predicted boxes to GT boxes
        tp_img = [0] * len(predicted_boxes)
        fp_img = [0] * len(predicted_boxes)
        detected = False

        for i, pred_box in enumerate(predicted_boxes):
            iou_value = iou(pred_box, gt_box)
            if iou_value >= iou_threshold and not detected:
                tp_img[i] = 1  # True Positive
                detected = True
                tp += 1
            else:
                fp_img[i] = 1  # False Positive

        # If no predictions match the GT box, count as False Negative
        if not detected:
            fn += 1

        # Count False Positives for unmatched predictions
        fp += len(predicted_boxes) - sum(tp_img)

        # Cumulative sums for precision and recall
        tp_cumsum = np.cumsum(tp_img)
        fp_cumsum = np.cumsum(fp_img)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / 1  # Only 1 GT box in this dataset

        # Store precision and recall for this image
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)

    precision, recall, f1_score = calculate_precision_recall(tp, fp, fn)

    # Interpolating the precision-recall curve
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    sorted_indices = np.argsort(all_recalls)
    sorted_recalls = all_recalls[sorted_indices]
    sorted_precisions = all_precisions[sorted_indices]

    # Ensure recall and precision arrays cover full range
    sorted_recalls = np.concatenate(([0], sorted_recalls, [1]))
    sorted_precisions = np.concatenate(([sorted_precisions[0]], sorted_precisions, [0]))

    sorted_precisions = np.flip(np.maximum.accumulate(np.flip(sorted_precisions)))

    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, sorted_recalls, sorted_precisions), x)  # integrate
    else:  # 'continuous'
        i = np.where(sorted_recalls[1:] != sorted_recalls[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((sorted_recalls[i + 1] - sorted_recalls[i]) * sorted_precisions[i + 1])  # area under curve

    # FPS 계산
    fps = num_images / total_time

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"mAP: {ap:.4f}")
    print(f"FPS: {fps:.2f}")

    return precision, recall, f1_score, ap, fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='', help='Input Image folder')
    opt = parser.parse_args()

    prefix = opt.data
    img_paths = [a for a in os.listdir(prefix) if a.endswith('.jpg')]

    d_net = load_model_VinLPD('./weight')  # VIN_LPD 사용 준비
    loader = DatasetLoader_WebCrawl(prefix)

    precision, recall, f1_score, map_score, fps = calculate_metrics(d_net, loader, img_paths, prefix)
