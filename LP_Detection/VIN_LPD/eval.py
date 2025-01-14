import argparse
import os
import time

import cv2
import numpy as np

from Data_Labeling.Dataset_Loader.DatasetLoader_WebCrawl import DatasetLoader_WebCrawl
from LP_Detection.VIN_LPD import load_model_VinLPD
from Utils import imread_uni, iou, add_text_with_background


def visualize(img, d_out):
    img_bb = img.copy()
    for _, bb in enumerate(d_out):
        cv2.rectangle(img_bb, (bb.x, bb.y, bb.w, bb.h), (255, 255, 0), 3)  # bounding box
        font_size = bb.w // 5  # magic number
        img_bb = add_text_with_background(img_bb, f'{bb.class_str}_{bb.conf:.3f}', position=(bb.x, bb.y - font_size), font_size=font_size, padding=0).astype(np.uint8)
    cv2.namedWindow('img_bb', cv2.WINDOW_NORMAL)
    cv2.imshow('img_bb', img_bb)
    cv2.waitKey()
    print("2이상")


def calculate_precision_recall(tp, fp, num):
    # Calculate Precision, Recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / num if num > 0 else 0
    # f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall


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


def val(d_net, loader, img_paths, prefix, iou_threshold=0.5):
    """
    Calculate Precision, Recall, F1-score, and mAP@0.5 for the VIN_LPD detection model on the test dataset.
    Handles multiple classes.
    """
    # Initialize metrics for each class
    classes = d_net.classes
    count_classes = {cls: 0 for cls in classes}

    pred_results = []

    total_time = 0
    num_images = len(img_paths)

    for img_path in img_paths:
        img = imread_uni(os.path.join(prefix, img_path))

        start_time = time.time()
        d_out = d_net.resize_N_forward(img)  # d_net 검출
        total_time += time.time() - start_time

        # Load ground truth (GT)
        plate_type, plate_number, xy1, xy2, xy3, xy4, left, top, right, bottom = loader.parse_json(img_path[:-4] + '.json')
        gt_box = [left, top, right, bottom]
        gt_class = plate_type[:2]  # VIN_LPD는 'P1', 'P2', 'P3', 'P4', 등등..
        count_classes[gt_class] += 1

        if gt_class not in classes:
            print(f"Unknown class '{gt_class}' in ground truth. Skipping this image.")
            continue

        # Parse predicted boxes and classes
        predicted_boxes = []
        predicted_classes = []
        predicted_confidences = []

        # if len(d_out) > 1:  # d_out 두개 이상 일때 사진 확인
        #     visualize(img, d_out)

        if d_out:  # predictions 존재 하면
            for det in d_out:
                x_min, y_min, x_max, y_max = det.x, det.y, det.x + det.w, det.y + det.h
                predicted_boxes.append([x_min, y_min, x_max, y_max])
                predicted_classes.append(det.class_str)
                predicted_confidences.append(det.conf)
        predictions = [
            (pb, pc, pcf, gt_box, gt_class, iou(pb, gt_box))
            for pb, pc, pcf in zip(predicted_boxes, predicted_classes, predicted_confidences)
        ]
        pred_results.append(predictions)

    fps = num_images / total_time

    # GT에 있는 class만 고려
    classes = [cls for cls, num in count_classes.items() if count_classes[cls] > 0]
    all_tp = {cls: [] for cls in classes}
    all_conf = {cls: [] for cls in classes}
    all_precisions = {cls: [] for cls in classes}
    all_recalls = {cls: [] for cls in classes}

    # 이미지 Predict 결과 비교 (IoU)
    for pred_img in pred_results:
        detected = False
        for pb, pc, pcf, gb, gc, i in pred_img:
            if pc not in classes:
                print(f"Warning: Predicted class '{pc}' not in known classes. False Positive.")
                continue
            if pc == gc:  # Only match boxes of the same class
                if i >= iou_threshold and not detected:
                    detected = True
                    all_tp[gc].append(True)  # True Positive
                    all_conf[gc].append(pcf)
                else:
                    all_tp[gc].append(False)  # False Positive
                    all_conf[gc].append(pcf)
            else:
                all_tp[pc].append(False)  # False Positive
                all_conf[pc].append(pcf)

    # Confidence Score별 Precision, Recall 계산
    confidence_thresholds = [0.9 - 0.1 * i for i in range(9)]
    for conf_threshold in confidence_thresholds:
        for cls in classes:
            if count_classes[cls] == 0:
                continue
            valid = np.array(all_conf[cls]) >= conf_threshold
            tpc = np.sum(np.array(all_tp[cls])[valid])
            fpc = np.sum(~np.array(all_tp[cls])[valid])

            precision = tpc / (tpc + fpc) if (tpc + fpc) > 0 else 0
            recall = tpc / count_classes[cls] if count_classes[cls] > 0 else 0

            all_precisions[cls].append(precision)
            all_recalls[cls].append(recall)

    ap_per_class = {}
    for cls in classes:
        if len(all_precisions[cls]) == 0 or len(all_recalls[cls]) == 0:
            # print(f"Class {cls}: No data to calculate AP.")
            ap_per_class[cls] = 0
            continue

        # Get precision and recall values for this class
        precisions = np.array(all_precisions[cls])
        recalls = np.array(all_recalls[cls])

        # Sort by recall to ensure a proper PR curve
        sorted_indices = np.argsort(recalls)
        precisions = precisions[sorted_indices]
        recalls = recalls[sorted_indices]

        # Compute AP for the class
        ap = compute_ap(precisions, recalls)
        ap_per_class[cls] = ap
        # print(f"{cls} : {ap}")

    valid_aps = [ap for cls, ap in ap_per_class.items() if count_classes[cls] > 0 and ap > 0]
    if valid_aps:
        mean_ap = np.mean(valid_aps)
    else:
        mean_ap = 0

    for cls in classes:
        print(f"AP@0.5 for {cls}: {ap_per_class[cls]:.4f}")
    print(f"mAP@0.5: {mean_ap:.4f}")
    print(f"FPS: {fps:.2f}")

    return ap_per_class, mean_ap, fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='', help='Input Image folder')
    opt = parser.parse_args()

    prefix = opt.data
    img_paths = [a for a in os.listdir(prefix) if a.endswith('.jpg')]

    d_net = load_model_VinLPD('./weight')  # VIN_LPD 사용 준비
    loader = DatasetLoader_WebCrawl(prefix)

    ap_per_class, mean_ap, fps = val(d_net, loader, img_paths, prefix)
