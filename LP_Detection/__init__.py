from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BBox:
    x: int = None
    y: int = None
    w: int = None
    h: int = None
    class_str: str = None
    class_idx: int = None
    conf: float = None


class OcvYoloBase:
    def __init__(self, _model_path, _weight_path, _classes_path, _in_w=416, _in_h=416, _conf_thresh=0.5, _iou_thresh=0.5):
        self.classes = open(_classes_path).read().strip().split('\n')
        self.net = cv2.dnn.readNetFromDarknet(_model_path, _weight_path)
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.in_w = _in_w
        self.in_h = _in_h
        self.conf_thresh = _conf_thresh
        self.iou_thresh = _iou_thresh

    def forward(self, _img):
        blob = cv2.dnn.blobFromImage(_img, 1 / 255, (self.in_w, self.in_h), [0, 0, 0], True, False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        classIDs = []
        h, w = _img.shape[:2]

        for output in outputs:
            for row in output:
                scores = row[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.conf_thresh:
                    box = row[:4] * np.array([w, h, w, h])
                    (cx, cy, bw, bh) = box.astype("int")
                    bx = cx - (bw // 2)
                    by = cy - (bh // 2)
                    box = [bx, by, int(bw), int(bh)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.conf_thresh, nms_threshold=self.iou_thresh)
        bboxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                b = BBox()
                b.x, b.y, b.w, b.h = boxes[i][:4]
                b.conf = confidences[i]
                b.class_idx = classIDs[i]
                b.class_str = self.classes[classIDs[i]]
                bboxes.append(b)
        return bboxes


def imread_uni(filename, flags=cv2.IMREAD_COLOR):
    """
    경로에 유니코드가 섞여있으면 이 함수를 사용하라.\n
    cv2.imread로는 이미지를 불러올 수 없다.
    :param filename: jpg, png, bmp 등 이미지 파일의 절대경로
    :param flags: cv2.IMREAD_COLOR(default), cv.IMREAD_GRAYSCALE, cv.IMREAD_UNCHANGED, ..., 참고(https://docs.opencv.org/4.10.0/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80)
    :return:
    """
    img_temp = np.fromfile(filename, np.uint8)
    return cv2.imdecode(img_temp, flags)
