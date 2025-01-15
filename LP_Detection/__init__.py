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

    def round_(self):
        self.x = round(self.x)
        self.y = round(self.y)
        self.w = round(self.w)
        self.h = round(self.h)


@dataclass
class Quadrilateral:
    xy1: tuple = None
    xy2: tuple = None
    xy3: tuple = None
    xy4: tuple = None
    class_str: str = None
    class_idx: int = None
    conf: float = None


class OcvYoloBase:
    def __init__(self, _model_path, _weight_path, _classes_path, _in_w=416, _in_h=416, _conf_thresh=0.5, _iou_thresh=0.5):
        self.classes = open(_classes_path).read().strip().split('\n')
        self.net = cv2.dnn.readNetFromDarknet(_model_path, _weight_path)
        self.net.enableWinograd(True)
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.in_w = _in_w
        self.in_h = _in_h
        self.conf_thresh = _conf_thresh
        self.iou_thresh = _iou_thresh

    def forward(self, _img):
        if type(_img) is not list:
            _img = [_img]
        # blob = cv2.dnn.blobFromImage(_img, 1 / 255, (self.in_w, self.in_h), [0, 0, 0], True, False)
        blob = cv2.dnn.blobFromImages(_img, 1 / 255, (self.in_w, self.in_h), [0, 0, 0], True, False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        bs = len(_img)
        boxes = [[] for _ in range(bs)]
        confidences = [[] for _ in range(bs)]
        classIDs = [[] for _ in range(bs)]

        for output in outputs:
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            for b, batch in enumerate(output):
                for row in batch:
                    scores = row[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > self.conf_thresh:
                        h, w = _img[b].shape[:2]
                        box = row[:4] * np.array([w, h, w, h])
                        (cx, cy, bw, bh) = box.astype("int")
                        bx = cx - (bw // 2)
                        by = cy - (bh // 2)
                        box = [bx, by, int(bw), int(bh)]
                        boxes[b].append(box)
                        confidences[b].append(float(confidence))
                        classIDs[b].append(classID)

        multi_batch_bboxes = [[] for _ in range(bs)]
        for b in range(bs):
            indices = list(cv2.dnn.NMSBoxes(boxes[b], confidences[b], score_threshold=self.conf_thresh, nms_threshold=self.iou_thresh))
            for i in indices:
                bb = BBox(*boxes[b][i][:4])
                bb.class_str = self.classes[classIDs[b][i]]
                bb.class_idx = classIDs[b][i]
                bb.conf = confidences[b][i]
                multi_batch_bboxes[b].append(bb)
        if bs == 1:
            return multi_batch_bboxes[0]
        return multi_batch_bboxes
