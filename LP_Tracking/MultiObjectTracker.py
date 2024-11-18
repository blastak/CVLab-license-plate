import cv2

from LP_Detection.VIN_LPD import load_model_VinLPD
from LinearKalmanFilter import LinearKalmanFilter

import numpy as np
from scipy.optimize import linear_sum_assignment
from Utils import iou

def xywh2xyxy(xywh):
    assert len(xywh) == 4
    xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
    return xyxy

def xyxy2xywh(xyxy):
    assert len(xyxy) == 4
    xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
    return xywh

def xywh2cxcywh(xywh):
    assert len(xywh) == 4
    w, h = xywh[2], xywh[3]
    cx = xywh[0] + w / 2
    cy = xywh[1] + h / 2
    cxcywh = [cx, cy, w, h]
    return cxcywh

def cxcywh2xywh(cxcywh):
    assert len(cxcywh) == 4
    w, h = cxcywh[2], cxcywh[3]
    x = cxcywh[0] - w / 2
    y = cxcywh[1] - h / 2
    xywh = [x, y, w, h]
    return xywh

def cxcywh2cxcysfar(cxcywh):
    assert len(cxcywh) == 4
    w, h = cxcywh[2], cxcywh[3]
    sf = w * h
    ar = w / h
    cxcysfar = [cxcywh[0], cxcywh[1], sf, ar]
    return cxcysfar

def cxcysfar2cxcywh(cxcysfar):
    assert len(cxcysfar) == 4
    sf, ar = cxcysfar[2], cxcysfar[3]
    w = (sf * ar) ** 0.5
    h = sf / w
    cxcywh = [cxcysfar[0], cxcysfar[1], w, h]
    return cxcywh

class Track:
    __id: int = 0

    def __init__(self, cx, cy, sf, ar):
        # system model : x1=Ax0 where x=[cx;cy;sf;ar;cx';cy';sf'] i.e. cx1=cx0+cx'0, z=[cx;cy;sf;ar]
        dim_x, dim_z = 7, 4
        A = np.eye(dim_x, dtype=np.float32)
        A[:3, 4:] = np.eye(3)
        H = np.eye(dim_z, dim_x, dtype=np.float32)
        Q = np.eye(dim_x, dtype=np.float32) * 1
        R = np.eye(dim_z, dtype=np.float32) * 1
        x0 = np.float32([cx, cy, sf, ar, 0, 0, 0])
        self.kf = LinearKalmanFilter(dim_x, A, H, Q, R, x0)

        Track.__id += 1
        self.id = Track.__id
        self.age = 1
        self.cnt_total_vis = 0
        self.cnt_consecutive_invis = 0

        self.bounding_box = x0[:4].tolist()
        self.trajectory = [self.bounding_box]

    def predict(self):
        x_ = self.kf.predict()
        cx, cy = x_[:2]
        return cx, cy

    def correct(self, z):
        x = self.kf.correct(z)


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []

    def track(self, detections):
        predictions = self.predict_new_location_of_tracks()
        assignments, unassignedTracks, unassignedDetections = self.assign_detection_to_track(predictions, detections)
        self.update_assigned_tracks()
        self.update_unassigned_tracks()
        self.delete_lost_tracks()
        self.create_new_tracks()

    def predict_new_location_of_tracks(self):
        predictions = []
        for tr in self.tracks:
            cx, cy = tr.predict()
            predictions.append([cx, cy])
        return predictions

    def assign_detection_to_track(self, predictions, detections):
        self.detections = detections
        if len(predictions) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(predictions)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(predictions):
                iou_matrix[d, t] = iou(det, trk)

        cost_matrix = 1 - iou_matrix
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(predictions):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update_assigned_tracks(self, td_pairs):
        for t,d in td_pairs:
            cx,cy,sf,ar = cxcywh2cxcysfar(xywh2cxcywh(xyxy2xywh(self.detections[d])))
            z = np.float32([cx, cy, sf, ar])
            self.tracks[t].correct(z)
            self.tracks[t].age += 1
            self.tracks[t].cnt_consecutive_invis = 0

    def update_unassigned_tracks(self, t_lists):
        for t in t_lists:
            self.tracks[t].age += 1
            self.tracks[t].cnt_consecutive_invis += 1

    def delete_lost_tracks(self):
        for i,trk in enumerate(self.tracks[:]):  # colon ":" 안 붙이면 del과 충돌
            THRESH_CONSECUTIVE_INVISIBLE = 10
            if trk.cnt_consecutive_invis >THRESH_CONSECUTIVE_INVISIBLE:
                del self.tracks[i]

    def create_new_tracks(self, d_lists):
        for d in d_lists:
            xyxy = self.detections[d]




if __name__ == '__main__':
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    cap = cv2.VideoCapture('./sample_video/sample1.avi')
    # cap = cv2.VideoCapture(r"D:\백업_비젼인_테스트동영상\20181129_테크윈카메라_인하대입구_2미터\009_20181129_085645_30대.avi")
    tracks = []

    while cap.isOpened():
        is_grabbed = cap.grab()
        if not is_grabbed:
            continue
        succ, img_orig = cap.retrieve()
        if not succ:
            break

        # detection
        d_out = d_net.resize_N_forward(img_orig)

        detections = []
        for _, d in enumerate(d_out):
            detections.append((d.x, d.y, d.w, d.h))
        # detections = detect_objects(frame)
        predictions = predict_new_locations_of_tracks(tracks)
        matches, unmatched_detections, unmatched_tracks = detection_to_track_assignment(detections, predictions)

        update_assigned_tracks(tracks, detections, matches)
        update_unassigned_tracks(tracks, unmatched_tracks)
        tracks = delete_lost_tracks(tracks)
        create_new_tracks(tracks, detections, unmatched_detections)

        display_tracking_results(frame, tracks)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# # Track 객체 정의
# class Track:
#     def __init__(self, cx, cy):
#         # LinearKalmanFilter 초기화
#         A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
#         H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
#         Q = np.eye(4, dtype=np.float32) * 0.03
#         R = np.eye(2, dtype=np.float32) * 1
#         x0 = np.array([bbox[0], bbox[1], 0, 0], dtype=np.float32)
#         self.kf = LinearKalmanFilter(dim_x=4, A=A, H=H, Q=Q, R=R, x0=x0)
#         self.bbox = list(bbox)
#         self.age = 1
#         self.consecutive_invisible_count = 0
#         self.color = (randint(0, 255), randint(0, 255), randint(0, 255))
#
#     def predict(self):
#         pred = self.kf.predict()
#         return (pred[0], pred[1])
#
#     def correct(self, measurement):
#         x_hat = self.kf.correct(np.array([measurement[0], measurement[1]], dtype=np.float32))
#         self.bbox = [*x_hat[:2], 550 // 4, 110 // 4]
#         self.age += 1
#         self.consecutive_invisible_count = 0
#
#
# # 객체 탐지 함수
# def detect_objects(frame):
#     # YCbCr 색 공간으로 변환
#     ycbcr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#
#     # 녹청색 중심값 (166, 70, 130)을 기준으로 범위 설정
#     y_center, cb_center, cr_center = 166, 70, 130
#     y_range, cb_range, cr_range = 50, 20, 20  # Y, Cb, Cr 범위 설정
#
#     lower_cyan = np.array([y_center - y_range, cb_center - cb_range, cr_center - cr_range], dtype=np.uint8)
#     upper_cyan = np.array([y_center + y_range, cb_center + cb_range, cr_center + cr_range], dtype=np.uint8)
#
#     # 녹청색 픽셀 마스크 생성
#     cyan_mask = cv2.inRange(ycbcr_image, lower_cyan, upper_cyan)
#     cyan_mask = cv2.dilate(cyan_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
#
#     # 마스크로부터 윤곽선 탐지
#     contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 윤곽선마다 bounding box 그리기
#     bboxes = [cv2.boundingRect(contour) for contour in contours]
#
#     return bboxes
#
#
# # 새 위치 예측 함수
# def predict_new_locations_of_tracks(tracks):
#     predictions = [track.predict() for track in tracks]
#     return predictions
#
#
# # 탐지 결과와 트랙의 매칭 함수
# def detection_to_track_assignment(detections, predictions):
#     cost_matrix = np.zeros((len(detections), len(predictions)))
#     for i, det in enumerate(detections):
#         for j, pred in enumerate(predictions):
#             cost_matrix[i][j] = np.linalg.norm(np.array(det[:2]) - np.array(pred))
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     matches, unmatched_detections, unmatched_tracks = [], [], []
#     for i, det in enumerate(detections):
#         if i in row_ind:
#             matches.append((i, col_ind[np.where(row_ind == i)[0][0]]))
#         else:
#             unmatched_detections.append(i)
#     for j, pred in enumerate(predictions):
#         if j not in col_ind:
#             unmatched_tracks.append(j)
#     return matches, unmatched_detections, unmatched_tracks
#
#
# # 매칭된 트랙 업데이트
# def update_assigned_tracks(tracks, detections, matches):
#     for det_idx, track_idx in matches:
#         tracks[track_idx].correct(detections[det_idx][:2])
#
#
# # 매칭되지 않은 트랙 업데이트
# def update_unassigned_tracks(tracks, unmatched_tracks):
#     for idx in unmatched_tracks:
#         tracks[idx].consecutive_invisible_count += 1
#
#
# # 오래된 트랙 삭제
# def delete_lost_tracks(tracks):
#     return [track for track in tracks if track.consecutive_invisible_count < 5]
#
#
# # 새로운 트랙 생성
# def create_new_tracks(tracks, detections, unmatched_detections):
#     for i in unmatched_detections:
#         new_track = Track(detections[i])
#         tracks.append(new_track)
#
#
# # 결과 표시
# def display_tracking_results(frame, tracks):
#     for track in tracks:
#         x, y, w, h = track.bbox
#         cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), track.color, 2)
#     cv2.imshow('Tracking', frame)
#
#
