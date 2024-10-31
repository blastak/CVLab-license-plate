from random import randint

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from LP_Detection.VIN_LPD import load_model_VinLPD
from LinearKalmanFilter import LinearKalmanFilter


# Track 객체 정의
class Track:
    def __init__(self, bbox):
        # LinearKalmanFilter 초기화
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        Q = np.eye(4, dtype=np.float32) * 0.03
        R = np.eye(2, dtype=np.float32) * 1
        x0 = np.array([bbox[0], bbox[1], 0, 0], dtype=np.float32)
        self.kf = LinearKalmanFilter(dim_x=4, A=A, H=H, Q=Q, R=R, x0=x0)
        self.bbox = list(bbox)
        self.age = 1
        self.consecutive_invisible_count = 0
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def predict(self):
        pred = self.kf.predict()
        return (pred[0], pred[1])

    def correct(self, measurement):
        x_hat = self.kf.correct(np.array([measurement[0], measurement[1]], dtype=np.float32))
        self.bbox = [*x_hat[:2], 550 // 4, 110 // 4]
        self.age += 1
        self.consecutive_invisible_count = 0


# 객체 탐지 함수
def detect_objects(frame):
    # YCbCr 색 공간으로 변환
    ycbcr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # 녹청색 중심값 (166, 70, 130)을 기준으로 범위 설정
    y_center, cb_center, cr_center = 166, 70, 130
    y_range, cb_range, cr_range = 50, 20, 20  # Y, Cb, Cr 범위 설정

    lower_cyan = np.array([y_center - y_range, cb_center - cb_range, cr_center - cr_range], dtype=np.uint8)
    upper_cyan = np.array([y_center + y_range, cb_center + cb_range, cr_center + cr_range], dtype=np.uint8)

    # 녹청색 픽셀 마스크 생성
    cyan_mask = cv2.inRange(ycbcr_image, lower_cyan, upper_cyan)
    cyan_mask = cv2.dilate(cyan_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # 마스크로부터 윤곽선 탐지
    contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선마다 bounding box 그리기
    bboxes = [cv2.boundingRect(contour) for contour in contours]

    return bboxes


# 새 위치 예측 함수
def predict_new_locations_of_tracks(tracks):
    predictions = [track.predict() for track in tracks]
    return predictions


# 탐지 결과와 트랙의 매칭 함수
def detection_to_track_assignment(detections, predictions):
    cost_matrix = np.zeros((len(detections), len(predictions)))
    for i, det in enumerate(detections):
        for j, pred in enumerate(predictions):
            cost_matrix[i][j] = np.linalg.norm(np.array(det[:2]) - np.array(pred))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_detections, unmatched_tracks = [], [], []
    for i, det in enumerate(detections):
        if i in row_ind:
            matches.append((i, col_ind[np.where(row_ind == i)[0][0]]))
        else:
            unmatched_detections.append(i)
    for j, pred in enumerate(predictions):
        if j not in col_ind:
            unmatched_tracks.append(j)
    return matches, unmatched_detections, unmatched_tracks


# 매칭된 트랙 업데이트
def update_assigned_tracks(tracks, detections, matches):
    for det_idx, track_idx in matches:
        tracks[track_idx].correct(detections[det_idx][:2])


# 매칭되지 않은 트랙 업데이트
def update_unassigned_tracks(tracks, unmatched_tracks):
    for idx in unmatched_tracks:
        tracks[idx].consecutive_invisible_count += 1


# 오래된 트랙 삭제
def delete_lost_tracks(tracks):
    return [track for track in tracks if track.consecutive_invisible_count < 5]


# 새로운 트랙 생성
def create_new_tracks(tracks, detections, unmatched_detections):
    for i in unmatched_detections:
        new_track = Track(detections[i])
        tracks.append(new_track)


# 결과 표시
def display_tracking_results(frame, tracks):
    for track in tracks:
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), track.color, 2)
    cv2.imshow('Tracking', frame)


if __name__ == '__main__':
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    cap = cv2.VideoCapture('./sample_video/sample1.avi')
    tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        d_out = d_net.resize_N_forward(frame)
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
