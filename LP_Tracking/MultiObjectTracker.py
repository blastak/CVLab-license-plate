import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Tracking.LinearKalmanFilter import LinearKalmanFilter
from Utils import iou, colors, xywh2xyxy, cxcywh2xywh, cxcysfar2cxcywh, cxcywh2cxcysfar, xywh2cxcywh, xyxy2xywh


class Track:
    __id: int = 0

    def __init__(self, cx, cy, sf, ar):
        # system model : x1=Ax0 where x=[cx;cy;sf;ar;cx';cy';sf'] i.e. cx1=cx0+cx'0, z=[cx;cy;sf;ar]
        dim_x, dim_z = 7, 4
        A = np.eye(dim_x, dtype=np.float32)
        A[:3, 4:] = np.eye(3)
        H = np.eye(dim_z, dim_x, dtype=np.float32)
        Q = np.eye(dim_x, dtype=np.float32) * 0.01
        R = np.eye(dim_z, dtype=np.float32) * np.float32([0.1, 0.1, 0.5, 0.5])
        x0 = np.float32([cx, cy, sf, ar, 0, 0, 100])
        self.kf = LinearKalmanFilter(dim_x, A, H, Q, R, x0)

        Track.__id += 1
        self.id = Track.__id
        self.age = 1
        self.cnt_total_vis = 0
        self.cnt_consecutive_invis = 0

        self.last_xyxy = xywh2xyxy(cxcywh2xywh(cxcysfar2cxcywh(x0[:4])))
        self.trajectories = []
        self.color = colors[self.id % len(colors)]

    def predict(self):
        x_ = self.kf.predict()
        x_[2] = max(1, x_[2])  # sf 가 음수가 되는 경우를 방지
        self.last_xyxy = xywh2xyxy(cxcywh2xywh(cxcysfar2cxcywh(x_[:4])))
        return self.last_xyxy

    def correct(self, z):
        x = self.kf.correct(z)
        self.last_xyxy = xywh2xyxy(cxcywh2xywh(cxcysfar2cxcywh(x[:4])))
        return self.last_xyxy

    def update_trajectory(self):
        self.trajectories.append(self.last_xyxy)
        return self.trajectories


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.detections = []
        self.predictions = []

    def track(self, detections):
        self.detections = detections
        self.predict_new_location_of_tracks()
        assigned_td_pairs, unassigned_trks, unassigned_dets = self.assign_detection_to_track()
        self.update_assigned_tracks(assigned_td_pairs)
        self.update_unassigned_tracks(unassigned_trks)
        self.delete_lost_tracks()
        self.create_new_tracks(unassigned_dets)

    def predict_new_location_of_tracks(self):
        self.predictions.clear()
        for tr in self.tracks:
            xyxy = tr.predict()
            self.predictions.append(xyxy)

    def assign_detection_to_track(self):
        N = len(self.predictions)
        M = len(self.detections)

        assigned_td_pairs = []
        unassigned_trks = list(range(N))
        unassigned_dets = list(range(M))

        if N * M != 0:
            mat_iou = np.zeros((N, M), dtype=np.float32)
            for i, trk in enumerate(self.predictions):
                for j, det in enumerate(self.detections):
                    mat_iou[i, j] = iou(trk, det)
            mat_cost = 1 - mat_iou
            matched = linear_sum_assignment(mat_cost)

            for t, d in list(zip(*matched)):
                THRESH_IOU = 0.05
                if mat_iou[t, d] >= THRESH_IOU:
                    assigned_td_pairs.append([t, d])
                    unassigned_trks.remove(t)
                    unassigned_dets.remove(d)

        return assigned_td_pairs, unassigned_trks, unassigned_dets

    def update_assigned_tracks(self, td_pairs):
        for t, d in td_pairs:
            cx, cy, sf, ar = cxcywh2cxcysfar(xywh2cxcywh(xyxy2xywh(self.detections[d])))
            z = np.float32([cx, cy, sf, ar])
            self.tracks[t].correct(z)
            self.tracks[t].age += 1
            self.tracks[t].cnt_consecutive_invis = 0

    def update_unassigned_tracks(self, t_idxs):
        for t in t_idxs:
            self.tracks[t].age += 1
            self.tracks[t].cnt_consecutive_invis += 1

    def delete_lost_tracks(self):
        for trk in self.tracks:
            THRESH_CONSECUTIVE_INVISIBLE = 10
            if trk.cnt_consecutive_invis > THRESH_CONSECUTIVE_INVISIBLE:
                self.tracks.remove(trk)

    def create_new_tracks(self, d_idxs):
        for d in d_idxs:
            xyxy = self.detections[d]
            cxcysfar = cxcywh2cxcysfar(xywh2cxcywh(xyxy2xywh(xyxy)))
            new_track = Track(*cxcysfar)
            self.tracks.append(new_track)


if __name__ == '__main__':
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    cap = cv2.VideoCapture('./sample_video/sample1.avi')

    myTracker = Tracker()

    cnt_continue = 0
    cnt_frame = 0
    delay = 1
    while cap.isOpened():
        is_grabbed = cap.grab()
        cnt_frame += 1
        if not is_grabbed:
            cnt_continue += 1
            if cnt_continue > 3:
                break
            continue
        succ, img_orig = cap.retrieve()
        if not succ:
            break
        cnt_continue = 0

        # detection
        d_out = d_net.forward(img_orig)

        img_disp = img_orig.copy()
        xyxys = []
        for _, d in enumerate(d_out):
            xyxys.append(xywh2xyxy([d.x, d.y, d.w, d.h]))
            cv2.rectangle(img_disp, (d.x, d.y, d.w, d.h), color=(255, 255, 255), thickness=1)
            cv2.circle(img_disp, (d.x + d.w // 2, d.y + d.h // 2), 1, (0, 255, 0), 3)

        myTracker.track(xyxys)

        for _, trk in enumerate(myTracker.tracks):
            trajectories = trk.update_trajectory()
            for traj in trajectories:
                traj = list(map(int, traj))
                cv2.rectangle(img_disp, traj[:2], traj[2:], color=trk.color, thickness=2)
                cv2.circle(img_disp, ((traj[0] + traj[2]) // 2, (traj[1] + traj[3]) // 2), 2, trk.color, 3)

        cv2.putText(img_disp, f'{cnt_frame}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('img_disp', img_disp)

        key_in = cv2.waitKey(delay)
        if key_in == 27:  # ESC
            break
        elif key_in == 32:  # space
            delay = 1 - delay

    cap.release()
    cv2.destroyAllWindows()
