import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from LP_Detection.VIN_LPD import load_model_VinLPD
from LinearKalmanFilter import LinearKalmanFilter
from Utils import iou

colors = [(0, 0, 255),
          (0, 128, 255),
          (0, 255, 255),
          (0, 255, 0),
          (255, 0, 0),
          (128, 0, 0),
          (255, 0, 255)
          ]


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
        Q = np.eye(dim_x, dtype=np.float32) * 0.01
        R = np.eye(dim_z, dtype=np.float32) * 1
        x0 = np.float32([cx, cy, sf, ar, 0, 0, 0])
        self.kf = LinearKalmanFilter(dim_x, A, H, Q, R, x0)

        Track.__id += 1
        self.id = Track.__id
        self.age = 1
        self.cnt_total_vis = 0
        self.cnt_consecutive_invis = 0

        self.last_xyxy = xywh2xyxy(cxcywh2xywh(cxcysfar2cxcywh(x0[:4])))
        self.trajectory = []
        self.color = colors[Track.__id % len(colors)]

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
        self.trajectory.append(self.last_xyxy)
        return self.trajectory


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
                THRESH_IOU = 0.1
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
    while cap.isOpened():
        is_grabbed = cap.grab()
        if not is_grabbed:
            cnt_continue += 1
            if cnt_continue > 3:
                break
            continue
        succ, img_orig = cap.retrieve()
        if not succ:
            break
        cnt_continue = 0
        cnt_frame += 1

        # detection
        d_out = d_net.resize_N_forward(img_orig)

        xyxys = []
        for _, d in enumerate(d_out):
            xyxys.append(xywh2xyxy([d.x, d.y, d.w, d.h]))
            cv2.rectangle(img_orig, (d.x, d.y, d.w, d.h), color=(255, 255, 255), thickness=2)

        myTracker.track(xyxys)

        for _, trk in enumerate(myTracker.tracks):
            trajectories = trk.update_trajectory()
            for traj in trajectories:
                traj = list(map(int, traj))
                cv2.rectangle(img_orig, traj[:2], traj[2:], color=trk.color, thickness=2)
        cv2.putText(img_orig, f'{cnt_frame}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('img_orig', img_orig)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
