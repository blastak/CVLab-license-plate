from collections import Counter

import cv2
import numpy as np

from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from LP_Tracking.MultiObjectTracker import Track, Tracker
from Utils import trans_eng2kor_v1p3, add_text_with_background
from Utils import xywh2xyxy, cxcywh2xywh, cxcysfar2cxcywh, cxcywh2cxcysfar, xywh2cxcywh, xyxy2xywh


class TrackWithPlateNumber(Track):
    def __init__(self, cx, cy, sf, ar, p_type, p_number):
        super().__init__(cx, cy, sf, ar)
        self.plate_types = Counter({p_type: 1})
        self.plate_numbers = Counter({p_number: 1})
        self.voted_type = p_type
        self.voted_number = p_number

    def correct2(self, z, p_type, p_number):
        x = self.kf.correct(z)
        self.last_xyxy = xywh2xyxy(cxcywh2xywh(cxcysfar2cxcywh(x[:4])))

        self.plate_types[p_type] += 1
        self.plate_numbers[p_number] += 1

        self.vote()

        return self.last_xyxy

    def vote(self):
        self.voted_type = self.plate_types.most_common()[0][0]
        self.voted_number = self.plate_numbers.most_common()[0][0]


class TrackerWithVoting(Tracker):
    def __init__(self):
        super().__init__()
        self.tracks: list[TrackWithPlateNumber] = []

    def track2(self, detections, p_types, p_numbers):
        self.detections = detections
        self.predict_new_location_of_tracks()
        assigned_td_pairs, unassigned_trks, unassigned_dets = self.assign_detection_to_track()
        self.update_assigned_tracks2(assigned_td_pairs, p_types, p_numbers)
        self.update_unassigned_tracks(unassigned_trks)
        self.delete_lost_tracks()
        self.create_new_tracks2(unassigned_dets, p_types, p_numbers)

    def update_assigned_tracks2(self, td_pairs, p_types, p_numbers):
        for t, d in td_pairs:
            cx, cy, sf, ar = cxcywh2cxcysfar(xywh2cxcywh(xyxy2xywh(self.detections[d])))
            z = np.float32([cx, cy, sf, ar])
            self.tracks[t].correct2(z, p_types[d], p_numbers[d])
            self.tracks[t].age += 1
            self.tracks[t].cnt_consecutive_invis = 0

    def create_new_tracks2(self, d_idxs, p_types, p_numbers):
        for d in d_idxs:
            xyxy = self.detections[d]
            cxcysfar = cxcywh2cxcysfar(xywh2cxcywh(xyxy2xywh(xyxy)))
            new_track = TrackWithPlateNumber(*cxcysfar, p_types[d], p_numbers[d])
            self.tracks.append(new_track)


if __name__ == '__main__':
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    cap = cv2.VideoCapture('./sample_video/sample1.avi')

    myTracker = TrackerWithVoting()

    cnt_continue = 0
    cnt_frame = 0
    delay = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, cnt_frame)
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

        img_disp = img_orig.copy()

        # detection
        d_out = d_net.resize_N_forward(img_orig)

        xyxys = []
        types = []
        numbers = []
        for _, d in enumerate(d_out):
            cv2.rectangle(img_disp, (d.x, d.y, d.w, d.h), color=(255, 255, 255), thickness=1)
            cv2.circle(img_disp, (d.x + d.w // 2, d.y + d.h // 2), 1, (0, 255, 0), 3)

            # recognition
            img_crop = r_net.crop_resize_with_padding(img_orig, d)
            r_out = r_net.resize_N_forward(img_crop)
            list_char, _ = r_net.check_align(r_out, d.class_idx + 1)
            list_char_kr = trans_eng2kor_v1p3(list_char)
            label = ''.join(list_char_kr)

            if len(label) >= 7:
                types.append(d.class_str)
                numbers.append(label)
                xyxys.append(xywh2xyxy([d.x, d.y, d.w, d.h]))

        myTracker.track2(xyxys, types, numbers)

        for _, trk in enumerate(myTracker.tracks):
            trajectories = trk.update_trajectory()
            for traj in trajectories:
                traj = list(map(int, traj))
                cv2.rectangle(img_disp, traj[:2], traj[2:], color=trk.color, thickness=2)
                cv2.circle(img_disp, ((traj[0] + traj[2]) // 2, (traj[1] + traj[3]) // 2), 2, trk.color, 3)
            w = trk.last_xyxy[2] - trk.last_xyxy[0]
            font_size = max(1, w // 5)
            img_disp = add_text_with_background(img_disp, '(%s)%s'%(trk.voted_type,trk.voted_number),position=(trk.last_xyxy[0], trk.last_xyxy[1] - font_size), font_size=font_size, padding=0).astype(np.uint8)
        cv2.putText(img_disp, f'{cnt_frame}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('img_disp', img_disp)

        key_in = cv2.waitKey(delay)
        if key_in == 27:
            break
        elif key_in == 32:
            delay = 1 - delay

    cap.release()
    cv2.destroyAllWindows()
