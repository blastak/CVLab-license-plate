from LP_Recognition.VIN_OCR import load_model_VinOCR
from MultiObjectTracker import *
from Utils import trans_eng2kor_v1p3

class TrackWithLabel(Track):
    def __init__(self, cx, cy, sf, ar, label):
        super().__init__(cx, cy, sf, ar)
        self.labels = []
        self.plate_types = []


class TrackerWithVoting(Tracker):
    def __init__(self):
        super().__init__()
        self.tracks: list[TrackWithLabel] = []


if __name__ == '__main__':
    d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
    r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
    # cap = cv2.VideoCapture('./sample_video/sample1.avi')
    cap = cv2.VideoCapture(r"D:\Dataset\01_LicensePlate\06_Techwin_B1_in_\20180523_171757_합본(in).avi")  # 테크윈B1 - in

    myTracker = TrackerWithVoting()

    cnt_continue = 0
    cnt_frame = 0
    delay = 0
    while cap.isOpened():
        is_grabbed = cap.grab()
        cnt_frame += 1
        if cnt_frame < 500: continue
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
        for _, d in enumerate(d_out):
            # recognition
            img_crop = r_net.crop_resize_with_padding(img_orig, d)
            r_out = r_net.resize_N_forward(img_crop)
            list_char = r_net.check_align(r_out, d.class_idx + 1)
            list_char_kr = trans_eng2kor_v1p3(list_char)
            label = ''.join(list_char_kr)
            print(label)

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
        if key_in == 27:
            break
        elif key_in == 32:
            delay = 1 - delay

    cap.release()
    cv2.destroyAllWindows()
