import time

import cv2
import numpy as np

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation_3points
from LP_Detection.Bases import BBox
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from LP_Detection.VIN_LPD_ONNX.VinLPD_Onnx import load_model_VinLPD_Onnx
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from LP_Recognition.VIN_OCR_ONNX.VinOCR_Onnx import load_model_VinOCR_Onnx
from LP_Swapping.swap import Swapper
from LP_Swapping.utils import crop_img_square
from LP_Tracking.MultiObjectTrackerWithVoting import TrackerWithVoting
from Utils import encrypt_number
from Utils import trans_eng2kor_v1p3, iou_4corner
from Utils import xywh2xyxy, xyxy2xywh, xywh2cxcywh


class Demo_Runner:
    def __init__(self):
        using_onnx = True
        if using_onnx:
            self.d_net = load_model_VinLPD_Onnx('../LP_Detection/VIN_LPD/weight')  # Onnx 사용 준비
            self.r_net = load_model_VinOCR_Onnx('../LP_Recognition/VIN_OCR/weight')
        else:
            self.d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
            self.r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')

        self.tracker = TrackerWithVoting()
        self.gm_generator = Graphical_Model_Generator_KOR()  # 반복문 안에서 객체 생성 시 오버헤드가 발생

        # self.swapper = Swapper('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try004/ckpt_epoch000200.pth')
        self.swapper = Swapper('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try005_server/ckpt_best_loss_G.pth')

        self.using_iwpod = False
        if self.using_iwpod:
            self.iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

    def setup(self):
        self.tracker.tracks.clear()

    def loop(self, frame, password1to2, password2to3):
        st0 = time.time()  #######################

        # 검출
        st = time.time()  #######################
        d_out = self.d_net.forward(frame)[0]
        print('%s: %.1fms' % ('detection', (time.time() - st) * 1000))  #######################

        # 검출2
        if self.using_iwpod:
            st = time.time()  #######################
            parallelograms, prob = find_lp_corner(frame, self.iwpod_tf)
            print('%s: %.1fms' % ('detection2', (time.time() - st) * 1000))  #######################

        # 인식
        xyxys = []
        types = []
        numbers = []
        img_crops = []
        st = time.time()  #######################
        for d in d_out:
            img_crops.append(self.r_net.keep_ratio_padding(frame, d))
        if len(img_crops) != 0:
            r_outs = self.r_net.forward(img_crops)
            for i, r_out in enumerate(r_outs):
                list_char = self.r_net.check_align(r_out, d_out[i].class_idx + 1)
                list_char_kr = trans_eng2kor_v1p3(list_char)
                label = ''.join(list_char_kr)

                # label이 7이상인 경우만
                if len(label) >= 7:
                    types.append(d_out[i].class_str)
                    numbers.append(label)
                    xyxys.append(xywh2xyxy([d_out[i].x, d_out[i].y, d_out[i].w, d_out[i].h]))
        print('%s: %.1fms' % ('recognition', (time.time() - st) * 1000))  #######################

        # 추적
        st = time.time()  #######################
        self.tracker.track2(xyxys, types, numbers)
        print('%s: %.1fms' % ('tracking', (time.time() - st) * 1000))  #######################

        # voting 결과로 GraphicalModel 만들고 Homography 계산
        mat_Ts = []
        types_numbers = []
        dst_xys = []
        cxcywhs = []
        for trk in self.tracker.tracks:
            if trk.last_xyxy[0] < 0 or trk.last_xyxy[1] < 0 or trk.last_xyxy[2] >= frame.shape[1] or trk.last_xyxy[3] >= frame.shape[0]:
                continue
            bb = BBox(*xyxy2xywh(trk.last_xyxy))
            p_type = trk.voted_type
            p_number = trk.voted_number

            p_type = 'P1-1' if p_type == 'P1' else p_type

            try:
                st = time.time()  #######################
                img_gen0 = self.gm_generator.make_LP(p_number, p_type)
                print('%s %s: %.1fms' % ('Create Graphical Model', p_type, (time.time() - st) * 1000))  #######################
            except:
                continue
            cxcywhs.append(xywh2cxcywh(xyxy2xywh(trk.last_xyxy)))
            types_numbers.append((p_type, p_number))  # 암호화를 위해 저장
            img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
            g_h, g_w = img_gened.shape[:2]
            st = time.time()  #######################
            if self.using_iwpod:
                L = [iou_4corner(bb, qb) for qb in parallelograms]
                pt_src = np.float32([[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]])
                if len(L) == 0:
                    x1, y1, x2, y2 = xywh2xyxy([bb.x, bb.y, bb.w, bb.h])
                    pt_dst = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                else:
                    qb_idx = L.index(max(L))
                    pt_dst = np.float32(parallelograms[qb_idx])
                mat_T = cv2.getPerspectiveTransform(pt_src, pt_dst)
            else:
                mat_T = find_total_transformation_3points(img_gened, frame, bb, p_type, self.gm_generator)
            print('%s: %.1fms' % ('Compute Homography', (time.time() - st) * 1000))  #######################
            mat_Ts.append(mat_T)  # 암호화 후 superimposing을 위해 저장

            # 꼭지점 좌표 계산
            # st = time.time()  #######################
            # dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), mat_T)
            # print('%s: %.1fms' % ('Find LP corners', (time.time() - st) * 1000))  #######################
            # dst_xys.append(dst_xy)  # 복호화 후 superimposing을 위해 저장

        img_disp2 = frame.copy()
        types_numbers2 = []
        for i, (mat_T, (p_type, p_number)) in enumerate(zip(mat_Ts, types_numbers)):
            st = time.time()  #######################
            new_number = encrypt_number(p_type, p_number, password1to2)
            print('#2 %s: %.1fms' % ('Encrypt LP number', (time.time() - st) * 1000))  #######################
            types_numbers2.append((p_type, new_number))

            st = time.time()  #######################
            # p_type = 'P1-2' if p_type == 'P1-1' else p_type
            img_gen0 = self.gm_generator.make_LP(new_number, p_type)
            img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
            print('#2 %s %s: %.1fms' % ('Make new graphical model', p_type, (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            # graphical model을 전체 이미지 좌표계로 warping
            img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])
            # 해당 영역 mask 생성
            mask_white = img_gen_recon[:, :, 3]
            print('#2 %s: %.1fms' % ('superimposing - warpPerspective', (time.time() - st) * 1000))  #######################

            # 정방형 crop
            cx = int(cxcywhs[i][0])
            cy = int(cxcywhs[i][1])
            margin = int(cxcywhs[i][2])
            frame_roi, tblr = crop_img_square(frame, cx, cy, margin)

            # 영상 합성
            st = time.time()  #######################
            # img1 = cv2.bitwise_and(img_cond2, img_cond2, mask=cv2.bitwise_not(mask_white))
            # img2 = cv2.bitwise_and(img_gen_recon[:, :, :3], img_gen_recon[:, :, :3], mask=mask_white)
            # img_cond2 = img1 + img2
            mask_white_roi, _ = crop_img_square(mask_white, cx, cy, margin)
            img_gen_recon_roi, _ = crop_img_square(img_gen_recon[:, :, :3], cx, cy, margin)
            img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
            img2 = cv2.bitwise_and(img_gen_recon_roi, img_gen_recon_roi, mask=mask_white_roi)
            print('#2 %s: %.1fms' % ('superimposing', (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            A = img1 + img2
            B = frame_roi
            M = mask_white_roi
            inputs = self.swapper.make_tensor(A, B, M)
            print('#2 %s: %.1fms' % ('crop and make tensor ABM', (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            img_swapped = self.swapper.swap(inputs)
            # img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
            img_swapped_unshrink = cv2.resize(img_swapped, (tblr[3] - tblr[2], tblr[1] - tblr[0]))
            # try:
            img_disp2[tblr[0]:tblr[1], tblr[2]:tblr[3], ...] = img_swapped_unshrink.copy()
            # except Exception as e:
            #     pass
            print('#2 %s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################

        img_disp3 = frame.copy()
        for i, (mat_T, (p_type, p_number)) in enumerate(zip(mat_Ts, types_numbers2)):
            st = time.time()  #######################
            new_number = encrypt_number(p_type, p_number, password2to3, reverse=True)
            print(' #3 %s: %.1fms' % ('Decrypt LP number', (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            img_gen0 = self.gm_generator.make_LP(new_number, p_type)
            img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
            print(' #3 %s %s: %.1fms' % ('Make new graphical model', p_type, (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            # graphical model을 전체 이미지 좌표계로 warping
            img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])
            # 해당 영역 mask 생성
            mask_white = img_gen_recon[:, :, 3]
            print(' #3 %s: %.1fms' % ('superimposing - warpPerspective', (time.time() - st) * 1000))  #######################

            # 정방형 crop
            cx = int(cxcywhs[i][0])
            cy = int(cxcywhs[i][1])
            margin = int(cxcywhs[i][2])

            # 영상 합성
            st = time.time()  #######################
            frame_roi, tblr = crop_img_square(frame, cx, cy, margin)
            mask_white_roi, _ = crop_img_square(mask_white, cx, cy, margin)
            img_gen_recon_roi, _ = crop_img_square(img_gen_recon[:, :, :3], cx, cy, margin)
            img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
            img2 = cv2.bitwise_and(img_gen_recon_roi, img_gen_recon_roi, mask=mask_white_roi)
            print(' #3 %s: %.1fms' % ('superimposing', (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            A = img1 + img2
            B = frame_roi
            M = mask_white_roi
            inputs = self.swapper.make_tensor(A, B, M)
            print(' #3 %s: %.1fms' % ('crop and make tensor ABM', (time.time() - st) * 1000))  #######################

            st = time.time()  #######################
            img_swapped = self.swapper.swap(inputs)
            img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
            img_disp3[tblr[0]:tblr[1], tblr[2]:tblr[3], ...] = img_swapped_unshrink.copy()
            print(' #3 %s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################

        print('%s: %.1fms' % ('--------------total--------------', (time.time() - st0) * 1000))  #######################

        return img_disp2, img_disp3
