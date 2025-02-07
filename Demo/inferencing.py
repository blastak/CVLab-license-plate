import random
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation
from LP_Detection.Bases import BBox
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD.VinLPD import load_model_VinLPD
from LP_Detection.VIN_LPD_ONNX.VinLPD_Onnx import load_model_VinLPD_Onnx
from LP_Recognition.VIN_OCR.VinOCR import load_model_VinOCR
from LP_Recognition.VIN_OCR_ONNX.VinOCR_Onnx import load_model_VinOCR_Onnx
from LP_Swapping.models.masked_pix2pix_model import Masked_Pix2pixModel
from LP_Tracking.MultiObjectTrackerWithVoting import TrackerWithVoting
from Utils import trans_eng2kor_v1p3, kor_complete_form, plate_number_tokenizer, iou_4corner
from Utils import xywh2xyxy, xyxy2xywh, xywh2cxcywh


class Swapping_Runner():
    image_width = 256
    tf_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    tf_mask = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(self, ckpt_path):
        ########## torch environment settings
        gpu_ids = [0]
        self.device = torch.device('cuda:%d' % gpu_ids[0] if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
        torch.set_default_device(self.device)  # working on torch>2.0.0
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            torch.multiprocessing.set_start_method('spawn')
        ########## model settings
        self.model = Masked_Pix2pixModel(4, 3, gpu_ids)
        self.model.load_checkpoints(ckpt_path)
        self.model.eval()

    def make_tensor(self, A, B, M):
        A_ = Image.fromarray(A)
        B_ = Image.fromarray(B)
        M_ = Image.fromarray(M)
        t_cond = self.tf_img(A_)
        t_real = self.tf_img(B_)
        t_mask = self.tf_mask(M_)
        t_cond = torch.cat((t_cond, t_mask), dim=0)

        t_cond = torch.unsqueeze(t_cond, dim=0)
        t_real = torch.unsqueeze(t_real, dim=0)
        sample = {'condition_image': t_cond, 'real_image': t_real}
        return sample

    def swap(self, inputs):
        self.model.input_data(inputs)
        self.model.testing()
        detached = self.model.fake_image.detach().cpu()
        bs = 1
        montage = make_grid(detached, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
        montage = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        return montage


class Demo_Runner():
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

        self.swapper = Swapping_Runner('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try004/ckpt_epoch000200.pth')

        self.using_iwpod = False
        if self.using_iwpod:
            self.iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

    def setup(self):
        self.tracker.tracks.clear()

    def encrypt_number(self, p_type, p_number, password, reverse=False):
        if password == '':
            return p_number

        random.seed(password)
        while True:
            adder = random.randint(100, 999999)
            if adder % 10 != 0:
                break
        if reverse:
            adder = -adder

        tokens = plate_number_tokenizer(p_number)
        new_tokens = []
        if p_type in ['P3', 'P4', 'P5']:
            i0 = kor_complete_form[p_type + 'prov'].index(tokens[0])
            j0 = (i0 + adder) % len(kor_complete_form[p_type + 'prov'])
            new_tokens.append(kor_complete_form[p_type + 'prov'][j0])
        for ch in ''.join(tokens[1:]):
            if ch.isdigit():
                i0 = int(ch)
                j0 = (i0 + adder) % 10
                new_tokens.append(str(j0))
            else:
                i0 = kor_complete_form[p_type].index(ch)
                j0 = (i0 + adder) % len(kor_complete_form[p_type])
                new_tokens.append(kor_complete_form[p_type][j0])

        new_number = ''.join(new_tokens)
        return new_number

    def loop(self, frame, password1to2, password2to3):
        # 검출
        st = time.time()  #######################
        d_out = self.d_net.forward(frame)[0]
        print('%s: %.1fms' % ('detection', (time.time() - st) * 1000))  #######################

        # 검출2
        if self.using_iwpod:
            st = time.time()  #######################
            parallelograms = find_lp_corner(frame, self.iwpod_tf)
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
                mat_T = find_total_transformation(img_gened, self.gm_generator, p_type, frame, bb)
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
            new_number = self.encrypt_number(p_type, p_number, password1to2)
            print('#2 %s: %.1fms' % ('Encrypt LP number', (time.time() - st) * 1000))  #######################
            types_numbers2.append((p_type, new_number))

            st = time.time()  #######################
            p_type = 'P1-2' if p_type == 'P1-1' else p_type
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
            sq_lr = np.array([cx - margin, cx + margin])
            sq_tb = np.array([cy - margin, cy + margin])
            if sq_lr[0] < 0:
                sq_lr -= sq_lr[0]
            if frame.shape[1] - sq_lr[1] <= 0:
                sq_lr += (frame.shape[1] - sq_lr[1])
            if sq_tb[0] < 0:
                sq_tb -= sq_tb[0]
            if frame.shape[0] - sq_tb[1] <= 0:
                sq_tb += (frame.shape[0] - sq_tb[1])

            # 영상 합성
            st = time.time()  #######################
            # img1 = cv2.bitwise_and(img_cond2, img_cond2, mask=cv2.bitwise_not(mask_white))
            # img2 = cv2.bitwise_and(img_gen_recon[:, :, :3], img_gen_recon[:, :, :3], mask=mask_white)
            # img_cond2 = img1 + img2
            frame_roi = frame[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
            mask_white_roi = mask_white[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
            img_gen_recon_roi = img_gen_recon[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], :3]
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
            img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
            img_disp2[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...] = img_swapped_unshrink.copy()
            print('#2 %s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################

        img_disp3 = frame.copy()
        # for i, (mat_T, (p_type, p_number)) in enumerate(zip(mat_Ts, types_numbers2)):
        #     st = time.time()  #######################
        #     new_number = self.encrypt_number(p_type, p_number, password2to3, reverse=True)
        #     print(' #3 %s: %.1fms' % ('Decrypt LP number', (time.time() - st) * 1000))  #######################
        #
        #     st = time.time()  #######################
        #     img_gen0 = self.gm_generator.make_LP(new_number, p_type)
        #     img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
        #     print(' #3 %s %s: %.1fms' % ('Make new graphical model', p_type, (time.time() - st) * 1000))  #######################
        #
        #     st = time.time()  #######################
        #     # graphical model을 전체 이미지 좌표계로 warping
        #     img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])
        #     # 해당 영역 mask 생성
        #     mask_white = img_gen_recon[:, :, 3]
        #     print(' #3 %s: %.1fms' % ('superimposing - warpPerspective', (time.time() - st) * 1000))  #######################
        #
        #     # 정방형 crop
        #     cx = int(cxcywhs[i][0])
        #     cy = int(cxcywhs[i][1])
        #     margin = int(cxcywhs[i][2])
        #     sq_lr = np.array([cx - margin, cx + margin])
        #     sq_tb = np.array([cy - margin, cy + margin])
        #     if sq_lr[0] < 0:
        #         sq_lr -= sq_lr[0]
        #     if img_gen_recon.shape[1] - sq_lr[1] <= 0:
        #         sq_lr += (img_gen_recon.shape[1] - sq_lr[1])
        #     if sq_tb[0] < 0:
        #         sq_tb -= sq_tb[0]
        #     if img_gen_recon.shape[0] - sq_tb[1] <= 0:
        #         sq_tb += (img_gen_recon.shape[0] - sq_tb[1])
        #
        #     # 영상 합성
        #     st = time.time()  #######################
        #     frame_roi = frame[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
        #     mask_white_roi = mask_white[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
        #     img_gen_recon_roi = img_gen_recon[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], :3]
        #     img1 = cv2.bitwise_and(frame_roi, frame_roi, mask=cv2.bitwise_not(mask_white_roi))
        #     img2 = cv2.bitwise_and(img_gen_recon_roi, img_gen_recon_roi, mask=mask_white_roi)
        #     print(' #3 %s: %.1fms' % ('superimposing', (time.time() - st) * 1000))  #######################
        #
        #     st = time.time()  #######################
        #     A = img1 + img2
        #     B = frame_roi
        #     M = mask_white_roi
        #     inputs = self.swapper.make_tensor(A, B, M)
        #     print(' #3 %s: %.1fms' % ('crop and make tensor ABM', (time.time() - st) * 1000))  #######################
        #
        #     st = time.time()  #######################
        #     img_swapped = self.swapper.swap(inputs)
        #     img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
        #     img_disp3[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...] = img_swapped_unshrink.copy()
        #     print(' #3 %s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################

        return img_disp2, img_disp3
