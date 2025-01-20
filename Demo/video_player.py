import random
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from torchvision.utils import make_grid

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from LP_Detection import BBox
from LP_Detection.IWPOD_tf.iwpod_plate_detection_Min import find_lp_corner
from LP_Detection.IWPOD_tf.src.keras_utils import load_model_tf
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
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


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super(VideoPlayer, self).__init__()
        uic.loadUi("video_player.ui", self)

        self.center_window()

        # QGraphicsScene 초기화
        self.video_scenes = [QGraphicsScene(self) for _ in range(3)]
        self.video_views = [self.video1View, self.video2View, self.video3View]

        for view, scene in zip(self.video_views, self.video_scenes):
            view.setScene(scene)
            view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # 가로 스크롤바 제거
            view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # 세로 스크롤바 제거
            view.setBackgroundBrush(QtCore.Qt.black)  # 배경색 검정으로 설정
            view.setFrameShape(QtWidgets.QFrame.NoFrame)  # 경계선 제거

        self.video_capture = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.qtimer_update_frames)

        self.is_playing = False

        self.loadButton.clicked.connect(self.load_videos)
        self.playPauseButton.clicked.connect(self.play_pause_videos)

        # 입력 필드의 텍스트 변경 시 콜백
        self.editBox1.textChanged.connect(self.get_text_from_editbox)
        self.editBox2.textChanged.connect(self.get_text_from_editbox)
        self.password1to2 = ''
        self.password2to3 = ''

        self.d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
        self.r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
        self.tracker = TrackerWithVoting()
        self.gm_generator = Graphical_Model_Generator_KOR()  # 반복문 안에서 객체 생성 시 오버헤드가 발생

        self.swapper = Swapping_Runner('../LP_Swapping/checkpoints/Masked_Pix2pix_CondRealMask_try004/ckpt_epoch000200.pth')

        self.iwpod_tf = load_model_tf('../LP_Detection/IWPOD_tf/weights/iwpod_net')  # iwpod_tf 사용 준비

    def center_window(self):
        screen = QtWidgets.QApplication.primaryScreen()  # 기본 화면 가져오기
        available_geometry = screen.availableGeometry()  # 작업 표시줄을 제외한 사용 가능한 영역
        screen_width = available_geometry.width()
        screen_height = available_geometry.height()
        window_width = self.frameGeometry().width()
        window_height = self.frameGeometry().height()

        x = (screen_width - window_width) // 2 + available_geometry.x()
        y = (screen_height - window_height) // 2 + available_geometry.y()

        self.move(x, y)

    def load_videos(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "동영상 파일 선택", "", "동영상 파일 (*.mp4 *.avi *.mkv)")
        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.tracker.tracks.clear()
            if not self.is_playing:
                self.play_pause_videos()

    def play_pause_videos(self):
        if not self.video_capture:
            return

        if self.is_playing:
            self.timer.stop()
        else:
            self.timer.start(30)  # 약 30 FPS
        self.is_playing = not self.is_playing

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

    def qtimer_update_frames(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                st0 = time.time()  #######################
                # 검출
                st = time.time()  #######################
                d_out = self.d_net.forward(frame)
                print('%s: %.1fms' % ('detection', (time.time() - st) * 1000))  #######################

                # 검출2
                parallelograms = find_lp_corner(frame, self.iwpod_tf)

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
                    if len(img_crops) != len(r_outs):
                        r_outs = [r_outs]
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
                    # mat_T = find_total_transformation(img_gened, self.gm_generator, p_type, frame, bb)
                    L = [iou_4corner(bb, qb) for qb in parallelograms]
                    pt_src = np.float32([[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]])
                    if len(L) == 0:
                        x1, y1, x2, y2 = xywh2xyxy([bb.x, bb.y, bb.w, bb.h])
                        pt_dst = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    else:
                        qb_idx = L.index(max(L))
                        pt_dst = np.float32(parallelograms[qb_idx])
                    mat_T = cv2.getPerspectiveTransform(pt_src, pt_dst)
                    print('%s: %.1fms' % ('Compute Homography', (time.time() - st) * 1000))  #######################
                    mat_Ts.append(mat_T)  # 암호화 후 superimposing을 위해 저장

                    # 꼭지점 좌표 계산
                    st = time.time()  #######################
                    dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), mat_T)
                    print('%s: %.1fms' % ('Find LP corners', (time.time() - st) * 1000))  #######################
                    dst_xys.append(dst_xy)  # 복호화 후 superimposing을 위해 저장

                img_cond2 = frame.copy()
                img_disp2 = frame.copy()
                types_numbers2 = []
                for i, (mat_T, (p_type, p_number)) in enumerate(zip(mat_Ts, types_numbers)):
                    st = time.time()  #######################
                    new_number = self.encrypt_number(p_type, p_number, self.password1to2)
                    print('%s: %.1fms' % ('Encrypt LP number', (time.time() - st) * 1000))  #######################
                    types_numbers2.append((p_type, new_number))

                    p_type_temp = 'P1-2' if p_type == 'P1' else p_type
                    st = time.time()  #######################
                    img_gen0 = self.gm_generator.make_LP(new_number, p_type_temp)
                    print('%s %s: %.1fms' % ('Make new graphical model', p_type_temp, (time.time() - st) * 1000))  #######################
                    img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
                    st = time.time()  #######################
                    # graphical model을 전체 이미지 좌표계로 warping
                    img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])

                    # 해당 영역 mask 생성
                    mask_white = img_gen_recon[:, :, 3]

                    # 영상 합성
                    img1 = cv2.bitwise_and(img_cond2, img_cond2, mask=cv2.bitwise_not(mask_white))
                    img2 = cv2.bitwise_and(img_gen_recon[:, :, :3], img_gen_recon[:, :, :3], mask=mask_white)
                    img_cond2 = img1 + img2
                    print('%s: %.1fms' % ('superimposing', (time.time() - st) * 1000))  #######################

                    # 정방형 crop
                    cx = int(cxcywhs[i][0])
                    cy = int(cxcywhs[i][1])
                    margin = int(cxcywhs[i][2])
                    sq_lr = np.array([cx - margin, cx + margin])
                    sq_tb = np.array([cy - margin, cy + margin])
                    if sq_lr[0] < 0:
                        sq_lr -= sq_lr[0]
                    if img_gen_recon.shape[1] - sq_lr[1] <= 0:
                        sq_lr += (img_gen_recon.shape[1] - sq_lr[1])
                    if sq_tb[0] < 0:
                        sq_tb -= sq_tb[0]
                    if img_gen_recon.shape[0] - sq_tb[1] <= 0:
                        sq_tb += (img_gen_recon.shape[0] - sq_tb[1])

                    # A = crop_img_square(img_cond1, int(cxcywhs[i][0]), int(cxcywhs[i][1]), margin=int(cxcywhs[i][2]))
                    # B = crop_img_square(frame, int(cxcywhs[i][0]), int(cxcywhs[i][1]), margin=int(cxcywhs[i][2]))
                    # M = crop_img_square(mask_white, int(cxcywhs[i][0]), int(cxcywhs[i][1]), margin=int(cxcywhs[i][2]))
                    A = img_cond2[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    B = frame[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    M = mask_white[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    st = time.time()  #######################
                    inputs = self.swapper.make_tensor(A, B, M)
                    print('%s: %.1fms' % ('crop and make tensor ABM', (time.time() - st) * 1000))  #######################
                    st = time.time()  #######################
                    img_swapped = self.swapper.swap(inputs)
                    print('%s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################
                    img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
                    img_disp2[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...] = img_swapped_unshrink.copy()

                img_cond3 = frame.copy()
                img_disp3 = frame.copy()
                for i, (mat_T, (p_type, p_number)) in enumerate(zip(mat_Ts, types_numbers2)):
                    st = time.time()  #######################
                    new_number = self.encrypt_number(p_type, p_number, self.password2to3, reverse=True)
                    print('%s: %.1fms' % ('Decrypt LP number', (time.time() - st) * 1000))  #######################

                    st = time.time()  #######################
                    img_gen0 = self.gm_generator.make_LP(new_number, p_type)
                    print('%s %s: %.1fms' % ('Make new graphical model', p_type, (time.time() - st) * 1000))  #######################
                    img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
                    st = time.time()  #######################
                    # graphical model을 전체 이미지 좌표계로 warping
                    img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])

                    # 해당 영역 mask 생성
                    mask_white = img_gen_recon[:, :, 3]

                    # 영상 합성
                    img1 = cv2.bitwise_and(img_cond3, img_cond3, mask=cv2.bitwise_not(mask_white))
                    img2 = cv2.bitwise_and(img_gen_recon[:, :, :3], img_gen_recon[:, :, :3], mask=mask_white)
                    img_cond3 = img1 + img2
                    print('%s: %.1fms' % ('superimposing', (time.time() - st) * 1000))  #######################

                    # 정방형 crop
                    cx = int(cxcywhs[i][0])
                    cy = int(cxcywhs[i][1])
                    margin = int(cxcywhs[i][2])
                    sq_lr = np.array([cx - margin, cx + margin])
                    sq_tb = np.array([cy - margin, cy + margin])
                    if sq_lr[0] < 0:
                        sq_lr -= sq_lr[0]
                    if img_gen_recon.shape[1] - sq_lr[1] <= 0:
                        sq_lr += (img_gen_recon.shape[1] - sq_lr[1])
                    if sq_tb[0] < 0:
                        sq_tb -= sq_tb[0]
                    if img_gen_recon.shape[0] - sq_tb[1] <= 0:
                        sq_tb += (img_gen_recon.shape[0] - sq_tb[1])

                    A = img_cond3[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    B = frame[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    M = mask_white[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...]
                    st = time.time()  #######################
                    inputs = self.swapper.make_tensor(A, B, M)
                    print('%s: %.1fms' % ('crop and make tensor ABM', (time.time() - st) * 1000))  #######################
                    st = time.time()  #######################
                    img_swapped = self.swapper.swap(inputs)
                    print('%s: %.1fms' % ('swapping', (time.time() - st) * 1000))  #######################
                    img_swapped_unshrink = cv2.resize(img_swapped, (margin * 2, margin * 2))
                    img_disp3[sq_tb[0]:sq_tb[1], sq_lr[0]:sq_lr[1], ...] = img_swapped_unshrink.copy()

                displays = [frame, img_disp2, img_disp3]
                st = time.time()  #######################
                for i, (scene, disp) in enumerate(zip(self.video_scenes, displays)):
                    img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                    height, width, channel = img.shape
                    qimg = QtGui.QImage(img.data, width, height, channel * width, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)

                    # QGraphicsScene에 Pixmap 추가
                    scene.clear()
                    item = scene.addPixmap(pixmap)

                    # Aspect Ratio 유지하며 Stretch
                    view_width = self.video_views[i].width()
                    view_height = self.video_views[i].height()
                    scale_x = view_width / width
                    scale_y = view_height / height
                    scale = min(scale_x, scale_y)
                    item.setScale(scale)

                    # View 중심에 Pixmap 배치
                    item.setPos((view_width - width * scale) / 2, (view_height - height * scale) / 2)
                print('%s: %.1fms' % ('display', (time.time() - st) * 1000))  #######################
                print('%s: %.1fms\n' % ('whole', (time.time() - st0) * 1000))  #######################
            else:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상 반복 재생

    def get_text_from_editbox(self):
        self.password1to2 = self.editBox1.text()
        self.password2to3 = self.editBox2.text()
        # self.setWindowTitle(f"{text1} {text2}")

    def closeEvent(self, event):
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
        super(VideoPlayer, self).closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
