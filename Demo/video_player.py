import sys

import cv2
import numpy as np
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene

from Data_Labeling.Graphical_Model_Generation.Graphical_Model_Generator_KOR import Graphical_Model_Generator_KOR
from Data_Labeling.find_homography_iteratively import find_total_transformation
from LP_Detection import BBox
from LP_Detection.VIN_LPD import load_model_VinLPD
from LP_Recognition.VIN_OCR import load_model_VinOCR
from LP_Tracking.MultiObjectTrackerWithVoting import TrackerWithVoting
from Utils import trans_eng2kor_v1p3
from Utils import xywh2xyxy, xyxy2xywh


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
        self.timer.timeout.connect(self.t_update_frames)

        self.is_playing = False

        self.loadButton.clicked.connect(self.load_videos)
        self.playPauseButton.clicked.connect(self.play_pause_videos)

        # 입력 필드의 텍스트 변경 시 콜백
        self.editBox1.textChanged.connect(self.get_text_from_editbox)
        self.editBox2.textChanged.connect(self.get_text_from_editbox)
        self.password1 = ''
        self.password2 = ''

        self.d_net = load_model_VinLPD('../LP_Detection/VIN_LPD/weight')  # VIN_LPD 사용 준비
        self.r_net = load_model_VinOCR('../LP_Recognition/VIN_OCR/weight')
        self.tracker = TrackerWithVoting()
        self.gm_generator = Graphical_Model_Generator_KOR()  # 반복문 안에서 객체 생성 시 오버헤드가 발생

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

    def t_update_frames(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                # 검출
                d_out = self.d_net.resize_N_forward(frame)

                # 인식
                xyxys = []
                types = []
                numbers = []
                for _, d in enumerate(d_out):
                    img_crop = self.r_net.crop_resize_with_padding(frame, d)
                    r_out = self.r_net.resize_N_forward(img_crop)
                    list_char = self.r_net.check_align(r_out, d.class_idx + 1)
                    list_char_kr = trans_eng2kor_v1p3(list_char)
                    label = ''.join(list_char_kr)

                    # label이 7이상인 경우만
                    if len(label) >= 7:
                        types.append(d.class_str)
                        numbers.append(label)
                        xyxys.append(xywh2xyxy([d.x, d.y, d.w, d.h]))

                # 추적
                self.tracker.track2(xyxys, types, numbers)

                # voting 결과로 GraphicalModel 만들고 Homography 계산
                mat_Ts = []
                img_geneds = []
                for trk in self.tracker.tracks:
                    if trk.last_xyxy[0] < 0 or trk.last_xyxy[1] < 0 or trk.last_xyxy[2] >= frame.shape[1] or trk.last_xyxy[3] >= frame.shape[0]:
                        continue
                    bb = BBox(*xyxy2xywh(trk.last_xyxy))
                    p_type = trk.voted_type
                    p_number = trk.voted_number

                    if p_type == 'P1':
                        p_type += '-1'

                    try:
                        img_gen0 = self.gm_generator.make_LP(p_number, p_type)
                    except:
                        continue
                    img_gened = cv2.resize(img_gen0, None, fx=0.5, fy=0.5)
                    mat_T = find_total_transformation(img_gened, self.gm_generator, p_type, frame, bb)

                    mat_Ts.append(mat_T)
                    img_geneds.append(img_gened)

                img_cond1 = frame.copy()
                for mat_T, img_gened in zip(mat_Ts, img_geneds):
                    # graphical model을 전체 이미지 좌표계로 warping
                    img_gen_recon = cv2.warpPerspective(img_gened, mat_T, frame.shape[1::-1])

                    # 해당 영역 mask 생성
                    img_gened_white = np.full_like(img_gened[:, :, 0], 255, dtype=np.uint8)
                    mask_white = cv2.warpPerspective(img_gened_white, mat_T, frame.shape[1::-1])

                    # 영상 합성
                    img1 = cv2.bitwise_and(img_cond1, img_cond1, mask=cv2.bitwise_not(mask_white))
                    img2 = cv2.bitwise_and(img_gen_recon, img_gen_recon, mask=mask_white)
                    img_cond1 = img1 + img2
                cv2.imshow('img_cond1',img_cond1)
                cv2.waitKey(1)

                # # 좌표 계산
                # g_h, g_w = img_gened.shape[:2]
                # dst_xy = cv2.perspectiveTransform(np.float32([[[0, 0], [g_w, 0], [g_w, g_h], [0, g_h]]]), mat_T)
                # cv2.polylines(img_superimposed, [np.int32(dst_xy)], True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)  # quadrilateral box

                # img_results.append([img_superimposed, img])
                # dst_xy_list.append(dst_xy)

                for i, scene in enumerate(self.video_scenes):
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
            else:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상 반복 재생

    def get_text_from_editbox(self):
        self.password1 = self.editBox1.text()
        self.password2 = self.editBox2.text()
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
