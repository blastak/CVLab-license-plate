import sys

import cv2
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene


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
                for i, scene in enumerate(self.video_scenes):
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # todo here

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
