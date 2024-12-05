import sys
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
import cv2


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super(VideoPlayer, self).__init__()
        uic.loadUi("video_player.ui", self)

        self.video_widgets = [self.video1, self.video2, self.video3]
        self.video_paths = [None, None, None]
        self.video_captures = [None, None, None]
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.is_playing = False

        self.loadButton.clicked.connect(self.load_videos)
        self.playPauseButton.clicked.connect(self.play_pause_videos)

    def load_videos(self):
        for i in range(3):
            video_path, _ = QFileDialog.getOpenFileName(self, "동영상 파일 선택", "", "동영상 파일 (*.mp4 *.avi *.mkv)")
            if video_path:
                self.video_paths[i] = video_path
                self.video_captures[i] = cv2.VideoCapture(video_path)

    def play_pause_videos(self):
        if not any(self.video_captures):
            return

        if self.is_playing:
            self.timer.stop()
        else:
            self.timer.start(30)  # 약 30 FPS
        self.is_playing = not self.is_playing

    def update_frames(self):
        for i, capture in enumerate(self.video_captures):
            if capture and capture.isOpened():
                ret, frame = capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    step = channel * width
                    qimg = QtGui.QImage(frame.data, width, height, step, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)
                    self.video_widgets[i].setPixmap(pixmap)
                else:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상 반복 재생

    def closeEvent(self, event):
        self.timer.stop()
        for capture in self.video_captures:
            if capture:
                capture.release()
        super(VideoPlayer, self).closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
