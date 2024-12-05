import sys
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
import cv2


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super(VideoPlayer, self).__init__()
        uic.loadUi("video_player.ui", self)

        # QGraphicsScene 초기화
        self.video_scenes = [QGraphicsScene(self) for _ in range(3)]
        self.video_views = [self.video1View, self.video2View, self.video3View]
        for view, scene in zip(self.video_views, self.video_scenes):
            view.setScene(scene)

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
        for i, (capture, scene) in enumerate(zip(self.video_captures, self.video_scenes)):
            if capture and capture.isOpened():
                ret, frame = capture.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    qimg = QtGui.QImage(frame.data, width, height, channel * width, QtGui.QImage.Format_RGB888)
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
