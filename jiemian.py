import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, \
    QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
from ultralytics import YOLO
import time


class Worker:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt)")
        if model_path:
            self.model = YOLO(model_path)
            return self.model is not None
        return False

    def detect_image(self, image):
        start_time = time.time()
        results = self.model.predict(image)
        end_time = time.time()
        detection_time = end_time - start_time
        return results, detection_time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.setWindowTitle("@author：笑脸惹桃花")
        # self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(300, 150, 800, 600)

        # 添加显示时间和用户信息的标签
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.time_user_label = QLabel(f"Current Date and Time (UTC): {current_time}\nCurrent User: wangjiahao528")
        self.time_user_label.setAlignment(Qt.AlignCenter)
        self.time_user_label.setStyleSheet('border:1px solid #000000; background-color: white;')

        # 创建两个 QLabel 分别显示左右图像
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumSize(580, 450)  # 设置大小
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumSize(580, 450)  # 设置大小
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        # 显示检测时间和坐标的 QLabel
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet('border:1px solid #000000; background-color: white;')

        # 水平布局，用于放置左右两个 QLabel
        layout = QVBoxLayout()
        layout.addWidget(self.time_user_label)  # 添加时间和用户信息标签

        hbox_video = QHBoxLayout()
        hbox_video.addWidget(self.label1)  # 左侧显示原始图像
        hbox_video.addWidget(self.label2)  # 右侧显示检测后的图像
        layout.addLayout(hbox_video)
        layout.addWidget(self.info_label)
        self.worker = Worker()

        # 创建按钮布局
        hbox_buttons = QHBoxLayout()
        # 添加模型选择按钮
        self.load_model_button = QPushButton("📁模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_model_button)

        # 添加图片检测按钮
        self.image_detect_button = QPushButton("💾图片检测")
        self.image_detect_button.clicked.connect(self.detect_image)
        self.image_detect_button.setEnabled(False)
        self.image_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.image_detect_button)

        # 添加视频检测按钮
        self.video_detect_button = QPushButton("🎥视频检测")
        self.video_detect_button.clicked.connect(self.detect_video)
        self.video_detect_button.setEnabled(False)
        self.video_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.video_detect_button)

        # 添加摄像头检测按钮
        self.webcam_detect_button = QPushButton("📷摄像头检测")
        self.webcam_detect_button.clicked.connect(self.detect_webcam)
        self.webcam_detect_button.setEnabled(False)
        self.webcam_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.webcam_detect_button)

        # 添加开始/停止检测按钮
        self.detect_toggle_button = QPushButton("▶️开始检测")
        self.detect_toggle_button.clicked.connect(self.toggle_detection)
        self.detect_toggle_button.setEnabled(False)
        self.detect_toggle_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.detect_toggle_button)

        # 添加显示检测物体按钮
        self.display_objects_button = QPushButton("🔍显示检测物体")
        self.display_objects_button.clicked.connect(self.show_detected_objects)
        self.display_objects_button.setEnabled(False)
        self.display_objects_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.display_objects_button)

        # 添加退出按钮
        self.exit_button = QPushButton("❌退出")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.exit_button)

        layout.addLayout(hbox_buttons)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.current_results = None

        # 添加摄像头相关变量
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_detecting = False

        # 添加时间更新定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # 每秒更新一次

    def update_time(self):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.time_user_label.setText(f"Current Date and Time (UTC): {current_time}\nCurrent User: wangjiahao528")

    def detect_image(self):
        image_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)")
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                self.current_results, detection_time = self.worker.detect_image(image)
                if self.current_results:
                    annotated_image = self.current_results[0].plot()
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height1, width1, channel1 = image_rgb.shape
                    bytesPerLine1 = 3 * width1
                    qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
                    pixmap1 = QPixmap.fromImage(qimage1)
                    self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height2, width2, channel2 = annotated_image.shape
                    bytesPerLine2 = 3 * width2
                    qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
                    pixmap2 = QPixmap.fromImage(qimage2)
                    self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

                    # 显示检测时间和坐标信息
                    self.info_label.setText(
                        f"检测时间: {detection_time:.2f} 秒\n坐标信息: {self.get_coordinates_info(self.current_results[0].boxes.xyxy)}")
                else:
                    self.show_message_box("检测结果", "未检测到物体")

    def detect_video(self):
        video_path, _ = QFileDialog.getOpenFileName(None, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame)
                cv2.waitKey(1)
            cap.release()

    def detect_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.detect_toggle_button.setEnabled(True)
                self.timer.start(30)  # 30ms 更新一次，约等于 33fps
                self.webcam_detect_button.setText("📷关闭摄像头")
            else:
                self.show_message_box("错误", "无法打开摄像头")
        else:
            self.stop_webcam()

    def stop_webcam(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.detect_toggle_button.setEnabled(False)
            self.is_detecting = False
            self.detect_toggle_button.setText("▶️开始检测")
            self.webcam_detect_button.setText("📷摄像头检测")
            # 清空显示
            self.label1.clear()
            self.label2.clear()
            self.info_label.clear()

    def toggle_detection(self):
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.detect_toggle_button.setText("⏸停止检测")
        else:
            self.detect_toggle_button.setText("▶️开始检测")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            return

        # 转换为RGB并显示原始画面
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label1.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.label1.size(), Qt.KeepAspectRatio))

        # 如果开启了检测，则进行检测
        if self.is_detecting:
            self.current_results, detection_time = self.worker.detect_image(frame)
            if self.current_results:
                annotated_image = self.current_results[0].plot()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                height2, width2, channel2 = annotated_image.shape
                bytes_per_line2 = 3 * width2
                q_image2 = QImage(annotated_image.data, width2, height2,
                                  bytes_per_line2, QImage.Format_RGB888)
                self.label2.setPixmap(QPixmap.fromImage(q_image2).scaled(
                    self.label2.size(), Qt.KeepAspectRatio))

                # 更新检测信息
                self.info_label.setText(
                    f"检测时间: {detection_time:.2f} 秒\n"
                    f"坐标信息: {self.get_coordinates_info(self.current_results[0].boxes.xyxy)}")
        else:
            # 不检测时，右侧显示同样的画面
            self.label2.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.label2.size(), Qt.KeepAspectRatio))
            self.info_label.clear()

    def process_frame(self, frame):
        self.current_results, detection_time = self.worker.detect_image(frame)
        if self.current_results:
            annotated_image = self.current_results[0].plot()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            height1, width1, channel1 = frame_rgb.shape
            bytesPerLine1 = 3 * width1
            qimage1 = QImage(frame_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimage1)
            self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            height2, width2, channel2 = annotated_image.shape
            bytesPerLine2 = 3 * width2
            qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qimage2)
            self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

            # 显示检测时间和坐标信息
            self.info_label.setText(
                f"检测时间: {detection_time:.2f} 秒\n坐标信息: {self.get_coordinates_info(self.current_results[0].boxes.xyxy)}")

    def get_coordinates_info(self, coordinates):
        coords_info = ""
        for coord in coordinates:
            coords_info += f"({int(coord[0])}, {int(coord[1])}) - ({int(coord[2])}, {int(coord[3])})\n"
        return coords_info

    def show_detected_objects(self):
        if self.current_results:
            det_info = self.current_results[0].boxes.cls
            object_count = len(det_info)
            object_info = f"识别到的物体总个数：{object_count}\n"
            object_dict = {}
            class_names_dict = self.current_results[0].names
            for class_id in det_info:
                class_name = class_names_dict[int(class_id)]
                if class_name in object_dict:
                    object_dict[class_name] += 1
                else:
                    object_dict[class_name] = 1
            sorted_objects = sorted(object_dict.items(), key=lambda x: x[1], reverse=True)
            for obj_name, obj_count in sorted_objects:
                object_info += f"{obj_name}: {obj_count}\n"
            self.show_message_box("识别结果", object_info)
        else:
            self.show_message_box("识别结果", "未检测到物体")

    def show_message_box(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def load_model(self):
        if self.worker.load_model():
            self.image_detect_button.setEnabled(True)
            self.video_detect_button.setEnabled(True)
            self.webcam_detect_button.setEnabled(True)
            self.display_objects_button.setEnabled(True)

    def exit_application(self):
        if self.cap is not None:
            self.stop_webcam()
        sys.exit()

    def closeEvent(self, event):
        # 窗口关闭时确保摄像头被正确释放
        if self.cap is not None:
            self.stop_webcam()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())