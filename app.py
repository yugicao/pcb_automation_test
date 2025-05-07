
import sys
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from script import Ui_MainWindow
from image_preprocess import ImageProcessingThread, AutoTuningThread
import logging
import os

# CAMERA_INDEX = "http://192.168.1.9:8080/video"
CAMERA_INDEX = 0
class VideoStream(QtCore.QThread):
    frame_updated = QtCore.pyqtSignal(QtGui.QImage)
    captured_image = None 

    def __init__(self, camera_index=CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        self.last_frame = None

    def run(self):
        while self.running:
            ret, self.last_frame = self.cap.read()
            if ret:

                bgr_frame = self.last_frame.copy()

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                self.frame_updated.emit(qt_image)
            QtCore.QThread.msleep(5)

    def capture(self):
        if self.last_frame is not None:

            self.captured_image = cv2.cvtColor(self.last_frame.copy(), cv2.COLOR_BGR2RGB)
            return self.captured_image
        return None

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()


# ----------------------- Giao Diện Ứng Dụng ------------------------
requirement = [("MCU", 1), ("res", 9), ("cap", 5), ("button", 1), ("header-dip", 2), 
               ("mosfet", 2), ("ht7333", 1), ("diode", 1), ("sensor", 1), ("header-smd", 1)]
current_dir = os.path.dirname(os.path.abspath(__file__))

class NotificationDialog:
    def __init__(self, parent=None):
        self.parent = parent
    
    def show_info(self, title: str, message: str):
        self._show_message(title, message, QtWidgets.QMessageBox.Icon.Information)
    
    def show_warning(self, title: str, message: str):
        self._show_message(title, message, QtWidgets.QMessageBox.Icon.Warning)
    
    def show_error(self, title: str, message: str):
        self._show_message(title, message, QtWidgets.QMessageBox.Icon.Critical)
    
    def _show_message(self, title: str, message: str, icon: QtWidgets.QMessageBox.Icon):
        msg_box = QtWidgets.QMessageBox(self.parent)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.dialog = NotificationDialog(self)
        self.setupUi(self)
        self.video_thread = VideoStream()
        self.video_thread.frame_updated.connect(self.update_frame, QtCore.Qt.ConnectionType.DirectConnection)
        self.video_thread.start()
        self.capture_button.clicked.connect(self.capture_image)
        self.tuning_button.clicked.connect(self.auto_tuning)


        self.tuning_progress_dialog = None

    def update_frame(self, qt_image):
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,  
            QtCore.Qt.TransformationMode.SmoothTransformation  
        ))

    def auto_tuning(self):

        # self.tuning_progress_dialog = QtWidgets.QProgressDialog("Đang hiệu chỉnh tham số...", "Hủy bỏ", 0, 0, self)
        self.tuning_progress_dialog = QtWidgets.QProgressDialog("Đang hiệu chỉnh tham số...", None, 0, 0, self)
        self.tuning_progress_dialog.setWindowTitle("Đang xử lý")
        self.tuning_progress_dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        # self.tuning_progress_dialog.setCancelButtonText("Hủy bỏ")
        self.tuning_progress_dialog.resize(300, 100)



        # progress_bar = self.tuning_progress_dialog.findChild(QtWidgets.QProgressBar)
        # if progress_bar:
        #     # Chỉnh alignment của progress bar trong layout
        #     layout = self.tuning_progress_dialog.layout()
        #     if layout:
        #         layout.setAlignment(progress_bar, QtCore.Qt.AlignHCenter)

        # # Căn giữa dialog trên màn hình (tùy chọn)
        # screen_geo = QtWidgets.QApplication.primaryScreen().geometry()
        # dialog_geo = self.tuning_progress_dialog.frameGeometry()
        # dialog_geo.moveCenter(screen_geo.center())
        # self.tuning_progress_dialog.move(dialog_geo.topLeft())

        # Căn giữa tiến trình
        self.tuning_progress_dialog.setModal(True)
        # self.tuning_progress_dialog.setRange(0, 100)
        # self.tuning_progress_dialog.setValue(50)  # Cập nhật tiến trình

        self.tuning_progress_dialog.show()
        
        frame = self.video_thread.capture()
        if frame is None:
            self.dialog.show_warning("Cảnh báo", "Không có frame để xử lý.")
            self.tuning_progress_dialog.close()
            return
        
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess", "template.jpg")

        self.processing_thread = AutoTuningThread(frame, template_path)
        print("Starting auto tuning...")
        self.processing_thread.tuning_done.connect(self.on_tuning_done)
        self.processing_thread.start()

    def on_tuning_done(self, best_params):

        if self.tuning_progress_dialog is not None:
            self.tuning_progress_dialog.close()
            self.tuning_progress_dialog = None

        if best_params is None:
            self.dialog.show_info("Thông báo", "Auto tuning không thành công.")
            print("Auto tuning không thành công.")
            return

        self.dialog.show_info("Thông báo", f"Auto tuning thành công!\nBest params: {best_params}")
        print("Best parameters:", best_params)

    def capture_image(self):
        frame = self.video_thread.capture()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.show_loading()
        if frame is not None:
            template_path = os.path.join(current_dir, "preprocess", "template.jpg")
            self.processing_thread = ImageProcessingThread(frame, template_path)
            self.processing_thread.processing_done.connect(self.on_processing_done)
            self.processing_thread.start()

    def on_processing_done(self, processed_image, detected_objects):
        if processed_image is None:
            self.result_label.setStyleSheet("background-color: black;")
            self.show_icon_status(False)
            print("Xử lý ảnh thất bại.")
            self.dialog.show_info("Thông báo", "Xử lý ảnh thất bại.")
            return

        h, w, ch = processed_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(processed_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation
        ))

        if detected_objects is None:
            self.show_icon_status(False)
            print("Không tìm thấy vật thể.")
            self.dialog.show_info("Thông báo", "Không tìm thấy vật thể.")
            return
        
        print("Vật thể phát hiện được:", detected_objects)
        
        if self.update_result_tree(detected_objects):
            self.show_icon_status(True)
            print("Đạt yêu cầu.")
        else:
            self.show_icon_status(False)
            print("Không đạt yêu cầu.")

    def update_result_tree(self, detected_objects) -> bool:
        name_mapping = {
            "MCU": "MCU",
            "res": "Res",
            "cap": "Cap",
            "button": "Button",
            "header-dip": "Header-dip",
            "mosfet": "Mosfet",
            "ht7333": "HT7333",
            "diode": "Diode",
            "sensor": "Sensor",
            "header-smd": "Header-smd",
        }
        detected_dict = {name_mapping.get(name, name): count for name, count in detected_objects}
        is_pass = True
        root = self.result_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            component_name = item.text(0)
            detected_count = detected_dict.get(component_name, 0)
            required = int(item.text(1).strip())
            if detected_count < required:
                item.setForeground(2, QtGui.QColor("red"))
                is_pass = False
                item.setText(2, f"{detected_count}")
            else:
                item.setForeground(2, QtGui.QColor("green"))
                item.setText(2, f"{required}")
        return is_pass

    def show_loading(self):
        self.result_label.setStyleSheet("background-color: white;")
        loading_path = os.path.join(current_dir, "ui_graphics", "loading.png")
        pixmap = QtGui.QPixmap(loading_path)
        scaled_pixmap = pixmap.scaled(
            int(self.result_label.width() * 0.5),
            int(self.result_label.height() * 0.5),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
    def show_icon_status(self, isPass):
        if isPass:
            icon_path = os.path.join(current_dir, "ui_graphics", "pass.png")
        else:
            icon_path = os.path.join(current_dir, "ui_graphics", "fail.png")
        pixmap = QtGui.QPixmap(icon_path)
        scaled_pixmap = pixmap.scaled(
            self.result_icon_label.width(),
            self.result_icon_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.result_icon_label.setPixmap(scaled_pixmap)
        self.result_icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

# --------------------------- MAIN -------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


