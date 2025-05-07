import cv2
from ultralytics import YOLO
import numpy as np


def crop_pcb_image(image):
    """ Cắt vùng PCB từ ảnh và trả về ảnh đã cắt """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Không tìm thấy viền PCB.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[y:y+h, x:x+w]
    return cropped



model = YOLO(r"E:/PythonProgramming/detect_components_yolov11x/runs/detect/train/weights/best.pt")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.9:8080/video")

captured_image = None  # Lưu ảnh chụp để xử lý YOLO

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    if not ret:
        print("Không thể truy cập camera.")
        break
    cv2.imshow("Stream video", frame)  # Chỉ stream camera, không chạy YOLO

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Nhấn 'c' để chụp ảnh
        captured_image = frame.copy()
        print("Đã chụp ảnh.")
        cropped_image = crop_pcb_image(captured_image)
        if cropped_image is not None:
            print("Đã cắt ảnh PCB.")
            captured_image = cropped_image  # Lưu ảnh PCB đã cắt để xử lý YOLO

    if key == ord('q'):  # Nhấn 'q' để thoát
        break

    if captured_image is not None:
        # results = model(captured_image)
        results = model(captured_image, conf=0.5)

        img_result = results[0].plot()
        cv2.imshow("Detection Result", img_result)
        captured_image = None
    # Kiểm tra nếu cửa sổ bị đóng
    if cv2.getWindowProperty("Stream video", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
