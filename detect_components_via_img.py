# import cv2
# from ultralytics import YOLO

# # model = YOLO("runs/detect/train2/weights/best.pt")
# model = YOLO(r"E:/PythonProgramming/detect_components_yolov11x/runs/detect/train/weights/best.pt")
# cap = cv2.VideoCapture('http://192.168.1.9:8080/video')
# # cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (640, 480))
#     if not ret:
#         print("Không thể truy cập camera.")
#         break

#     results = model(frame)

#     img_result = results[0].plot()

#     cv2.imshow("Stream video - YOLO detection", img_result)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2

# # Load model
# model = YOLO(r"E:/PythonProgramming/detect_components_yolov11x_v2/runs/detect/train/weights/best.pt")

# # Path to the image
# # image_path = r"E:/PythonProgramming/detect_components_yolov11x_v2/test/z6379209590131_b7715078b21082542a6de364ceb4f6d0.jpg"
# # image_path = r"E:/PythonProgramming/detect_components_yolov11x_v2/test/z6228844731591_44148e6be58c73fe9fb4aa39b6fd7431.jpg"
# image_path = r"E:/PythonProgramming/detect_components_yolov11x_v2/preprocess/template.jpg"
# # image_path = r"E:/PythonProgramming/detect_components_yolov11x_v2/corrected_pcb.jpg"

# # Perform detection
# results = model(image_path)
# # results = model(image_path, conf=0.25)

# # Display the results
# for result in results:
#     print(result)  # In ra thông tin phát hiện, bounding boxes, confidence, class IDs, v.v.

# # Optional: Save or display the detected image
# annotated_frame = results[0].plot()  # Vẽ bounding boxes lên ảnh
# annotated_frame = cv2.resize(annotated_frame, (800, 600))
# cv2.imshow("Detection", annotated_frame)
# cv2.moveWindow("Detection", 100, 100)  # Đặt cửa sổ tại tọa độ (x, y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






from ultralytics import YOLO
import cv2
#train train2 
# Load model
model = YOLO(r"E:\PythonProgramming\.Download\train_down_70-110\runs\content\runs\detect\train3\weights\best.pt")
# model = YOLO(r"E:\PythonProgramming\.Download\train_down_90\runs\content\runs\detect\train\weights\best.pt")
# model = YOLO(r"E:\PythonProgramming\.Download\train_down\runs\content\runs\detect\train\weights\best.pt")

# Đường dẫn ảnh cần nhận diện
image_path = r"E:/PythonProgramming/detect_components_yolov11x_v3/test/z6228844731591_44148e6be58c73fe9fb4aa39b6fd7431.jpg"
# image_path = r"E:/PythonProgramming/detect_components_yolov11x_v3/test/miss_ex.jpg"
# image_path = r"E:/PythonProgramming/detect_components_yolov11x_v3/test/z6228844718618_d10a22edb95076d13d4e2ec4a114bd79.jpg"

# image_path = r"E:/PythonProgramming/detect_components_yolov11x_v3/test/miss.jpg"


# Lấy danh sách các class từ model
class_names = model.names  # model.names là dictionary {id: "class_name"}
UNWANTED_CLASSES = ["miss"]  # Danh sách các class cần ẩn
# UNWANTED_CLASSES = []  # Danh sách các class cần ẩn
# UNWANTED_CLASSES = ["MCU", "res", "cap", "button", "header-dip", "mosfet", "ht7333", "diode", "sensor", "header-smd"]  # Danh sách các class cần ẩn

CONFIDENCE_THRESHOLD = 0.25


# Xác định danh sách class ID cần ẩn
unwanted_class_ids = [i for i, name in class_names.items() if name in UNWANTED_CLASSES]

# Nếu có class cần ẩn, chỉ hiển thị các class còn lại
allowed_class_ids = [i for i in class_names if i not in unwanted_class_ids]

# Thực hiện nhận diện chỉ với các class mong muốn
results = model(image_path, classes=allowed_class_ids, conf=CONFIDENCE_THRESHOLD)

# Hiển thị kết quả nhận diện
for result in results:
    print(result)  # In ra thông tin phát hiện (bounding boxes, confidence, class IDs,...)

# Vẽ kết quả lên ảnh sau khi đã lọc
annotated_frame = results[0].plot()
annotated_frame = cv2.resize(annotated_frame, (800, 600))
# annotated_frame = cv2.resize(annotated_frame, (800, 1000))

cv2.imshow("Detection", annotated_frame)
cv2.moveWindow("Detection", 100, 100)
cv2.waitKey(0)
cv2.destroyAllWindows()
