
import cv2
import numpy as np
import sys
import os
from PyQt6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# ============================= THAM SỐ ===========================
GAUSSIAN_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD_LOW = 10
CANNY_THRESHOLD_HIGH = 100
DILATION_KERNEL_SIZE = (50, 50)
DILATION_ITERATIONS = 1
ENABLE_PCB_OUTLINE = True
CROP_EXPAND_RATIO = 1.1

# ================================================================

# ------------------- Model Preload & Warm-up ----------------------
# Load model khi ứng dụng khởi động (đường dẫn model thay đổi theo cấu hình của bạn)

MODEL_PATH = r"E:\PythonProgramming\detect_components_yolov11x_v3\runs\content\runs\detect\train\weights\best.pt"

model = YOLO(MODEL_PATH)

if hasattr(model, 'eval'):
    model.eval()

dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)

_ = model(dummy_image)
print("Model đã được preload và warm-up.")

# ========================= XỬ LÝ ẢNH ============================
def crop_and_rotate_pcb(image):
    """Cắt vùng PCB theo hình chữ nhật nghiêng và xoay vuông góc"""
    try:
        print("crop_and_rotate_pcb: Starting", flush=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, 0)
        edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)
        edges = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("crop_and_rotate_pcb: No PCB edge detected.", flush=True)
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        pcb_outline = image.copy()
        cv2.drawContours(pcb_outline, [largest_contour], -1, (0, 255, 0), 2)
        if ENABLE_PCB_OUTLINE:
            cv2.imwrite("E:/PythonProgramming/detect_components_yolov11x_v3/debug/pcb_outline.jpg", pcb_outline)
        rect = cv2.minAreaRect(largest_contour)
        width = int(rect[1][0]*CROP_EXPAND_RATIO)
        height = int(rect[1][1]*CROP_EXPAND_RATIO)
        angle = rect[2]
        if width < height:
            angle += 90
            width, height = height, width
        center = tuple(map(int, rect[0]))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        x, y = center
        x1 = max(0, x - width // 2)
        y1 = max(0, y - height // 2)
        x2 = min(rotated.shape[1], x + width // 2)
        y2 = min(rotated.shape[0], y + height // 2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            print("crop_and_rotate_pcb: Invalid crop region.", flush=True)
            return None
        cropped = rotated[y1:y2, x1:x2]
        return cropped
    except Exception as e:
        print("crop_and_rotate_pcb: Exception: " + str(e), flush=True)
        return None

def correct_orientation(pcb_image, template_patch):
    """
    So khớp Template Matching để xoay ảnh PCB về chiều đúng.
    Nếu kích thước của template lớn hơn ảnh PCB, hoặc sau xoay ảnh nhỏ hơn template,
    sẽ tiến hành resize cho phù hợp.
    """
    # Nếu template lớn hơn ảnh PCB ban đầu, resize template về kích thước ảnh PCB
    if pcb_image.shape[0] < template_patch.shape[0] or pcb_image.shape[1] < template_patch.shape[1]:
        print("Warning: Template lớn hơn ảnh PCB, resize template về kích thước ảnh PCB.", flush=True)
        template_patch = cv2.resize(template_patch, (pcb_image.shape[1], pcb_image.shape[0]))

    orientations = [0, 90, 180, 270]
    best_angle = 0
    max_val = -1

    for angle in orientations:
        if angle == 0:
            rotated = pcb_image
        else:
            rotated = cv2.rotate(pcb_image, {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }[angle])
            
        # Nếu ảnh sau khi xoay nhỏ hơn template, tiến hành resize
        if rotated.shape[0] < template_patch.shape[0] or rotated.shape[1] < template_patch.shape[1]:
            print(f"Warning: Ảnh sau khi xoay {angle}° nhỏ hơn template, resize rotated image...", flush=True)
            rotated = cv2.resize(rotated, (template_patch.shape[1], template_patch.shape[0]))
            
        result = cv2.matchTemplate(rotated, template_patch, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, _ = cv2.minMaxLoc(result)
        if maxVal > max_val:
            max_val = maxVal
            best_angle = angle

    if best_angle != 0:
        pcb_image = cv2.rotate(pcb_image, {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }[best_angle])
    return pcb_image, best_angle

def process_image(image, template_patch):
    """Xử lý ảnh: cắt, xoay PCB và xác định hướng đúng."""
    print("process_image: Starting", flush=True)
    cropped_image = crop_and_rotate_pcb(image)
    if cropped_image is not None and cropped_image.size != 0:
        print("process_image: PCB image cropped.", flush=True)
        # Nếu kích thước của cropped_image nhỏ hơn template_patch, resize template_patch
        if (cropped_image.shape[0] < template_patch.shape[0] or 
            cropped_image.shape[1] < template_patch.shape[1]):
            print("process_image: Template lớn hơn ảnh cắt được, resize template...", flush=True)
            template_patch = cv2.resize(template_patch, (cropped_image.shape[1], cropped_image.shape[0]))
        corrected_image, angle_found = correct_orientation(cropped_image, template_patch)
        print("process_image: Hướng đúng: {}°".format(angle_found), flush=True)

        cv2.imwrite("E:/PythonProgramming/detect_components_yolov11x_v3/debug/crop_img.jpg", corrected_image)
        return corrected_image
    else:
        print("process_image: Không thể cắt PCB!", flush=True)
        return None

# def auto_tune_crop_params(image, template_image):
#     best_params = None
#     best_score = -1
    
#     global GAUSSIAN_KERNEL_SIZE, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH, DILATION_KERNEL_SIZE
#     original_params = (GAUSSIAN_KERNEL_SIZE, (CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH), DILATION_KERNEL_SIZE)
    
#     GAUSSIAN_SIZES = [(3,3), (5,5), (7,7), (9,9), (11,11)]
#     CANNY_THRESHOLDS = [(10,50), (30,100), (50,150), (100,200)]
#     DILATION_SIZES = [(30,30), (50,50), (70,70), (100,100)]
    
#     for g_size in GAUSSIAN_SIZES:
#         for c_thresh in CANNY_THRESHOLDS:
#             for d_size in DILATION_SIZES:
#                 try:
#                     # Cập nhật tham số tạm thời
#                     GAUSSIAN_KERNEL_SIZE = g_size
#                     CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH = c_thresh
#                     DILATION_KERNEL_SIZE = d_size
                    
#                     # Xử lý ảnh và kiểm tra contour
#                     cropped = process_image(image, template_image)
#                     if cropped is None:
#                         print(">> Skipped: cropped is None")
#                         continue
#                     if cropped.size == 0:
#                         print(">> Skipped: cropped.size == 0")
#                         continue
                    
#                     # Kiểm tra contour lớn nhất có hợp lệ không
#                     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#                     blurred = cv2.GaussianBlur(gray, g_size, 0)
#                     edges = cv2.Canny(blurred, c_thresh[0], c_thresh[1])
#                     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, d_size)
#                     dilated = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
#                     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
#                     if not contours:
#                         print(">> No contours found")
#                         continue
                    
#                     largest_contour = max(contours, key=cv2.contourArea)
#                     area = cv2.contourArea(largest_contour)
#                     img_area = cropped.shape[0] * cropped.shape[1]
                    
#                     # Loại bỏ contour quá nhỏ (<20% diện tích ảnh)
#                     if area < img_area * 0.3:
#                         continue
                    
#                     # Kiểm tra tỉ lệ khung hình
#                     x, y, w, h = cv2.boundingRect(largest_contour)
#                     aspect_ratio = max(w, h) / min(w, h)
#                     if aspect_ratio > 8:  # Tỉ lệ không hợp lý
#                         continue
                    
#                     # Kiểm tra hình dạng gần với hình chữ nhật
#                     epsilon = 0.04 * cv2.arcLength(largest_contour, True)
#                     approx = cv2.approxPolyDP(largest_contour, epsilon, True)
#                     if len(approx) != 4:  # Không phải tứ giác
#                         continue
                    
#                     # Resize template và ảnh về cùng kích thước
#                     h, w = cropped.shape[:2]
#                     resized_template = cv2.resize(template_image, (w, h), interpolation=cv2.INTER_AREA)
                    
#                     # Tính điểm TM_CCOEFF_NORMED
#                     result = cv2.matchTemplate(cropped, resized_template, cv2.TM_CCOEFF_NORMED)
#                     _, max_corr, _, _ = cv2.minMaxLoc(result)
                    
#                     # Tính SSIM
#                     gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#                     gray_template = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
#                     ssim_score = ssim(gray_cropped, gray_template, data_range=255)
                    
#                     # Kết hợp điểm số (trọng số 50-50)
#                     combined_score = 0.5 * max_corr + 0.5 * ssim_score
#                     print(f"Current score: {combined_score:.2f}")
#                     if combined_score > best_score:
#                         best_score = combined_score
#                         best_params = (g_size, c_thresh, d_size)
                        
#                 except Exception as e:
#                     print(f"Error with params {g_size}, {c_thresh}, {d_size}: {str(e)}")
#                     continue
#                 finally:
#                     # Khôi phục tham số gốc
#                     GAUSSIAN_KERNEL_SIZE, (CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH), DILATION_KERNEL_SIZE = original_params
    
#     if best_params:
#         print(f"Best params: Gaussian={best_params[0]}, Canny={best_params[1]}, Dilation={best_params[2]}, Score={best_score:.2f}")
#         # Cập nhật tham số toàn cục
#         GAUSSIAN_KERNEL_SIZE = best_params[0]
#         CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH = best_params[1]
#         DILATION_KERNEL_SIZE = best_params[2]
#     return best_params

def auto_tune_crop_params(image, template_image):
    best_params = None
    best_score = -np.inf
    
    # Điều chỉnh tham số thử nghiệm
    GAUSSIAN_SIZES = [(5, 5), (7, 7), (9, 9)]
    CANNY_THRESHOLDS = [(30, 150), (50, 200), (80, 250)]
    DILATION_SIZES = [(40, 40), (60, 60), (80, 80)]
    
    # Thông số template
    target_h, target_w = template_image.shape[:2]
    template_area = target_w * target_h
    template_aspect = target_w / target_h
    
    # Chuẩn bị HOG descriptor
    win_size = (64, 128)  # Kích thước cửa sổ HOG
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    n_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    
    # Tính HOG cho template
    template_resized = cv2.resize(template_image, win_size)
    template_hog = hog.compute(template_resized).flatten()
    
    # Tạo thư mục debug
    debug_dir = "E:/detect_components_yolov11x_v3/debug/auto_tune"
    os.makedirs(debug_dir, exist_ok=True)

    for g_size in GAUSSIAN_SIZES:
        for c_thresh in CANNY_THRESHOLDS:
            for d_size in DILATION_SIZES:
                try:
                    # Cập nhật tham số tạm thời
                    global GAUSSIAN_KERNEL_SIZE, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH, DILATION_KERNEL_SIZE
                    GAUSSIAN_KERNEL_SIZE = g_size
                    CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH = c_thresh
                    DILATION_KERNEL_SIZE = d_size

                    # Phát hiện contour
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL_SIZE, 0)
                    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)
                    edges = cv2.dilate(edges, kernel, iterations=2)

                    # Tìm và lọc contour
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    
                    min_contour_area = image.shape[0] * image.shape[1] * 0.3
                    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
                    if not valid_contours:
                        continue
                    
                    largest_contour = max(valid_contours, key=cv2.contourArea)

                    # Kiểm tra aspect ratio
                    rect = cv2.minAreaRect(largest_contour)
                    _, (w, h), _ = rect
                    current_aspect = max(w, h) / min(w, h)
                    if abs(current_aspect - template_aspect) > 0.5:
                        continue

                    # Cắt và xoay PCB
                    cropped = crop_and_rotate_pcb(image)
                    if cropped is None or cropped.size == 0:
                        continue

                    # Hiệu chỉnh hướng
                    cropped_corrected, _ = correct_orientation(cropped, template_image)
                    
                    # Resize và padding
                    h_crop, w_crop = cropped_corrected.shape[:2]
                    scale = min(target_w/w_crop, target_h/h_crop)
                    new_w, new_h = int(w_crop*scale), int(h_crop*scale)
                    resized = cv2.resize(cropped_corrected, (new_w, new_h))
                    
                    resized_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    x_offset = (target_w - new_w) // 2
                    y_offset = (target_h - new_h) // 2
                    resized_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

                    # Tính đặc trưng HOG và NCC
                    processed = cv2.resize(resized_padded, win_size)
                    processed_hog = hog.compute(processed).flatten()
                    
                    # Tính toán độ tương đồng
                    ncc_score = cv2.compareHist(template_hog, processed_hog, cv2.HISTCMP_CORREL)
                    aspect_score = 1 - abs((w_crop/h_crop) - template_aspect)
                    area_score = min(1, (new_w*new_h)/template_area)
                    
                    # Tính điểm tổng hợp
                    total_score = 0.6*ncc_score + 0.3*aspect_score + 0.1*area_score

                    # Debug
                    debug_img = np.hstack([template_image, resized_padded])
                    cv2.imwrite(f"{debug_dir}/match_{g_size}_{c_thresh}_{d_size}.jpg", debug_img)

                    if total_score > best_score:
                        best_score = total_score
                        best_params = (g_size, c_thresh, d_size)
                        print(f"New best params: {best_params} | Score: {total_score:.2f}")

                except Exception as e:
                    print(f"Error with params {g_size}, {c_thresh}, {d_size}: {str(e)}")
                    continue

    # Cập nhật tham số
    if best_params:
        GAUSSIAN_KERNEL_SIZE = best_params[0]
        CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH = best_params[1]
        DILATION_KERNEL_SIZE = best_params[2]
        print(f"Auto-tuning successful! Best params: {best_params}")
    else:
        print("Auto-tuning failed. Using safe defaults.")
        GAUSSIAN_KERNEL_SIZE = (7, 7)
        CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH = (50, 200)
        DILATION_KERNEL_SIZE = (60, 60)
    
    return best_params



def is_inside(inner_box, outer_box):
    """Kiểm tra xem inner_box có nằm hoàn toàn trong outer_box không"""
    return (inner_box[0] >= outer_box[0] and
            inner_box[1] >= outer_box[1] and
            inner_box[2] <= outer_box[2] and
            inner_box[3] <= outer_box[3])

# --------------------- Thread Xử Lý Ảnh --------------------------
class ImageProcessingThread(QtCore.QThread):
    processing_done = QtCore.pyqtSignal(object, object)  # Emit processed image và danh sách các object

    def __init__(self, image, template_path, parent=None):
        super().__init__(parent)
        print("ImageProcessingThread: Initializing", flush=True)
        self.image = image.copy()
        self.template_path = template_path


    def YOLO_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = model(image)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = model.names
        
        keep_mask = np.ones(len(class_ids), dtype=bool)
        

        header_smd_indices = [
            i for i, cls_id in enumerate(class_ids) 
            if class_names[cls_id] == "header-smd"
        ]
        
        for i in header_smd_indices:
            header_box = boxes[i]
            for j in range(len(class_ids)):
                if j == i or not keep_mask[j]:
                    continue
                obj_box = boxes[j]
                if is_inside(obj_box, header_box):
                    keep_mask[j] = False
        

        filtered_class_ids = class_ids[keep_mask]
        filtered_boxes = boxes[keep_mask]
        

        results[0] = results[0][keep_mask]  # Cập nhật boxes và class trong kết quả
        
        # Đếm số lượng
        detected_counts = {}
        for class_id in filtered_class_ids:
            class_name = class_names[class_id]
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        detected_objects = [(name, count) for name, count in detected_counts.items()]
        formatted_result = ", ".join([f"{count} {name}" for name, count in detected_objects])
        print(formatted_result)
        
        return results[0].plot(), detected_objects


# def YOLO_detection(self, image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Lấy kết quả từ model
#     results = model(image)
    
#     # Lấy thông tin boxes và class
#     boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
#     class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
#     class_names = model.names
    
#     # Tạo mask để đánh dấu các object cần giữ
#     keep_mask = np.ones(len(class_ids), dtype=bool)
    
#     # Tìm tất cả các header-smd
#     header_smd_indices = [
#         i for i, cls_id in enumerate(class_ids) 
#         if class_names[cls_id] == "header-smd"
#     ]
    
#     # Lọc các object bên trong header-smd
#     for i in header_smd_indices:
#         header_box = boxes[i]
#         for j in range(len(class_ids)):
#             if j == i or not keep_mask[j]:
#                 continue
#             obj_box = boxes[j]
#             if is_inside(obj_box, header_box):
#                 keep_mask[j] = False
    
#     # Giữ lại các object không bị loại
#     filtered_class_ids = class_ids[keep_mask]
#     filtered_boxes = boxes[keep_mask]
    
#     # Cập nhật kết quả vẽ
#     results[0] = results[0][keep_mask]  # Cập nhật boxes và class trong kết quả
    
#     # Đếm số lượng
#     detected_counts = {}
#     for class_id in filtered_class_ids:
#         class_name = class_names[class_id]
#         detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
    
#     detected_objects = [(name, count) for name, count in detected_counts.items()]
#     formatted_result = ", ".join([f"{count} {name}" for name, count in detected_objects])
#     print(formatted_result)
    
#     return results[0].plot(), detected_objects

    def run(self):
        try:
            print("ImageProcessingThread: Starting", flush=True)
            template_image = cv2.imread(self.template_path)
            if template_image is None:
                self.processing_done.emit(None, None)
                return
            height, width = template_image.shape[:2]
            # template_patch = template_image[height // 2:height, width // 2:width]
            template_patch = template_image[:, :]

            processed_image = process_image(self.image, template_patch)
            if processed_image is None:
                self.processing_done.emit(None, None)
                return
            processed_image, detected_objects = self.YOLO_detection(processed_image)
            # cv2.imwrite("E:/PythonProgramming/detect_components_yolov11x_v3/debug_img.jpg", processed_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            self.processing_done.emit(processed_image, detected_objects)
            # cv2.imwrite("E:/PythonProgramming/detect_components_yolov11x_v3/corrected_pcb.jpg", processed_image)
        except Exception as e:
            print("ImageProcessingThread: Exception: " + str(e), flush=True)
            self.processing_done.emit(None, None)


class AutoTuningThread(QtCore.QThread):
    tuning_done = QtCore.pyqtSignal(object)  # best_params (tuple) hoặc None nếu thất bại

    def __init__(self, image, template_path, parent=None):
        super().__init__(parent)
        self.image = image.copy()
        self.template_path = template_path

    def run(self):
        try:
            template_image = cv2.imread(self.template_path)
            if template_image is None:
                self.tuning_done.emit(None)
                return

            best_params = auto_tune_crop_params(self.image, template_image)
            cv2.imwrite("E:/PythonProgramming/detect_components_yolov11x_v3/debug/best_params.jpg", process_image(self.image, template_image))
            self.tuning_done.emit(best_params)
        except Exception as e:
            print("AutoTuningThread: Exception:", e)
            self.tuning_done.emit(None)



