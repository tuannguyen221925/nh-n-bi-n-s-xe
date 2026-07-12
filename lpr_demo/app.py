"""
Backend demo — Nhận diện ký tự biển số bằng SVM
=================================================
Pipeline khớp đúng 3 notebook:
    02_Khử_nhiễu.ipynb                    -> tiền xử lý (blur + adaptive + canny)
    03_Phat_hien_va_Phan_doan_ki_tu.ipynb -> detect_license_plate() (HSV mask + scoring)
                                              + segment_characters() (tách ký tự dính nhau)
    04_...SVM...ipynb                     -> preprocess_image() cho đặc trưng SVM

*** LƯU Ý QUAN TRỌNG ***
Notebook 04 (huấn luyện SVM) CHƯA được cung cấp khi viết lại file này, nên hàm
`char_to_feature()` bên dưới là suy luận tốt nhất dựa trên `segment_characters()`
của notebook 03: ảnh ký tự được lưu ra `char_images/` đã là ảnh NHỊ PHÂN 20x30
(letterbox, giữ tỉ lệ) — nên feature ở đây chỉ flatten ảnh đó, KHÔNG blur/threshold
lại từ đầu. Nếu notebook 04 xử lý khác (vd. blur lại, chuẩn hóa khác, resize méo
tỉ lệ thay vì letterbox...) thì cần sửa lại đúng hàm `char_to_feature()` theo đó.

Chạy:
    pip install -r requirements.txt
    uvicorn app:app --reload

Sau đó mở http://127.0.0.1:8000
"""

import base64
import pickle
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------
# Cấu hình đường dẫn model
# --------------------------------------------------------------------------
# app.py nằm ở  E:\Nhom A\lpr_demo\app.py
# model nằm ở    E:\Nhom A\model_data\  (ngang hàng với lpr_demo, KHÔNG nằm trong)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model_data"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.pkl"
SVM_SCALER_PATH = MODEL_DIR / "scaler.npz"

CHAR_WIDTH = 20
CHAR_HEIGHT = 30

app = FastAPI(title="LPR SVM Demo")

_svm = None
_is_cv2_svm = True     # True nếu model là cv2.ml_SVM, False nếu là model sklearn (pickle)
_scaler_mean = None
_scaler_std = None


def load_model():
    """
    Nạp model + scaler một lần, dùng lại cho mọi request.

    File model là .pkl nên có thể là 1 trong 2 dạng:
      (a) pickle.dump() của model sklearn (vd. sklearn.svm.SVC)
      (b) file YAML/XML do cv2 SVM.save() ghi ra nhưng đặt tên đuôi .pkl
    Hàm này tự thử pickle trước, nếu lỗi thì fallback sang cv2.ml.SVM_load().
    """
    global _svm, _is_cv2_svm, _scaler_mean, _scaler_std

    if not SVM_MODEL_PATH.exists() or not SVM_SCALER_PATH.exists():
        _svm = None
        return

    try:
        with open(SVM_MODEL_PATH, "rb") as f:
            loaded = pickle.load(f)
        _svm = loaded
        _is_cv2_svm = isinstance(_svm, cv2.ml_SVM)
    except Exception as pickle_error:
        try:
            # không phải pickle hợp lệ -> thử đọc như file cv2 SVM (yml/xml)
            _svm = cv2.ml.SVM_load(str(SVM_MODEL_PATH))
            _is_cv2_svm = True
        except Exception as cv2_error:
            # cả 2 cách đều thất bại -> chạy tiếp ở chế độ "chưa có model"
            # thay vì crash cả server khi khởi động
            print(f"[load_model] Không đọc được model: {SVM_MODEL_PATH}")
            print(f"  - pickle.load lỗi: {pickle_error}")
            print(f"  - cv2.ml.SVM_load lỗi: {cv2_error}")
            _svm = None
            return

    try:
        scaler = np.load(SVM_SCALER_PATH)
        _scaler_mean = scaler["mean"]
        _scaler_std = scaler["std"]
    except Exception as scaler_error:
        print(f"[load_model] Không đọc được scaler: {SVM_SCALER_PATH} ({scaler_error})")
        _svm = None


load_model()


# --------------------------------------------------------------------------
# Tiện ích ảnh
# --------------------------------------------------------------------------
def to_base64_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Không encode được ảnh")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh tải lên")
    return image


# ==========================================================================
# NHÓM 2 — 02_Khử_nhiễu.ipynb
# Gaussian Blur -> Adaptive Threshold (Gaussian, block=19, C=9) -> Canny(50,150)
# Ảnh Canny này chỉ để MINH HỌA bước bàn giao cho Nhóm 3 (đúng như notebook),
# bản thân detect_license_plate() ở dưới KHÔNG dùng ảnh này để dò biển số.
# ==========================================================================
def denoise_and_edge(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 19, 9,
    )
    canny = cv2.Canny(thresh, 50, 150)
    return blur, thresh, canny


# ==========================================================================
# NHÓM 3 — Bước 2-4: Phát hiện vùng biển số (HSV mask + chấm điểm)
# Port 1:1 từ hàm detect_license_plate() / is_valid_plate_crop() /
# count_char_like_regions() trong 03_Phat_hien_va_Phan_doan_ki_tu.ipynb
# ==========================================================================
def is_valid_plate_crop(plate_img: np.ndarray) -> bool:
    """Kiểm tra vùng crop có giống biển số không (loại mặt đường, kính xe, viền xe)."""
    if plate_img is None or plate_img.size == 0:
        return False

    h, w = plate_img.shape[:2]
    if h == 0 or w == 0:
        return False

    aspect = w / float(h)
    is_long_plate = 2.0 <= aspect <= 7.5
    is_square_plate = 0.6 <= aspect <= 2.2
    if not (is_long_plate or is_square_plate):
        return False

    if w < 40 or h < 15:
        return False

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def count_char_like(binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < 5:
                continue
            if ch < h * 0.22:
                continue
            if ch > h * 0.95:
                continue
            if cw < max(2, w * 0.01):
                continue
            if cw > w * 0.35:
                continue
            ratio = ch / float(cw)
            if ratio < 0.6 or ratio > 10:
                continue
            count += 1
        return count

    char_count = max(count_char_like(th_inv), count_char_like(th_bin))
    if char_count < 3:
        return False

    white_ratio_inv = np.sum(th_inv == 255) / float(th_inv.size)
    white_ratio_bin = np.sum(th_bin == 255) / float(th_bin.size)
    density = max(white_ratio_inv, white_ratio_bin)
    if density < 0.03 or density > 0.75:
        return False

    return True


def detect_license_plate(img_canny, img_raw):
    """
    Phát hiện biển số từ ảnh gốc.
    (img_canny được giữ trong chữ ký hàm cho khớp đúng notebook 03, nhưng
    không được dùng trong thân hàm — bản gốc cũng vậy.)

    Hướng xử lý: lọc màu biển số (trắng/vàng) trên HSV -> morphology nối
    vùng -> tìm contour -> lọc kích thước/tỉ lệ -> chấm điểm ứng viên ->
    bỏ qua nếu điểm không đủ tin cậy -> crop + validate lần cuối.
    """
    H, W = img_raw.shape[:2]

    # Bước 1: lọc màu biển số
    hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 70, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Bước 2: morphology nối vùng biển số
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Bước 3: tìm contour ứng viên
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = -1
    plate_rect = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if w == 0 or h == 0:
            continue
        if w < 30 or h < 8:
            continue
        if w > W * 0.85 or h > H * 0.60:
            continue

        aspect = w / float(h)
        is_1_row = 2.2 <= aspect <= 7.5
        is_2_row = 0.6 <= aspect <= 2.0
        if not (is_1_row or is_2_row):
            continue

        plate_area_ratio = (w * h) / float(H * W)
        if plate_area_ratio < 0.0008 or plate_area_ratio > 0.25:
            continue

        # Bước 4: chấm điểm ứng viên
        score_area = min(plate_area_ratio * 120, 25)

        cy = y + h / 2
        score_pos = 30 if cy > H * 0.30 else 0

        if 3.0 <= aspect <= 6.5:
            score_aspect = 45
        elif 0.8 <= aspect <= 1.7:
            score_aspect = 40
        else:
            score_aspect = 10

        extent = area / float(w * h)
        score_extent = extent * 25

        penalty_size = 0
        if w > W * 0.45:
            penalty_size += 70
        if h > H * 0.30:
            penalty_size += 70

        penalty_pos = 0
        if cy < H * 0.20:
            penalty_pos += 40

        total_score = (
            score_area + score_pos + score_aspect + score_extent
            - penalty_size - penalty_pos
        )

        if total_score > best_score:
            best_score = total_score
            best_cnt = c
            plate_rect = (x, y, w, h)

    # Bước 5: bỏ qua nếu không đủ tin cậy
    if best_cnt is None or plate_rect is None:
        return None, None, None

    x, y, w, h = plate_rect
    rw = w / float(W)
    rh = h / float(H)

    if rw > 0.45 or rh > 0.30:
        return None, None, None
    if rw < 0.035 or rh < 0.015:
        return None, None, None

    MIN_ACCEPT_SCORE = 95
    if best_score < MIN_ACCEPT_SCORE:
        return None, None, None

    # Bước 6: cắt biển số
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.12)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    plate_img = img_raw[y1:y2, x1:x2]
    plate_rect = (x1, y1, x2 - x1, y2 - y1)

    if not is_valid_plate_crop(plate_img):
        return None, None, None

    return plate_img, plate_rect, best_cnt


# ==========================================================================
# NHÓM 3 — Bước 5: Phân đoạn ký tự (xử lý ký tự dính nhau)
# Port 1:1 từ segment_characters() / choose_best_threshold() /
# split_wide_box() / sort_character_boxes() / normalize_char()
# ==========================================================================
def normalize_char(char_img: np.ndarray, out_size=(CHAR_WIDTH, CHAR_HEIGHT)):
    """Đưa ký tự về kích thước 20x30, GIỮ TỈ LỆ và căn giữa (letterbox)."""
    target_w, target_h = out_size
    h, w = char_img.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def choose_best_threshold(gray: np.ndarray):
    """
    Tạo 2 ảnh nhị phân (BINARY_INV cho biển nền sáng/chữ tối, BINARY cho
    biển nền tối/xanh chữ sáng) rồi chọn ảnh có mật độ nét hợp lý hơn.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def score_binary(th):
        white_ratio = np.sum(th == 255) / th.size
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = th.shape[:2]
        count = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < h_img * 0.20:
                continue
            if h > h_img * 0.98:
                continue
            if w < w_img * 0.01:
                continue
            if w > w_img * 0.60:
                continue
            count += 1
        if white_ratio < 0.02 or white_ratio > 0.70:
            count -= 5
        return count

    if score_binary(th_bin) > score_binary(th_inv):
        return th_bin
    return th_inv


def split_wide_box(binary: np.ndarray, box):
    """Tách contour quá rộng (nghi ngờ nhiều ký tự dính nhau) bằng projection theo cột."""
    x, y, w, h = box

    if w <= h * 0.85:
        return [box]

    roi = binary[y:y + h, x:x + w]
    if roi.size == 0:
        return [box]

    col_sum = np.sum(roi > 0, axis=0)
    threshold = max(1, int(h * 0.04))
    has_ink = col_sum > threshold

    runs = []
    start = None
    for i, value in enumerate(has_ink):
        if value and start is None:
            start = i
        if (not value or i == len(has_ink) - 1) and start is not None:
            end = i if not value else i + 1
            if end - start >= 2:
                runs.append((start, end))
            start = None

    if len(runs) >= 2:
        boxes = []
        for start, end in runs:
            nx = x + start
            nw = end - start
            if nw >= 2:
                boxes.append((nx, y, nw, h))
        return boxes

    # không có khe rõ ràng -> tách đều theo ước lượng độ rộng ký tự
    estimated_char_width = h * 0.50
    n_chars = int(round(w / estimated_char_width))
    n_chars = max(1, min(n_chars, 4))
    if n_chars <= 1:
        return [box]

    split_boxes = []
    step = w / n_chars
    for i in range(n_chars):
        nx1 = int(x + i * step)
        nx2 = int(x + (i + 1) * step)
        nw = nx2 - nx1
        if nw >= 2:
            split_boxes.append((nx1, y, nw, h))
    return split_boxes


def sort_character_boxes(boxes, plate_h):
    """Sắp xếp ký tự theo thứ tự đọc: hỗ trợ cả biển 1 dòng và 2 dòng."""
    if not boxes:
        return []

    centers_y = [y + h / 2 for x, y, w, h in boxes]

    if max(centers_y) - min(centers_y) > plate_h * 0.28:
        top, bottom = [], []
        mid_y = np.median(centers_y)
        for box in boxes:
            x, y, w, h = box
            cy = y + h / 2
            (top if cy < mid_y else bottom).append(box)
        top = sorted(top, key=lambda b: b[0])
        bottom = sorted(bottom, key=lambda b: b[0])
        return top + bottom

    return sorted(boxes, key=lambda b: b[0])


def segment_characters(plate_img: np.ndarray, padding: int = 2):
    """
    Phân đoạn ký tự từ ảnh biển số đã cắt:
    phóng to 3x -> equalizeHist -> choose_best_threshold -> xóa viền ->
    morphology open -> tìm contour -> tách box quá rộng (dính ký tự) ->
    sắp xếp theo thứ tự đọc -> chuẩn hóa 20x30 (giữ tỉ lệ).

    Trả về (char_imgs, thresh, char_boxes) — char_imgs là list ảnh NHỊ PHÂN 20x30,
    char_boxes là toạ độ (x1,y1,x2,y2) trên ảnh `thresh` để vẽ minh họa.
    """
    if plate_img is None or plate_img.size == 0:
        return [], None, []

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    scale = 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)

    thresh = choose_best_threshold(gray)
    pH, pW = thresh.shape[:2]

    # xóa viền ngoài biển số để tránh cắt nhầm khung biển
    margin_x = int(pW * 0.06)
    margin_y = int(pH * 0.12)
    thresh[:margin_y, :] = 0
    thresh[pH - margin_y:, :] = 0
    thresh[:, :margin_x] = 0
    thresh[:, pW - margin_x:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 8:
            continue
        if h < pH * 0.22:
            continue
        if h > pH * 0.95:
            continue
        if w < pW * 0.008:
            continue
        if w > pW * 0.75:
            continue
        boxes.append((x, y, w, h))

    split_boxes = []
    for box in boxes:
        split_boxes.extend(split_wide_box(thresh, box))

    split_boxes = sort_character_boxes(split_boxes, pH)

    char_imgs = []
    char_boxes_kept = []
    for x, y, w, h in split_boxes:
        near_left_edge = x < pW * 0.04
        near_right_edge = (x + w) > pW * 0.96
        near_top_edge = y < pH * 0.08
        near_bottom_edge = (y + h) > pH * 0.94

        if (near_left_edge or near_right_edge) and h > pH * 0.40:
            continue
        if (near_top_edge or near_bottom_edge) and w > pW * 0.20:
            continue

        ratio = h / float(w)
        if ratio > 8.5 and (near_left_edge or near_right_edge):
            continue

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(pW, x + w + padding)
        y2 = min(pH, y + h + padding)

        roi = thresh[y1:y2, x1:x2]
        normalized = normalize_char(roi, out_size=(CHAR_WIDTH, CHAR_HEIGHT))
        if normalized is not None:
            char_imgs.append(normalized)
            char_boxes_kept.append((x1, y1, x2, y2))

    return char_imgs, thresh, char_boxes_kept


# ==========================================================================
# NHÓM 4 — đặc trưng cho SVM (SUY LUẬN — chưa có notebook 04 để đối chiếu)
# ==========================================================================
def char_to_feature(binary_char_20x30: np.ndarray):
    """
    Ảnh đầu vào ĐÃ là ảnh nhị phân 20x30 (letterbox) do segment_characters()
    tạo ra — đúng những gì được lưu vào char_images/ để bàn giao cho nhóm 4.
    Ở đây chỉ flatten + chuẩn hóa [0,1], KHÔNG blur/threshold lại.

    ⚠️ Cần đối chiếu với hàm preprocess_image() thật trong notebook 04 khi có
    file đó — nếu notebook 04 xử lý khác (vd. tính thêm đặc trưng HOG, không
    dùng ảnh letterbox mà resize méo tỉ lệ, v.v.) thì phải sửa lại đúng ở đây.
    """
    feature = binary_char_20x30.reshape(1, -1).astype(np.float32) / 255.0
    return feature


def predict_char(feature: np.ndarray):
    if _svm is None:
        return "?", False

    scaled = ((feature - _scaler_mean) / _scaler_std).astype(np.float32)

    if _is_cv2_svm:
        _, result = _svm.predict(scaled)
        code = int(result.ravel()[0])
    else:
        # model sklearn: predict() trả thẳng mảng nhãn, không phải tuple
        result = _svm.predict(scaled)
        code = int(np.asarray(result).ravel()[0])

    return chr(code), True


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/api/process")
async def process_plate(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        original = read_upload_to_bgr(raw_bytes)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    steps = []

    # Bước 1 — Ảnh gốc
    steps.append({
        "title": "1. Ảnh đầu vào",
        "description": "Ảnh do người dùng tải lên (có thể là ảnh cả xe hoặc biển số đã crop sẵn).",
        "image": to_base64_png(original),
    })

    # Bước 2 — Nhóm 2: khử nhiễu + tách biên (chỉ để minh họa bàn giao,
    # không dùng trực tiếp để dò biển số — đúng như notebook 03)
    gray_full = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur_full, thresh_full, canny_full = denoise_and_edge(gray_full)
    steps.append({
        "title": "2. Khử nhiễu & tách biên (Nhóm 2)",
        "description": (
            "Gaussian Blur (5x5) -> Adaptive Threshold Gaussian (block=19, C=9) "
            "-> Canny(50, 150). Đây là bước bàn giao giữa Nhóm 2 và Nhóm 3; "
            "bước dò biển số ở dưới dùng lọc màu HSV trên ảnh gốc, không dùng "
            "trực tiếp ảnh Canny này."
        ),
        "image": to_base64_png(canny_full),
    })

    # Bước 3 — Nhóm 3: phát hiện vùng biển số (HSV mask + chấm điểm)
    plate_img, plate_rect, best_cnt = detect_license_plate(canny_full, original)

    if plate_img is not None and plate_rect is not None:
        x, y, w, h = plate_rect
        boxed_full = original.copy()
        cv2.rectangle(boxed_full, (x, y), (x + w, y + h), (0, 255, 0), 3)
        steps.append({
            "title": "3. Phát hiện vùng biển số (HSV + chấm điểm)",
            "description": (
                "Lọc màu biển số trắng/vàng trên không gian HSV -> morphology "
                "nối vùng -> tìm contour -> lọc kích thước/tỉ lệ -> chấm điểm "
                "ứng viên (diện tích, vị trí, aspect ratio, extent, phạt vùng "
                "quá lớn/quá cao) -> chọn ứng viên điểm cao nhất nếu đủ tin cậy."
            ),
            "image": to_base64_png(boxed_full),
        })
    else:
        # Không tìm được vùng biển số đủ tin cậy -> coi cả ảnh là biển số
        plate_img = original
        steps.append({
            "title": "3. Phát hiện vùng biển số (HSV + chấm điểm)",
            "description": (
                "Không tìm được ứng viên đủ tin cậy (điểm số hoặc kiểm tra crop "
                "không đạt) — có thể ảnh đã là biển số crop sẵn. Dùng nguyên ảnh "
                "cho bước phân đoạn ký tự."
            ),
            "image": to_base64_png(original),
        })

    # Bước 4 — Nhóm 3: phân đoạn ký tự (xử lý ký tự dính nhau)
    char_imgs, thresh_plate, char_boxes = segment_characters(plate_img)

    boxed_plate = (
        cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)
        if thresh_plate is not None else plate_img.copy()
    )
    for (x1, y1, x2, y2) in char_boxes:
        cv2.rectangle(boxed_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)

    steps.append({
        "title": "4. Phân đoạn ký tự",
        "description": (
            f"Phóng to 3x -> cân bằng histogram -> nhị phân hóa Otsu (tự chọn "
            f"BINARY hoặc BINARY_INV) -> xóa viền khung biển -> tìm contour -> "
            f"tách contour quá rộng nghi dính ký tự (projection theo cột) -> "
            f"sắp xếp trên→dưới, trái→phải. Tìm được {len(char_imgs)} ký tự."
        ),
        "image": to_base64_png(boxed_plate),
    })

    # Bước 5 — Đặc trưng + nhận diện SVM
    char_results = []
    predicted_text = ""

    for char_img in char_imgs:
        feature = char_to_feature(char_img)
        predicted_char, _ = predict_char(feature)
        predicted_text += predicted_char

        char_results.append({
            "processed": to_base64_png(char_img),
            "prediction": predicted_char,
        })

    model_warning = None
    if _svm is None:
        model_warning = (
            "Chưa tìm thấy model_data/svm_model.pkl và scaler.npz — "
            "hãy chạy notebook 04 để train và lưu model trước, "
            "ký tự sẽ tạm hiển thị '?'."
        )

    return {
        "steps": steps,
        "characters": char_results,
        "predicted_text": predicted_text,
        "character_count": len(char_imgs),
        "model_warning": model_warning,
    }


@app.get("/api/health")
async def health():
    return {
        "model_loaded": _svm is not None,
        "model_type": ("cv2_svm" if _is_cv2_svm else "sklearn") if _svm is not None else None,
        "model_path": str(SVM_MODEL_PATH),
        "scaler_path": str(SVM_SCALER_PATH),
    }


# Phục vụ luôn frontend tĩnh tại "/"
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")