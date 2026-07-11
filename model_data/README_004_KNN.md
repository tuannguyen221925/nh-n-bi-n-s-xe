# PHẦN 4 – NHẬN DIỆN KÝ TỰ BIỂN SỐ BẰNG KNN

## 1. Mục tiêu

Phần 4 sử dụng thuật toán **K-Nearest Neighbors (KNN)** để nhận diện từng ký tự đã được tách từ ảnh biển số xe.

Đầu vào của phần này là các ảnh ký tự riêng lẻ, ví dụ:

```text
char_0.png
char_1.png
char_2.png
...
```

Đầu ra gồm:

- Ký tự được dự đoán.
- Mô hình KNN đã huấn luyện.
- Các thông số đánh giá mô hình.
- Confusion Matrix.
- Danh sách các mẫu dự đoán sai.

---

## 2. Quy trình xử lý

Quy trình nhận diện ký tự gồm các bước:

1. Đọc ảnh ký tự ở dạng ảnh xám.
2. Chuẩn hóa kích thước ảnh về `20 x 30` pixel.
3. Làm mờ ảnh bằng Gaussian Blur.
4. Nhị phân hóa ảnh bằng ngưỡng Otsu.
5. Chuyển ảnh thành vector gồm `600` đặc trưng.
6. Chia dữ liệu thành tập train và test theo từng lớp.
7. Huấn luyện mô hình KNN.
8. Dự đoán ký tự trên tập test.
9. Đánh giá kết quả và lưu mô hình.

Mỗi ảnh sau tiền xử lý có:

```text
20 x 30 = 600 đặc trưng
```

---

## 3. Cấu trúc thư mục

Cấu trúc đề xuất:

```text
project/
│
├── data/
│   └── char_images/
│       ├── bien_so_1/
│       │   ├── char_0.png
│       │   ├── char_1.png
│       │   └── ...
│       └── bien_so_2/
│           ├── char_0.png
│           └── ...
│
├── model_data/
│   ├── labels.csv
│   ├── classifications.txt
│   ├── flattened_images.txt
│   ├── knn_model.yml
│   ├── evaluation_report.txt
│   └── confusion_matrix.csv
│
├── group4_knn_can_bang.ipynb
└── README_GROUP4_KNN.md
```

---

## 4. Các thư viện cần thiết

Cài đặt bằng lệnh:

```bash
python -m pip install numpy pandas matplotlib opencv-python jupyter
```

Các thư viện chính:

- `OpenCV`: đọc ảnh, tiền xử lý và xây dựng KNN.
- `NumPy`: xử lý ma trận và vector đặc trưng.
- `Pandas`: hiển thị bảng kết quả.
- `Matplotlib`: vẽ biểu đồ và Confusion Matrix.
- `Jupyter`: chạy notebook.

---

## 5. Chuẩn bị dữ liệu

Các ảnh ký tự phải có tên bắt đầu bằng:

```text
char_
```

Ví dụ:

```text
char_0.png
char_1.png
char_2.png
```

Notebook tìm ảnh theo cấu trúc:

```python
data/char_images/*/char_*.png
```

---

## 6. Gán nhãn dữ liệu

File nhãn mặc định:

```text
model_data/labels.csv
```

Cấu trúc file:

```csv
image_path,label
data/char_images/bien_so_1/char_0.png,5
data/char_images/bien_so_1/char_1.png,1
data/char_images/bien_so_1/char_2.png,A
```

Trong đó:

- `image_path`: đường dẫn ảnh ký tự.
- `label`: ký tự thật tương ứng với ảnh.
- Mỗi nhãn chỉ được chứa đúng một ký tự.
- Nhãn có thể là chữ số `0-9` hoặc chữ cái `A-Z`.

Có thể dùng hàm sau để tạo hoặc cập nhật file nhãn:

```python
init_labels(
    char_dir=DEFAULT_CHAR_DIR,
    model_dir=DEFAULT_MODEL_DIR
)
```

Sau đó mở `labels.csv` và điền cột `label`.

---

## 7. Chạy huấn luyện và đánh giá

Mở file:

```text
group4_knn_can_bang.ipynb
```

Chạy lần lượt toàn bộ các cell từ trên xuống.

Cell chính:

```python
results = evaluate_knn(
    model_dir=DEFAULT_MODEL_DIR,
    k=3,
    test_size=0.25,
    seed=42
)
```

Ý nghĩa tham số:

| Tham số | Ý nghĩa |
|---|---|
| `model_dir` | Thư mục chứa nhãn và mô hình |
| `k` | Số láng giềng gần nhất |
| `test_size` | Tỷ lệ dữ liệu dùng để kiểm tra |
| `seed` | Giá trị cố định để chia dữ liệu có thể lặp lại |

Thiết lập mặc định:

```text
K = 3
Test size = 0.25
Seed = 42
```

---

## 8. Các thông số được hiển thị

Notebook hiển thị các thông số sau:

### Thông số dữ liệu

- Tổng số mẫu.
- Số lớp ký tự.
- Số mẫu train.
- Số mẫu test.
- Số chiều đặc trưng.
- Phân bố số lượng mẫu theo từng nhãn.

### Thông số mô hình

- Giá trị `K`.
- Tỷ lệ chia test.
- Seed.
- Số dự đoán đúng.
- Số dự đoán sai.
- Accuracy.

### Chỉ số đánh giá

- Precision.
- Recall.
- F1-score.
- Macro Precision.
- Macro Recall.
- Macro F1-score.
- Support của từng lớp.
- Confusion Matrix.
- Khoảng cách trung bình của các láng giềng KNN.
- Danh sách các ảnh bị dự đoán sai.

---

## 9. Ý nghĩa các chỉ số

### Accuracy

Tỷ lệ tổng số mẫu được dự đoán đúng:

```text
Accuracy = Số mẫu đúng / Tổng số mẫu test
```

### Precision

Trong các mẫu được mô hình dự đoán là một lớp, Precision cho biết có bao nhiêu mẫu thực sự thuộc lớp đó.

### Recall

Trong toàn bộ mẫu thật của một lớp, Recall cho biết mô hình nhận diện đúng được bao nhiêu mẫu.

### F1-score

Trung bình điều hòa giữa Precision và Recall.

### Macro Average

Tính trung bình chỉ số của tất cả các lớp, mỗi lớp có trọng số như nhau.

### Support

Số mẫu thật của từng lớp trong tập test.

### Confusion Matrix

- Hàng: nhãn thật.
- Cột: nhãn dự đoán.
- Các giá trị trên đường chéo chính là dự đoán đúng.
- Các giá trị ngoài đường chéo là dự đoán sai.

---

## 10. Các file đầu ra

Sau khi huấn luyện, chương trình có thể tạo các file:

| File | Nội dung |
|---|---|
| `knn_model.yml` | Mô hình KNN đã huấn luyện |
| `labels.csv` | Danh sách ảnh và nhãn |
| `classifications.txt` | Nhãn được chuyển sang dạng số |
| `flattened_images.txt` | Vector đặc trưng của ảnh |
| `evaluation_report.txt` | Báo cáo đánh giá |
| `confusion_matrix.csv` | Ma trận nhầm lẫn |

Mô hình chính được lưu tại:

```text
model_data/knn_model.yml
```

---

## 11. Dự đoán một thư mục ký tự

Sau khi có mô hình, có thể dự đoán các ảnh trong một thư mục:

```python
from types import SimpleNamespace

args = SimpleNamespace(
    folder="duong_dan_den_thu_muc_anh",
    model_dir="model_data",
    k=3
)

predict_folder(args)
```

Thư mục phải chứa các ảnh:

```text
char_0.png
char_1.png
char_2.png
...
```

Kết quả được ghép theo thứ tự tên file thành một chuỗi ký tự.

---

## 12. Lỗi thường gặp

### Không tìm thấy `labels.csv`

```text
FileNotFoundError: Chua co model_data/labels.csv
```

Cách xử lý:

1. Chạy hàm `init_labels`.
2. Mở `model_data/labels.csv`.
3. Điền nhãn cho các ảnh.
4. Chạy lại quá trình huấn luyện.

### Không có mẫu test

Nguyên nhân:

- Dữ liệu quá ít.
- Một số lớp chỉ có một ảnh.

Cách xử lý:

- Bổ sung thêm ảnh cho mỗi lớp.
- Mỗi lớp nên có nhiều mẫu khác nhau.

### `k` lớn hơn số mẫu train

Giảm giá trị `k`, ví dụ:

```python
k=3
```

### Không tìm thấy ảnh

Kiểm tra:

- Đường dẫn ảnh trong `labels.csv`.
- Tên file có bắt đầu bằng `char_` hay không.
- Notebook có đang được mở ở đúng thư mục gốc dự án hay không.

### Chạy cell nhưng không có kết quả

Cell chứa `def ...` chỉ dùng để định nghĩa hàm.

Cần chạy thêm cell gọi hàm:

```python
results = evaluate_knn(...)
```

---

## 13. Kết quả đạt được

Phần 4 đã thực hiện được:

- Tiền xử lý và chuẩn hóa ảnh ký tự.
- Chuyển ảnh thành vector đặc trưng.
- Huấn luyện mô hình KNN.
- Nhận diện chữ số và chữ cái.
- Chia tập train/test theo từng lớp.
- Đánh giá bằng nhiều chỉ số.
- Hiển thị Confusion Matrix.
- Phân tích các trường hợp dự đoán sai.
- Lưu mô hình để tái sử dụng.

---

## 14. Hạn chế

Một số hạn chế hiện tại:

- Kết quả phụ thuộc nhiều vào chất lượng ảnh ký tự được tách.
- Dữ liệu ít hoặc mất cân bằng làm giảm độ chính xác.
- KNN có thể chậm khi tập dữ liệu lớn.
- Ảnh nghiêng, mờ, thiếu nét hoặc dính ký tự dễ bị nhận diện sai.
- KNN sử dụng trực tiếp pixel nên chưa khai thác được đặc trưng hình dạng phức tạp.

---

## 15. Hướng phát triển

Có thể cải tiến bằng các hướng:

- Tăng số lượng và độ đa dạng của ảnh huấn luyện.
- Cân bằng số mẫu giữa các lớp.
- Bổ sung xoay, dịch chuyển, làm mờ và thay đổi độ sáng để tăng cường dữ liệu.
- Tối ưu giá trị `K`.
- So sánh KNN với SVM, Random Forest hoặc CNN.
- Kết hợp toàn bộ quy trình phát hiện biển số, tách ký tự và nhận diện thành một hệ thống hoàn chỉnh.

---

## 16. Kết luận

KNN là phương pháp đơn giản, dễ triển khai và phù hợp để xây dựng mô hình nhận diện ký tự cơ bản. Phần 4 hoàn thiện bước cuối của quy trình nhận diện biển số: biến các ảnh ký tự đã tách thành chuỗi ký tự có ý nghĩa.

Độ chính xác của mô hình phụ thuộc trực tiếp vào chất lượng dữ liệu, bước tách ký tự và số lượng mẫu được gán nhãn.
