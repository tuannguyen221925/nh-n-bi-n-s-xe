# README – NHẬN DIỆN KÝ TỰ BIỂN SỐ BẰNG SVM

## 1. Giới thiệu

Phần này sử dụng thuật toán **Support Vector Machine (SVM)** để nhận diện các ký tự chữ và số đã được tách từ ảnh biển số xe.

SVM là một thuật toán học máy có giám sát, phù hợp với các bài toán phân loại dữ liệu có số chiều lớn. Trong dự án này, mỗi ảnh ký tự được chuyển thành một vector gồm 600 đặc trưng, sau đó được đưa vào mô hình SVM để huấn luyện và dự đoán.

Notebook sử dụng:

```text
group4_svm_hoan_chinh.ipynb
```

Mô hình được triển khai bằng thư viện OpenCV.

## 2. Mục tiêu

- Nhận diện chữ cái và chữ số từ ảnh ký tự biển số.
- Đánh giá khả năng phân loại của SVM.
- Khảo sát ảnh hưởng của tham số `C` và `gamma`.
- Tìm cấu hình SVM tốt nhất.
- So sánh kết quả với phương pháp KNN.
- Lưu mô hình để tái sử dụng khi dự đoán dữ liệu mới.

## 3. Giả thuyết

SVM sử dụng kernel RBF có thể phân loại tốt các ký tự có hình dạng gần giống nhau nhờ khả năng tạo ranh giới phân loại phi tuyến.

Chất lượng mô hình phụ thuộc chủ yếu vào:

```text
C
gamma
```

Giả thuyết của nhóm là một giá trị `C` và `gamma` trung bình sẽ cho kết quả tốt hơn các giá trị quá nhỏ hoặc quá lớn.

## 4. Dữ liệu đầu vào

Dữ liệu huấn luyện được khai báo trong file:

```text
model_data/labels.csv
```

Cấu trúc:

```csv
image_path,label
data/char_images/bien_so_1/char_0.png,5
data/char_images/bien_so_1/char_1.png,1
data/char_images/bien_so_1/char_2.png,A
```

Trong đó:

- `image_path`: đường dẫn đến ảnh ký tự.
- `label`: nhãn thật của ảnh.
- Mỗi nhãn chỉ chứa một ký tự.
- Nhãn có thể là chữ số `0-9` hoặc chữ cái `A-Z`.

## 5. Cấu trúc thư mục

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
│   ├── svm_model.yml
│   └── svm_scaler.npz
│
├── group4_svm_hoan_chinh.ipynb
└── README_SVM.md
```

## 6. Thư viện cần thiết

Cài đặt bằng lệnh:

```bash
python -m pip install numpy pandas matplotlib opencv-python jupyter
```

Các thư viện chính:

- `OpenCV`: đọc ảnh, tiền xử lý và xây dựng mô hình SVM.
- `NumPy`: xử lý dữ liệu số và ma trận.
- `Pandas`: hiển thị bảng kết quả.
- `Matplotlib`: vẽ biểu đồ và Confusion Matrix.
- `Jupyter`: chạy notebook.

## 7. Quy trình tiền xử lý

Mỗi ảnh ký tự được xử lý theo các bước:

1. Đọc ảnh xám.
2. Resize về `20 x 30` pixel.
3. Gaussian Blur để giảm nhiễu.
4. Otsu Threshold để nhị phân hóa.
5. Đảo ảnh nếu cần để ký tự sáng trên nền tối.
6. Trải phẳng thành vector 600 chiều.
7. Chia pixel cho 255 để đưa dữ liệu về khoảng `[0, 1]`.

Mỗi ảnh sau tiền xử lý có:

```text
20 x 30 = 600 đặc trưng
```

## 8. Chia dữ liệu train và test

Thiết lập mặc định:

```text
test_size = 0.25
seed = 42
```

Ý nghĩa:

- 75% dữ liệu dùng để huấn luyện.
- 25% dữ liệu dùng để đánh giá.
- `seed = 42` giúp kết quả chia dữ liệu có thể lặp lại.

## 9. Chuẩn hóa đặc trưng

SVM nhạy với thang đo dữ liệu nên chương trình chuẩn hóa theo công thức:

```text
X_scaled = (X - mean) / standard deviation
```

`mean` và `standard deviation` chỉ được tính trên tập train để tránh rò rỉ dữ liệu.

## 10. Mô hình SVM

Mô hình sử dụng:

```text
Loại SVM: C-Support Vector Classification
Kernel: RBF
```

Khởi tạo bằng OpenCV:

```python
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
```

## 11. Ý nghĩa tham số C

`C` quyết định mức phạt khi mô hình phân loại sai.

### C nhỏ

- Cho phép nhiều lỗi hơn.
- Ranh giới phân loại đơn giản hơn.
- Có thể bị underfitting.

### C lớn

- Cố gắng phân loại đúng nhiều mẫu train hơn.
- Ranh giới phức tạp hơn.
- Có thể bị overfitting.

Các giá trị được khảo sát:

```text
C = 0.1
C = 1
C = 10
```

## 12. Ý nghĩa tham số gamma

`gamma` quyết định phạm vi ảnh hưởng của mỗi mẫu dữ liệu.

### Gamma nhỏ

- Một mẫu ảnh hưởng đến vùng rộng.
- Ranh giới phân loại mượt hơn.
- Có thể chưa phân biệt đủ chi tiết.

### Gamma lớn

- Một mẫu chỉ ảnh hưởng đến vùng nhỏ.
- Ranh giới phân loại phức tạp hơn.
- Có nguy cơ overfitting.

Các giá trị được khảo sát:

```text
gamma = 0.001
gamma = 0.01
gamma = 0.1
```

## 13. Khảo sát tham số

Notebook chạy tổng cộng 9 cấu hình:

```text
C = 0.1, 1, 10
gamma = 0.001, 0.01, 0.1
```

Cấu hình tốt nhất được chọn theo:

1. Macro F1-score cao nhất.
2. Nếu Macro F1 bằng nhau, chọn Accuracy cao hơn.

## 14. Các chỉ số đánh giá

Notebook hiển thị:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Macro Precision.
- Macro Recall.
- Macro F1-score.
- Support.
- Confusion Matrix.
- Số mẫu đúng và sai.

Macro F1 đặc biệt quan trọng khi số lượng mẫu giữa các lớp không cân bằng.

## 15. Confusion Matrix

Trong Confusion Matrix:

```text
Hàng: nhãn thật
Cột: nhãn dự đoán
```

- Đường chéo chính: dự đoán đúng.
- Ngoài đường chéo: dự đoán sai.

Các cặp ký tự dễ nhầm:

```text
0 và O
1 và I
2 và Z
5 và S
8 và B
6 và G
```

## 16. Huấn luyện mô hình cuối

Sau khi chọn được `C` và `gamma` tốt nhất, chương trình:

1. Chuẩn hóa toàn bộ dữ liệu.
2. Train lại SVM trên toàn bộ dữ liệu.
3. Lưu mô hình và scaler.

Các file đầu ra:

```text
model_data/svm_model.yml
model_data/svm_scaler.npz
```

## 17. Cách chạy notebook

Mở file:

```text
group4_svm_hoan_chinh.ipynb
```

Sau đó chọn:

```text
Run All
```

Trước khi chạy, cần bảo đảm:

```text
model_data/labels.csv
```

đã tồn tại và đã được gán nhãn đầy đủ.

## 18. Dự đoán một thư mục ký tự

Sau khi train xong:

```python
predicted_text, prediction_table = predict_folder_svm(
    folder=r"data/char_images/thu_muc_bien_so_can_du_doan"
)
```

Thư mục phải chứa các ảnh:

```text
char_0.png
char_1.png
char_2.png
...
```

Chương trình sẽ đọc ảnh, tiền xử lý, chuẩn hóa, dự đoán và ghép thành chuỗi biển số.

## 19. Lỗi thường gặp

### Không tìm thấy `labels.csv`

Kiểm tra:

- Notebook có nằm trong thư mục gốc dự án không.
- Có đúng thư mục `model_data` không.
- File `labels.csv` đã được tạo chưa.

### Không tìm thấy ảnh

Kiểm tra đường dẫn trong `labels.csv` và tên file ảnh.

### Không có mẫu test

Nguyên nhân thường do dữ liệu quá ít hoặc mỗi lớp chỉ có một mẫu.

### F1-score bằng 0

Có thể do:

- Lớp có quá ít dữ liệu.
- Mô hình không dự đoán đúng mẫu nào.
- Nhãn bị gán sai.
- Ảnh của các lớp quá giống nhau.

## 20. Ưu điểm của SVM

- Phù hợp với dữ liệu có số chiều lớn.
- Hoạt động tốt với dữ liệu vừa và nhỏ.
- Kernel RBF hỗ trợ ranh giới phi tuyến.
- Có khả năng tổng quát hóa tốt khi chọn tham số phù hợp.
- Dự đoán không cần so sánh với toàn bộ tập train như KNN.

## 21. Hạn chế của SVM

- Phụ thuộc nhiều vào `C` và `gamma`.
- Train chậm hơn KNN.
- Cần chuẩn hóa dữ liệu.
- Khó giải thích trực quan hơn KNN.
- Pixel thô chưa mô tả tốt hình dạng ký tự.
- Kết quả vẫn phụ thuộc vào chất lượng ảnh tách ký tự.

## 22. So sánh ngắn với KNN

### KNN

- Train nhanh.
- Dễ hiểu và dễ triển khai.
- Dự đoán phải tìm các láng giềng gần nhất.
- Thời gian dự đoán tăng khi tập train lớn.

### SVM

- Train chậm hơn.
- Cần tối ưu `C` và `gamma`.
- Có thể xây dựng ranh giới phi tuyến tốt hơn.
- Thường ổn định hơn khi số chiều đặc trưng lớn.

Không nên kết luận SVM luôn tốt hơn KNN. Cần dựa trên kết quả thực nghiệm của chính bộ dữ liệu.

## 23. Kết quả đạt được

Phần SVM đã thực hiện:

- Đọc dữ liệu từ `labels.csv`.
- Tiền xử lý ảnh ký tự.
- Chuẩn hóa ảnh về `20 x 30`.
- Chuyển ảnh thành vector 600 chiều.
- Chia train/test theo lớp.
- Chuẩn hóa đặc trưng.
- Huấn luyện SVM-RBF.
- Khảo sát 9 tổ hợp `C` và `gamma`.
- Tự động chọn cấu hình tốt nhất.
- Đánh giá bằng Accuracy, Precision, Recall và F1.
- Hiển thị Confusion Matrix.
- Phân tích các mẫu dự đoán sai.
- Lưu mô hình và scaler.
- Dự đoán dữ liệu mới.

## 24. Hướng phát triển

- Tăng số lượng dữ liệu.
- Cân bằng số mẫu giữa các lớp.
- Áp dụng data augmentation.
- Thử nhiều giá trị `C` và `gamma` hơn.
- Dùng HOG thay cho pixel thô.
- So sánh Linear SVM, RBF SVM và Polynomial SVM.
- So sánh với KNN, Random Forest hoặc CNN.
- Tích hợp vào toàn bộ pipeline nhận diện biển số.

## 25. Kết luận

SVM-RBF là phương pháp phù hợp cho bài toán nhận diện ký tự biển số với dữ liệu vừa và nhỏ.

Mô hình được khảo sát tham số, đánh giá bằng nhiều chỉ số và phân tích các trường hợp dự đoán sai. Kết quả cuối cùng cần được so sánh với KNN trên cùng dữ liệu để lựa chọn phương pháp phù hợp nhất.
