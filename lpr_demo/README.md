# LPR SVM Demo — Web đơn giản hiển thị pipeline nhận diện biển số

## Cấu trúc
```
lpr_demo/
├── app.py              # Backend FastAPI, xử lý toàn bộ pipeline
├── requirements.txt
├── model_data/          # <-- copy svm_model.yml + svm_scaler.npz từ notebook 04 vào đây
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Cách chạy

1. Cài thư viện:
   ```
   pip install -r requirements.txt
   ```

2. Copy 2 file model đã train từ notebook 04 (`svm_model.yml`, `svm_scaler.npz`)
   vào thư mục `model_data/`. Nếu chưa copy, web vẫn chạy được nhưng ký tự
   sẽ hiển thị `?` và có banner cảnh báo.

3. Chạy server:
   ```
   uvicorn app:app --reload
   ```

4. Mở trình duyệt: http://127.0.0.1:8000

## Pipeline hiển thị trên web

1. Ảnh gốc người dùng tải lên
2. Ảnh xám (grayscale)
3. Làm mờ Gaussian
4. Nhị phân hóa Otsu
5. Phát hiện & phân đoạn ký tự (vẽ bounding box xanh, sắp xếp trái → phải)
6. Từng ký tự được resize 20x30, đưa qua SVM → ghép thành kết quả cuối

Bước tiền xử lý ký tự (`preprocess_char_feature`) và scaler dùng **đúng logic**
với notebook `04_Train_SVM_va_Nhan_dien.ipynb` để model dự đoán chính xác
như lúc train.

## Lưu ý

- Ảnh đầu vào nên là ảnh **biển số đã crop gọn** (chưa làm bước tách biển số
  từ ảnh toàn cảnh — nếu cần bước đó, nói mình làm thêm).
- Ngưỡng lọc contour ký tự (chiều cao, tỉ lệ khung hình, số lượng tối đa)
  nằm ở đầu file `app.py`, chỉnh lại nếu ảnh biển số của nhóm có tỉ lệ khác.
