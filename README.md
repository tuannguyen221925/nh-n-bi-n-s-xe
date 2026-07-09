Bảng phân công công việc chi tiết (8 người - 4 nhóm)
Quy trình sẽ đi tuần tự từ dữ liệu → tiền xử lý → phân đoạn kí tự → nhận diện và đánh giá kết quả. 
Nhóm 1 (2 người): Thu thập dữ liệu & Tiền xử lý ảnh (Bước quan trọng nhất)
	Nhiệm vụ:
	Thu thập và làm sạch tập dữ liệu ảnh/video xe từ các nguồn trên. 
	Xử lý không gian màu: Chuyển đổi ảnh màu sang ảnh xám từ hệ màu HSV (lấy kênh Value - cường độ sáng) thay vì hệ màu RGB thông thường để thích nghi tốt hơn với môi trường thay đổi ánh sáng. 
	Tăng độ tương phản (Cực kỳ quan trọng): Ứng dụng các phép toán hình thái học (Morphological Transformations): Thực hiện phép toán Top Hat và Black Hat, sau đó tính toán theo công thức: Ảnh đầu ra = Ảnh gốc + Top Hat - Black Hat để làm nổi bật rõ rệt vùng biển số so với phông nền. 
	Sản phẩm đầu ra: Hàm tiền xử lý ảnh giúp chuẩn hóa ảnh đầu vào thành ảnh xám có độ tương phản cao, sẵn sàng cho các bước lọc tiếp theo. 
Nhóm 2 (2 người): Giảm nhiễu, Nhị phân hóa & Phát hiện cạnh
	Nhiệm vụ:
	Giảm nhiễu: Áp dụng bộ lọc Gauss (Gaussian Blur) với kích thước kernel phù hợp (ví dụ 5x5) để làm mượt ảnh và loại bỏ các hạt nhiễu li ti. 
	Nhị phân hóa: Sử dụng phương pháp ngưỡng động (Adaptive Thresholding), cụ thể là toán tử ADAPTIVE_THRESH_GAUSSIAN_C trong OpenCV giúp xử lý hiệu quả ngay cả khi ảnh bị hiện tượng chói sáng hoặc shadow (vùng tối/sáng không đều). 
	Phát hiện cạnh Canny: Triển khai giải thuật Canny Edge Detection (tính Gradient, loại bỏ điểm không cực đại và lọc ngưỡng kép) nhằm trích xuất toàn bộ các đường biên cạnh sắc nét của biển số xe. 
	Sản phẩm đầu ra: Ảnh nhị phân chỉ giữ lại các đường nét cạnh biên phục vụ cho việc tìm khung biển số. 
Nhóm 3 (2 người): Trích xuất, Lọc tọa độ biển số & Phân đoạn kí tự
	Nhiệm vụ:
	Tìm Contour biển số: Sử dụng thuật toán Suzuki's (hàm findContours trong OpenCV) để lấy các đường biên, sau đó tiến hành xấp xỉ đa giác để tìm các khung hình có đúng 4 đỉnh. 
	Lọc điều kiện hình học: Thiết lập thuật toán lọc diện tích và tỉ lệ cao/rộng chuẩn của biển số Việt Nam: biển dài (3.5≤"cao/rộng"≤6.5) và biển vuông (0.8≤"cao/rộng"≤1.5). 
	Xoay chỉnh & Cắt kí tự: Viết thuật toán tính góc nghiêng dựa trên 2 đỉnh đáy của biển số để xoay ảnh thẳng lại (tránh thuật toán nhận diện nhầm số 1 với 7, chữ B với số 8...). Tiếp tục dùng Contour để tìm vùng trắng của các kí tự trong biển số, cắt và chuẩn hóa chúng về kích thước cố định là 20×30 pixels. 
	Sản phẩm đầu ra: Tập hợp các file ảnh nhỏ chứa từng kí tự riêng biệt đã được làm sạch và căn chỉnh thẳng thắn. 
Nhóm 4 (2 người): Gán nhãn, Huấn luyện mô hình KNN & Đánh giá kết quả
	Nhiệm vụ:
	Xây dựng Dataset cho KNN: Tạo hoặc thu thập tập dữ liệu mẫu gồm đầy đủ các chữ cái và chữ số (sử dụng font biển số xe tiêu chuẩn), sau đó làm phẳng ma trận ảnh kí tự (20×30=600 pixel) thành vector hàng. 
	Gán nhãn & Train: Thực hiện gán nhãn thủ công thông qua mã ASCII để xuất ra hai file cơ sở dữ liệu classifications.txt và flattened_images.txt. Tiến hành lập trình thuật toán K-Nearest Neighbor (KNN) bằng OpenCV. 
	Tích hợp & Đánh giá: Nhận diện các kí tự được chuyển từ Nhóm 3 qua mô hình KNN, gộp chuỗi kí tự và vẽ đè kết quả văn bản lên hình ảnh gốc ban đầu. Thống kê tỉ lệ nhận diện sai (ví dụ lỗi nhận diện sai giữa 1 ↔ 7, 6 ↔ 0) để viết phần kết luận cho đồ án. 
	Sản phẩm đầu ra: Mô hình học máy nhận diện kí tự hoàn chỉnh và bảng thống kê, đánh giá độ chính xác của toàn bộ hệ thống. 
Lời khuyên để đạt điểm cao với Thầy:
Vì thầy rất xem trọng bước xử lý ảnh, trong phần báo cáo thuyết phục, Nhóm 1 và Nhóm 2 cần phải lưu lại hình ảnh minh họa sau mỗi bước biến đổi (Ảnh gốc → Ảnh xám HSV → Ảnh tăng tương phản Top Hat/Black Hat → Ảnh lọc Gauss → Ảnh nhị phân Adaptive → Ảnh biên Canny). Khi thầy thấy các bước trung gian được xử lý cực kỳ trực quan và mạch lạc, điểm số của nhóm chắc chắn sẽ rất cao
Nhóm bạn (Dùng OpenCV + KNN) nên xử lý tập dữ liệu này ra sao?
Vì thầy của bạn rất quan trọng bước xử lý ảnh căn bản, nhóm bạn không thể bê nguyên tập dữ liệu này đi train như Deep Learning được, mà phải biến thư mục test (hoặc cả train) thành nguồn "Raw Data" để chạy các thuật toán xử lý ảnh hình thái học.
Tuấn hãy phân công các nhóm xử lý đống dữ liệu này theo hướng sau:
📂 Coi toàn bộ ảnh trong thư mục test hoặc train là "Raw Data" để tìm biển số
	Nhóm 1, 2 và 3 sẽ lấy các ảnh xe nguyên bản (ảnh chụp thực tế có cả đầu xe/đuôi xe) từ thư mục này để chạy code qua các bước: Đổi màu HSV → Top Hat/Black Hat → Lọc Gauss → Canny → Cắt lấy khung biển số → Tách rời từng kí tự.
📂 Việc "Train" và "Test" của mô hình KNN nhóm bạn sẽ hoạt động thế nào?
Đối với thuật toán KNN mà nhóm đang làm (như trong file .docx và .pptx của bạn mô tả):
	Dữ liệu huấn luyện (Train) thực sự của KNN: Không phải là ảnh cả chiếc xe, mà là ảnh của các kí tự mẫu đơn lẻ (các chữ từ A-Z và số từ 0-9) đã được cắt chuẩn hóa về kích thước 20×30, được lưu dưới dạng vector trong file flattened_images.txt và nhãn tương ứng trong classifications.txt.
	Quá trình Test: Chính là lúc nhóm bạn lấy một ảnh xe bất kỳ trong thư mục dữ liệu Kaggle kia ra, cho chạy qua thuật toán xử lý ảnh để cắt ra biển số, phân đoạn kí tự, rồi bắt mô hình KNN đoán xem kí tự đó là số mấy, chữ gì.
💡 Hướng dẫn gom thư mục cho cả nhóm dễ làm việc:
Để không bị rối bởi cấu trúc của Kaggle, Tuấn bảo Nhóm 1 làm như sau:
	Tạo một thư mục đặt tên là data/raw_data/ như cấu trúc cũ mình thống nhất.
	Bạn có thể chọn lấy toàn bộ ảnh nằm trong thư mục test của Kaggle (khoảng vài trăm ảnh xe thực tế Việt Nam là quá đủ làm bài rồi) copy hết thảy vào thư mục raw_data này.
	Bỏ qua cấu trúc train/test mặc định của Kaggle đi, cứ coi đống ảnh xe đó là ảnh thô đầu vào để hệ thống của mình tự xử lý, tự cắt biển số và tự nhận diện từ đầu đến cuối.
Làm cách này, cấu trúc các file Notebook (01_Tien_xu_ly.ipynb, 02_Phan_doan.ipynb) của nhóm bạn vẫn chạy mượt mà đúng chuẩn bài toán Computer Vision mà thầy yêu cầu!

