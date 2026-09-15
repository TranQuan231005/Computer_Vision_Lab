# Yêu Cầu Bài Tập: Wavelet Hashing & Image Matching

## I. Mục Tiêu Bài Tập
* **Biết cách sử dụng biến đổi Wavelet**: Trích xuất thông tin đặc trưng và so sánh sự tương đồng giữa các hình ảnh.
* **Làm quen với thư viện PyWavelets (`pywt`)**: Và các công cụ xử lý ảnh trong Python (OpenCV, PIL, NumPy, Matplotlib,...).
* **Đánh giá hiệu năng**: Đánh giá kết quả của phương pháp hàm băm wavelet (Wavelet Hashing - wHash) trong việc nhận diện và đối sánh hình ảnh tương đồng.

---

## II. Bài Toán Cụ Thể

### 1. Chuẩn bị dữ liệu
* Chuẩn bị tập dữ liệu ảnh gồm:
  * **Cặp ảnh tương đồng (Similar pairs)**: Cùng một đối tượng nhưng chụp ở các góc độ khác nhau, độ sáng khác nhau, có thêm nhiễu (noise), resize, crop nhẹ, nén JPEG, v.v.
  * **Cặp ảnh không tương đồng (Non-similar pairs)**: Các hình ảnh của những đối tượng/khung cảnh hoàn toàn khác nhau.

### 2. Trích xuất đặc trưng Wavelet
* Sử dụng biến đổi Wavelet (thường là 2D Discrete Wavelet Transform - DWT, ví dụ Haar/Daubechies) để phân rã ảnh thành các dải tần số (LL, LH, HL, HH).
* Trích xuất ma trận hệ số xấp xỉ (LL subband) mang thông tin cấu trúc chủ đạo của ảnh.

### 3. Tạo mã băm (Hash code)
* Lượng tử hóa các hệ số Wavelet (ví dụ: so sánh với giá trị trung bình/trung vị để tạo chuỗi nhị phân 0/1 hoặc chuỗi hex) để tạo mã băm đại diện cho từng ảnh.

### 4. So sánh hàm băm
* Tính khoảng cách Hamming giữa các mã băm nhị phân để đánh giá mức độ tương đồng giữa các cặp hình ảnh.

### 5. Đánh giá kết quả
* **Độ chính xác (Accuracy)**: Tỷ lệ các cặp ảnh được phân loại đúng (tương đồng / không tương đồng).
* **Độ nhạy (Sensitivity / Recall)**: Tỷ lệ các cặp ảnh thực sự tương đồng được hệ thống nhận diện đúng.
* **Độ đặc hiệu (Specificity)**: Tỷ lệ các cặp ảnh không tương đồng được hệ thống phân loại đúng.
* **Đường cong ROC & AUC**: Vẽ đường cong ROC (Receiver Operating Characteristic) theo các ngưỡng khoảng cách Hamming khác nhau để đánh giá toàn diện hiệu suất của thuật toán.

---

## III. Bài Tập Nâng Cao

1. **Khảo sát & So sánh các phương pháp băm**:
   * So sánh Wavelet Hash (wHash) với các phương pháp perceptual hashing khác: Average Hash (aHash), Difference Hash (dHash), Perceptual Hash (pHash - DCT).
   * Đánh giá độ bền vững (robustness) trước các biến đổi: xoay ảnh, lật ảnh, thay đổi tỷ lệ, mờ Gaussian, nhiễu muối tiêu, thay đổi màu sắc.

2. **Xây dựng ứng dụng tìm kiếm ảnh dựa trên Wavelet Hash**:
   * Xây dựng giao diện web/ứng dụng cho phép người dùng tải lên ảnh truy vấn (Query image).
   * Tìm kiếm và trả về top các hình ảnh tương đồng nhất trong cơ sở dữ liệu dựa trên khoảng cách Hamming của mã băm Wavelet.
