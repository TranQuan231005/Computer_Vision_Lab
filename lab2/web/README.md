# Interactive Image Processing & Canny Edge Detection Studio (Web App)

Ứng dụng web trực quan hóa tương tác và xử lý ảnh số thời gian thực (Real-time Image Processing & Edge Detection Web Studio), được thiết kế hoàn toàn bằng HTML5 Canvas, Vanilla CSS (Dark Cyberpunk / Glassmorphic UI) và JavaScript thuần (không dùng thư viện nặng).

---

## 🌟 Tính năng chính

Ứng dụng được chia thành **5 giai đoạn (Phases)** tương ứng với lộ trình xử lý ảnh số và thị giác máy tính:

### 1. Phase 1: Biến đổi điểm ảnh & Đoạn cơ bản (Point Operations)
* **Độ sáng ($\beta$) & Độ tương phản ($\alpha$):** $g(x,y) = \alpha \cdot f(x,y) + \beta$.
* **Ảnh âm bản (Invert):** Đảo ngược mức xám $255 - I$.
* **Phân ngưỡng nhị phân (Binary Thresholding):** Phân ngưỡng cứng với ngưỡng cắt tùy chỉnh trực quan.
* **Biểu đồ Histogram mức xám:** Hiển thị phân bố cường độ sáng thời gian thực.

### 2. Phase 2: Lọc tuyến tính & Làm sắc nét (Linear Filtering & Sharpening)
* **Lọc trung bình (Box Blur):** Làm mịn ảnh cơ bản với kích thước kernel tùy chỉnh ($3\times3, 5\times5, 7\times7$).
* **Lọc Gaussian (Gaussian Blur):** Tạo ma trận nhân Gaussian 2D động theo kích thước kernel và bán kính lệch chuẩn $\sigma$.
* **Làm sắc nét (Laplacian Sharpening):** Tăng cường chi tiết và biên với bộ lọc Laplacian 4 láng giềng hoặc 8 láng giềng.

### 3. Phase 3: Phát hiện biên cơ bản & Bộ thiết kế ma trận Kernel tùy chỉnh
* **Toán tử Gradient Sobel & Prewitt:** Tách thành phần đạo hàm theo phương ngang $G_x$ và phương dọc $G_y$, kết hợp độ lớn biên gradient $G = \sqrt{G_x^2 + G_y^2}$.
* **Trình thiết kế Kernel 3x3 tùy chỉnh (Custom Convolution Matrix Designer):**
  * Nhập trực tiếp 9 trọng số ma trận tùy ý.
  * Tùy chỉnh hệ số chia (Divisor) và độ lệch bù (Bias).
  * Đi kèm **12 Presets mẫu** thông dụng: *Identity, Box Blur 3x3, Gaussian 3x3, Sharpen, Edge Enhance, Ridge Detection, Sobel X/Y, Prewitt X/Y, Emboss, Outline*.

### 4. Phase 4: Khử nhiễu phi tuyến & Bơm nhiễu thực nghiệm (Non-linear Filtering)
* **Lọc trung vị (Median Filter):** Khử nhiễu muối tiêu (Salt & Pepper) hiệu quả mà không làm nhòe cạnh quá mức.
* **Lọc song phương (Bilateral Filter):** Làm mịn bề mặt đồng nhất nhưng bảo toàn biên cạnh sắc nét (Edge-preserving Smoothing) với tham số $\sigma_{spatial}$ và $\sigma_{intensity}$.
* **Công cụ tạo nhiễu (Noise Generator):** Bơm nhiễu muối tiêu (Salt & Pepper) và nhiễu Gauss (Gaussian Noise) trực tiếp vào ảnh để so sánh hiệu năng các bộ lọc.

### 5. Phase 5: Thuật toán Canny Edge Detection & Pipeline trực quan
* Tự cài đặt và mô phỏng trực quan trọn vẹn **5 bước của thuật toán Canny**:
  1. **Bước 1:** Ảnh mức xám (Grayscale).
  2. **Bước 2:** Khử nhiễu với Gaussian Blur ($\sigma$, kernel).
  3. **Bước 3:** Tính Gradient Sobel ($G_x, G_y$, Magnitude & Angle).
  4. **Bước 4:** Triệt tiêu không cực đại (Non-Maximum Suppression - NMS) theo 4 hướng góc chuẩn ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
  5. **Bước 5:** Phân ngưỡng kép và nối biên Hysteresis (Double Thresholding & Edge Tracking).
* **2 Chế độ hiển thị:**
  * **Step-by-Step Viewer:** Xem từng bước một qua dropdown với thanh so sánh kéo trượt.
  * **6-Step Grid Mode:** Hiển thị đồng thời toàn bộ 6 bước trên lưới ảnh để dễ đối chiếu.

---

## 🛠️ Công cụ tiện ích & Trải nghiệm người dùng (UX)

* **Thanh trượt so sánh Trước / Sau (Split Curtain Comparison Slider):** Kéo chuột qua lại để so sánh trực tiếp ảnh gốc và ảnh đã qua xử lý.
* **Nhập ảnh linh hoạt:**
  * Kéo & thả file ảnh trực tiếp vào giao diện.
  * Nút chọn file ảnh từ máy tính.
  * Dán ảnh nhanh từ Clipboard bằng phím tắt `Ctrl + V`.
  * Bộ ảnh mẫu có sẵn (Lena, Cameraman, Shapes, Coins, ...) để kiểm thử nhanh.
* **Thanh thông tin & Trạng thái (Live Status & FPS):** Hiển thị kích thước ảnh, tỉ lệ zoom và thời gian xử lý / FPS thực tế.
* **Xuất ảnh kết quả:** Tải ảnh kết quả đã xử lý về máy ở định dạng `.png` chất lượng cao.
* **Trình tạo mã nguồn Python OpenCV tương ứng (Code Snippet Viewer):** Tự động sinh đoạn mã Python tương ứng với mọi cấu hình tham số đang chọn, kèm nút Copy mã nguồn 1-click.

---

## 🚀 Hướng dẫn khởi chạy

Do ứng dụng là **Client-side Single Page Application (SPA)** hoàn toàn tĩnh, bạn không cần cài đặt môi trường backend hay database.

### Cách 1: Mở trực tiếp
Nhấp đúp chuột vào file `index.html` để mở ngay bằng bất kỳ trình duyệt web nào (Chrome, Edge, Firefox, Brave, Safari).

### Cách 2: Chạy qua Web Server cục bộ (Khuyên dùng)
Nếu muốn chạy qua HTTP Server cục bộ:

```bash
# Đứng tại thư mục web và khởi chạy máy chủ Python:
cd web
python -m http.server 8000
```
Hoặc đứng từ thư mục gốc `lab2`:
```bash
python -m http.server 8000 --directory web
```
Sau đó mở trình duyệt và truy cập: **[http://localhost:8000](http://localhost:8000)**

---

## 📁 Cấu trúc thư mục

```
web/
├── index.html         # Giao diện chính chứa toàn bộ Layout, Controls, Presets & Canvas
├── style.css          # Phong cách thiết kế Dark Theme, Glassmorphism, Layout 3 cột & Responsive
├── js/
│   ├── app.js         # Quản lý sự kiện giao diện, Canvas rendering, Split slider, Zoom/Pan & State
│   ├── filters.js     # Thuật toán xử lý ảnh cơ bản, làm mịn, làm sắc nét, tích chập ma trận & khử nhiễu
│   ├── canny.js       # Triển khai thuật toán Canny 5 bước và các hàm trích xuất dữ liệu trung gian
│   └── presets.js     # Thư viện ảnh mẫu SVG/Base64 và 12 ma trận tích chập chuẩn
└── README.md          # Tài liệu hướng dẫn sử dụng thư mục Web
```
