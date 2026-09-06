e# Canny Edge Detection — Tu cai dat tu dau

Tu cai dat 4 buoc cua thuat toan Canny (Gaussian Blur -> Sobel Gradient -> Non-Maximum Suppression -> Hysteresis Thresholding), so sanh voi ham `cv2.Canny()` co san.

## Cau truc thu muc

```
lab2/
├── README.md                  # Tài liệu hướng dẫn & giải thích dự án
├── plan.md                    # Kế hoạch & checklist công việc
├── visioncanny.py             # Script Python chứa các hàm cài đặt Canny
├── canny_edge_detection.ipynb # Notebook chính: Cài đặt từng bước & so sánh Canny
├── LinearFiltering.ipynb      # Notebook bài tập về lọc tuyến tính (Linear Filtering)
├── baitapnangcao.ipynb        # Các bài tập/thử nghiệm nâng cao
├── web/                       # Ứng dụng Web trực quan hóa tương tác (Phase 1 -> Phase 5)
│   ├── index.html             # Giao diện chính Studio trực quan hóa
│   ├── style.css              # Giao diện Dark theme, responsive & split slider
│   └── js/                    # Mã nguồn xử lý ảnh (app.js, filters.js, canny.js, presets.js)
└── output/                    # Thư mục chứa ảnh kết quả xuất ra
```

## Trang thai

- [x] Buoc 0: Doc anh, chuyen xam
- [x] Buoc 1: Gaussian Blur
- [x] Buoc 2: Tinh gradient (Sobel)
- [x] Buoc 3: Non-Maximum Suppression
- [x] Buoc 4: Hysteresis Thresholding
- [x] So sanh voi `cv2.Canny()`
- [x] Xây dựng Web App trực quan hóa tương tác (Phase 1 đến Phase 5)
- [ ] dùng Vectorization với NumPy, Numba, hoặc Cython
- [ ] Thu nghiem tren nhieu anh/nguong khac nhau

## Cach chay

### 1. Chạy Ứng dụng Web trực quan hóa tương tác (Interactive Studio)

Ứng dụng web được xây dựng hoàn toàn phía Client (Static SPA), không yêu cầu cài đặt backend phức tạp:

- **Cách 1 (Mở trực tiếp):** Nhấp đúp chuột để mở trực tiếp file `web/index.html` trong bất kỳ trình duyệt web nào (Chrome, Edge, Firefox, Brave, Safari).
- **Cách 2 (Khởi chạy qua Local Web Server):**
  ```bash
  # Chạy máy chủ tĩnh cục bộ bằng Python
  python -m http.server 8080 --directory web
  ```
  Sau đó mở trình duyệt và truy cập: [http://localhost:8080](http://localhost:8080)

**Các tính năng nổi bật của Web Studio:**
* **Phase 1:** Điều chỉnh độ sáng ($\beta$), tương phản ($\alpha$), âm bản, cắt ngưỡng nhị phân với thanh trượt mượt mà.
* **Phase 2:** Lọc trung bình (Box blur), lọc Gaussian (tùy chỉnh $\sigma$, kernel size), và làm sắc nét (Laplacian 4/8 láng giềng).
* **Phase 3:** Phát hiện cạnh Sobel, Prewitt, và **Bộ thiết kế ma trận Kernel tùy chỉnh 3x3** (nhập trực tiếp trọng số vào ô với 12 presets mẫu, Divisor, Bias).
* **Phase 4:** Lọc phi tuyến Median (khử nhiễu muối tiêu), Bilateral (làm mịn giữ biên), kèm công cụ bơm nhiễu để kiểm thử trực tiếp.
* **Phase 5:** Trực quan hóa từng bước thuật toán Canny (Grayscale $\rightarrow$ Blur $\rightarrow$ Gradient $\rightarrow$ NMS $\rightarrow$ Hysteresis) và chế độ lưới 6 bước.
* **So sánh & Trình bày:** Thanh trượt so sánh **Before / After Split Curtain**, hỗ trợ kéo thả ảnh / dán từ Clipboard (`Ctrl+V`), tải ảnh kết quả PNG, và thẻ mã nguồn Python OpenCV tương ứng.

---

### 2. Chạy Scripts & Notebooks Python

1. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install opencv-python numpy matplotlib scikit-image
   ```
2. **Chạy script Canny độc lập:**
   ```bash
   python visioncanny.py
   ```
3. **Mở và thực thi các Notebook:**
   * `canny_edge_detection.ipynb`: Cài đặt chi tiết từng bước thuật toán Canny và so sánh đối chiếu với OpenCV `cv2.Canny()`.
   * `LinearFiltering.ipynb`: Các bài tập về lọc tuyến tính.
   * `baitapnangcao.ipynb`: Các bài tập nâng cao mở rộng.
4. **Kết quả:** Ảnh so sánh các bước sẽ được hiển thị trên cửa sổ đồ thị Matplotlib hoặc lưu trong thư mục `output/`.

---

## Ghi chu

- Ngưỡng hysteresis mặc định trong script: `low_ratio=0.05`, `high_ratio=0.15` (tính theo `magnitude.max()` sau NMS) — có thể điều chỉnh để thay đổi độ nhạy bắt cạnh.
- `cv2.Canny(blurred, 50, 150)` dùng để đối chiếu kết quả tự cài đặt với hàm chuẩn của OpenCV.

