# Lab 3: So sánh sự tương đồng hình ảnh sử dụng Wavelet (Wavelet Hashing)

Dự án thực hành chuyên sâu về kỹ thuật **Băm hình ảnh dựa trên Biến đổi Wavelet (Wavelet Hashing — wHash)**, đo lường độ tương đồng hình ảnh bằng **Khoảng cách Hamming**, đánh giá hiệu năng thống kê qua **Đường cong ROC & AUC**, khảo sát đa phương pháp băm và xây dựng **Ứng dụng Web Studio tìm kiếm ảnh tương đồng (CBIR)** tương tác thời gian thực.

---

## 📖 1. Tổng quan bài thực hành (Overview)

### 1.1 Nguyên lý Biến đổi Wavelet 2D (2D Discrete Wavelet Transform)
Khác với Biến đổi Fourier (chỉ phân tích tần số mà mất thông tin vị trí không gian), **Biến đổi Wavelet** cho phép phân tích đồng thời cả miền **tần số** và miền **không gian** của hình ảnh.

Khi áp dụng 2D DWT cấp 1 (`level=1`) trên ảnh mức xám, ảnh được phân tách thành 4 băng tần con (subbands):
- **$LL$ (Approximation — Tần số thấp):** Chứa phần lớn năng lượng và thông tin cấu trúc tổng thể của bức ảnh. Băng tần này rất bền vững với nhiễu và biến dạng nhẹ.
- **$LH$ (Horizontal Details — Chi tiết ngang):** Làm nổi bật các biên/cạnh theo phương ngang.
- **$HL$ (Vertical Details — Chi tiết dọc):** Làm nổi bật các biên/cạnh theo phương dọc.
- **$HH$ (Diagonal Details — Chi tiết chéo):** Chứa các góc và chi tiết tần số cao theo đường chéo.

### 1.2 Thuật toán Wavelet Hashing (wHash)
1. **Tiền xử lý:** Chuyển ảnh sang mức xám (Grayscale) và co giãn về kích thước chuẩn (ví dụ $32\times32$ hoặc $64\times64$).
2. **Biến đổi Wavelet:** Áp dụng biến đổi Wavelet 2D (`pywt.wavedec2` với họ Wavelet như `db4` hoặc `haar`) để thu được ma trận xấp xỉ tần số thấp $LL$.
3. **Lượng tử hóa & Sinh mã băm:**
   - Trích xuất ma trận hệ số $LL$ (ví dụ kích thước $8\times8 = 64$ phần tử hoặc $16\times16 = 256$ phần tử).
   - Tính giá trị ngưỡng trung vị (Median) hoặc trung bình (Mean) của các hệ số trong ma trận $LL$.
   - Mỗi hệ số $\ge \text{Median}$ được gán bit `1`, ngược lại gán bit `0`.
   - Kết quả thu được một chuỗi bit nhị phân (Hash code) đặc trưng cho bức ảnh.
4. **So sánh bằng Khoảng cách Hamming:**
   - Khoảng cách Hamming $D_H(\text{hash}_1, \text{hash}_2)$ là số lượng vị trí bit khác nhau giữa 2 chuỗi mã băm.
   - Khoảng cách càng nhỏ $\implies$ Hai ảnh càng giống nhau.
   - Tỉ lệ tương đồng: $\text{Similarity} = \left(1 - \frac{D_H}{N_{\text{bits}}}\right) \times 100\%$.

### 1.3 Đánh giá Thống kê & Đường cong ROC
- **Chỉ số đo lường:**
  - **Độ chính xác (Accuracy):** Tỉ lệ các cặp ảnh (tương tự & khác biệt) được phân loại đúng.
  - **Độ nhạy (Sensitivity / Recall):** Tỉ lệ nhận diện đúng các cặp ảnh tương tự thực sự ($\frac{TP}{TP + FN}$).
  - **Độ đặc hiệu (Specificity):** Tỉ lệ nhận diện đúng các cặp ảnh khác biệt ($\frac{TN}{TN + FP}$).
  - **Precision:** Tỉ lệ ảnh thực sự tương tự trong số các cặp được dự đoán tương tự ($\frac{TP}{TP + FP}$).
- **Đường cong ROC & AUC:** Quét biến thiên ngưỡng khoảng cách $\theta \in [0, N_{\text{bits}}]$ để vẽ đường cong giữa True Positive Rate (TPR) và False Positive Rate (FPR), tính diện tích dưới đường cong (AUC) và xác định ngưỡng cắt tối ưu qua chỉ số Youden's J Statistic ($J = \text{TPR} - \text{FPR}$).

---

## 📁 2. Cấu trúc thư mục (Directory Structure)

```text
lab3/
├── README.md                  # Tài liệu hướng dẫn, lý thuyết & tổng kết dự án
├── plan.md                    # Bảng phân công chi tiết công việc cho các thành viên
├── vision_wavelet.py          # Thư viện Python lõi (wHash, Hamming, Augmentation, Metrics, ROC, CBIR)
├── Lab3.ipynb                 # Jupyter Notebook thực nghiệm chi tiết từng bước & biểu đồ phân tích
├── data/                      # Thư mục dữ liệu kiểm thử
│   ├── original/              # Tập ảnh gốc mẫu (chân dung, phong cảnh, đồ vật, ký tự)
│   ├── augmented_similar/     # Tập ảnh biến dạng tương tự (xoay, đổi sáng, nhiễu, nén JPEG, làm mờ)
│   ├── dissimilar/            # Tập ảnh khác biệt
│   └── dataset_pairs.json     # File chỉ mục danh sách cặp ảnh và nhãn Ground Truth (1/0)
└── web/                       # Ứng dụng Web Studio trực quan hóa tương tác (100% Client-Side)
    ├── index.html             # Giao diện chính Studio (HTML5 Canvas + Dark Glassmorphic Theme)
    ├── style.css              # Phong cách Cyberpunk / Modern Dark Glassmorphism UI
    ├── README.md              # Hướng dẫn chi tiết sử dụng Web Studio
    └── js/                    # Mã nguồn JavaScript xử lý ảnh & Wavelet
        ├── wavelet_core.js    # Thuật toán 2D Haar Wavelet DWT & wHash thuần JS Canvas
        ├── visualizer.js      # Trực quan hóa 4 băng tần subbands LL/LH/HL/HH & ma trận Bit
        ├── matcher.js         # Bộ so khớp 2 ảnh trực tiếp, split slider, đo khoảng cách Hamming
        ├── stress_lab.js      # Phòng thí nghiệm sinh nhiễu & biến dạng trực tiếp trên Canvas
        ├── cbir_engine.js     # Bộ máy tìm kiếm ảnh tương đồng Top-K (Content-Based Image Retrieval)
        └── app.js             # Điều phối sự kiện và quản lý giao diện
```

---

## ⚙️ 3. Hướng dẫn Cài đặt & Môi trường (How to Setup)

### 3.1 Yêu cầu hệ thống
- **Python:** Phiên bản `>= 3.8` (khuyên dùng Python 3.10 hoặc 3.11).
- **Trình duyệt Web:** Bất kỳ trình duyệt hiện đại nào hỗ trợ HTML5 Canvas (Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Safari).

### 3.2 Cài đặt các thư viện Python
Mở terminal hoặc command prompt và cài đặt các thư viện phụ thuộc:

```bash
pip install numpy scipy matplotlib pywavelets opencv-python scikit-learn pillow seaborn
```

*Giải thích các thư viện chính:*
- `pywavelets` (`pywt`): Thư viện chuyên dụng biến đổi Wavelet 1D/2D trong Python.
- `opencv-python` (`cv2`): Xử lý ảnh số, biến đổi hình học và lọc ma trận.
- `numpy`: Thao tác mảng nhiều chiều và tính toán đại số ma trận siêu tốc.
- `scikit-learn`: Tính toán ma trận nhầm lẫn, đường cong ROC và chỉ số AUC.
- `matplotlib` & `seaborn`: Trực quan hóa biểu đồ phân tích khoa học.
- `pillow` (`PIL`): Đọc và lưu trữ định dạng ảnh.

---

## 🚀 4. Hướng dẫn Thực thi & Triển khai (How to Implement & Run)

### 4.1 Khởi chạy Ứng dụng Web Studio Tương tác (Interactive Web Studio)
Ứng dụng Web Studio được thiết kế chạy **100% Client-Side Static SPA**, hoàn toàn độc lập và không cần cài đặt backend phức tạp:

- **Cách 1 (Mở trực tiếp):** Nhấp đúp chuột vào tệp [`web/index.html`](file:///d:/Computer%20Vision/Lap/lab3/web/index.html) trên máy tính.
- **Cách 2 (Khởi chạy qua HTTP Server nội bộ):**
  ```bash
  python -m http.server 8080 --directory web
  ```
  Sau đó mở trình duyệt và truy cập: [http://localhost:8080](http://localhost:8080)

**Các tính năng nổi bật của Web Studio:**
1. **Trực quan hóa 4 Băng tần Wavelet (Subband Visualizer):** Tách và hiển thị trực tiếp 4 băng tần $LL, LH, HL, HH$ kèm biểu đồ ma trận bit nhị phân thời gian thực.
2. **So khớp 2 Ảnh Trực quan (Dual Matcher):** Tải 2 ảnh bất kỳ $\rightarrow$ Tự động tính wHash $\rightarrow$ Hiển thị số bit lệch, khoảng cách Hamming và kết luận Đồng dạng (Matched) / Khác nhau (Dissimilar).
3. **Phòng thí nghiệm Biến dạng (Live Robustness Stress Lab):** Kéo thanh trượt để thêm nhiễu Gauss, nhiễu muối tiêu, xoay góc, chỉnh sáng $\rightarrow$ Kiểm chứng trực tiếp tính bất biến của wHash.
4. **Công cụ Tìm kiếm Ảnh Tương tự (CBIR Gallery):** Tải ảnh truy vấn $\rightarrow$ Tìm kiếm và xếp hạng Top 5 / Top 10 ảnh giống nhất trong kho cơ sở dữ liệu mẫu.

---

### 4.2 Chạy Thư viện Lõi Python & Kiểm thử tự động
Chạy script Python độc lập để kiểm tra toàn bộ pipeline từ tạo dữ liệu, trích xuất wHash, tính khoảng cách Hamming đến đánh giá thống kê:

```bash
python vision_wavelet.py
```

*Script sẽ tự động:*
1. Khởi tạo tập ảnh mẫu và sinh các biến thể ảnh tương tự / khác biệt trong `data/`.
2. Tính toán mã băm wHash cho toàn bộ tập dữ liệu.
3. Xuất bảng chỉ số đánh giá (Accuracy, Recall, Specificity, Precision, F1-Score, AUC).
4. Lưu biểu đồ 4 băng tần Wavelet, ma trận nhầm lẫn và đường cong ROC vào thư mục kết quả.

---

### 4.3 Thực thi Jupyter Notebook Thực nghiệm (`Lab3.ipynb`)
Mở Jupyter Notebook hoặc VS Code / Google Colab để chạy từng bước thực nghiệm:

```bash
jupyter notebook Lab3.ipynb
```

**Nội dung chi tiết trong Notebook:**
- **Phần 1:** Giới thiệu lý thuyết biến đổi Wavelet 2D DWT và các họ Wavelet cơ bản.
- **Phần 2:** Chuẩn bị tập dữ liệu ảnh và sinh các biến dạng hình học, nhiễu, nén.
- **Phần 3:** Cài đặt thuật toán wHash và trực quan hóa 4 băng tần $LL, LH, HL, HH$.
- **Phần 4:** Đo lường hiệu suất thống kê, vẽ Ma trận nhầm lẫn và Đường cong ROC / AUC.
- **Phần 5 (Nâng cao 1):** Khảo sát thực nghiệm so sánh các họ Wavelet (`haar`, `db2`, `db4`, `sym4`, `bior2.2`), các cấp độ phân giải và đối đầu với `aHash`, `dHash`, `pHash`.
- **Phần 6 (Nâng cao 2):** Xây dựng bộ máy tìm kiếm ảnh tương đồng (Content-Based Image Retrieval).

---

## 👥 5. Phân công Nhóm thực hiện (Team Assignment)

| Thành viên | Nhiệm vụ chính |
|---|---|
| **Ngọc Nhân** | **Kỹ sư Dữ liệu & wHash Core:** Thu thập ảnh, sinh biến dạng tương tự/khác biệt, cài đặt 2D DWT (`pywt.wavedec2`), lượng tử hóa và sinh mã băm wHash, hàm khoảng cách Hamming. |
| **Thanh Nguyên** | **Kỹ sư Đánh giá & Khảo sát:** Đo lường Accuracy, Recall, Specificity, F1-Score, vẽ Đường cong ROC & tính AUC, xác định ngưỡng tối ưu, khảo sát các họ Wavelet và so sánh aHash/dHash/pHash. |
| **Minh Quân** | **Kiến trúc sư & Web Studio UI:** Thiết kế Web Studio Dark Glassmorphism, bộ trực quan hóa 4 subbands, Dual Matcher, Live Stress Lab, CBIR Top-K Image Search, quản trị Notebook và tài liệu. |

---

## 📚 6. Tài liệu Tham khảo (References)
1. TS. Nguyễn Thị Khánh Tiên, *Chương 3 (Part 1): Phát hiện đặc trưng và đối sánh ảnh*, Trường ĐH Giao thông vận tải TP.HCM (UTH).
2. PyWavelets Documentation: [https://pywavelets.readthedocs.io/](https://pywavelets.readthedocs.io/)
3. ImageHash Python Library: [https://github.com/JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash)
4. Scikit-image Wavelet Transform: [https://scikit-image.org/docs/stable/auto_examples/transform/plot_dwt.html](https://scikit-image.org/docs/stable/auto_examples/transform/plot_dwt.html)
