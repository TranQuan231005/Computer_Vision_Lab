# Lab 3: So sánh sự tương đồng hình ảnh sử dụng Wavelet (Wavelet Hashing)

## Bản bàn giao — Trần Ngọc Nhân

**Đã triển khai Phase 1–2:** dữ liệu, biến dạng, DWT, wHash và Hamming.
Các phần mô tả đánh giá, benchmark và Web Studio phía dưới là **kế hoạch của cả nhóm**, chưa được triển khai trong phần bàn giao này.

### Chạy phần đã hoàn thành

Python >=3.10, mở terminal tại `lab3`:

```powershell
python -m pip install -r requirements.txt
python vision_wavelet.py
python -m unittest -v test_vision_wavelet.py
```

Mở [Lab3.ipynb](Lab3.ipynb) bằng VS Code/Jupyter, chọn Python đã cài thư viện rồi **Run All**.
Notebook có phần giải thích tiếng Việt, ảnh minh họa và đầu ra thực nghiệm Phase 1–2.
`create_notebook.py` tạo lại cấu trúc notebook và sẽ bỏ đầu ra cũ.

Trong phiên hiện tại, thư viện bổ sung nằm tại `.lab3_deps` ở thư mục chứa `Computer_Vision_Lab-main`.
Có thể chạy tại thư mục đó bằng:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.lab3_deps'
python Computer_Vision_Lab-main/lab3/vision_wavelet.py
```

### Dữ liệu đã đóng gói

- 15 ảnh nguồn: 10 mẫu từ `scikit-image` và 5 chữ số bảy đoạn tự vẽ, ghi rõ `synthetic: true`.
- 225 biến thể: mỗi ảnh có xoay ±5°/±10°, resize 0.8/1.2, sáng 0.8/1.2, Gauss σ=10,
  muối tiêu 2%, JPEG quality 30/50/70, blur bán kính 1.5 và crop 5% mỗi phía.
- 330 cặp: 225 cùng nguồn (nhãn 1), 105 tổ hợp hai nguồn khác nhau (nhãn 0).
- `data/dataset_pairs.json` chứa `schema_version`, `seed`, `originals`, `pairs`, `label_definition`.
  Mỗi cặp có `id`, `image1`, `image2`, `label`, `source_id1`, `source_id2`, `transform`, `parameters`.
  Đường dẫn ảnh tương đối với thư mục chứa JSON. Cặp âm tham chiếu trực tiếp `original/`, không nhân bản vào `dissimilar/`.
- Nguồn ảnh và liên kết tra cứu được lưu trong `originals`; xem [scikit-image data](https://scikit-image.org/docs/stable/api/skimage.data.html)
  để tra tác giả và điều kiện sử dụng từng ảnh. Ảnh mẫu có chân dung, động vật, đồ vật, phong cảnh và kết cấu.
- Seed mặc định 42, nhiễu từng ảnh dùng seed cộng thứ tự nguồn. JPEG được giải mã và lưu PNG sau một lần nén.
- Xoay giữ khung, nền trắng; resize thay độ phân giải. Dữ liệu không mô phỏng đầy đủ góc chụp mới.

Chạy `python vision_wavelet.py --rebuild` để tái tạo các tệp mẫu do chương trình quản lý;
các tệp trùng tên sẽ được ghi lại. Đọc dữ liệu đã đóng gói không cần mạng;
tái tạo có thể cần mạng nếu phiên bản scikit-image tải ảnh mẫu theo yêu cầu.

### API bàn giao

```python
from vision_wavelet import WaveletHasher, hamming_distance, similarity_percentage

hasher = WaveletHasher()  # Haar, ảnh 32×32, level 2, hash 8×8, median
a = hasher.hash('data/original/astronaut.png')
b = hasher.hash('data/augmented_similar/astronaut__jpeg_30.png')
print(hamming_distance(a, b), similarity_percentage(a, b))
details = hasher.analyze('data/original/astronaut.png')
large = WaveletHasher(hash_size=16, image_size=64)  # 256 bit
```

Đầu vào: đường dẫn, PIL hoặc NumPy `uint8` RGB. Xử lý EXIF và ghép alpha nền trắng trước khi chuyển xám.
`analyze` trả `gray`, `LL`, `LH`, `HL`, `HH`, `coeffs`, `features`, `cutoff`, `bits`, `hex`.
`hash` trả vector boolean theo từng hàng; `compare` trả `distance`, `similarity`.

DWT dùng `periodization`, bốn băng lấy cùng cấp sâu nhất. Theo [PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html),
LL=cA, LH=cH, HL=cV, HH=cD; cH/cV theo quy ước trục của thư viện.
Nếu LL lớn hơn kích thước hash thì lấy trung bình khối đều nhau; nếu nhỏ hơn thì báo lỗi.
Bit dùng phép **>= median/mean**. Chỉ so sánh hash có cùng cấu hình; lưu `hasher.config()` cùng hash.
Với wavelet bộ lọc dài hoặc level cao cần tăng kích thước ảnh đầu vào để không vượt `dwt_max_level`.
Triển khai theo đề bài, không cam kết trùng bit với ImageHash vì không có bước loại DC.

`compute_pair_distances` băm mỗi ảnh một lần. Script xuất `results/pair_distances.json` gồm
`config`, `n_bits`, `hashes` (hex) và `pairs` có nhãn, khoảng cách và phần trăm bit trùng nhau.
Notebook xuất `originals.png`, `augmentations.png`, `subbands.png`, `hash_bits.png` vào `results/`.

### Kiểm chứng và giới hạn

Kiểm thử dùng ma trận Haar biết trước, tái dựng nghịch đảo trên 6 họ wavelet,
64/256 bit, median/mean, Hamming, alpha, ảnh đơn sắc, tham số sai, tái lập nhiễu và tính hợp lệ của mọi cặp/ảnh.
Đã chạy đạt **7/7 kiểm thử** và **7/7 ô mã notebook**, đã xem kiểm tra bốn hình kết quả.
Notebook thực nghiệm với Haar 64 bit cho khoảng cách trung bình **4.64/64** ở 225 cặp cùng nguồn,
và **31.03/64** ở 105 cặp khác nguồn. Hai khoảng giá trị vẫn chồng lấn (0–28 và 8–52),
nên không thể kết luận phân loại hoàn hảo. Sai số tái dựng ảnh mẫu là khoảng `7.77e-16`.
15 nguồn ảnh là tập minh họa nhỏ; hash có thể va chạm và không bất biến hoàn toàn với xoay/crop.
Phần trăm bit trùng **không phải xác suất nhận dạng**. Chưa chọn ngưỡng hay báo cáo ROC/AUC.

Thanh Nguyên tiếp tục Phase 3–4, Minh Quân tiếp tục Phase 5. Khi chia tập chọn ngưỡng/kiểm thử,
**chia theo nguồn trước khi tạo cặp**; cặp âm phải có cả hai nguồn thuộc cùng tập để tránh rò rỉ dữ liệu.
Phase 6 chỉ mới hoàn thành tích hợp và kiểm tra phạm vi của Nhân, chưa đánh dấu xong toàn nhóm.

---

## Đặc tả tổng thể của nhóm (bao gồm các phần dự kiến)

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
- **Python:** Phiên bản `>= 3.10` cho mã nguồn đã triển khai.
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

### 4.1 Dự kiến khởi chạy Web Studio (chưa triển khai — Minh Quân)
Ứng dụng Web Studio được thiết kế chạy **100% Client-Side Static SPA**, hoàn toàn độc lập và không cần cài đặt backend phức tạp:

- **Cách 1 (Mở trực tiếp, sau khi triển khai):** Nhấp đúp chuột vào tệp `web/index.html` trên máy tính.
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
Chạy script Python để tạo dữ liệu khi chưa có chỉ mục, trích xuất wHash và tính khoảng cách Hamming:

```bash
python vision_wavelet.py
```

*Script sẽ tự động:*
1. Khởi tạo tập ảnh mẫu và sinh các biến thể ảnh tương tự / khác biệt trong `data/`.
2. Tính toán mã băm wHash cho toàn bộ tập dữ liệu.
3. Xuất `results/pair_distances.json` để bàn giao cho Phase 3.
4. Chạy notebook để lưu ảnh nguồn, biến dạng, bốn băng tần và ma trận bit. Chỉ số phân loại và ROC chưa triển khai.

---

### 4.3 Thực thi Jupyter Notebook Thực nghiệm (`Lab3.ipynb`)
Mở Jupyter Notebook hoặc VS Code / Google Colab để chạy từng bước thực nghiệm:

```bash
jupyter notebook Lab3.ipynb
```

**Cấu trúc dự kiến cho notebook toàn nhóm (bản hiện tại chỉ có dữ liệu, DWT, hash, Hamming và bàn giao):**
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
