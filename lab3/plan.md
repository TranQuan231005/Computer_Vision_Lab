# Trạng thái & Kế hoạch Lab 3: So sánh sự tương đồng hình ảnh sử dụng Wavelet (PyWavelets)

> `[ ]` Chưa làm · `[x]` Đã xong

## Cập nhật phần Thanh Nguyên — 16/09/2026

- Phase 3–4 dùng dataset và API của Nhân trong `vision_wavelet.py`.
- Đánh giá chia theo nguồn, chọn ngưỡng trên hiệu chỉnh bằng Youden J,
  báo cáo ROC/AUC, confusion matrix và metrics trên kiểm thử.
- Khảo sát 55 cấu hình; xuất bảng CSV, độ bền theo biến dạng và latency.
- Báo cáo thực nghiệm nằm ở mục 7–8 của `Lab3.ipynb`, đầu ra ở
  `results/evaluation/`; kiểm thử bổ sung trong `test_evaluation.py`.
- Phase 6 tiếp tục để trống vì chưa tích hợp Web Studio của cả nhóm.

## Cập nhật phần Trần Ngọc Nhân — 15/09/2026

- Hoàn thành các mục **1.1–1.5 và 2.1–2.5** trong `vision_wavelet.py`.
- Đóng gói 15 ảnh nguồn (10 mẫu scikit-image, 5 chữ số tự vẽ), 225 biến thể,
  225 cặp cùng nguồn và 105 cặp khác nguồn trong `data/dataset_pairs.json`.
- Có `Lab3.ipynb` cho Phase 1–2, kiểm thử `test_vision_wavelet.py` và
  `results/pair_distances.json` để bàn giao cho phần đánh giá.
- Đã đóng góp Phase 6 bằng tài liệu và notebook phần Nhân; các mục Phase 6
  vẫn để trống vì chưa tích hợp phần đánh giá và Web Studio của cả nhóm.
- Chi tiết chạy, API và giới hạn dữ liệu xem phần **Bản bàn giao** trong `README.md`.

---

## 🎯 Mục tiêu Lab 3 (What does this Lab do?)

Bài thực hành tập trung vào việc áp dụng **Biến đổi Wavelet 2D (2D Discrete Wavelet Transform - DWT)** để trích xuất đặc trưng tần số không gian của hình ảnh, tạo **Mã băm hình ảnh Wavelet (Wavelet Hash - wHash)**, và đánh giá mức độ tương đồng giữa các ảnh:

1. **Hiểu và làm chủ biến đổi Wavelet**: Sử dụng thư viện `PyWavelets` (`pywt`) để phân tách ảnh thành 4 băng tần: $LL$ (xấp xỉ tần số thấp), $LH$ (chi tiết ngang), $HL$ (chi tiết dọc), $HH$ (chi tiết chéo).
2. **Thuật toán băm ảnh Wavelet (wHash)**: Lượng tử hóa ma trận hệ số $LL$ dựa trên ngưỡng trung vị (Median) / trung bình (Mean) để sinh vector bit nhị phân 64-bit hoặc 256-bit bất biến với các biến dạng nhỏ.
3. **So sánh & Phân loại tương đồng**: Sử dụng **Khoảng cách Hamming (Hamming Distance)** giữa các chuỗi mã băm để quyết định 2 ảnh là tương tự (Matched/Similar) hay khác biệt (Dissimilar).
4. **Đánh giá thống kê & Đường cong ROC**:
   - Đo lường: Độ chính xác (Accuracy), Độ nhạy / Recall (Sensitivity), Độ đặc hiệu (Specificity; khác với Precision), F1-Score.
   - Vẽ đường cong ROC (Receiver Operating Characteristic) và tính diện tích dưới đường cong (AUC).
   - Xác định ngưỡng cắt tối ưu (Optimal Decision Threshold).
5. **Bài tập nâng cao**:
   - Khảo sát thực nghiệm so sánh các họ Wavelet (`haar`, `db2`, `db4`, `sym4`, `bior2.2`, `coif2`), các mức level 1–4, và đối đầu với `aHash`, `dHash`, `pHash`.
   - Xây dựng ứng dụng tìm kiếm ảnh tương đồng (Content-Based Image Retrieval - CBIR Web Studio).

---

## 👥 Bảng phân công nhiệm vụ (Assignment & Task Breakdown)

### Phase 1 — Chuẩn bị Dữ liệu & Biến dạng thực nghiệm (Dataset Preparation)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [x] | 1.1 | Thu thập tập ảnh gốc đa dạng (chân dung, đồ vật, phong cảnh, kết cấu, chữ số). | Ngọc Nhân |
| [x] | 1.2 | Xây dựng bộ sinh biến thể ảnh tương tự (xoay góc nhỏ $\pm 5^\circ, \pm 10^\circ$, phóng to/thu nhỏ 0.8x-1.2x, đổi sáng $\pm 20\%$). | Ngọc Nhân |
| [x] | 1.3 | Sinh biến thể nhiễu và nén ảnh (nhiễu Gauss, nhiễu muối tiêu Salt & Pepper, nén JPEG quality 30-70%, làm mờ Gaussian Blur). | Ngọc Nhân |
| [x] | 1.4 | Tạo các cặp ảnh không tương tự (Dissimilar negative pairs) từ các đối tượng khác nhau. | Ngọc Nhân |
| [x] | 1.5 | Đóng gói tập dữ liệu cặp ảnh và gán nhãn chuẩn Ground Truth (1: Tương tự, 0: Khác biệt) vào file chỉ mục `dataset_pairs.json`. | Ngọc Nhân |

### Phase 2 — Thuật toán Wavelet Hashing & Khoảng cách Hamming (wHash Core)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [x] | 2.1 | Cài đặt tiền xử lý ảnh: chuyển đổi mức xám (Grayscale), chuẩn hóa kích thước cố định ($32\times32, 64\times64$). | Ngọc Nhân |
| [x] | 2.2 | Cài đặt 2D DWT sử dụng `pywt.wavedec2` trích xuất 4 băng tần subbands $LL, LH, HL, HH$. | Ngọc Nhân |
| [x] | 2.3 | Cài đặt thuật toán lượng tử hóa hệ số $LL$ theo ngưỡng Median/Mean để sinh mã băm nhị phân (64-bit / 256-bit wHash). | Ngọc Nhân |
| [x] | 2.4 | Cài đặt hàm tính khoảng cách Hamming (`hamming_distance`) giữa 2 mã băm và quy đổi sang % tương đồng (Similarity). | Ngọc Nhân |
| [x] | 2.5 | Đóng gói các hàm thành module tái sử dụng `WaveletHasher` trong file `vision_wavelet.py`. | Ngọc Nhân |

### Phase 3 — Đánh giá Hiệu suất & Đường cong ROC (Evaluation & ROC Analysis)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [x] | 3.1 | Tính toán ma trận khoảng cách Hamming cho toàn bộ các cặp ảnh trong tập dữ liệu kiểm thử. | Thanh Nguyên |
| [x] | 3.2 | Quét dải ngưỡng phân loại $\theta \in [0, N_{\text{bits}}]$ để phân loại Tương tự / Khác biệt. | Thanh Nguyên |
| [x] | 3.3 | Tính toán các chỉ số thống kê định lượng: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score. | Thanh Nguyên |
| [x] | 3.4 | Vẽ đường cong ROC (True Positive Rate vs. False Positive Rate) và tính chỉ số AUC. | Thanh Nguyên |
| [x] | 3.5 | Xác định ngưỡng phân loại tối ưu (Optimal Threshold) bằng chỉ số Youden's J Statistic và vẽ Ma trận nhầm lẫn (Confusion Matrix). | Thanh Nguyên |

### Phase 4 — Bài tập Nâng cao 1: Khảo sát Đa phương pháp Băm (Survey & Benchmark)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [x] | 4.1 | Khảo sát thực nghiệm các họ Wavelet trong PyWavelets (`haar`, `db2`, `db4`, `sym4`, `bior2.2`, `coif2`) trên cùng tập dữ liệu. | Thanh Nguyên |
| [x] | 4.2 | Khảo sát ảnh hưởng của cấp độ phân giải (Level 1, 2, 3, 4) và kích thước Hash ($8\times8$ vs $16\times16$). | Thanh Nguyên |
| [x] | 4.3 | Cài đặt & so sánh đối đầu với các thuật toán băm ảnh phổ biến khác: Average Hash (`aHash`), Difference Hash (`dHash`), Perceptual Hash (`pHash`). | Thanh Nguyên |
| [x] | 4.4 | Kiểm thử độ bền vững (Robustness Test) theo từng loại biến dạng (Noise, Rotation, Scale, Lighting, JPEG Blur) và đo thời gian thực thi (Latency Benchmark). | Thanh Nguyên |
| [x] | 4.5 | Tổng hợp bảng số liệu so sánh toàn diện và vẽ biểu đồ radar / cột phân tích trong notebook. | Thanh Nguyên |

### Phase 5 — Bài tập Nâng cao 2 & Web Studio tương tác (Web UI & CBIR Engine)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [ ] | 5.1 | Thiết kế giao diện Web Studio hiện đại (HTML5 Canvas + Vanilla CSS Dark Glassmorphism, kéo thả ảnh, paste `Ctrl+V`, chọn ảnh mẫu). | Minh Quân |
| [ ] | 5.2 | Xây dựng bộ trực quan hóa 4 băng tần Wavelet ($LL, LH, HL, HH$) và ma trận bit nhị phân thời gian thực bằng JS Canvas. | Minh Quân |
| [ ] | 5.3 | Xây dựng bộ so khớp 2 ảnh trực tiếp (Dual Image Matcher): live Hamming distance, số bit lệch, % tương đồng, nhãn Matched / Dissimilar. | Minh Quân |
| [ ] | 5.4 | Xây dựng phòng thí nghiệm kiểm thử biến dạng trực tiếp (Live Robustness Stress Lab) với thanh trượt thêm nhiễu, xoay, chỉnh sáng, làm mờ. | Minh Quân |
| [ ] | 5.5 | Xây dựng công cụ tìm kiếm ảnh tương đồng (Content-Based Image Retrieval - CBIR): lập chỉ mục kho ảnh, tìm kiếm & hiển thị Top-K ảnh tương đồng nhất. | Minh Quân |

### Phase 6 — Tích hợp Hệ thống & Đóng gói Báo cáo (System Integration)
| Xong | ID | Công việc chi tiết | Thành viên đảm nhận |
|:---:|:---:|---|---|
| [ ] | 6.1 | Đóng gói trọn vẹn thư viện `lab3/vision_wavelet.py` sạch đẹp, tối ưu hóa, chú thích 100% tiếng Việt. | Cả nhóm |
| [ ] | 6.2 | Hoàn thiện Notebook thực nghiệm `lab3/Lab3.ipynb` với đầy đủ code, hình ảnh trực quan và diễn giải phân tích khoa học. | Cả nhóm |
| [ ] | 6.3 | Cập nhật hoàn chỉnh tài liệu `lab3/README.md` (hướng dẫn cài đặt, lý thuyết, kết quả thực nghiệm). | Minh Quân |
| [ ] | 6.4 | Kiểm thử toàn bộ hệ thống (Zero-error execution trên Python script, Jupyter notebook và Web Studio). | Cả nhóm |

---

## 📌 Quy tắc Thực hiện
- Toàn bộ code xử lý lõi đặt trong `vision_wavelet.py` để dễ tái sử dụng và kiểm thử.
- Toàn bộ quá trình thực nghiệm, biểu đồ, ma trận phân tích được trình bày chi tiết trong `Lab3.ipynb`.
- Giao diện Web Studio đặt tại `web/` chạy 100% Client-Side không phụ thuộc backend nặng.
- Chú thích code, giải thích markdown và tài liệu sử dụng 100% tiếng Việt rõ ràng, mạch lạc.
- Khi hoàn thành công việc nào, thay `[ ]` bằng `[x]` tương ứng trong bảng trên.
