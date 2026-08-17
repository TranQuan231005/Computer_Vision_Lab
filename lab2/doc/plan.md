# Trạng thái Lab 2
[ ] Chưa làm · [x] Đã xong

## Phase 1 — Toán tử điểm ảnh
| Xong | ID | Công việc |
|---|---|---|
| [ ] | 1.1 | Điều chỉnh độ sáng (tăng/giảm). |
| [ ] | 1.2 | Điều chỉnh độ tương phản (nhân với hằng số). |
| [ ] | 1.3 | Tạo ảnh âm bản. |
| [ ] | 1.4 | Áp dụng cắt ngưỡng để tạo ảnh nhị phân. |

## Phase 2 — Lọc tuyến tính
| Xong | ID | Công việc |
|---|---|---|
| [ ] | 2.1 | Áp dụng lọc trung bình. |
| [ ] | 2.2 | Áp dụng lọc Gaussian. |
| [ ] | 2.3 | Áp dụng lọc làm sắc nét (kernel Laplacian). |

## Phase 3 — Phát hiện cạnh cơ bản
| Xong | ID | Công việc |
|---|---|---|
| [ ] | 3.1 | Áp dụng toán tử Sobel. |
| [ ] | 3.2 | Áp dụng toán tử Prewitt. |
| [ ] | 3.3 | Tự thiết kế kernel và so sánh hiệu ứng. |
| [ ] | 3.4 | So sánh kết quả của tất cả các loại lọc đã dùng. |

## Phase 4 — Lọc phi tuyến tính
| Xong | ID | Công việc |
|---|---|---|
| [ ] | 4.1 | Áp dụng lọc trung vị (nhiễu muối tiêu). |
| [ ] | 4.2 | Áp dụng lọc bilateral. |

## Phase 5 — Thuật toán Canny
| Xong | ID | Công việc |
|---|---|---|
| [ ] | 5.1 | Cài đặt Canny bằng OpenCV (`cv2.Canny`). |
| [ ] | 5.2 | Cài đặt Canny bằng Scikit-image (`skimage.feature.canny`). |
| [ ] | 5.3 | Thay đổi sigma, ngưỡng thấp/cao và quan sát kết quả. |
| [ ] | 5.4 | Áp dụng Canny cho ảnh nhiễu / độ tương phản thấp / nhiều chi tiết. |
| [ ] | 5.5 | Kết hợp Canny với phân đoạn hoặc nhận dạng hình dạng. |

## Yêu cầu bài tập

1. Toán tử điểm ảnh
   * Thay đổi độ sáng và độ tương phản của ảnh.
   * Tạo phiên bản âm bản của ảnh.
   * Áp dụng cắt ngưỡng để tạo ảnh nhị phân.
2. Lọc tuyến tính
   * Áp dụng lọc trung bình và lọc Gaussian để làm mờ ảnh.
   * Áp dụng kernel làm sắc nét để tăng cường cạnh.
3. Phát hiện cạnh cơ bản
   * Dùng toán tử Sobel và Prewitt để phát hiện cạnh.
   * Tự thiết kế kernel cho một hiệu ứng khác.
   * So sánh kết quả của các loại lọc khác nhau.
4. Lọc phi tuyến tính
   * Áp dụng lọc trung vị và lọc bilateral.
5. Thuật toán Canny
   * Cài đặt Canny bằng cả OpenCV và Scikit-image.
   * Thử các giá trị tham số khác nhau và so sánh kết quả.
   * Áp dụng Canny cho các loại ảnh khác nhau.

## Quy tắc Notebook

- Giữ toàn bộ code trong file `Lap2.ipynb`.
- Thêm giải thích ngắn cho mỗi công việc.
- Dùng code đơn giản, nhãn kết quả (output) rõ ràng.
- Thay `__/__` bằng ngày hoàn thành.
