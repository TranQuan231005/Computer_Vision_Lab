# Canny Edge Detection — Tu cai dat tu dau

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
└── output/                    # Thư mục chứa ảnh kết quả xuất ra
```

## Trang thai

- [x] Buoc 0: Doc anh, chuyen xam
- [x] Buoc 1: Gaussian Blur
- [x] Buoc 2: Tinh gradient (Sobel)
- [x] Buoc 3: Non-Maximum Suppression
- [x] Buoc 4: Hysteresis Thresholding
- [x] So sanh voi `cv2.Canny()`
- [ ] dùng Vectorization với NumPy, Numba, hoặc Cython
- [ ] Thu nghiem tren nhieu anh/nguong khac nhau

## Cach chay

1. Cai thu vien can thiet:
   ```
   pip install opencv-python numpy matplotlib
   ```
2. Bo anh dau vao vao thu muc `input/` (mac dinh notebook doc `input/tên-ảnh.jpg` — doi ten file hoac sua bien `INPUT_PATH` trong notebook neu dung ten khac).
3. Mo va chay `notebooks/canny_edge_detection.ipynb` tu tren xuong duoi.
4. Ket qua (hinh so sanh 6 buoc) se tu dong luu vao `output/canny_steps.png`.

## Ghi chu

- Nguong hysteresis hien tai: `low_ratio=0.05`, `high_ratio=0.15` (tinh theo `magnitude.max()` sau NMS) — co the chinh de thay doi do nhay bat canh.
- `cv2.Canny(blurred, 50, 150)` dung de doi chieu ket qua tu cai dat voi ham chuan cua OpenCV.
