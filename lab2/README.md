# Canny Edge Detection — Tu cai dat tu dau

Tu cai dat 4 buoc cua thuat toan Canny (Gaussian Blur -> Sobel Gradient -> Non-Maximum Suppression -> Hysteresis Thresholding), so sanh voi ham `cv2.Canny()` co san.

## Cau truc thu muc

```
canny-edge-detection/
├── README.md
├── input/              # Anh dau vao (vd: circle.jpg)
├── output/             # Ket qua xuat ra (hinh so sanh 6 buoc)
└── notebooks/
    └── canny_edge_detection.ipynb   # Notebook chinh, chia theo tung buoc
```

## Trang thai

- [x] Buoc 0: Doc anh, chuyen xam
- [x] Buoc 1: Gaussian Blur
- [x] Buoc 2: Tinh gradient (Sobel)
- [x] Buoc 3: Non-Maximum Suppression
- [x] Buoc 4: Hysteresis Thresholding
- [x] So sanh voi `cv2.Canny()`
- [ ] Toi uu toc do (vong lap thuan Python con cham voi anh lon)
- [ ] Thu nghiem tren nhieu anh/nguong khac nhau

## Cach chay

1. Cai thu vien can thiet:
   ```
   pip install opencv-python numpy matplotlib
   ```
2. Bo anh dau vao vao thu muc `input/` (mac dinh notebook doc `input/circle.jpg` — doi ten file hoac sua bien `INPUT_PATH` trong notebook neu dung ten khac).
3. Mo va chay `notebooks/canny_edge_detection.ipynb` tu tren xuong duoi.
4. Ket qua (hinh so sanh 6 buoc) se tu dong luu vao `output/canny_steps.png`.

## Ghi chu

- Nguong hysteresis hien tai: `low_ratio=0.05`, `high_ratio=0.15` (tinh theo `magnitude.max()` sau NMS) — co the chinh de thay doi do nhay bat canh.
- `cv2.Canny(blurred, 50, 150)` dung de doi chieu ket qua tu cai dat voi ham chuan cua OpenCV.
