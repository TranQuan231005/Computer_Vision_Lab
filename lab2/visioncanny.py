import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ==============================================================================
# 1. CAC PHUONG PHAP THUC THI CANNY (LOOP vs NUMPY VECTORIZED vs NUMBA JIT)
# ==============================================================================

# --- A. Non-Maximum Suppression (NMS) ---

def nms_loop(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """NMS su dung 2 vong lap for truyen thong."""
    H, W = magnitude.shape
    result = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                neighbor1, neighbor2 = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= a < 67.5:
                neighbor1, neighbor2 = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif 67.5 <= a < 112.5:
                neighbor1, neighbor2 = magnitude[i+1, j], magnitude[i-1, j]
            else:  # 112.5 <= a < 157.5
                neighbor1, neighbor2 = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                result[i, j] = magnitude[i, j]

    return result

def nms_vectorized(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """NMS su dung NumPy Array Slicing & Boolean Masking (Vectorization)."""
    H, W = magnitude.shape
    result = np.zeros((H, W), dtype=np.float32)

    # Cat cac khung nhin tam (1:-1, 1:-1) va 8 lang gieng
    mag_c = magnitude[1:-1, 1:-1]
    ang_c = angle[1:-1, 1:-1]

    # Khoi tao ma tran chua gia tri 2 lang gieng
    q = np.zeros_like(mag_c)
    r = np.zeros_like(mag_c)

    # Huong 0 do (Ngang: E-W)
    mask_0 = ((0 <= ang_c) & (ang_c < 22.5)) | ((157.5 <= ang_c) & (ang_c <= 180))
    q[mask_0] = magnitude[1:-1, 2:][mask_0]
    r[mask_0] = magnitude[1:-1, :-2][mask_0]

    # Huong 45 do (Cheo phu: NE-SW)
    mask_45 = (22.5 <= ang_c) & (ang_c < 67.5)
    q[mask_45] = magnitude[2:, :-2][mask_45]
    r[mask_45] = magnitude[:-2, 2:][mask_45]

    # Huong 90 do (Doc: N-S)
    mask_90 = (67.5 <= ang_c) & (ang_c < 112.5)
    q[mask_90] = magnitude[2:, 1:-1][mask_90]
    r[mask_90] = magnitude[:-2, 1:-1][mask_90]

    # Huong 135 do (Cheo chinh: NW-SE)
    mask_135 = (112.5 <= ang_c) & (ang_c < 157.5)
    q[mask_135] = magnitude[:-2, :-2][mask_135]
    r[mask_135] = magnitude[2:, 2:][mask_135]

    # Giu lai cac diem cuc dai cuc bo
    keep = (mag_c >= q) & (mag_c >= r)
    result[1:-1, 1:-1][keep] = mag_c[keep]

    return result

if HAS_NUMBA:
    @jit(nopython=True, fastmath=True)
    def nms_numba(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """NMS bien dich JIT voi Numba."""
        H, W = magnitude.shape
        result = np.zeros((H, W), dtype=np.float32)

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                a = angle[i, j]
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    n1 = magnitude[i, j+1]
                    n2 = magnitude[i, j-1]
                elif 22.5 <= a < 67.5:
                    n1 = magnitude[i+1, j-1]
                    n2 = magnitude[i-1, j+1]
                elif 67.5 <= a < 112.5:
                    n1 = magnitude[i+1, j]
                    n2 = magnitude[i-1, j]
                else:
                    n1 = magnitude[i-1, j-1]
                    n2 = magnitude[i+1, j+1]

                if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                    result[i, j] = magnitude[i, j]

        return result
else:
    nms_numba = nms_vectorized


# --- B. Hysteresis Thresholding & Edge Tracking ---

def hysteresis_loop(img: np.ndarray, low_ratio=0.05, high_ratio=0.15) -> np.ndarray:
    """Hysteresis su dung 2 vong lap for de duyet 8-connectivity."""
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    H, W = img.shape
    result = np.zeros((H, W), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img >= low_threshold) & (img < high_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if result[i, j] == weak:
                if strong in (
                    result[i-1, j-1], result[i-1, j], result[i-1, j+1],
                    result[i, j-1],                   result[i, j+1],
                    result[i+1, j-1], result[i+1, j], result[i+1, j+1]
                ):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    return result

def hysteresis_vectorized(img: np.ndarray, low_ratio=0.05, high_ratio=0.15) -> np.ndarray:
    """
    Hysteresis su dung Vectorization & Scipy/OpenCV Dilation:
    Lan truyen cac canh manh (strong edges) vao tap hop cac canh yeu (weak edges) qua Connected Components.
    """
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    strong_edges = img >= high_threshold
    weak_edges = (img >= low_threshold) & (img < high_threshold)

    # Ma tran cau truc 8 lang gieng
    struct_8 = np.ones((3, 3), dtype=bool)

    # Lan truyen iterative dilation tren weak edges
    connected = strong_edges.copy()
    while True:
        dilated = binary_dilation(connected, structure=struct_8)
        new_connected = dilated & weak_edges
        if not np.any(new_connected & (~connected)):
            break
        connected |= new_connected

    result = np.zeros(img.shape, dtype=np.uint8)
    result[connected] = 255
    return result

if HAS_NUMBA:
    @jit(nopython=True, fastmath=True)
    def hysteresis_numba(img: np.ndarray, low_ratio=0.05, high_ratio=0.15) -> np.ndarray:
        """Hysteresis bien dich JIT voi Numba + Queue/Stack Edge Tracking."""
        high_threshold = img.max() * high_ratio
        low_threshold = high_threshold * low_ratio

        H, W = img.shape
        result = np.zeros((H, W), dtype=np.uint8)
        
        # Queue de DFS/BFS noi bien
        stack_r = np.zeros(H * W, dtype=np.int32)
        stack_c = np.zeros(H * W, dtype=np.int32)
        top = 0

        for r in range(H):
            for c in range(W):
                val = img[r, c]
                if val >= high_threshold:
                    result[r, c] = 255
                    stack_r[top] = r
                    stack_c[top] = c
                    top += 1
                elif val >= low_threshold:
                    result[r, c] = 75

        # Lan truyen tu cac pixel strong edge
        while top > 0:
            top -= 1
            cr = stack_r[top]
            cc = stack_c[top]

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if result[nr, nc] == 75:
                            result[nr, nc] = 255
                            stack_r[top] = nr
                            stack_c[top] = nc
                            top += 1

        # Xoa cac weak edge khong duoc noi
        for r in range(H):
            for c in range(W):
                if result[r, c] == 75:
                    result[r, c] = 0

        return result
else:
    hysteresis_numba = hysteresis_vectorized


# ==============================================================================
# 2. HAM PIPELINE CANNY HOAN CHINH
# ==============================================================================

def canny_custom(gray_image: np.ndarray, 
                 sigma: float = 1.4, 
                 ksize: int = 5, 
                 low_ratio: float = 0.05, 
                 high_ratio: float = 0.15,
                 method: str = 'vectorized'):
    """
    Pipeline thuat toan Canny tuy chon cach thuc thi:
    method: 'loop', 'vectorized', hoac 'numba'
    """
    # 1. Gaussian Blur
    blurred = cv2.GaussianBlur(gray_image, (ksize, ksize), sigmaX=sigma)

    # 2. Sobel Gradient
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    angle[angle < 0] += 180

    # 3. NMS
    if method == 'numba' and HAS_NUMBA:
        nms = nms_numba(magnitude.astype(np.float32), angle.astype(np.float32))
    elif method == 'vectorized':
        nms = nms_vectorized(magnitude, angle)
    else:
        nms = nms_loop(magnitude, angle)

    # 4. Hysteresis Thresholding
    if method == 'numba' and HAS_NUMBA:
        edges = hysteresis_numba(nms, low_ratio=low_ratio, high_ratio=high_ratio)
    elif method == 'vectorized':
        edges = hysteresis_vectorized(nms, low_ratio=low_ratio, high_ratio=high_ratio)
    else:
        edges = hysteresis_loop(nms, low_ratio=low_ratio, high_ratio=high_ratio)

    return {
        'blurred': blurred,
        'sobel_x': sobel_x,
        'sobel_y': sobel_y,
        'magnitude': magnitude,
        'angle': angle,
        'nms': nms,
        'edges': edges
    }


# ==============================================================================
# 3. BENCHMARK PERFORMANCE (LOOP vs VECTORIZED vs NUMBA)
# ==============================================================================

def run_performance_benchmark(gray_img: np.ndarray, num_runs: int = 5):
    print("=" * 65)
    print("BENCHMARK HIEU NANG CANNY (Kich thuoc anh: {}x{})".format(gray_img.shape[1], gray_img.shape[0]))
    print("=" * 65)

    # Blur + Gradient tinh truoc
    blurred = cv2.GaussianBlur(gray_img, (5, 5), sigmaX=1.4)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2).astype(np.float32)
    ang = (np.arctan2(sy, sx) * 180 / np.pi)
    ang[ang < 0] += 180
    ang = ang.astype(np.float32)

    # Warmup Numba JIT
    if HAS_NUMBA:
        _ = nms_numba(mag, ang)
        _ = hysteresis_numba(mag, 0.05, 0.15)

    methods = [('Loop (Co ban)', nms_loop, hysteresis_loop),
               ('Vectorized (NumPy/SciPy)', nms_vectorized, hysteresis_vectorized)]
    if HAS_NUMBA:
        methods.append(('Numba JIT (Tang toc)', nms_numba, hysteresis_numba))

    for name, nms_fn, hys_fn in methods:
        t_nms, t_hys = 0.0, 0.0
        for _ in range(num_runs):
            t0 = time.perf_counter()
            nms_out = nms_fn(mag, ang)
            t1 = time.perf_counter()
            _ = hys_fn(nms_out, 0.05, 0.15)
            t2 = time.perf_counter()
            t_nms += (t1 - t0)
            t_hys += (t2 - t1)

        avg_nms = (t_nms / num_runs) * 1000
        avg_hys = (t_hys / num_runs) * 1000
        total = avg_nms + avg_hys
        print(f"[{name:26s}] NMS: {avg_nms:6.2f} ms | Hysteresis: {avg_hys:6.2f} ms | Tong: {total:6.2f} ms")

    # OpenCV Canny chuan
    t_cv = 0.0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = cv2.Canny(blurred, 50, 150)
        t1 = time.perf_counter()
        t_cv += (t1 - t0)
    print(f"[{'OpenCV cv2.Canny()':26s}] Tong: {(t_cv / num_runs)*1000:6.2f} ms (Toi uu C++)")
    print("=" * 65 + "\n")


# ==============================================================================
# 4. THU NGHIEM TREN NHIEU LOAI ANH & BO NGUONG KHAC NHAU
# ==============================================================================

def create_synthetic_test_images():
    """Tao 4 loai anh test dai dien cho cac tinh huong thuc te."""
    images = {}
    
    # 1. Anh Geometric Shapes & Lines (Tuong phan ro net)
    img_shapes = np.full((300, 300), 230, dtype=np.uint8)
    cv2.circle(img_shapes, (150, 150), 70, (40,), -1)
    cv2.rectangle(img_shapes, (40, 40), (100, 100), (90,), -1)
    cv2.line(img_shapes, (30, 260), (270, 260), (20,), 4)
    images['Shapes & Lines (Chuan)'] = img_shapes

    # 2. Anh Low Contrast (Do tuong phan thap)
    img_low_contrast = (img_shapes.astype(np.float32) * 0.25 + 100).clip(0, 255).astype(np.uint8)
    images['Low Contrast (Tuong phan thap)'] = img_low_contrast

    # 3. Anh Noisy (Bi nhieu muoi tieu & nhieu Gauss)
    img_noisy = img_shapes.copy().astype(np.float32)
    noise_gauss = np.random.normal(0, 25, img_noisy.shape)
    img_noisy = np.clip(img_noisy + noise_gauss, 0, 255).astype(np.uint8)
    # Them salt-and-pepper
    num_sp = int(0.02 * img_noisy.size)
    coords_s = [np.random.randint(0, i - 1, num_sp) for i in img_noisy.shape]
    coords_p = [np.random.randint(0, i - 1, num_sp) for i in img_noisy.shape]
    img_noisy[tuple(coords_s)] = 255
    img_noisy[tuple(coords_p)] = 0
    images['Noisy Image (Nhieu Gauss & Muoi tieu)'] = img_noisy

    # 4. Anh Complex / Gradient (Ket cau chuyen dong muot)
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + yy**2)
    img_texture = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
    images['Complex Texture (Song van giao thoa)'] = img_texture

    return images

def test_multiple_images_and_thresholds(output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    test_imgs = create_synthetic_test_images()

    print(">>> Dang chay thu nghiem tren nhieu anh va bo nguong khac nhau...")

    # --- Test 1: So sanh Canny tren cac loai anh khac nhau ---
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    fig.suptitle('THU NGHIEM CANNY TREN CAC LOAI ANH THUC TE', fontsize=18, fontweight='bold')

    for idx, (title, img) in enumerate(test_imgs.items()):
        # Vectorized Canny (Default: sigma=1.4, low=0.05, high=0.15)
        res_default = canny_custom(img, sigma=1.4, low_ratio=0.05, high_ratio=0.15, method='vectorized')
        
        # Canny voi Blur manh hon (sigma=2.5)
        res_smooth = canny_custom(img, sigma=2.5, ksize=7, low_ratio=0.05, high_ratio=0.15, method='vectorized')
        
        # OpenCV Canny
        cv_canny = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 1.4), 50, 150)

        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f"Goc: {title}", fontsize=11)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(res_default['edges'], cmap='gray')
        axes[idx, 1].set_title("Canny Tu cai dat (sigma=1.4)", fontsize=11)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(res_smooth['edges'], cmap='gray')
        axes[idx, 2].set_title("Canny Loc nhieu (sigma=2.5)", fontsize=11)
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(cv_canny, cmap='gray')
        axes[idx, 3].set_title("OpenCV cv2.Canny()", fontsize=11)
        axes[idx, 3].axis('off')

    plt.tight_layout()
    save_path_types = os.path.join(output_dir, 'canny_image_types_test.png')
    plt.savefig(save_path_types, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Da luu ket qua thu nghiem loai anh tai: {save_path_types}")

    # --- Test 2: Khao sat anh huong cua bo nguong Hysteresis (Threshold Pairs) ---
    sample_img = test_imgs['Shapes & Lines (Chuan)']
    threshold_pairs = [
        (0.02, 0.08, "Nguong Rat Thap (Bat ca bien mo/nhieu)"),
        (0.05, 0.15, "Nguong Tieu Chuan (Can bang)"),
        (0.10, 0.25, "Nguong Cao (Chi giu bien manh)"),
        (0.18, 0.40, "Nguong Rat Cao (Mat nhieu chi tiet)")
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle('KHAO SAT ANH HUONG CUA BO NGUONG HYSTERESIS (Low Ratio / High Ratio)', fontsize=15, fontweight='bold')

    axes[0].imshow(sample_img, cmap='gray')
    axes[0].set_title('Anh Dau Vao', fontsize=12)
    axes[0].axis('off')

    for idx, (low_r, high_r, desc) in enumerate(threshold_pairs):
        res = canny_custom(sample_img, sigma=1.4, low_ratio=low_r, high_ratio=high_r, method='vectorized')
        axes[idx + 1].imshow(res['edges'], cmap='gray')
        axes[idx + 1].set_title(f"Low={low_r}, High={high_r}\n{desc}", fontsize=10)
        axes[idx + 1].axis('off')

    plt.tight_layout()
    save_path_thresh = os.path.join(output_dir, 'canny_threshold_variation_test.png')
    plt.savefig(save_path_thresh, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Da luu ket qua thu nghiem bo nguong tai: {save_path_thresh}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # 1. Chay test anh goc
    base_img = None
    for candidate in ['circle.jpg', os.path.join('..', 'circle.jpg'), os.path.join('..', 'Lap1', 'circle.jpg')]:
        if os.path.exists(candidate):
            base_img = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            break
            
    if base_img is None:
        base_img = create_synthetic_test_images()['Shapes & Lines (Chuan)']

    # 2. Benchmark Hieu nang
    run_performance_benchmark(base_img, num_runs=5)

    # 3. Thu nghiem tren nhieu anh & nguong
    test_multiple_images_and_thresholds(output_dir='output')

    # 4. Chay pipeline chi tiet 6 buoc cho anh circle
    print(">>> Chay pipeline chi tiet 6 buoc va luu anh output...")
    res = canny_custom(base_img, sigma=1.4, low_ratio=0.05, high_ratio=0.15, method='vectorized')
    cv_canny = cv2.Canny(res['blurred'], 50, 150)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(base_img, cmap='gray'); axes[0, 0].set_title('0. Anh xam goc'); axes[0, 0].axis('off')
    axes[0, 1].imshow(res['blurred'], cmap='gray'); axes[0, 1].set_title('1. Gaussian Blur (sigma=1.4)'); axes[0, 1].axis('off')
    axes[0, 2].imshow(np.uint8(255 * res['magnitude'] / np.max(res['magnitude'])), cmap='gray'); axes[0, 2].set_title('2. Sobel Magnitude'); axes[0, 2].axis('off')
    axes[1, 0].imshow(np.uint8(255 * res['nms'] / (np.max(res['nms']) + 1e-8)), cmap='gray'); axes[1, 0].set_title('3. Non-Max Suppression (Vectorized)'); axes[1, 0].axis('off')
    axes[1, 1].imshow(res['edges'], cmap='gray'); axes[1, 1].set_title('4. Hysteresis (Tu cai dat)'); axes[1, 1].axis('off')
    axes[1, 2].imshow(cv_canny, cmap='gray'); axes[1, 2].set_title('So sanh: cv2.Canny()'); axes[1, 2].axis('off')

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/output_canny_vectorized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(">>> Hoan thanh toan bo thu nghiem va benchmark thanh cong!")