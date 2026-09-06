/**
 * presets.js - Built-in Sample Image Generators, Kernel Matrix Presets & Code Snippets
 * Computer Vision Lab 2 Static Visualizer
 */

export const KERNEL_PRESETS = {
  '3x3': {
    'identity': {
      name: 'Identity (Nguyên bản)',
      matrix: [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
      ],
      divisor: 1,
      bias: 0,
      description: 'Giữ nguyên ảnh gốc không thay đổi.'
    },
    'box_blur': {
      name: 'Mean / Box Blur (Lọc trung bình)',
      matrix: [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
      ],
      divisor: 9,
      bias: 0,
      description: 'Làm mờ tuyến tính bằng cách lấy trung bình cộng 9 điểm lân cận.'
    },
    'gaussian_blur': {
      name: 'Gaussian Blur 3x3 (Lọc Gaussian)',
      matrix: [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
      ],
      divisor: 16,
      bias: 0,
      description: 'Làm mờ Gauss xấp xỉ phân phối chuẩn 2 chiều.'
    },
    'sharpen_4': {
      name: 'Sharpen (Làm sắc nét 4-hướng)',
      matrix: [
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
      ],
      divisor: 1,
      bias: 0,
      description: 'Tăng cường tương phản cạnh bằng Laplacian 4 láng giềng.'
    },
    'sharpen_8': {
      name: 'Sharpen (Làm sắc nét 8-hướng)',
      matrix: [
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
      ],
      divisor: 1,
      bias: 0,
      description: 'Tăng cường chi tiết cạnh mạnh mẽ với 8 láng giềng.'
    },
    'laplacian': {
      name: 'Laplacian Edge (Đạo hàm bậc 2)',
      matrix: [
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
      ],
      divisor: 1,
      bias: 128,
      description: 'Toán tử vi phân bậc hai phát hiện biên đẳng hướng.'
    },
    'sobel_x': {
      name: 'Sobel X (Biên dọc dI/dx)',
      matrix: [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
      ],
      divisor: 1,
      bias: 128,
      description: 'Đạo hàm bậc nhất theo trục hoành (cạnh đứng).'
    },
    'sobel_y': {
      name: 'Sobel Y (Biên ngang dI/dy)',
      matrix: [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
      ],
      divisor: 1,
      bias: 128,
      description: 'Đạo hàm bậc nhất theo trục tung (cạnh ngang).'
    },
    'prewitt_x': {
      name: 'Prewitt X (Biên dọc)',
      matrix: [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
      ],
      divisor: 1,
      bias: 128,
      description: 'Toán tử Prewitt tìm cạnh đứng với trọng số đều.'
    },
    'prewitt_y': {
      name: 'Prewitt Y (Biên ngang)',
      matrix: [
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
      ],
      divisor: 1,
      bias: 128,
      description: 'Toán tử Prewitt tìm cạnh ngang với trọng số đều.'
    },
    'emboss': {
      name: 'Emboss (Chạm nổi 3D)',
      matrix: [
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2]
      ],
      divisor: 1,
      bias: 128,
      description: 'Tạo hiệu ứng nổi khối nổi 3D theo hướng đường chéo.'
    },
    'ridge': {
      name: 'Ridge Detection (Phát hiện đường gờ)',
      matrix: [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
      ],
      divisor: 1,
      bias: 0,
      description: 'Nổi bật các chi tiết mảnh và điểm đặc trưng.'
    }
  },
  '5x5': {
    'gaussian_5x5': {
      name: 'Gaussian Blur 5x5',
      matrix: [
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
      ],
      divisor: 256,
      bias: 0,
      description: 'Làm mờ Gauss ma trận 5x5 chất lượng cao.'
    },
    'box_blur_5x5': {
      name: 'Box Blur 5x5',
      matrix: [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
      ],
      divisor: 25,
      bias: 0,
      description: 'Làm mờ trung bình trên vùng lân cận 25 pixel.'
    },
    'unsharp_mask_5x5': {
      name: 'Unsharp Mask 5x5',
      matrix: [
        [ 1,  4,    6,  4,  1],
        [ 4, 16,   24, 16,  4],
        [ 6, 24, -476, 24,  6],
        [ 4, 16,   24, 16,  4],
        [ 1,  4,    6,  4,  1]
      ],
      divisor: -256,
      bias: 0,
      description: 'Bộ lọc Unsharp Masking 5x5 tăng độ nét tinh tế.'
    }
  }
};

/**
 * Procedural Test Images Generator to guarantee zero broken images offline
 */
export function generateSampleImage(type = 'geometric', width = 512, height = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (type === 'geometric') {
    // High-contrast geometric shapes with clean edges
    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, '#1a202c');
    grad.addColorStop(1, '#2d3748');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // Large Circle
    ctx.beginPath();
    ctx.arc(width * 0.35, height * 0.4, width * 0.22, 0, Math.PI * 2);
    ctx.fillStyle = '#edf2f7';
    ctx.fill();
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#3182ce';
    ctx.stroke();

    // Rotated Rectangle
    ctx.save();
    ctx.translate(width * 0.65, height * 0.35);
    ctx.rotate((30 * Math.PI) / 180);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(-70, -70, 140, 140);
    ctx.lineWidth = 5;
    ctx.strokeStyle = '#e53e3e';
    ctx.strokeRect(-70, -70, 140, 140);
    ctx.restore();

    // Triangle
    ctx.beginPath();
    ctx.moveTo(width * 0.2, height * 0.85);
    ctx.lineTo(width * 0.5, height * 0.6);
    ctx.lineTo(width * 0.8, height * 0.85);
    ctx.closePath();
    ctx.fillStyle = '#cbd5e0';
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#38a169';
    ctx.stroke();

    // Slanted & Curved fine lines for edge testing
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      ctx.moveTo(30 + i * 20, height - 30);
      ctx.lineTo(130 + i * 20, height - 130);
      ctx.stroke();
    }
  } else if (type === 'portrait') {
    // Stylized Cameraman / Portrait test pattern
    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, '#718096');
    grad.addColorStop(1, '#2d3748');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // Ground & Building skyline
    ctx.fillStyle = '#1a202c';
    ctx.fillRect(0, height * 0.65, width, height * 0.35);
    ctx.fillRect(width * 0.7, height * 0.3, width * 0.25, height * 0.35);
    ctx.fillRect(width * 0.55, height * 0.45, width * 0.12, height * 0.2);

    // Silhouette Figure with coat & tripod
    ctx.fillStyle = '#0d1117';
    // Head & Coat
    ctx.beginPath();
    ctx.arc(width * 0.35, height * 0.35, 30, 0, Math.PI * 2);
    ctx.fill();
    // Coat
    ctx.beginPath();
    ctx.moveTo(width * 0.3, height * 0.39);
    ctx.lineTo(width * 0.4, height * 0.39);
    ctx.lineTo(width * 0.48, height * 0.75);
    ctx.lineTo(width * 0.22, height * 0.75);
    ctx.closePath();
    ctx.fill();

    // Tripod & Camera
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#0d1117';
    ctx.beginPath();
    ctx.moveTo(width * 0.45, height * 0.42);
    ctx.lineTo(width * 0.35, height * 0.85);
    ctx.moveTo(width * 0.45, height * 0.42);
    ctx.lineTo(width * 0.45, height * 0.85);
    ctx.moveTo(width * 0.45, height * 0.42);
    ctx.lineTo(width * 0.55, height * 0.85);
    ctx.stroke();

    // Camera box
    ctx.fillRect(width * 0.42, height * 0.38, 30, 20);
    ctx.fillRect(width * 0.48, height * 0.41, 15, 12);
  } else if (type === 'low_contrast') {
    // Low-contrast subtle textured scene
    ctx.fillStyle = '#7a8288';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = '#8a9299';
    ctx.beginPath();
    ctx.arc(width * 0.5, height * 0.5, width * 0.3, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#6e767c';
    ctx.fillRect(width * 0.2, height * 0.3, width * 0.6, height * 0.1);
    ctx.fillRect(width * 0.3, height * 0.6, width * 0.4, height * 0.15);

    // Subtle texture noise
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const n = (Math.random() - 0.5) * 20;
      data[i] = Math.min(255, Math.max(0, data[i] + n));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + n));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + n));
    }
    ctx.putImageData(imgData, 0, 0);
  } else if (type === 'noisy') {
    // Base image + Heavy Salt & Pepper noise
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = '#2b6cb0';
    ctx.fillRect(50, 50, width - 100, height - 100);

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, width * 0.25, 0, Math.PI * 2);
    ctx.fill();

    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    const noiseRatio = 0.08;
    for (let i = 0; i < data.length; i += 4) {
      const r = Math.random();
      if (r < noiseRatio / 2) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
      } else if (r < noiseRatio) {
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  return canvas.toDataURL('image/png');
}

/**
 * Theory Explanations & Python Code Snippets corresponding to Lab 2 Requirements
 */
export const THEORY_AND_CODE = {
  phase1: {
    title: 'Phase 1 — Toán tử điểm ảnh (Point Operations)',
    math: `1. Độ sáng & Tương phản: g(x,y) = α · f(x,y) + β (α: hệ số tương phản, β: độ sáng)
2. Âm bản (Negative): s = (L - 1) - r = 255 - r
3. Cắt ngưỡng nhị phân (Thresholding): g(x,y) = 255 nếu f(x,y) ≥ T, ngược lại g(x,y) = 0`,
    python: `# Phase 1: Toán tử điểm ảnh (OpenCV / NumPy)
import cv2
import numpy as np

# Đọc ảnh xám
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 1.1 Điều chỉnh độ sáng (+beta) và 1.2 Độ tương phản (*alpha)
alpha = 1.5  # Contrast (1.0 - 3.0)
beta = 30    # Brightness (-100 to 100)
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 1.3 Ảnh âm bản (Negative)
negative = 255 - img

# 1.4 Cắt ngưỡng nhị phân (Binary Threshold)
thresh_value = 127
_, binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)`
  },

  phase2: {
    title: 'Phase 2 — Lọc tuyến tính (Linear Filtering)',
    math: `1. Phép tích chập không gian: g(x,y) = ∑∑ w(s,t) · f(x+s, y+t)
2. Lọc trung bình (Box Blur): K = (1/k²) [ma trận 1 toàn bộ]
3. Lọc Gaussian 2D: G(x,y) = (1 / (2πσ²)) · exp(-(x²+y²)/(2σ²))
4. Laplacian Sharpen: g(x,y) = f(x,y) - c · ∇²f(x,y)`,
    python: `# Phase 2: Lọc tuyến tính (OpenCV)
import cv2
import numpy as np

img = cv2.imread('input.jpg', cv2.IMREAD_COLOR)

# 2.1 Lọc trung bình (Box / Mean filter)
mean_blur = cv2.blur(img, (5, 5))

# 2.2 Lọc Gaussian
gaussian_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)

# 2.3 Làm sắc nét với kernel Laplacian
laplacian_kernel = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(img, -1, laplacian_kernel)`
  },

  phase3: {
    title: 'Phase 3 — Phát hiện cạnh cơ bản (Basic Edge Detection)',
    math: `1. Toán tử Sobel: Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
2. Toán tử Prewitt: Gx = [[-1,0,1],[-1,0,1],[-1,0,1]], Gy = [[-1,-1,-1],[0,0,0],[1,1,1]]
3. Độ lớn Gradient: M(x,y) = √(Gx² + Gy²)
4. Hướng Gradient: θ(x,y) = atan2(Gy, Gx)`,
    python: `# Phase 3: Phát hiện cạnh cơ bản (OpenCV)
import cv2
import numpy as np

gray = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 3.1 Toán tử Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

# 3.2 Toán tử Prewitt
kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
prewitt_mag = np.uint8(np.clip(np.sqrt(prewittx**2 + prewitty**2), 0, 255))

# 3.3 Custom Kernel tự thiết kế (ví dụ Emboss 3x3)
custom_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
custom_result = cv2.filter2D(gray, -1, custom_kernel, delta=128)`
  },

  phase4: {
    title: 'Phase 4 — Lọc phi tuyến tính (Non-linear Filtering)',
    math: `1. Lọc trung vị (Median Filter): g(x,y) = median { f(x+s, y+t) | (s,t) ∈ Sxy }
   - Hiệu quả vượt trội khử nhiễu Muối tiêu (Salt & Pepper) mà bảo toàn biên ảnh.
2. Lọc Bilateral (Bảo toàn biên): BF[I]p = (1/Wp) ∑ Gσs(||p-q||) · Gσr(|Ip - Iq|) · Iq
   - Kết hợp khoảng cách không gian (Spatial) và độ chênh lệch màu sắc (Range/Color).`,
    python: `# Phase 4: Lọc phi tuyến tính (OpenCV)
import cv2

img = cv2.imread('input.jpg', cv2.IMREAD_COLOR)

# 4.1 Lọc trung vị (Median Filter - Khử nhiễu muối tiêu)
median_filtered = cv2.medianBlur(img, ksize=5)

# 4.2 Lọc Bilateral (Làm mịn giữ nét cạnh)
# d: Đường kính láng giềng, sigmaColor: Không gian màu, sigmaSpace: Không gian tọa độ
bilateral_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)`
  },

  phase5: {
    title: 'Phase 5 — Thuật toán Canny Edge Detection',
    math: `Quy trình 5 bước kinh điển của Canny:
1. Làm mịn Gaussian giảm nhiễu: I_σ = I * G_σ
2. Tính Gradient cường độ & hướng: Gx, Gy (Sobel) => M = √(Gx² + Gy²), θ = atan2(Gy, Gx)
3. Non-Maximum Suppression (NMS): Làm mỏng biên thành 1-pixel theo 4 hướng (0°, 45°, 90°, 135°)
4. Phân ngưỡng kép (Double Threshold): Phân loại Strong Edges (M ≥ Thigh) và Weak Edges (Tlow ≤ M < Thigh)
5. Hysteresis Edge Tracking: Giữ pixel yếu nếu liên thông với ít nhất 1 pixel mạnh 8-láng giềng.`,
    python: `# Phase 5: Canny Edge Detection (OpenCV & Scikit-image)
import cv2
from skimage import feature
from skimage.color import rgb2gray

# 5.1 Cài đặt Canny bằng OpenCV
img_gray = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
low_threshold = 50
high_threshold = 150
canny_cv = cv2.Canny(img_gray, low_threshold, high_threshold, apertureSize=3, L2gradient=True)

# 5.2 Cài đặt Canny bằng Scikit-image (5.2 & 5.3)
img_float = rgb2gray(cv2.imread('input.jpg'))
canny_skimage = feature.canny(img_float, sigma=1.5, low_threshold=0.1, high_threshold=0.3)`
  },

  comparison: {
    title: 'Tổng hợp so sánh các phương pháp lọc & phát hiện cạnh',
    math: `So sánh đặc tính:
- Mean / Gaussian Blur: Giảm nhiễu Gaussia / làm mờ, nhưng làm mờ cạnh.
- Median Filter: Khử nhiễu muối tiêu cực tốt, bảo toàn biên sắc nét.
- Bilateral Filter: Làm mịn vùng phẳng nhưng giữ nguyên cạnh tương phản cao.
- Sobel / Prewitt: Đạo hàm bậc 1 phát hiện hướng dốc cường độ, biên còn dày và nhạy với nhiễu.
- Laplacian: Đạo hàm bậc 2 tìm điểm zero-crossing, nhạy nhiễu mạnh.
- Canny: Tối ưu theo 3 tiêu chuẩn Canny (tỷ lệ phát hiện cao, định vị chính xác, đáp ứng đơn).`,
    python: `# So sánh đa bộ lọc trong Python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

box = cv2.blur(gray, (5, 5))
gauss = cv2.GaussianBlur(gray, (5, 5), 1.5)
median = cv2.medianBlur(gray, 5)
bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
sobel = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
canny = cv2.Canny(gray, 50, 150)`
  }
};
