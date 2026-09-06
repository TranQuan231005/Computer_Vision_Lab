/**
 * filters.js - Pixel-level Image Processing Algorithms for Lab 2
 * Includes: Point Operations, 2D Convolution, Linear Filtering,
 * Edge Detection (Sobel/Prewitt), Non-linear Filtering (Median/Bilateral), Noise Generators.
 */

// Helper to create an empty ImageData with matching dimensions
export function createBlankImageData(width, height) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  return ctx.createImageData(width, height);
}

// Clone an ImageData object
export function cloneImageData(src) {
  const blank = createBlankImageData(src.width, src.height);
  blank.data.set(src.data);
  return blank;
}

/**
 * Phase 1: Point Operations
 * - Brightness: val + beta
 * - Contrast: alpha * val
 * - Negative: 255 - val
 * - Thresholding: val >= thresh ? 255 : 0
 */
export function applyPointOps(src, {
  brightness = 0,
  contrast = 1.0,
  negative = false,
  thresholdEnabled = false,
  thresholdValue = 128,
  invertBinary = false
}) {
  const dst = createBlankImageData(src.width, src.height);
  const sData = src.data;
  const dData = dst.data;
  const len = sData.length;

  for (let i = 0; i < len; i += 4) {
    let r = sData[i];
    let g = sData[i + 1];
    let b = sData[i + 2];
    const a = sData[i + 3];

    // Contrast & Brightness: g(x,y) = alpha * f(x,y) + beta
    r = contrast * r + brightness;
    g = contrast * g + brightness;
    b = contrast * b + brightness;

    // Clamp
    r = Math.min(255, Math.max(0, r));
    g = Math.min(255, Math.max(0, g));
    b = Math.min(255, Math.max(0, b));

    // Negative / Inversion
    if (negative) {
      r = 255 - r;
      g = 255 - g;
      b = 255 - b;
    }

    // Binary Thresholding
    if (thresholdEnabled) {
      // Standard grayscale luminance formula: 0.299R + 0.587G + 0.114B
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      let binVal = gray >= thresholdValue ? 255 : 0;
      if (invertBinary) binVal = 255 - binVal;
      r = g = b = binVal;
    }

    dData[i] = r;
    dData[i + 1] = g;
    dData[i + 2] = b;
    dData[i + 3] = a;
  }

  return dst;
}

/**
 * General 2D Spatial Convolution with boundary clamping
 */
export function convolve2D(src, kernel, divisor = 1, bias = 0, grayscaleOutput = false) {
  const width = src.width;
  const height = src.height;
  const sData = src.data;
  const dst = createBlankImageData(width, height);
  const dData = dst.data;

  const kRows = kernel.length;
  const kCols = kernel[0].length;
  const halfH = Math.floor(kRows / 2);
  const halfW = Math.floor(kCols / 2);

  if (divisor === 0) divisor = 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let rSum = 0;
      let gSum = 0;
      let bSum = 0;

      for (let ky = 0; ky < kRows; ky++) {
        const py = Math.min(height - 1, Math.max(0, y + ky - halfH));
        const rowOffset = py * width * 4;

        for (let kx = 0; kx < kCols; kx++) {
          const px = Math.min(width - 1, Math.max(0, x + kx - halfW));
          const idx = rowOffset + px * 4;
          const weight = kernel[ky][kx];

          rSum += sData[idx] * weight;
          gSum += sData[idx + 1] * weight;
          bSum += sData[idx + 2] * weight;
        }
      }

      let rOut = rSum / divisor + bias;
      let gOut = gSum / divisor + bias;
      let bOut = bSum / divisor + bias;

      rOut = Math.min(255, Math.max(0, rOut));
      gOut = Math.min(255, Math.max(0, gOut));
      bOut = Math.min(255, Math.max(0, bOut));

      const dIdx = (y * width + x) * 4;
      if (grayscaleOutput) {
        const gray = 0.299 * rOut + 0.587 * gOut + 0.114 * bOut;
        dData[dIdx] = gray;
        dData[dIdx + 1] = gray;
        dData[dIdx + 2] = gray;
      } else {
        dData[dIdx] = rOut;
        dData[dIdx + 1] = gOut;
        dData[dIdx + 2] = bOut;
      }
      dData[dIdx + 3] = sData[dIdx + 3];
    }
  }

  return dst;
}

/**
 * Phase 2: Box / Mean Blur
 */
export function applyBoxBlur(src, kernelSize = 5) {
  const k = Math.max(3, kernelSize % 2 === 0 ? kernelSize + 1 : kernelSize);
  const matrix = Array.from({ length: k }, () => Array(k).fill(1));
  const divisor = k * k;
  return convolve2D(src, matrix, divisor, 0);
}

/**
 * Generate 2D Gaussian Kernel
 */
export function generateGaussianKernel(size = 5, sigma = 1.4) {
  const k = Math.max(3, size % 2 === 0 ? size + 1 : size);
  const half = Math.floor(k / 2);
  const kernel = [];
  let sum = 0;
  const twoSigmaSq = 2 * sigma * sigma;

  for (let y = -half; y <= half; y++) {
    const row = [];
    for (let x = -half; x <= half; x++) {
      const val = Math.exp(-(x * x + y * y) / twoSigmaSq) / (Math.PI * twoSigmaSq);
      row.push(val);
      sum += val;
    }
    kernel.push(row);
  }

  // Normalize so sum equals 1
  for (let y = 0; y < k; y++) {
    for (let x = 0; x < k; x++) {
      kernel[y][x] /= sum;
    }
  }

  return { kernel, divisor: 1 };
}

/**
 * Phase 2: Gaussian Blur
 */
export function applyGaussianBlur(src, kernelSize = 5, sigma = 1.4) {
  const { kernel } = generateGaussianKernel(kernelSize, sigma);
  return convolve2D(src, kernel, 1, 0);
}

/**
 * Phase 2: Laplacian Sharpening
 * g(x,y) = f(x,y) + alpha * [f(x,y) - Laplacian(f(x,y))]
 */
export function applyLaplacianSharpen(src, strength = 1.0, neighbor = '8') {
  let kernel;
  if (neighbor === '4') {
    // Center: 1 + 4*strength, láng giềng: -strength
    kernel = [
      [ 0,          -strength,           0],
      [-strength, 1 + 4 * strength, -strength],
      [ 0,          -strength,           0]
    ];
  } else {
    // 8-neighbor
    kernel = [
      [-strength,   -strength,      -strength],
      [-strength, 1 + 8 * strength, -strength],
      [-strength,   -strength,      -strength]
    ];
  }
  return convolve2D(src, kernel, 1, 0);
}

/**
 * Phase 3: Sobel Operator
 * Computes Gx, Gy, and Magnitude
 */
export function applySobel(src, mode = 'magnitude') {
  const width = src.width;
  const height = src.height;
  const sData = src.data;
  const dst = createBlankImageData(width, height);
  const dData = dst.data;

  // Convert to grayscale float array first
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * sData[idx] + 0.587 * sData[idx + 1] + 0.114 * sData[idx + 2];
  }

  const kx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ];
  const ky = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
  ];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let gx = 0;
      let gy = 0;

      for (let r = -1; r <= 1; r++) {
        const py = Math.min(height - 1, Math.max(0, y + r));
        const rOffset = py * width;
        for (let c = -1; c <= 1; c++) {
          const px = Math.min(width - 1, Math.max(0, x + c));
          const val = gray[rOffset + px];
          gx += val * kx[r + 1][c + 1];
          gy += val * ky[r + 1][c + 1];
        }
      }

      let res = 0;
      if (mode === 'x') {
        res = Math.min(255, Math.max(0, Math.abs(gx)));
      } else if (mode === 'y') {
        res = Math.min(255, Math.max(0, Math.abs(gy)));
      } else {
        // Magnitude
        res = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      }

      const outIdx = (y * width + x) * 4;
      dData[outIdx] = res;
      dData[outIdx + 1] = res;
      dData[outIdx + 2] = res;
      dData[outIdx + 3] = 255;
    }
  }

  return dst;
}

/**
 * Phase 3: Prewitt Operator
 */
export function applyPrewitt(src, mode = 'magnitude') {
  const width = src.width;
  const height = src.height;
  const sData = src.data;
  const dst = createBlankImageData(width, height);
  const dData = dst.data;

  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * sData[idx] + 0.587 * sData[idx + 1] + 0.114 * sData[idx + 2];
  }

  const kx = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
  ];
  const ky = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
  ];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let gx = 0;
      let gy = 0;

      for (let r = -1; r <= 1; r++) {
        const py = Math.min(height - 1, Math.max(0, y + r));
        const rOffset = py * width;
        for (let c = -1; c <= 1; c++) {
          const px = Math.min(width - 1, Math.max(0, x + c));
          const val = gray[rOffset + px];
          gx += val * kx[r + 1][c + 1];
          gy += val * ky[r + 1][c + 1];
        }
      }

      let res = 0;
      if (mode === 'x') {
        res = Math.min(255, Math.max(0, Math.abs(gx)));
      } else if (mode === 'y') {
        res = Math.min(255, Math.max(0, Math.abs(gy)));
      } else {
        res = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      }

      const outIdx = (y * width + x) * 4;
      dData[outIdx] = res;
      dData[outIdx + 1] = res;
      dData[outIdx + 2] = res;
      dData[outIdx + 3] = 255;
    }
  }

  return dst;
}

/**
 * Phase 4: Median Filter (Non-linear)
 * Highly effective at removing Salt & Pepper noise
 */
export function applyMedianFilter(src, kernelSize = 3) {
  const width = src.width;
  const height = src.height;
  const sData = src.data;
  const dst = createBlankImageData(width, height);
  const dData = dst.data;

  const k = Math.max(3, kernelSize % 2 === 0 ? kernelSize + 1 : kernelSize);
  const half = Math.floor(k / 2);
  const total = k * k;
  const midIdx = Math.floor(total / 2);

  const rBuf = new Uint8Array(total);
  const gBuf = new Uint8Array(total);
  const bBuf = new Uint8Array(total);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let count = 0;
      for (let ky = -half; ky <= half; ky++) {
        const py = Math.min(height - 1, Math.max(0, y + ky));
        const rOffset = py * width * 4;
        for (let kx = -half; kx <= half; kx++) {
          const px = Math.min(width - 1, Math.max(0, x + kx));
          const idx = rOffset + px * 4;
          rBuf[count] = sData[idx];
          gBuf[count] = sData[idx + 1];
          bBuf[count] = sData[idx + 2];
          count++;
        }
      }

      rBuf.sort();
      gBuf.sort();
      bBuf.sort();

      const outIdx = (y * width + x) * 4;
      dData[outIdx] = rBuf[midIdx];
      dData[outIdx + 1] = gBuf[midIdx];
      dData[outIdx + 2] = bBuf[midIdx];
      dData[outIdx + 3] = sData[outIdx + 3];
    }
  }

  return dst;
}

/**
 * Phase 4: Bilateral Filter (Non-linear, Edge-Preserving smoothing)
 */
export function applyBilateralFilter(src, diameter = 7, sigmaColor = 45, sigmaSpace = 45) {
  const width = src.width;
  const height = src.height;
  const sData = src.data;
  const dst = createBlankImageData(width, height);
  const dData = dst.data;

  const radius = Math.floor(diameter / 2);
  const spaceCoeff = -0.5 / (sigmaSpace * sigmaSpace);
  const colorCoeff = -0.5 / (sigmaColor * sigmaColor);

  // Precompute spatial Gaussian weights
  const spaceWeights = [];
  for (let dy = -radius; dy <= radius; dy++) {
    const row = [];
    for (let dx = -radius; dx <= radius; dx++) {
      row.push(Math.exp((dx * dx + dy * dy) * spaceCoeff));
    }
    spaceWeights.push(row);
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerIdx = (y * width + x) * 4;
      const cR = sData[centerIdx];
      const cG = sData[centerIdx + 1];
      const cB = sData[centerIdx + 2];

      let rSum = 0, gSum = 0, bSum = 0;
      let rW = 0, gW = 0, bW = 0;

      for (let dy = -radius; dy <= radius; dy++) {
        const py = Math.min(height - 1, Math.max(0, y + dy));
        const rOffset = py * width * 4;
        const sWeightRow = spaceWeights[dy + radius];

        for (let dx = -radius; dx <= radius; dx++) {
          const px = Math.min(width - 1, Math.max(0, x + dx));
          const idx = rOffset + px * 4;
          const pR = sData[idx];
          const pG = sData[idx + 1];
          const pB = sData[idx + 2];

          const sW = sWeightRow[dx + radius];

          // Intensity differences
          const rDiff = pR - cR;
          const gDiff = pG - cG;
          const bDiff = pB - cB;

          const rWeight = sW * Math.exp(rDiff * rDiff * colorCoeff);
          const gWeight = sW * Math.exp(gDiff * gDiff * colorCoeff);
          const bWeight = sW * Math.exp(bDiff * bDiff * colorCoeff);

          rSum += pR * rWeight;
          gSum += pG * gWeight;
          bSum += pB * bWeight;

          rW += rWeight;
          gW += gWeight;
          bW += bWeight;
        }
      }

      dData[centerIdx] = rSum / (rW || 1);
      dData[centerIdx + 1] = gSum / (gW || 1);
      dData[centerIdx + 2] = bSum / (bW || 1);
      dData[centerIdx + 3] = sData[centerIdx + 3];
    }
  }

  return dst;
}

/**
 * Noise Injector: Salt and Pepper Noise
 */
export function addSaltAndPepperNoise(src, density = 0.05) {
  const dst = cloneImageData(src);
  const data = dst.data;
  const len = data.length;

  for (let i = 0; i < len; i += 4) {
    const r = Math.random();
    if (r < density / 2) {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
    } else if (r < density) {
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
    }
  }
  return dst;
}
