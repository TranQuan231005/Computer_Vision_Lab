/**
 * canny.js - Pure JavaScript Step-by-Step Canny Edge Detection Pipeline
 * Matches the exact 4-step algorithm in visioncanny.py:
 * Step 1: Gaussian Blur
 * Step 2: Sobel Gradients (Magnitude & Angle)
 * Step 3: Non-Maximum Suppression (NMS)
 * Step 4: Hysteresis Thresholding (Double Threshold & Edge Tracking)
 */

import { createBlankImageData, generateGaussianKernel } from './filters.js';

/**
 * Executes full Canny pipeline and returns all intermediate step ImageDatas for visualization
 */
export function runCannyPipeline(src, {
  sigma = 1.4,
  kernelSize = 5,
  lowRatio = 0.05,
  highRatio = 0.15,
  directThresholdMode = false,
  lowThresh = 50,
  highThresh = 150
}) {
  const width = src.width;
  const height = src.height;
  const sData = src.data;

  // Step 0: Convert to Grayscale Float array [0, 255]
  const gray = new Float32Array(width * height);
  const grayImg = createBlankImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const g = 0.299 * sData[idx] + 0.587 * sData[idx + 1] + 0.114 * sData[idx + 2];
    gray[i] = g;
    grayImg.data[idx] = g;
    grayImg.data[idx + 1] = g;
    grayImg.data[idx + 2] = g;
    grayImg.data[idx + 3] = 255;
  }

  // Step 1: Gaussian Blur
  const { kernel } = generateGaussianKernel(kernelSize, sigma);
  const kSize = kernel.length;
  const half = Math.floor(kSize / 2);
  const blurred = new Float32Array(width * height);
  const blurredImg = createBlankImageData(width, height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let ky = 0; ky < kSize; ky++) {
        const py = Math.min(height - 1, Math.max(0, y + ky - half));
        const rOff = py * width;
        for (let kx = 0; kx < kSize; kx++) {
          const px = Math.min(width - 1, Math.max(0, x + kx - half));
          sum += gray[rOff + px] * kernel[ky][kx];
        }
      }
      const i = y * width + x;
      blurred[i] = sum;
      const idx = i * 4;
      blurredImg.data[idx] = sum;
      blurredImg.data[idx + 1] = sum;
      blurredImg.data[idx + 2] = sum;
      blurredImg.data[idx + 3] = 255;
    }
  }

  // Step 2: Sobel Gradients (Magnitude & Angle)
  const magnitude = new Float32Array(width * height);
  const angle = new Float32Array(width * height);
  const magImg = createBlankImageData(width, height);

  const kSobelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ];
  const kSobelY = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
  ];

  let maxMag = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let gx = 0;
      let gy = 0;

      for (let ky = -1; ky <= 1; ky++) {
        const py = Math.min(height - 1, Math.max(0, y + ky));
        const rOff = py * width;
        for (let kx = -1; kx <= 1; kx++) {
          const px = Math.min(width - 1, Math.max(0, x + kx));
          const val = blurred[rOff + px];
          gx += val * kSobelX[ky + 1][kx + 1];
          gy += val * kSobelY[ky + 1][kx + 1];
        }
      }

      const mag = Math.sqrt(gx * gx + gy * gy);
      const i = y * width + x;
      magnitude[i] = mag;
      if (mag > maxMag) maxMag = mag;

      // Angle in degrees: 0 to 180
      let ang = Math.atan2(gy, gx) * (180 / Math.PI);
      if (ang < 0) ang += 180;
      angle[i] = ang;
    }
  }

  // Normalize magnitude for display
  const normMagFactor = maxMag > 0 ? 255 / maxMag : 1;
  for (let i = 0; i < width * height; i++) {
    const val = Math.min(255, magnitude[i] * normMagFactor);
    const idx = i * 4;
    magImg.data[idx] = val;
    magImg.data[idx + 1] = val;
    magImg.data[idx + 2] = val;
    magImg.data[idx + 3] = 255;
  }

  // Step 3: Non-Maximum Suppression (NMS)
  const nms = new Float32Array(width * height);
  const nmsImg = createBlankImageData(width, height);
  let maxNms = 0;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const a = angle[i];
      const magCenter = magnitude[i];

      let n1 = 0, n2 = 0;

      // 4 Quantized Directions
      if ((a >= 0 && a < 22.5) || (a >= 157.5 && a <= 180)) {
        // Horizontal direction (0 deg) -> check left & right
        n1 = magnitude[y * width + (x + 1)];
        n2 = magnitude[y * width + (x - 1)];
      } else if (a >= 22.5 && a < 67.5) {
        // Diagonal 45 deg -> check top-right & bottom-left
        n1 = magnitude[(y - 1) * width + (x + 1)];
        n2 = magnitude[(y + 1) * width + (x - 1)];
      } else if (a >= 67.5 && a < 112.5) {
        // Vertical 90 deg -> check top & bottom
        n1 = magnitude[(y - 1) * width + x];
        n2 = magnitude[(y + 1) * width + x];
      } else {
        // Diagonal 135 deg -> check top-left & bottom-right
        n1 = magnitude[(y - 1) * width + (x - 1)];
        n2 = magnitude[(y + 1) * width + (x + 1)];
      }

      if (magCenter >= n1 && magCenter >= n2) {
        nms[i] = magCenter;
        if (magCenter > maxNms) maxNms = magCenter;
      } else {
        nms[i] = 0;
      }
    }
  }

  // Normalize NMS for display
  const normNmsFactor = maxNms > 0 ? 255 / maxNms : 1;
  for (let i = 0; i < width * height; i++) {
    const val = Math.min(255, nms[i] * normNmsFactor);
    const idx = i * 4;
    nmsImg.data[idx] = val;
    nmsImg.data[idx + 1] = val;
    nmsImg.data[idx + 2] = val;
    nmsImg.data[idx + 3] = 255;
  }

  // Step 4: Double Threshold & Hysteresis
  let highT, lowT;
  if (directThresholdMode) {
    highT = highThresh;
    lowT = lowThresh;
  } else {
    // Ratio-based thresholding (like visioncanny.py)
    highT = maxNms * highRatio;
    lowT = highT * lowRatio;
  }

  const doubleThreshImg = createBlankImageData(width, height);
  const resultEdges = new Uint8Array(width * height);
  const finalCannyImg = createBlankImageData(width, height);

  const STRONG = 255;
  const WEAK = 75;

  const queue = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const v = nms[i];
      const idx = i * 4;

      if (v >= highT) {
        resultEdges[i] = STRONG;
        queue.push(i);
        doubleThreshImg.data[idx] = 255;
        doubleThreshImg.data[idx + 1] = 255;
        doubleThreshImg.data[idx + 2] = 255;
      } else if (v >= lowT) {
        resultEdges[i] = WEAK;
        // Weak edge visual (gold / yellow-cyan highlight in double threshold view)
        doubleThreshImg.data[idx] = 120;
        doubleThreshImg.data[idx + 1] = 180;
        doubleThreshImg.data[idx + 2] = 255;
      } else {
        resultEdges[i] = 0;
        doubleThreshImg.data[idx] = 0;
        doubleThreshImg.data[idx + 1] = 0;
        doubleThreshImg.data[idx + 2] = 0;
      }
      doubleThreshImg.data[idx + 3] = 255;
    }
  }

  // Step 5: Hysteresis Edge Tracking (8-neighbor BFS connected component)
  const neighbors = [
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1],          [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1]
  ];

  let head = 0;
  while (head < queue.length) {
    const curr = queue[head++];
    const cy = Math.floor(curr / width);
    const cx = curr % width;

    for (let k = 0; k < 8; k++) {
      const ny = cy + neighbors[k][0];
      const nx = cx + neighbors[k][1];

      if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
        const ni = ny * width + nx;
        if (resultEdges[ni] === WEAK) {
          resultEdges[ni] = STRONG;
          queue.push(ni);
        }
      }
    }
  }

  // Suppress remaining isolated weak edges
  for (let i = 0; i < width * height; i++) {
    const val = resultEdges[i] === STRONG ? 255 : 0;
    const idx = i * 4;
    finalCannyImg.data[idx] = val;
    finalCannyImg.data[idx + 1] = val;
    finalCannyImg.data[idx + 2] = val;
    finalCannyImg.data[idx + 3] = 255;
  }

  return {
    gray: grayImg,
    blurred: blurredImg,
    magnitude: magImg,
    nms: nmsImg,
    doubleThreshold: doubleThreshImg,
    final: finalCannyImg,
    stats: {
      maxMagnitude: maxMag.toFixed(1),
      maxNMS: maxNms.toFixed(1),
      lowThreshold: lowT.toFixed(1),
      highThreshold: highT.toFixed(1)
    }
  };
}
