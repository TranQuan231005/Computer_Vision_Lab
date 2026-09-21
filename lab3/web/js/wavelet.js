/**
 * Wavelet Vision Studio — Pure JavaScript 2D Haar Wavelet Transform & wHash Core
 * Tương thích toán học với thư viện PyWavelets và module vision_wavelet.py của Lab 3.
 */

class WaveletCore {
  /**
   * Chuyển ImageData sang mảng 2D float64 [0, 1] theo chuẩn xám ITU-R BT.601
   */
  static imageToGrayscaleMatrix(imageData, targetSize = 32) {
    // Tạo canvas tạm để resize bilinear về targetSize x targetSize
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const ctx = tempCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Vẽ imageData lên canvas tạm kích thước targetSize
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = imageData.width;
    srcCanvas.height = imageData.height;
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

    ctx.drawImage(srcCanvas, 0, 0, targetSize, targetSize);
    const resizedData = ctx.getImageData(0, 0, targetSize, targetSize).data;

    const matrix = [];
    for (let r = 0; r < targetSize; r++) {
      const row = new Float64Array(targetSize);
      for (let c = 0; c < targetSize; c++) {
        const idx = (r * targetSize + c) * 4;
        const R = resizedData[idx];
        const G = resizedData[idx + 1];
        const B = resizedData[idx + 2];
        // Công thức Grayscale: 0.299 R + 0.587 G + 0.114 B chuẩn hóa [0, 1]
        row[c] = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0;
      }
      matrix.push(row);
    }
    return matrix;
  }

  /**
   * Biến đổi 2D Haar DWT 1 cấp trên ma trận N x N
   * Phân tách thành 4 băng tần: LL, LH (cH), HL (cV), HH (cD)
   * Hệ số chuẩn hóa sqrt(2) theo chuẩn PyWavelets
   */
  static haar2D(matrix) {
    const N = matrix.length;
    const half = Math.floor(N / 2);
    const SQRT2 = Math.SQRT2;

    // Bước 1: 1D Haar theo từng hàng
    const temp = [];
    for (let r = 0; r < N; r++) {
      const row = matrix[r];
      const newRow = new Float64Array(N);
      for (let c = 0; c < half; c++) {
        const a = row[2 * c];
        const b = row[2 * c + 1];
        newRow[c] = (a + b) / SQRT2;         // Low-pass
        newRow[half + c] = (a - b) / SQRT2;  // High-pass
      }
      temp.push(newRow);
    }

    // Bước 2: 1D Haar theo từng cột
    const LL = [];
    const LH = []; // cH: Low along col, High along row
    const HL = []; // cV: High along col, Low along row
    const HH = []; // cD: High along col, High along row

    for (let r = 0; r < half; r++) {
      LL.push(new Float64Array(half));
      LH.push(new Float64Array(half));
      HL.push(new Float64Array(half));
      HH.push(new Float64Array(half));
    }

    for (let c = 0; c < half; c++) {
      for (let r = 0; r < half; r++) {
        const topLow = temp[2 * r][c];
        const botLow = temp[2 * r + 1][c];
        LL[r][c] = (topLow + botLow) / SQRT2;
        HL[r][c] = (topLow - botLow) / SQRT2;

        const topHigh = temp[2 * r][half + c];
        const botHigh = temp[2 * r + 1][half + c];
        LH[r][c] = (topHigh + botHigh) / SQRT2;
        HH[r][c] = (topHigh - botHigh) / SQRT2;
      }
    }

    return { LL, LH, HL, HH, N, half };
  }

  /**
   * Tính Wavelet Hash (wHash) từ ma trận LL
   * Hỗ trợ hashSize = 8 (64-bit) hoặc 16 (256-bit)
   * Ngưỡng 'median' hoặc 'mean'
   */
  static computeWHashFromLL(LL, hashSize = 8, thresholdType = 'median') {
    const half = LL.length;
    const blockSize = Math.floor(half / hashSize);
    const features = [];
    const flatValues = [];

    for (let br = 0; br < hashSize; br++) {
      const fRow = new Float64Array(hashSize);
      for (let bc = 0; bc < hashSize; bc++) {
        let sum = 0;
        let count = 0;
        for (let r = br * blockSize; r < (br + 1) * blockSize; r++) {
          for (let c = bc * blockSize; c < (bc + 1) * blockSize; c++) {
            sum += LL[r][c];
            count++;
          }
        }
        const val = count > 0 ? sum / count : 0;
        fRow[bc] = val;
        flatValues.push(val);
      }
      features.push(fRow);
    }

    // Tính ngưỡng Cutoff
    let cutoff = 0;
    if (thresholdType === 'median') {
      const sorted = [...flatValues].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      cutoff = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    } else {
      const sum = flatValues.reduce((acc, v) => acc + v, 0);
      cutoff = sum / flatValues.length;
    }

    // Lượng tử hóa thành bit [0, 1]
    const bits = [];
    for (let r = 0; r < hashSize; r++) {
      for (let c = 0; c < hashSize; c++) {
        bits.push(features[r][c] >= cutoff ? 1 : 0);
      }
    }

    return {
      features,
      cutoff,
      bits,
      hex: WaveletCore.bitsToHex(bits),
      hashSize,
      totalBits: bits.length
    };
  }

  /**
   * Chuyển mảng bit [1, 0, ...] thành chuỗi Hex
   */
  static bitsToHex(bits) {
    let hex = '';
    for (let i = 0; i < bits.length; i += 4) {
      const chunk = bits.slice(i, i + 4);
      let val = 0;
      for (let j = 0; j < chunk.length; j++) {
        val = (val << 1) | chunk[j];
      }
      hex += val.toString(16);
    }
    return hex;
  }

  /**
   * Tính khoảng cách Hamming giữa 2 mảng bit cùng kích thước
   */
  static hammingDistance(bitsA, bitsB) {
    if (bitsA.length !== bitsB.length) {
      throw new Error(`Độ dài vector bit không khớp: ${bitsA.length} != ${bitsB.length}`);
    }
    let distance = 0;
    const diffMap = [];
    for (let i = 0; i < bitsA.length; i++) {
      const diff = bitsA[i] !== bitsB[i] ? 1 : 0;
      if (diff) distance++;
      diffMap.push(diff);
    }
    return {
      distance,
      diffMap,
      totalBits: bitsA.length,
      similarity: parseFloat(((1 - distance / bitsA.length) * 100).toFixed(2))
    };
  }

  /**
   * Chuyển ma trận hệ số 2D sang ImageData để vẽ lên Canvas
   * Hỗ trợ chuẩn hóa Min-Max tương phản cao
   */
  static matrixToImageData(matrix, ctx) {
    const H = matrix.length;
    const W = matrix[0].length;
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;

    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const val = matrix[r][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    const range = maxVal - minVal > 1e-8 ? maxVal - minVal : 1;

    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const norm = (matrix[r][c] - minVal) / range;
        const px = Math.min(255, Math.max(0, Math.round(norm * 255)));
        const idx = (r * W + c) * 4;
        data[idx] = px;
        data[idx + 1] = px;
        data[idx + 2] = px;
        data[idx + 3] = 255;
      }
    }
    return imgData;
  }
}

// Xuất cho môi trường Node hoặc trình duyệt
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WaveletCore };
}
