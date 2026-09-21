/**
 * Wavelet Vision Studio — Main Application Controller (Phase 5)
 * Kết nối DOM, xử lý sự kiện kéo thả, Canvas rendering, Dual Matcher, Stress Lab và CBIR.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ==========================================================================
  // STATE MANAGEMENT
  // ==========================================================================
  const state = {
    hashSize: parseInt(document.getElementById('global-hash-size').value, 10) || 8,
    thresholdType: document.getElementById('global-threshold').value || 'median',
    decisionThreshold: parseInt(document.getElementById('global-decision-thresh').value, 10) || 14,
    dataset: [...(typeof SAMPLE_DATASET !== 'undefined' ? SAMPLE_DATASET : [])],

    // Tab 1 state
    visImage: null,
    visHashData: null,

    // Tab 2 state
    matcherImageA: null,
    matcherImageB: null,
    matcherHashA: null,
    matcherHashB: null,

    // Tab 3 state
    stressBaseImage: null,
    stressBaseHash: null,
    stressParams: {
      rot: 0,
      bright: 100,
      contrast: 100,
      noise: 0,
      blur: 0
    },

    // Tab 4 state
    cbirQueryImage: null,
    cbirQueryHash: null,
    cbirTopK: parseInt(document.getElementById('cbir-topk-select').value, 10) || 8
  };

  // ==========================================================================
  // TAB NAVIGATION
  // ==========================================================================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const viewSections = document.querySelectorAll('.view-section');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      viewSections.forEach(v => v.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        targetSection.classList.add('active');
      }
    });
  });

  // Global Controls change
  document.getElementById('global-hash-size').addEventListener('change', (e) => {
    state.hashSize = parseInt(e.target.value, 10);
    document.getElementById('vis-bit-count').textContent = `${state.hashSize * state.hashSize}-bit`;
    recomputeAll();
  });

  document.getElementById('global-threshold').addEventListener('change', (e) => {
    state.thresholdType = e.target.value;
    recomputeAll();
  });

  document.getElementById('global-decision-thresh').addEventListener('change', (e) => {
    state.decisionThreshold = parseInt(e.target.value, 10);
    updateMatcherVerdict();
  });

  function recomputeAll() {
    if (state.visImage) processVisImage(state.visImage);
    if (state.matcherImageA && state.matcherImageB) {
      computeMatcherHashes();
    }
    if (state.stressBaseImage) {
      computeStressBaseHash();
      applyStressTransforms();
    }
    if (state.cbirQueryImage) {
      runCBIRSearch();
    }
  }

  // ==========================================================================
  // HELPER UTILITIES
  // ==========================================================================
  function loadImageElement(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function setupDropzone(dropzoneEl, fileInputEl, onImageLoaded) {
    dropzoneEl.addEventListener('click', () => fileInputEl.click());

    fileInputEl.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0], onImageLoaded);
      }
    });

    dropzoneEl.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzoneEl.classList.add('dragover');
    });

    dropzoneEl.addEventListener('dragleave', () => {
      dropzoneEl.classList.remove('dragover');
    });

    dropzoneEl.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzoneEl.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0], onImageLoaded);
      }
    });
  }

  function handleFile(file, callback) {
    if (!file.type.startsWith('image/')) {
      alert('Vui lòng chọn một tệp hình ảnh hợp lệ (PNG, JPG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      loadImageElement(e.target.result).then(callback);
    };
    reader.readAsDataURL(file);
  }

  // Clipboard Paste (Ctrl+V) anywhere on the page
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.type.indexOf('image') === 0) {
        const file = item.getAsFile();
        handleFile(file, (img) => {
          // Send to currently active tab
          const activeTab = document.querySelector('.tab-btn.active').getAttribute('data-target');
          if (activeTab === 'view-visualizer') {
            document.getElementById('vis-image-name').textContent = 'Pasted Image';
            processVisImage(img);
          } else if (activeTab === 'view-matcher') {
            state.matcherImageB = img;
            document.getElementById('matcher-name-b').textContent = 'Pasted Image';
            drawOntoCanvas(img, document.getElementById('matcher-canvas-b'));
            computeMatcherHashes();
          } else if (activeTab === 'view-stress') {
            processStressBaseImage(img);
          } else if (activeTab === 'view-cbir') {
            processCBIRQuery(img);
          }
        });
        break;
      }
    }
  });

  function drawOntoCanvas(img, canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const aspect = img.width / img.height;
    let dw = canvas.width;
    let dh = canvas.height;
    if (aspect > 1) {
      dh = canvas.width / aspect;
    } else {
      dw = canvas.height * aspect;
    }
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);
  }

  // ==========================================================================
  // TAB 1: WAVELET VISUALIZER
  // ==========================================================================
  const visInputCanvas = document.getElementById('vis-input-canvas');
  const canvasLL = document.getElementById('canvas-ll');
  const canvasLH = document.getElementById('canvas-lh');
  const canvasHL = document.getElementById('canvas-hl');
  const canvasHH = document.getElementById('canvas-hh');
  const visBitsCanvas = document.getElementById('vis-bits-canvas');
  const visHexVal = document.getElementById('vis-hex-val');
  const visCutoffVal = document.getElementById('vis-cutoff-val');
  const visSampleChips = document.getElementById('vis-sample-chips');

  setupDropzone(
    document.getElementById('vis-dropzone'),
    document.getElementById('vis-file-input'),
    (img) => {
      document.getElementById('vis-image-name').textContent = 'Custom Upload';
      processVisImage(img);
    }
  );

  // Copy Hex Button
  document.getElementById('vis-copy-hex-btn').addEventListener('click', () => {
    const hex = visHexVal.textContent;
    if (hex && hex !== '----------------') {
      navigator.clipboard.writeText(hex).then(() => {
        alert(`Đã sao chép mã wHash: ${hex}`);
      });
    }
  });

  function populateVisualizerSamples() {
    visSampleChips.innerHTML = '';
    state.dataset.slice(0, 15).forEach((item, idx) => {
      const chip = document.createElement('button');
      chip.className = `chip-btn ${idx === 0 ? 'active' : ''}`;
      chip.innerHTML = `<img src="${item.data_url}" alt="${item.name}"><span>${item.name}</span>`;
      chip.addEventListener('click', () => {
        visSampleChips.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        document.getElementById('vis-image-name').textContent = item.name;
        loadImageElement(item.data_url).then(processVisImage);
      });
      visSampleChips.appendChild(chip);
    });
  }

  function processVisImage(img) {
    state.visImage = img;
    drawOntoCanvas(img, visInputCanvas);

    // 1. Trích xuất ma trận Grayscale 64x64
    const grayMatrix = WaveletCore.imageToGrayscaleMatrix(
      visInputCanvas.getContext('2d').getImageData(0, 0, visInputCanvas.width, visInputCanvas.height),
      64
    );

    // 2. 2D Haar DWT
    const dwt = WaveletCore.haar2D(grayMatrix);

    // 3. Vẽ 4 Subbands lên canvas
    renderSubband(dwt.LL, canvasLL);
    renderSubband(dwt.LH, canvasLH);
    renderSubband(dwt.HL, canvasHL);
    renderSubband(dwt.HH, canvasHH);

    // 4. Lượng tử hóa wHash
    const hashData = WaveletCore.computeWHashFromLL(dwt.LL, state.hashSize, state.thresholdType);
    state.visHashData = hashData;

    // 5. Hiển thị Lưới Bit & Hex
    renderBitMatrix(hashData.bits, state.hashSize, visBitsCanvas);
    visHexVal.textContent = hashData.hex;
    visCutoffVal.textContent = hashData.cutoff.toFixed(4);
  }

  function renderSubband(matrix, canvas) {
    const ctx = canvas.getContext('2d');
    const imgData = WaveletCore.matrixToImageData(matrix, ctx);

    // Vẽ ImageData lên canvas tạm rồi phóng to vừa với canvas hiển thị
    const temp = document.createElement('canvas');
    temp.width = matrix[0].length;
    temp.height = matrix.length;
    temp.getContext('2d').putImageData(imgData, 0, 0);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false; // Giữ pixelation sắc nét
    ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
  }

  function renderBitMatrix(bits, size, canvas, highlightDiffMap = null) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cellW = W / size;
    const cellH = H / size;

    ctx.clearRect(0, 0, W, H);

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const idx = r * size + c;
        const bit = bits[idx];

        if (highlightDiffMap && highlightDiffMap[idx] === 1) {
          // Bit lệch trong XOR diff
          ctx.fillStyle = '#f43f5e';
        } else {
          // Bit chuẩn: 1 là xanh ngọc, 0 là xám đậm
          ctx.fillStyle = bit === 1 ? '#06b6d4' : '#1e293b';
        }

        ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
      }
    }
  }

  // ==========================================================================
  // TAB 2: DUAL MATCHER
  // ==========================================================================
  const matcherCanvasA = document.getElementById('matcher-canvas-a');
  const matcherCanvasB = document.getElementById('matcher-canvas-b');
  const matcherHexA = document.getElementById('matcher-hex-a');
  const matcherHexB = document.getElementById('matcher-hex-b');
  const matcherDiffCanvas = document.getElementById('matcher-diff-canvas');
  const matchGaugeBar = document.getElementById('match-gauge-bar');
  const matchSimVal = document.getElementById('match-sim-val');
  const matchDistVal = document.getElementById('match-dist-val');
  const matchDiffBits = document.getElementById('match-diff-bits');
  const matchVerdictBadge = document.getElementById('match-verdict-badge');

  setupDropzone(
    document.getElementById('matcher-dropzone-a'),
    document.getElementById('matcher-file-a'),
    (img) => {
      state.matcherImageA = img;
      document.getElementById('matcher-name-a').textContent = 'Ảnh A Tải Lên';
      drawOntoCanvas(img, matcherCanvasA);
      computeMatcherHashes();
    }
  );

  setupDropzone(
    document.getElementById('matcher-dropzone-b'),
    document.getElementById('matcher-file-b'),
    (img) => {
      state.matcherImageB = img;
      document.getElementById('matcher-name-b').textContent = 'Ảnh B Tải Lên';
      drawOntoCanvas(img, matcherCanvasB);
      computeMatcherHashes();
    }
  );

  function populateMatcherChips() {
    const chipsA = document.getElementById('matcher-chips-a');
    const chipsB = document.getElementById('matcher-chips-b');
    chipsA.innerHTML = '';
    chipsB.innerHTML = '';

    state.dataset.slice(0, 10).forEach((item, idx) => {
      const btnA = document.createElement('button');
      btnA.className = `chip-btn ${idx === 0 ? 'active' : ''}`;
      btnA.innerHTML = `<img src="${item.data_url}" alt="${item.name}"><span>${item.name}</span>`;
      btnA.addEventListener('click', () => {
        chipsA.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
        btnA.classList.add('active');
        document.getElementById('matcher-name-a').textContent = item.name;
        loadImageElement(item.data_url).then(img => {
          state.matcherImageA = img;
          drawOntoCanvas(img, matcherCanvasA);
          computeMatcherHashes();
        });
      });
      chipsA.appendChild(btnA);

      const btnB = document.createElement('button');
      btnB.className = `chip-btn ${idx === 1 ? 'active' : ''}`;
      btnB.innerHTML = `<img src="${item.data_url}" alt="${item.name}"><span>${item.name}</span>`;
      btnB.addEventListener('click', () => {
        chipsB.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
        btnB.classList.add('active');
        document.getElementById('matcher-name-b').textContent = item.name;
        loadImageElement(item.data_url).then(img => {
          state.matcherImageB = img;
          drawOntoCanvas(img, matcherCanvasB);
          computeMatcherHashes();
        });
      });
      chipsB.appendChild(btnB);
    });
  }

  function computeHashFromCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const gray = WaveletCore.imageToGrayscaleMatrix(imgData, 64);
    const dwt = WaveletCore.haar2D(gray);
    return WaveletCore.computeWHashFromLL(dwt.LL, state.hashSize, state.thresholdType);
  }

  function computeMatcherHashes() {
    if (!state.matcherImageA || !state.matcherImageB) return;

    state.matcherHashA = computeHashFromCanvas(matcherCanvasA);
    state.matcherHashB = computeHashFromCanvas(matcherCanvasB);

    matcherHexA.textContent = state.matcherHashA.hex;
    matcherHexB.textContent = state.matcherHashB.hex;

    updateMatcherVerdict();
  }

  function updateMatcherVerdict() {
    if (!state.matcherHashA || !state.matcherHashB) return;

    const result = WaveletCore.hammingDistance(state.matcherHashA.bits, state.matcherHashB.bits);
    const totalBits = state.hashSize * state.hashSize;

    matchDistVal.textContent = result.distance;
    matchDiffBits.textContent = `${result.distance} / ${totalBits}`;
    matchSimVal.textContent = `${result.similarity}%`;

    // Cập nhật SVG Gauge: stroke-dasharray = 377 (2 * pi * 55 ~ 345)
    const circumference = 2 * Math.PI * 55;
    const offset = circumference - (result.similarity / 100) * circumference;
    matchGaugeBar.style.strokeDasharray = `${circumference}`;
    matchGaugeBar.style.strokeDashoffset = `${offset}`;

    // Phân định Ngưỡng Youden's J
    // Tỉ lệ ngưỡng scale theo tổng số bit (14 bit cho 64-bit ~ 21.8%)
    const maxThreshold = Math.round((state.decisionThreshold / 64) * totalBits);
    const isMatched = result.distance <= maxThreshold;

    if (isMatched) {
      matchVerdictBadge.className = 'verdict-badge verdict-matched';
      matchVerdictBadge.innerHTML = `MATCHED (TƯƠNG TỰ — Lệch ≤ ${maxThreshold} bit)`;
      matchGaugeBar.style.stroke = 'var(--accent-emerald)';
    } else {
      matchVerdictBadge.className = 'verdict-badge verdict-dissimilar';
      matchVerdictBadge.innerHTML = `DISSIMILAR (KHÁC BIỆT — Lệch > ${maxThreshold} bit)`;
      matchGaugeBar.style.stroke = 'var(--accent-rose)';
    }

    // Vẽ XOR Diff Canvas
    renderBitMatrix(state.matcherHashA.bits, state.hashSize, matcherDiffCanvas, result.diffMap);
  }

  // Quick preset test pairs
  document.getElementById('btn-pair-similar-blur').addEventListener('click', () => {
    loadTestPair('astronaut', 'astronaut__blur', 'Astronaut', 'Astronaut (Mờ Gaussian)');
  });
  document.getElementById('btn-pair-similar-noise').addEventListener('click', () => {
    loadTestPair('camera', 'camera__jpeg_30', 'Camera', 'Camera (Nén JPEG Q30)');
  });
  document.getElementById('btn-pair-similar-rot').addEventListener('click', () => {
    loadTestPair('coffee', 'coffee__rotate_+5', 'Coffee', 'Coffee (Xoay +5°)');
  });
  document.getElementById('btn-pair-diff-1').addEventListener('click', () => {
    loadTestPair('astronaut', 'horse', 'Astronaut', 'Con Ngựa (Khác biệt)');
  });
  document.getElementById('btn-pair-diff-2').addEventListener('click', () => {
    loadTestPair('rocket', 'digit_3', 'Tên Lửa', 'Chữ Số 3 (Khác biệt)');
  });

  function loadTestPair(idA, idB, nameA, nameB) {
    const itemA = state.dataset.find(x => x.id === idA);
    const itemB = state.dataset.find(x => x.id === idB);
    if (!itemA || !itemB) return;

    document.getElementById('matcher-name-a').textContent = nameA;
    document.getElementById('matcher-name-b').textContent = nameB;

    Promise.all([loadImageElement(itemA.data_url), loadImageElement(itemB.data_url)]).then(([imgA, imgB]) => {
      state.matcherImageA = imgA;
      state.matcherImageB = imgB;
      drawOntoCanvas(imgA, matcherCanvasA);
      drawOntoCanvas(imgB, matcherCanvasB);
      computeMatcherHashes();
    });
  }

  // ==========================================================================
  // TAB 3: STRESS LAB
  // ==========================================================================
  const stressBaseCanvas = document.getElementById('stress-base-canvas');
  const stressLiveCanvas = document.getElementById('stress-live-canvas');
  const stressBaseHex = document.getElementById('stress-base-hex');
  const stressLiveDist = document.getElementById('stress-live-dist');
  const stressLiveSim = document.getElementById('stress-live-sim');
  const stressVerdictBadge = document.getElementById('stress-verdict-badge');
  const stressLiveStatus = document.getElementById('stress-live-status');

  const sliderRot = document.getElementById('slider-rot');
  const sliderBright = document.getElementById('slider-bright');
  const sliderContrast = document.getElementById('slider-contrast');
  const sliderNoise = document.getElementById('slider-noise');
  const sliderBlur = document.getElementById('slider-blur');

  function populateStressSamples() {
    const chips = document.getElementById('stress-sample-chips');
    chips.innerHTML = '';
    state.dataset.slice(0, 10).forEach((item, idx) => {
      const chip = document.createElement('button');
      chip.className = `chip-btn ${idx === 0 ? 'active' : ''}`;
      chip.innerHTML = `<img src="${item.data_url}" alt="${item.name}"><span>${item.name}</span>`;
      chip.addEventListener('click', () => {
        chips.querySelectorAll('.chip-btn').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        loadImageElement(item.data_url).then(processStressBaseImage);
      });
      chips.appendChild(chip);
    });
  }

  function processStressBaseImage(img) {
    state.stressBaseImage = img;
    drawOntoCanvas(img, stressBaseCanvas);
    computeStressBaseHash();
    applyStressTransforms();
  }

  function computeStressBaseHash() {
    if (!state.stressBaseImage) return;
    state.stressBaseHash = computeHashFromCanvas(stressBaseCanvas);
    stressBaseHex.textContent = state.stressBaseHash.hex;
  }

  function bindSlider(slider, labelId, suffix, paramKey) {
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById(labelId).textContent = `${val}${suffix}`;
      state.stressParams[paramKey] = val;
      applyStressTransforms();
    });
  }

  bindSlider(sliderRot, 'val-rot', '°', 'rot');
  bindSlider(sliderBright, 'val-bright', '%', 'bright');
  bindSlider(sliderContrast, 'val-contrast', '%', 'contrast');
  bindSlider(sliderNoise, 'val-noise', '%', 'noise');
  bindSlider(sliderBlur, 'val-blur', ' px', 'blur');

  document.getElementById('stress-reset-btn').addEventListener('click', () => {
    sliderRot.value = 0; document.getElementById('val-rot').textContent = '0°';
    sliderBright.value = 100; document.getElementById('val-bright').textContent = '100%';
    sliderContrast.value = 100; document.getElementById('val-contrast').textContent = '100%';
    sliderNoise.value = 0; document.getElementById('val-noise').textContent = '0%';
    sliderBlur.value = 0; document.getElementById('val-blur').textContent = '0 px';
    state.stressParams = { rot: 0, bright: 100, contrast: 100, noise: 0, blur: 0 };
    applyStressTransforms();
  });

  function applyStressTransforms() {
    if (!state.stressBaseImage) return;

    const ctx = stressLiveCanvas.getContext('2d');
    const W = stressLiveCanvas.width;
    const H = stressLiveCanvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.save();

    // 1. Áp dụng CSS filters: Brightness, Contrast, Blur
    ctx.filter = `brightness(${state.stressParams.bright}%) contrast(${state.stressParams.contrast}%) blur(${state.stressParams.blur}px)`;

    // 2. Áp dụng Xoay (Rotation) quanh tâm
    ctx.translate(W / 2, H / 2);
    ctx.rotate((state.stressParams.rot * Math.PI) / 180);
    ctx.translate(-W / 2, -H / 2);

    // 3. Vẽ ảnh
    const aspect = state.stressBaseImage.width / state.stressBaseImage.height;
    let dw = W, dh = H;
    if (aspect > 1) dh = W / aspect; else dw = H * aspect;
    ctx.drawImage(state.stressBaseImage, (W - dw) / 2, (H - dh) / 2, dw, dh);
    ctx.restore();

    // 4. Áp dụng Nhiễu hạt (Noise Simulation) nếu có
    if (state.stressParams.noise > 0) {
      const imgData = ctx.getImageData(0, 0, W, H);
      const data = imgData.data;
      const noiseAmp = (state.stressParams.noise / 100) * 120;
      for (let i = 0; i < data.length; i += 4) {
        const noise = (Math.random() - 0.5) * noiseAmp;
        data[i] = Math.min(255, Math.max(0, data[i] + noise));
        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
      }
      ctx.putImageData(imgData, 0, 0);
    }

    // 5. Tính wHash sau biến dạng và so sánh với ảnh gốc
    const liveHash = computeHashFromCanvas(stressLiveCanvas);
    if (state.stressBaseHash) {
      const result = WaveletCore.hammingDistance(state.stressBaseHash.bits, liveHash.bits);
      const totalBits = state.hashSize * state.hashSize;
      const maxThreshold = Math.round((state.decisionThreshold / 64) * totalBits);

      stressLiveDist.textContent = `${result.distance} bit`;
      stressLiveSim.textContent = `${result.similarity}%`;

      if (result.distance <= maxThreshold) {
        stressVerdictBadge.className = 'verdict-badge verdict-matched';
        stressVerdictBadge.textContent = 'VẪN TƯƠNG TỰ (ROBUST)';
        stressLiveStatus.textContent = 'Giữ nguyên nhận diện';
      } else {
        stressVerdictBadge.className = 'verdict-badge verdict-dissimilar';
        stressVerdictBadge.textContent = 'LỆCH MÃ BĂM (BIẾN DẠNG LỚN)';
        stressLiveStatus.textContent = 'Đã mất nhận diện';
      }
    }
  }

  // ==========================================================================
  // TAB 4: CBIR SEARCH STUDIO
  // ==========================================================================
  const cbirQueryImg = document.getElementById('cbir-query-img');
  const cbirSampleSelect = document.getElementById('cbir-sample-select');
  const cbirTopKSelect = document.getElementById('cbir-topk-select');
  const cbirResultsContainer = document.getElementById('cbir-results-container');
  const cbirDbCount = document.getElementById('cbir-db-count');

  setupDropzone(
    document.getElementById('cbir-dropzone'),
    document.getElementById('cbir-query-input'),
    processCBIRQuery
  );

  cbirTopKSelect.addEventListener('change', (e) => {
    state.cbirTopK = parseInt(e.target.value, 10);
    runCBIRSearch();
  });

  cbirSampleSelect.addEventListener('change', (e) => {
    const selectedId = e.target.value;
    const item = state.dataset.find(x => x.id === selectedId);
    if (item) {
      loadImageElement(item.data_url).then(processCBIRQuery);
    }
  });

  // Upload new images to dataset dynamically
  const cbirUploadBtn = document.getElementById('cbir-upload-btn');
  const cbirUploadInput = document.getElementById('cbir-upload-input');
  cbirUploadBtn.addEventListener('click', () => cbirUploadInput.click());

  cbirUploadInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      Array.from(e.target.files).forEach(file => {
        handleFile(file, (img) => {
          // Precompute 64-bit and 256-bit hashes
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = 64;
          tempCanvas.height = 64;
          drawOntoCanvas(img, tempCanvas);
          const ctx = tempCanvas.getContext('2d');
          const gray = WaveletCore.imageToGrayscaleMatrix(ctx.getImageData(0, 0, 64, 64), 64);
          const dwt = WaveletCore.haar2D(gray);
          const h64 = WaveletCore.computeWHashFromLL(dwt.LL, 8, 'median');
          const h256 = WaveletCore.computeWHashFromLL(dwt.LL, 16, 'median');

          const newId = `user_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
          state.dataset.unshift({
            id: newId,
            name: file.name.replace(/\.[^/.]+$/, "").replace(/_/g, " "),
            category: 'user-uploaded',
            is_original: false,
            data_url: tempCanvas.toDataURL('image/jpeg', 0.85),
            whash_64_bits: h64.bits,
            whash_64_hex: h64.hex,
            whash_256_bits: h256.bits,
            whash_256_hex: h256.hex
          });

          updateDatasetUI();
          runCBIRSearch();
        });
      });
    }
  });

  function populateCBIRSamples() {
    cbirSampleSelect.innerHTML = '';
    state.dataset.forEach((item, idx) => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = `${item.is_original ? '⭐ [Gốc] ' : '⚡ '}${item.name}`;
      if (idx === 0) opt.selected = true;
      cbirSampleSelect.appendChild(opt);
    });
    cbirDbCount.textContent = `Kho ảnh: ${state.dataset.length} ảnh`;
  }

  function updateDatasetUI() {
    cbirDbCount.textContent = `Kho ảnh: ${state.dataset.length} ảnh`;
    populateCBIRSamples();
  }

  function processCBIRQuery(img) {
    state.cbirQueryImage = img;
    cbirQueryImg.src = img.src;

    // Tính hash của ảnh query
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 64;
    tempCanvas.height = 64;
    drawOntoCanvas(img, tempCanvas);
    const ctx = tempCanvas.getContext('2d');
    const gray = WaveletCore.imageToGrayscaleMatrix(ctx.getImageData(0, 0, 64, 64), 64);
    const dwt = WaveletCore.haar2D(gray);
    state.cbirQueryHash = WaveletCore.computeWHashFromLL(dwt.LL, state.hashSize, state.thresholdType);

    runCBIRSearch();
  }

  function runCBIRSearch() {
    if (!state.cbirQueryHash) return;

    // So sánh khoảng cách với toàn bộ cơ sở dữ liệu
    const scores = state.dataset.map(item => {
      const dbBits = state.hashSize === 8 ? item.whash_64_bits : item.whash_256_bits;
      const res = WaveletCore.hammingDistance(state.cbirQueryHash.bits, dbBits);
      return {
        item,
        distance: res.distance,
        similarity: res.similarity
      };
    });

    // Sắp xếp tăng dần theo khoảng cách Hamming (ảnh giống nhất lên đầu)
    scores.sort((a, b) => a.distance - b.distance);

    // Cắt Top-K
    const topK = scores.slice(0, state.cbirTopK);
    renderCBIRResults(topK);
  }

  function renderCBIRResults(topResults) {
    cbirResultsContainer.innerHTML = '';

    topResults.forEach((res, rank) => {
      const card = document.createElement('div');
      card.className = `cbir-card ${rank === 0 ? 'match-top1' : ''}`;

      const isExact = res.distance === 0;
      const rankLabel = rank === 0 ? 'Top 1 (Khớp cao nhất)' : `#${rank + 1}`;

      card.innerHTML = `
        <div class="cbir-card-thumb-wrap">
          <img src="${res.item.data_url}" alt="${res.item.name}" loading="lazy">
          <span class="cbir-rank-tag">${rankLabel}</span>
        </div>
        <div class="cbir-card-body">
          <div class="cbir-card-title" title="${res.item.name}">${res.item.name}</div>
          <div class="cbir-card-meta">
            <span>Khoảng cách: <strong style="color: ${isExact ? 'var(--accent-emerald)' : 'var(--accent-cyan)'}">${res.distance} bit</strong></span>
            <span>Tương đồng: <strong>${res.similarity}%</strong></span>
          </div>
          <div class="sim-bar-bg">
            <div class="sim-bar-fill" style="width: ${res.similarity}%"></div>
          </div>
        </div>
      `;

      // Click card to open in Visualizer or Matcher
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => {
        loadImageElement(res.item.data_url).then(img => {
          document.getElementById('tab-btn-visualizer').click();
          document.getElementById('vis-image-name').textContent = res.item.name;
          processVisImage(img);
        });
      });

      cbirResultsContainer.appendChild(card);
    });
  }

  // ==========================================================================
  // INITIAL BOOTSTRAP
  // ==========================================================================
  populateVisualizerSamples();
  populateMatcherChips();
  populateStressSamples();
  populateCBIRSamples();

  // Load first image as initial demo
  if (state.dataset.length > 0) {
    const first = state.dataset[0];
    const second = state.dataset[1] || first;

    loadImageElement(first.data_url).then(img => {
      processVisImage(img);
      processStressBaseImage(img);
      processCBIRQuery(img);
    });

    loadImageElement(first.data_url).then(imgA => {
      state.matcherImageA = imgA;
      drawOntoCanvas(imgA, matcherCanvasA);
      loadImageElement(second.data_url).then(imgB => {
        state.matcherImageB = imgB;
        drawOntoCanvas(imgB, matcherCanvasB);
        computeMatcherHashes();
      });
    });
  }
});
