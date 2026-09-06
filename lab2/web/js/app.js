/**
 * app.js - Main Application Orchestrator for Computer Vision Lab 2 Visualizer
 */

import {
  KERNEL_PRESETS,
  THEORY_AND_CODE,
  generateSampleImage
} from './presets.js';

import {
  createBlankImageData,
  cloneImageData,
  applyPointOps,
  convolve2D,
  applyBoxBlur,
  applyGaussianBlur,
  applyLaplacianSharpen,
  applySobel,
  applyPrewitt,
  applyMedianFilter,
  applyBilateralFilter,
  addSaltAndPepperNoise
} from './filters.js';

import { runCannyPipeline } from './canny.js';

// ==========================================================================
// Application State
// ==========================================================================

const state = {
  activeTab: 'phase1',
  viewMode: 'split', // 'split' | 'sideBySide'
  splitPos: 50, // Percentage (0 - 100)
  isDraggingSplit: false,

  // Images
  rawImage: null,        // Original HTMLImageElement
  baseImageData: null,   // ImageData before current filter (e.g. might have noise injected)
  resultImageData: null, // ImageData after current filter

  // Phase 1 Params
  p1: {
    brightness: 0,
    contrast: 1.0,
    negative: false,
    thresholdEnabled: false,
    thresholdValue: 128,
    invertBinary: false
  },

  // Phase 2 Params
  p2: {
    type: 'gaussian', // 'gaussian' | 'box' | 'sharpen'
    kernelSize: 5,
    sigma: 1.5,
    sharpenStrength: 1.0,
    neighbor: '8'
  },

  // Phase 3 Params
  p3: {
    type: 'sobel', // 'sobel' | 'prewitt' | 'custom'
    direction: 'magnitude', // 'magnitude' | 'x' | 'y'
    customMatrix: [
      [-2, -1, 0],
      [-1,  1, 1],
      [ 0,  1, 2]
    ],
    customDivisor: 1,
    customBias: 128
  },

  // Phase 4 Params
  p4: {
    type: 'median', // 'median' | 'bilateral'
    medianSize: 5,
    bilateralDiameter: 9,
    bilateralColor: 75,
    bilateralSpace: 75,
    addNoise: false,
    noiseDensity: 0.06
  },

  // Phase 5 Params (Canny)
  p5: {
    step: 'final', // 'final' | 'step1_blur' | 'step2_grad' | 'step3_nms' | 'step4_thresh' | 'all_grid'
    threshMode: 'ratio', // 'ratio' | 'absolute'
    sigma: 1.4,
    kernelSize: 5,
    lowRatio: 0.05,
    highRatio: 0.15,
    absLow: 50,
    absHigh: 150
  },

  // Comparison Studio Params
  comp: {
    kernelSize: 5
  }
};

// ==========================================================================
// DOM Elements
// ==========================================================================

const DOM = {
  // Navigation
  navTabs: document.getElementById('navTabs'),
  tabButtons: document.querySelectorAll('.tab-btn'),
  phasePanels: document.querySelectorAll('.phase-panel'),

  // Toolbar
  btnUpload: document.getElementById('btnUpload'),
  fileInput: document.getElementById('fileInput'),
  btnReset: document.getElementById('btnReset'),
  btnDownload: document.getElementById('btnDownload'),
  presetChips: document.querySelectorAll('.preset-chip'),

  // Viewports
  stageContainer: document.getElementById('stageContainer'),
  splitWrapper: document.getElementById('splitWrapper'),
  canvasBefore: document.getElementById('canvasBefore'),
  canvasAfter: document.getElementById('canvasAfter'),
  splitDivider: document.getElementById('splitDivider'),
  badgeBefore: document.getElementById('badgeBefore'),
  badgeAfter: document.getElementById('badgeAfter'),
  dropOverlay: document.getElementById('dropOverlay'),
  multiGrid: document.getElementById('multiGrid'),
  viewportTitle: document.getElementById('viewportTitle'),
  modeSplit: document.getElementById('modeSplit'),
  modeSideBySide: document.getElementById('modeSideBySide'),

  // Theory & Code
  mathContent: document.getElementById('mathContent'),
  pythonCodeContent: document.getElementById('pythonCodeContent'),
  btnCopyCode: document.getElementById('btnCopyCode'),

  // Phase 1 Controls
  sliderBrightness: document.getElementById('sliderBrightness'),
  valBrightness: document.getElementById('valBrightness'),
  sliderContrast: document.getElementById('sliderContrast'),
  valContrast: document.getElementById('valContrast'),
  checkNegative: document.getElementById('checkNegative'),
  checkThreshold: document.getElementById('checkThreshold'),
  thresholdControls: document.getElementById('thresholdControls'),
  sliderThreshold: document.getElementById('sliderThreshold'),
  valThreshold: document.getElementById('valThreshold'),
  checkInvertBinary: document.getElementById('checkInvertBinary'),

  // Phase 2 Controls
  linearTypeBtns: document.querySelectorAll('[data-linear]'),
  sliderKernelSize: document.getElementById('sliderKernelSize'),
  valKernelSize: document.getElementById('valKernelSize'),
  sliderSigma: document.getElementById('sliderSigma'),
  valSigma: document.getElementById('valSigma'),
  groupKernelSize: document.getElementById('groupKernelSize'),
  groupSigma: document.getElementById('groupSigma'),
  groupSharpen: document.getElementById('groupSharpen'),
  sliderSharpenStrength: document.getElementById('sliderSharpenStrength'),
  valSharpenStrength: document.getElementById('valSharpenStrength'),
  neighborBtns: document.querySelectorAll('[data-neighbor]'),

  // Phase 3 Controls
  edgeTypeBtns: document.querySelectorAll('[data-edge]'),
  dirBtns: document.querySelectorAll('[data-dir]'),
  groupEdgeDirection: document.getElementById('groupEdgeDirection'),
  groupCustomKernel: document.getElementById('groupCustomKernel'),
  selectKernelPreset: document.getElementById('selectKernelPreset'),
  kernelGrid: document.getElementById('kernelGrid'),
  inputDivisor: document.getElementById('inputDivisor'),
  inputBias: document.getElementById('inputBias'),

  // Phase 4 Controls
  nonlinearTypeBtns: document.querySelectorAll('[data-nonlinear]'),
  groupMedian: document.getElementById('groupMedian'),
  groupBilateral: document.getElementById('groupBilateral'),
  sliderMedianSize: document.getElementById('sliderMedianSize'),
  valMedianSize: document.getElementById('valMedianSize'),
  sliderBilateralDiameter: document.getElementById('sliderBilateralDiameter'),
  valBilateralDiameter: document.getElementById('valBilateralDiameter'),
  sliderBilateralColor: document.getElementById('sliderBilateralColor'),
  valBilateralColor: document.getElementById('valBilateralColor'),
  sliderBilateralSpace: document.getElementById('sliderBilateralSpace'),
  valBilateralSpace: document.getElementById('valBilateralSpace'),
  checkAddNoise: document.getElementById('checkAddNoise'),
  groupNoiseDensity: document.getElementById('groupNoiseDensity'),
  sliderNoiseDensity: document.getElementById('sliderNoiseDensity'),
  valNoiseDensity: document.getElementById('valNoiseDensity'),

  // Phase 5 Controls
  selectCannyStep: document.getElementById('selectCannyStep'),
  btnThreshModeRatio: document.getElementById('btnThreshModeRatio'),
  btnThreshModeAbs: document.getElementById('btnThreshModeAbs'),
  labelCannyLow: document.getElementById('labelCannyLow'),
  labelCannyHigh: document.getElementById('labelCannyHigh'),
  sliderCannySigma: document.getElementById('sliderCannySigma'),
  valCannySigma: document.getElementById('valCannySigma'),
  sliderCannyLowRatio: document.getElementById('sliderCannyLowRatio'),
  valCannyLowRatio: document.getElementById('valCannyLowRatio'),
  sliderCannyHighRatio: document.getElementById('sliderCannyHighRatio'),
  valCannyHighRatio: document.getElementById('valCannyHighRatio'),
  cannyPresetBtns: document.querySelectorAll('[data-cannypreset]'),

  // Comparison Controls
  compKernelBtns: document.querySelectorAll('[data-compk]')
};

// ==========================================================================
// Initialization & Image Loading
// ==========================================================================

function init() {
  buildCustomKernelGrid();
  setupEventListeners();
  loadSampleImage('geometric');
}

/**
 * Loads a procedural sample image
 */
function loadSampleImage(presetKey) {
  const dataUrl = generateSampleImage(presetKey, 480, 480);
  const img = new Image();
  img.onload = () => {
    state.rawImage = img;
    processAndRender();
  };
  img.src = dataUrl;
}

/**
 * Loads an image from a user File object
 */
function loadUserFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      // Scale down large images to max 640px for ultra-smooth 60fps interaction
      const maxDim = 600;
      let w = img.naturalWidth;
      let h = img.naturalHeight;
      if (w > maxDim || h > maxDim) {
        if (w > h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = w;
      tempCanvas.height = h;
      const ctx = tempCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      const scaledImg = new Image();
      scaledImg.onload = () => {
        state.rawImage = scaledImg;
        // Unset preset chips active state
        DOM.presetChips.forEach(c => c.classList.remove('active'));
        processAndRender();
      };
      scaledImg.src = tempCanvas.toDataURL();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ==========================================================================
// Image Processing & Rendering Pipeline
// ==========================================================================

let renderPending = false;

function scheduleRender() {
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {
    processAndRender();
    renderPending = false;
  });
}

function processAndRender() {
  if (!state.rawImage) return;

  const w = state.rawImage.width;
  const h = state.rawImage.height;

  // 1. Prepare Base ImageData from rawImage
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(state.rawImage, 0, 0, w, h);
  let baseData = tempCtx.getImageData(0, 0, w, h);

  // In Phase 4 or if noise is toggled, inject synthetic noise into base
  if (state.activeTab === 'phase4' && state.p4.addNoise) {
    baseData = addSaltAndPepperNoise(baseData, state.p4.noiseDensity);
  }
  state.baseImageData = baseData;

  // 2. Compute Filter Result based on active Tab
  let resultData = null;
  let isMultiGridView = false;

  if (state.activeTab === 'phase1') {
    resultData = applyPointOps(baseData, state.p1);
  } else if (state.activeTab === 'phase2') {
    if (state.p2.type === 'gaussian') {
      resultData = applyGaussianBlur(baseData, state.p2.kernelSize, state.p2.sigma);
    } else if (state.p2.type === 'box') {
      resultData = applyBoxBlur(baseData, state.p2.kernelSize);
    } else if (state.p2.type === 'sharpen') {
      resultData = applyLaplacianSharpen(baseData, state.p2.sharpenStrength, state.p2.neighbor);
    }
  } else if (state.activeTab === 'phase3') {
    if (state.p3.type === 'sobel') {
      resultData = applySobel(baseData, state.p3.direction);
    } else if (state.p3.type === 'prewitt') {
      resultData = applyPrewitt(baseData, state.p3.direction);
    } else if (state.p3.type === 'custom') {
      resultData = convolve2D(baseData, state.p3.customMatrix, state.p3.customDivisor, state.p3.customBias);
    }
  } else if (state.activeTab === 'phase4') {
    if (state.p4.type === 'median') {
      resultData = applyMedianFilter(baseData, state.p4.medianSize);
    } else if (state.p4.type === 'bilateral') {
      resultData = applyBilateralFilter(
        baseData,
        state.p4.bilateralDiameter,
        state.p4.bilateralColor,
        state.p4.bilateralSpace
      );
    }
  } else if (state.activeTab === 'phase5') {
    const cannyRes = runCannyPipeline(baseData, {
      sigma: state.p5.sigma,
      kernelSize: state.p5.kernelSize,
      lowRatio: state.p5.lowRatio,
      highRatio: state.p5.highRatio,
      directThresholdMode: state.p5.threshMode === 'absolute',
      lowThresh: state.p5.absLow,
      highThresh: state.p5.absHigh
    });

    if (state.p5.step === 'all_grid') {
      isMultiGridView = true;
      renderCannyAllGrid(cannyRes);
    } else if (state.p5.step === 'step1_blur') {
      resultData = cannyRes.blurred;
    } else if (state.p5.step === 'step2_grad') {
      resultData = cannyRes.magnitude;
    } else if (state.p5.step === 'step3_nms') {
      resultData = cannyRes.nms;
    } else if (state.p5.step === 'step4_thresh') {
      resultData = cannyRes.doubleThreshold;
    } else {
      resultData = cannyRes.final;
    }
  } else if (state.activeTab === 'comparison') {
    isMultiGridView = true;
    renderComparisonGrid(baseData);
  }

  state.resultImageData = resultData;

  // 3. Render Views
  if (isMultiGridView) {
    DOM.splitWrapper.style.display = 'none';
    DOM.multiGrid.classList.add('active');
  } else {
    DOM.multiGrid.classList.remove('active');
    DOM.splitWrapper.style.display = 'flex';
    renderMainCanvases(baseData, resultData);
  }

  // 4. Update Theory and Code Snippets
  updateTheoryAndCode();
}

/**
 * Renders the Before and After Canvases
 */
function renderMainCanvases(baseData, resultData) {
  const w = baseData.width;
  const h = baseData.height;

  // Configure Canvas Dimensions
  DOM.canvasBefore.width = w;
  DOM.canvasBefore.height = h;
  DOM.canvasAfter.width = w;
  DOM.canvasAfter.height = h;

  const ctxBefore = DOM.canvasBefore.getContext('2d');
  const ctxAfter = DOM.canvasAfter.getContext('2d');

  ctxBefore.putImageData(baseData, 0, 0);
  ctxAfter.putImageData(resultData, 0, 0);

  updateSplitView();
}

/**
 * Updates Split Slider curtain clipping & handle positions
 */
function updateSplitView() {
  if (state.viewMode === 'sideBySide') {
    // Side by side rendering mode
    DOM.canvasBefore.style.clipPath = 'none';
    DOM.canvasBefore.style.position = 'static';
    DOM.canvasBefore.style.transform = 'none';
    DOM.canvasAfter.style.position = 'static';
    DOM.canvasAfter.style.transform = 'none';
    DOM.splitDivider.style.display = 'none';
    DOM.splitWrapper.style.display = 'grid';
    DOM.splitWrapper.style.gridTemplateColumns = '1fr 1fr';
    DOM.splitWrapper.style.gap = '10px';
    DOM.splitWrapper.style.padding = '10px';
    DOM.badgeBefore.style.display = 'block';
    DOM.badgeAfter.style.display = 'block';
    return;
  }

  // Split curtain mode
  DOM.splitWrapper.style.display = 'flex';
  DOM.splitWrapper.style.gridTemplateColumns = 'none';
  DOM.splitWrapper.style.gap = '0';
  DOM.splitWrapper.style.padding = '0';
  DOM.canvasBefore.style.position = 'absolute';
  DOM.canvasBefore.style.transform = 'translate(-50%, -50%)';
  DOM.canvasAfter.style.position = 'absolute';
  DOM.canvasAfter.style.transform = 'translate(-50%, -50%)';
  DOM.splitDivider.style.display = 'block';

  const pos = state.splitPos;
  // Clip the before canvas from 0 to pos%
  DOM.canvasBefore.style.clipPath = `polygon(0 0, ${pos}% 0, ${pos}% 100%, 0 100%)`;
  DOM.splitDivider.style.left = `${pos}%`;
}

/**
 * Multi-Grid for All 6 Canny Steps (Phase 5)
 */
function renderCannyAllGrid(cannyRes) {
  DOM.multiGrid.innerHTML = '';
  const steps = [
    { title: '0. Ảnh xám nguồn', img: cannyRes.gray },
    { title: '1. Gaussian Blur (σ=' + state.p5.sigma + ')', img: cannyRes.blurred },
    { title: '2. Sobel Gradient Magnitude', img: cannyRes.magnitude },
    { title: '3. Non-Max Suppression (NMS)', img: cannyRes.nms },
    { title: '4. Double Threshold (Hysteresis)', img: cannyRes.doubleThreshold },
    { title: '5. Canny Edges Hoàn thiện', img: cannyRes.final }
  ];

  steps.forEach(step => {
    const cell = document.createElement('div');
    cell.className = 'grid-cell';

    const header = document.createElement('div');
    header.className = 'grid-cell-header';
    header.textContent = step.title;

    const canvas = document.createElement('canvas');
    canvas.className = 'grid-canvas';
    canvas.width = step.img.width;
    canvas.height = step.img.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(step.img, 0, 0);

    cell.appendChild(header);
    cell.appendChild(canvas);
    DOM.multiGrid.appendChild(cell);
  });
}

/**
 * Multi-Grid for Comparison Studio
 */
function renderComparisonGrid(baseData) {
  DOM.multiGrid.innerHTML = '';
  const k = state.comp.kernelSize;

  const filters = [
    { title: 'Ảnh gốc (Original)', data: baseData },
    { title: `Gaussian Blur (${k}x${k}, σ=1.5)`, data: applyGaussianBlur(baseData, k, 1.5) },
    { title: `Median Filter (${k}x${k})`, data: applyMedianFilter(baseData, k) },
    { title: `Sobel Magnitude Edge`, data: applySobel(baseData, 'magnitude') },
    { title: `Laplacian Sharpen`, data: applyLaplacianSharpen(baseData, 1.0, '8') },
    {
      title: `Canny Edge Detection`,
      data: runCannyPipeline(baseData, { sigma: 1.4, kernelSize: 5, lowRatio: 0.05, highRatio: 0.15 }).final
    }
  ];

  filters.forEach(item => {
    const cell = document.createElement('div');
    cell.className = 'grid-cell';

    const header = document.createElement('div');
    header.className = 'grid-cell-header';
    header.textContent = item.title;

    const canvas = document.createElement('canvas');
    canvas.className = 'grid-canvas';
    canvas.width = item.data.width;
    canvas.height = item.data.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(item.data, 0, 0);

    cell.appendChild(header);
    cell.appendChild(canvas);
    DOM.multiGrid.appendChild(cell);
  });
}

// ==========================================================================
// Custom Kernel 3x3 Matrix Grid Builder
// ==========================================================================

function buildCustomKernelGrid() {
  DOM.kernelGrid.innerHTML = '';
  const matrix = state.p3.customMatrix;

  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const input = document.createElement('input');
      input.type = 'number';
      input.className = 'kernel-cell';
      input.value = matrix[r][c];
      input.dataset.row = r;
      input.dataset.col = c;

      input.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value) || 0;
        state.p3.customMatrix[r][c] = val;
        scheduleRender();
      });

      DOM.kernelGrid.appendChild(input);
    }
  }
}

function updateCustomKernelUI() {
  const cells = DOM.kernelGrid.querySelectorAll('.kernel-cell');
  cells.forEach(cell => {
    const r = parseInt(cell.dataset.row);
    const c = parseInt(cell.dataset.col);
    cell.value = state.p3.customMatrix[r][c];
  });
  DOM.inputDivisor.value = state.p3.customDivisor;
  DOM.inputBias.value = state.p3.customBias;
}

// ==========================================================================
// Theory & Code Snippets Synchronization
// ==========================================================================

function updateTheoryAndCode() {
  const theory = THEORY_AND_CODE[state.activeTab] || THEORY_AND_CODE.phase1;
  DOM.mathContent.textContent = theory.math;
  DOM.pythonCodeContent.textContent = theory.python;
}

// ==========================================================================
// Event Listeners & User Interactions
// ==========================================================================

function setupEventListeners() {
  // --- Navigation Tabs ---
  DOM.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;
      state.activeTab = targetTab;

      DOM.tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      DOM.phasePanels.forEach(p => {
        p.style.display = p.id === `panel-${targetTab}` ? 'flex' : 'none';
      });

      scheduleRender();
    });
  });

  // --- View Mode Toggles ---
  DOM.modeSplit.addEventListener('click', () => {
    state.viewMode = 'split';
    DOM.modeSplit.classList.add('active');
    DOM.modeSideBySide.classList.remove('active');
    updateSplitView();
  });

  DOM.modeSideBySide.addEventListener('click', () => {
    state.viewMode = 'sideBySide';
    DOM.modeSideBySide.classList.add('active');
    DOM.modeSplit.classList.remove('active');
    updateSplitView();
  });

  // --- Split Slider Dragging Interaction ---
  const onSplitMove = (clientX) => {
    if (!state.isDraggingSplit || state.viewMode !== 'split') return;
    const rect = DOM.stageContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    let pct = (x / rect.width) * 100;
    pct = Math.min(95, Math.max(5, pct));
    state.splitPos = pct;
    updateSplitView();
  };

  DOM.splitDivider.addEventListener('mousedown', (e) => {
    state.isDraggingSplit = true;
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => onSplitMove(e.clientX));
  window.addEventListener('mouseup', () => { state.isDraggingSplit = false; });

  // Touch support for mobile/tablets
  DOM.splitDivider.addEventListener('touchstart', (e) => {
    state.isDraggingSplit = true;
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (e.touches && e.touches.length > 0) {
      onSplitMove(e.touches[0].clientX);
    }
  }, { passive: true });

  window.addEventListener('touchend', () => { state.isDraggingSplit = false; });

  // --- Preset Image Chips ---
  DOM.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      DOM.presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      loadSampleImage(chip.dataset.preset);
    });
  });

  // --- Upload & File Chooser ---
  DOM.btnUpload.addEventListener('click', () => DOM.fileInput.click());
  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      loadUserFile(e.target.files[0]);
    }
  });

  // --- Drag and Drop onto Stage ---
  ['dragenter', 'dragover'].forEach(eventName => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropOverlay.classList.add('active');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropOverlay.classList.remove('active');
    });
  });

  window.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadUserFile(e.dataTransfer.files[0]);
    }
  });

  // --- Clipboard Paste (Ctrl+V) ---
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        loadUserFile(file);
        break;
      }
    }
  });

  // --- Reset & Download Actions ---
  DOM.btnReset.addEventListener('click', () => {
    const activeChip = document.querySelector('.preset-chip.active');
    const preset = activeChip ? activeChip.dataset.preset : 'geometric';
    loadSampleImage(preset);
  });

  DOM.btnDownload.addEventListener('click', () => {
    if (!state.resultImageData) return;
    const canvas = document.createElement('canvas');
    canvas.width = state.resultImageData.width;
    canvas.height = state.resultImageData.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(state.resultImageData, 0, 0);

    const link = document.createElement('a');
    link.download = `lab2_processed_${state.activeTab}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  });

  // --- Copy Code Button ---
  DOM.btnCopyCode.addEventListener('click', () => {
    const code = DOM.pythonCodeContent.textContent;
    navigator.clipboard.writeText(code).then(() => {
      DOM.btnCopyCode.textContent = 'Copied! ✓';
      setTimeout(() => { DOM.btnCopyCode.textContent = 'Copy code'; }, 2000);
    });
  });

  // ========================================================================
  // Phase 1 Controls Events
  // ========================================================================
  DOM.sliderBrightness.addEventListener('input', (e) => {
    state.p1.brightness = parseInt(e.target.value);
    DOM.valBrightness.textContent = state.p1.brightness > 0 ? `+${state.p1.brightness}` : state.p1.brightness;
    scheduleRender();
  });

  DOM.sliderContrast.addEventListener('input', (e) => {
    state.p1.contrast = parseFloat(e.target.value);
    DOM.valContrast.textContent = `${state.p1.contrast.toFixed(1)}x`;
    scheduleRender();
  });

  DOM.checkNegative.addEventListener('change', (e) => {
    state.p1.negative = e.target.checked;
    scheduleRender();
  });

  DOM.checkThreshold.addEventListener('change', (e) => {
    state.p1.thresholdEnabled = e.target.checked;
    DOM.thresholdControls.style.opacity = state.p1.thresholdEnabled ? '1' : '0.5';
    DOM.thresholdControls.style.pointerEvents = state.p1.thresholdEnabled ? 'auto' : 'none';
    scheduleRender();
  });

  DOM.sliderThreshold.addEventListener('input', (e) => {
    state.p1.thresholdValue = parseInt(e.target.value);
    DOM.valThreshold.textContent = state.p1.thresholdValue;
    scheduleRender();
  });

  DOM.checkInvertBinary.addEventListener('change', (e) => {
    state.p1.invertBinary = e.target.checked;
    scheduleRender();
  });

  // ========================================================================
  // Phase 2 Controls Events
  // ========================================================================
  DOM.linearTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.linearTypeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.p2.type = btn.dataset.linear;

      if (state.p2.type === 'sharpen') {
        DOM.groupKernelSize.style.display = 'none';
        DOM.groupSigma.style.display = 'none';
        DOM.groupSharpen.style.display = 'flex';
      } else if (state.p2.type === 'box') {
        DOM.groupKernelSize.style.display = 'flex';
        DOM.groupSigma.style.display = 'none';
        DOM.groupSharpen.style.display = 'none';
      } else {
        DOM.groupKernelSize.style.display = 'flex';
        DOM.groupSigma.style.display = 'flex';
        DOM.groupSharpen.style.display = 'none';
      }
      scheduleRender();
    });
  });

  DOM.sliderKernelSize.addEventListener('input', (e) => {
    state.p2.kernelSize = parseInt(e.target.value);
    DOM.valKernelSize.textContent = `${state.p2.kernelSize}x${state.p2.kernelSize}`;
    scheduleRender();
  });

  DOM.sliderSigma.addEventListener('input', (e) => {
    state.p2.sigma = parseFloat(e.target.value);
    DOM.valSigma.textContent = state.p2.sigma.toFixed(1);
    scheduleRender();
  });

  DOM.sliderSharpenStrength.addEventListener('input', (e) => {
    state.p2.sharpenStrength = parseFloat(e.target.value);
    DOM.valSharpenStrength.textContent = state.p2.sharpenStrength.toFixed(1);
    scheduleRender();
  });

  DOM.neighborBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.neighborBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.p2.neighbor = btn.dataset.neighbor;
      scheduleRender();
    });
  });

  // ========================================================================
  // Phase 3 Controls Events
  // ========================================================================
  DOM.edgeTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.edgeTypeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.p3.type = btn.dataset.edge;

      if (state.p3.type === 'custom') {
        DOM.groupEdgeDirection.style.display = 'none';
        DOM.groupCustomKernel.style.display = 'flex';
      } else {
        DOM.groupEdgeDirection.style.display = 'flex';
        DOM.groupCustomKernel.style.display = 'none';
      }
      scheduleRender();
    });
  });

  DOM.dirBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.dirBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.p3.direction = btn.dataset.dir;
      scheduleRender();
    });
  });

  DOM.selectKernelPreset.addEventListener('change', (e) => {
    const key = e.target.value;
    const preset = KERNEL_PRESETS['3x3'][key];
    if (preset) {
      state.p3.customMatrix = preset.matrix.map(row => [...row]);
      state.p3.customDivisor = preset.divisor;
      state.p3.customBias = preset.bias;
      updateCustomKernelUI();
      scheduleRender();
    }
  });

  DOM.inputDivisor.addEventListener('input', (e) => {
    state.p3.customDivisor = parseFloat(e.target.value) || 1;
    scheduleRender();
  });

  DOM.inputBias.addEventListener('input', (e) => {
    state.p3.customBias = parseFloat(e.target.value) || 0;
    scheduleRender();
  });

  // ========================================================================
  // Phase 4 Controls Events
  // ========================================================================
  DOM.nonlinearTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.nonlinearTypeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.p4.type = btn.dataset.nonlinear;

      if (state.p4.type === 'bilateral') {
        DOM.groupMedian.style.display = 'none';
        DOM.groupBilateral.style.display = 'flex';
      } else {
        DOM.groupMedian.style.display = 'flex';
        DOM.groupBilateral.style.display = 'none';
      }
      scheduleRender();
    });
  });

  DOM.sliderMedianSize.addEventListener('input', (e) => {
    state.p4.medianSize = parseInt(e.target.value);
    DOM.valMedianSize.textContent = `${state.p4.medianSize}x${state.p4.medianSize}`;
    scheduleRender();
  });

  DOM.sliderBilateralDiameter.addEventListener('input', (e) => {
    state.p4.bilateralDiameter = parseInt(e.target.value);
    DOM.valBilateralDiameter.textContent = state.p4.bilateralDiameter;
    scheduleRender();
  });

  DOM.sliderBilateralColor.addEventListener('input', (e) => {
    state.p4.bilateralColor = parseInt(e.target.value);
    DOM.valBilateralColor.textContent = state.p4.bilateralColor;
    scheduleRender();
  });

  DOM.sliderBilateralSpace.addEventListener('input', (e) => {
    state.p4.bilateralSpace = parseInt(e.target.value);
    DOM.valBilateralSpace.textContent = state.p4.bilateralSpace;
    scheduleRender();
  });

  DOM.checkAddNoise.addEventListener('change', (e) => {
    state.p4.addNoise = e.target.checked;
    DOM.groupNoiseDensity.style.opacity = state.p4.addNoise ? '1' : '0.5';
    DOM.groupNoiseDensity.style.pointerEvents = state.p4.addNoise ? 'auto' : 'none';
    scheduleRender();
  });

  DOM.sliderNoiseDensity.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    state.p4.noiseDensity = val / 100;
    DOM.valNoiseDensity.textContent = `${val}%`;
    scheduleRender();
  });

  // ========================================================================
  // Phase 5 Controls Events (Canny)
  // ========================================================================
  DOM.selectCannyStep.addEventListener('change', (e) => {
    state.p5.step = e.target.value;
    scheduleRender();
  });

  // Threshold Mode Switching (Ratio vs Absolute OpenCV)
  DOM.btnThreshModeRatio.addEventListener('click', () => {
    state.p5.threshMode = 'ratio';
    DOM.btnThreshModeRatio.classList.add('active');
    DOM.btnThreshModeAbs.classList.remove('active');

    DOM.labelCannyLow.textContent = 'Ngưỡng thấp (Low Ratio: 0% - 100%)';
    DOM.labelCannyHigh.textContent = 'Ngưỡng cao (High Ratio: 0% - 100%)';

    DOM.sliderCannyLowRatio.min = '0.0';
    DOM.sliderCannyLowRatio.max = '1.0';
    DOM.sliderCannyLowRatio.step = '0.01';
    DOM.sliderCannyLowRatio.value = state.p5.lowRatio;
    DOM.valCannyLowRatio.textContent = state.p5.lowRatio.toFixed(2);

    DOM.sliderCannyHighRatio.min = '0.0';
    DOM.sliderCannyHighRatio.max = '1.0';
    DOM.sliderCannyHighRatio.step = '0.01';
    DOM.sliderCannyHighRatio.value = state.p5.highRatio;
    DOM.valCannyHighRatio.textContent = state.p5.highRatio.toFixed(2);

    scheduleRender();
  });

  DOM.btnThreshModeAbs.addEventListener('click', () => {
    state.p5.threshMode = 'absolute';
    DOM.btnThreshModeAbs.classList.add('active');
    DOM.btnThreshModeRatio.classList.remove('active');

    DOM.labelCannyLow.textContent = 'Ngưỡng thấp (OpenCV Low: 0 - 255)';
    DOM.labelCannyHigh.textContent = 'Ngưỡng cao (OpenCV High: 0 - 255)';

    DOM.sliderCannyLowRatio.min = '0';
    DOM.sliderCannyLowRatio.max = '255';
    DOM.sliderCannyLowRatio.step = '1';
    DOM.sliderCannyLowRatio.value = state.p5.absLow;
    DOM.valCannyLowRatio.textContent = state.p5.absLow;

    DOM.sliderCannyHighRatio.min = '0';
    DOM.sliderCannyHighRatio.max = '255';
    DOM.sliderCannyHighRatio.step = '1';
    DOM.sliderCannyHighRatio.value = state.p5.absHigh;
    DOM.valCannyHighRatio.textContent = state.p5.absHigh;

    scheduleRender();
  });

  DOM.sliderCannySigma.addEventListener('input', (e) => {
    state.p5.sigma = parseFloat(e.target.value);
    DOM.valCannySigma.textContent = state.p5.sigma.toFixed(1);
    scheduleRender();
  });

  DOM.sliderCannyLowRatio.addEventListener('input', (e) => {
    if (state.p5.threshMode === 'ratio') {
      state.p5.lowRatio = parseFloat(e.target.value);
      DOM.valCannyLowRatio.textContent = `${state.p5.lowRatio.toFixed(2)} (${(state.p5.lowRatio * 100).toFixed(0)}%)`;
    } else {
      state.p5.absLow = parseInt(e.target.value);
      DOM.valCannyLowRatio.textContent = state.p5.absLow;
    }
    scheduleRender();
  });

  DOM.sliderCannyHighRatio.addEventListener('input', (e) => {
    if (state.p5.threshMode === 'ratio') {
      state.p5.highRatio = parseFloat(e.target.value);
      DOM.valCannyHighRatio.textContent = `${state.p5.highRatio.toFixed(2)} (${(state.p5.highRatio * 100).toFixed(0)}%)`;
    } else {
      state.p5.absHigh = parseInt(e.target.value);
      DOM.valCannyHighRatio.textContent = state.p5.absHigh;
    }
    scheduleRender();
  });

  DOM.cannyPresetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.cannyPresetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const p = btn.dataset.cannypreset;
      if (p === 'standard') {
        state.p5.sigma = 1.4;
        state.p5.lowRatio = 0.05;
        state.p5.highRatio = 0.15;
        state.p5.absLow = 50;
        state.p5.absHigh = 150;
      } else if (p === 'noisy') {
        state.p5.sigma = 2.4;
        state.p5.lowRatio = 0.08;
        state.p5.highRatio = 0.25;
        state.p5.absLow = 80;
        state.p5.absHigh = 200;
      } else if (p === 'low_contrast') {
        state.p5.sigma = 1.0;
        state.p5.lowRatio = 0.02;
        state.p5.highRatio = 0.08;
        state.p5.absLow = 20;
        state.p5.absHigh = 60;
      } else if (p === 'fine_detail') {
        state.p5.sigma = 0.8;
        state.p5.lowRatio = 0.04;
        state.p5.highRatio = 0.12;
        state.p5.absLow = 30;
        state.p5.absHigh = 100;
      }

      DOM.sliderCannySigma.value = state.p5.sigma;
      DOM.valCannySigma.textContent = state.p5.sigma.toFixed(1);

      if (state.p5.threshMode === 'ratio') {
        DOM.sliderCannyLowRatio.value = state.p5.lowRatio;
        DOM.valCannyLowRatio.textContent = state.p5.lowRatio.toFixed(2);
        DOM.sliderCannyHighRatio.value = state.p5.highRatio;
        DOM.valCannyHighRatio.textContent = state.p5.highRatio.toFixed(2);
      } else {
        DOM.sliderCannyLowRatio.value = state.p5.absLow;
        DOM.valCannyLowRatio.textContent = state.p5.absLow;
        DOM.sliderCannyHighRatio.value = state.p5.absHigh;
        DOM.valCannyHighRatio.textContent = state.p5.absHigh;
      }
      scheduleRender();
    });
  });

  // ========================================================================
  // Comparison Studio Events
  // ========================================================================
  DOM.compKernelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.compKernelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.comp.kernelSize = parseInt(btn.dataset.compk);
      scheduleRender();
    });
  });
}

// Start application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
