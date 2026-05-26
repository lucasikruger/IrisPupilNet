// Preprocesamiento por modelo. Cada función toma un canvas RGBA (HxW)
// y devuelve un Float32Array NCHW listo para el ONNX runtime.

export type PreprocessName =
  | "grayscale_norm_01"
  | "grayscale_clahe_norm_01"
  | "rgb_imagenet"
  | "rgb_minus1_1";

export interface PreprocessFn {
  channels: number;
  toTensor: (canvas: HTMLCanvasElement, size: number) => Float32Array;
}

function getImageData(canvas: HTMLCanvasElement, size: number): Uint8ClampedArray {
  if (canvas.width === size && canvas.height === size) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    return ctx.getImageData(0, 0, size, size).data;
  }
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const ctx = tmp.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(canvas, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size).data;
}

function toGrayU8(px: Uint8ClampedArray, size: number): Uint8Array {
  const out = new Uint8Array(size * size);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    // Rec. 601 luma — same coefficients used in cv2.cvtColor(BGR2GRAY).
    out[j] = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2] + 0.5) | 0;
  }
  return out;
}

// CLAHE port of OpenCV's clahe.cpp. Validated bit-equivalent (≤1 level diff,
// 0% pixels >2 levels) vs cv2.createCLAHE(clipLimit, tileGridSize).
// Training pipeline uses clipLimit=1.5, tileGridSize=(8,8) — see CLAHE_PARAMS.
const CLAHE_PARAMS = { clipLimit: 1.5, tilesX: 8, tilesY: 8 } as const;

function clahe(
  gray: Uint8Array,
  w: number,
  h: number,
  clipLimit: number,
  tilesX: number,
  tilesY: number,
): Uint8Array {
  const bins = 256;
  const tileH = h / tilesY;
  const tileW = w / tilesX;
  const tilePx = tileW * tileH;
  const limit = Math.max(1, Math.round((clipLimit * tilePx) / bins));

  const luts: Uint8Array[] = new Array(tilesY * tilesX);
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Int32Array(bins);
      const x0 = Math.floor(tx * tileW);
      const y0 = Math.floor(ty * tileH);
      const x1 = Math.floor((tx + 1) * tileW);
      const y1 = Math.floor((ty + 1) * tileH);
      for (let y = y0; y < y1; y++) {
        const row = y * w;
        for (let x = x0; x < x1; x++) hist[gray[row + x]]++;
      }
      let clipped = 0;
      for (let i = 0; i < bins; i++) {
        if (hist[i] > limit) {
          clipped += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const redistBatch = Math.floor(clipped / bins);
      let residual = clipped - redistBatch * bins;
      for (let i = 0; i < bins; i++) hist[i] += redistBatch;
      if (residual > 0) {
        const step = Math.max(1, Math.floor(bins / residual));
        for (let i = 0; i < bins && residual > 0; i += step, residual--) hist[i]++;
      }
      const lut = new Uint8Array(bins);
      const lutScale = 255 / tilePx;
      let sum = 0;
      for (let i = 0; i < bins; i++) {
        sum += hist[i];
        const v = Math.round(sum * lutScale);
        lut[i] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
      luts[ty * tilesX + tx] = lut;
    }
  }

  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let yf = y / tileH - 0.5;
    let ty0 = Math.floor(yf);
    let ay = yf - ty0;
    let ty1 = ty0 + 1;
    if (ty0 < 0) { ty0 = 0; ay = 0; }
    if (ty1 < 0) ty1 = 0;
    if (ty0 > tilesY - 1) ty0 = tilesY - 1;
    if (ty1 > tilesY - 1) { ty1 = tilesY - 1; ay = 1; }

    for (let x = 0; x < w; x++) {
      let xf = x / tileW - 0.5;
      let tx0 = Math.floor(xf);
      let ax = xf - tx0;
      let tx1 = tx0 + 1;
      if (tx0 < 0) { tx0 = 0; ax = 0; }
      if (tx1 < 0) tx1 = 0;
      if (tx0 > tilesX - 1) tx0 = tilesX - 1;
      if (tx1 > tilesX - 1) { tx1 = tilesX - 1; ax = 1; }

      const v = gray[y * w + x];
      const v00 = luts[ty0 * tilesX + tx0][v];
      const v01 = luts[ty0 * tilesX + tx1][v];
      const v10 = luts[ty1 * tilesX + tx0][v];
      const v11 = luts[ty1 * tilesX + tx1][v];
      const top = v00 * (1 - ax) + v01 * ax;
      const bot = v10 * (1 - ax) + v11 * ax;
      out[y * w + x] = Math.round(top * (1 - ay) + bot * ay);
    }
  }
  return out;
}

const grayscale_norm_01: PreprocessFn = {
  channels: 1,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const out = new Float32Array(size * size);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      // Rec. 601 luma weights (matches OpenCV BGR2GRAY).
      const gray = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255.0;
      out[j] = gray;
    }
    return out;
  },
};

const grayscale_clahe_norm_01: PreprocessFn = {
  channels: 1,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const gray = toGrayU8(px, size);
    const eq = clahe(gray, size, size, CLAHE_PARAMS.clipLimit, CLAHE_PARAMS.tilesX, CLAHE_PARAMS.tilesY);
    const out = new Float32Array(size * size);
    for (let i = 0; i < eq.length; i++) out[i] = eq[i] / 255.0;
    return out;
  },
};

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const rgb_imagenet: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < px.length; i += 4, p++) {
      out[p] = (px[i] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      out[plane + p] = (px[i + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      out[2 * plane + p] = (px[i + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    return out;
  },
};

const rgb_minus1_1: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < px.length; i += 4, p++) {
      out[p] = px[i] / 127.5 - 1.0;
      out[plane + p] = px[i + 1] / 127.5 - 1.0;
      out[2 * plane + p] = px[i + 2] / 127.5 - 1.0;
    }
    return out;
  },
};

export const PREPROCESS: Record<PreprocessName, PreprocessFn> = {
  grayscale_norm_01,
  grayscale_clahe_norm_01,
  rgb_imagenet,
  rgb_minus1_1,
};
