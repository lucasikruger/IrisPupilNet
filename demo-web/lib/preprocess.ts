// Preprocesamiento por modelo. Cada función toma un canvas RGBA (HxW)
// y devuelve un Float32Array NCHW listo para el ONNX runtime.

export type PreprocessName =
  | "grayscale_norm_01"
  | "grayscale_clahe_norm_01"
  | "grayscale_clahe_gamma_01"
  | "rgb_clahe_lab_gamma_01"
  | "rgb_imagenet"
  | "rgb_minus1_1";

export interface PreprocessFn {
  channels: number;
  toTensor: (canvas: HTMLCanvasElement, size: number) => Float32Array;
  /** Visual preview of what the model receives (for "preprocessed" view).
   *  Returns an RGBA ImageData of size×size. Optional. */
  toPreview?: (canvas: HTMLCanvasElement, size: number) => ImageData;
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

// Gamma LUT — matches training: out = (in/255) ** (1/gamma) * 255, gamma=0.8.
function gammaLut(gamma: number): Uint8Array {
  const inv = 1.0 / gamma;
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    const v = Math.round(Math.pow(i / 255, inv) * 255);
    lut[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return lut;
}
const GAMMA_08_LUT = gammaLut(0.8);

// ---- sRGB <-> CIE-LAB (D65) — matches cv2.cvtColor(BGR2LAB)/(LAB2RGB) for
// 8-bit images. cv2 stores L in [0, 255] (L*255/100), a/b in [0, 255] (a+128, b+128).
// We work in the same convention so we can run CLAHE on cv2-style L directly.
//
// Pipeline (per pixel):
//   sRGB-uint8 → linear RGB → XYZ → LAB (cv2 8-bit packing)
//   CLAHE on L only
//   LAB → XYZ → linear RGB → sRGB-uint8

// sRGB gamma (non-linear → linear)
function srgbToLinear(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function linearToSrgb(x: number): number {
  let y = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  y = y * 255;
  return y < 0 ? 0 : y > 255 ? 255 : Math.round(y);
}
// LAB f / f^-1
const LAB_EPS = 216 / 24389;
const LAB_KAPPA = 24389 / 27;
function labF(t: number): number {
  return t > LAB_EPS ? Math.cbrt(t) : (LAB_KAPPA * t + 16) / 116;
}
function labFInv(t: number): number {
  const t3 = t * t * t;
  return t3 > LAB_EPS ? t3 : (116 * t - 16) / LAB_KAPPA;
}
// D65 reference white (Xn, Yn, Zn)
const D65_Xn = 0.95047;
const D65_Yn = 1.0;
const D65_Zn = 1.08883;

function rgbU8ToLabPacked(
  px: Uint8ClampedArray,
  size: number,
): { L: Uint8Array; a: Uint8Array; b: Uint8Array } {
  const N = size * size;
  const L = new Uint8Array(N);
  const a = new Uint8Array(N);
  const b = new Uint8Array(N);
  for (let i = 0, j = 0; i < px.length; i += 4, j++) {
    const r = srgbToLinear(px[i]);
    const g = srgbToLinear(px[i + 1]);
    const bl = srgbToLinear(px[i + 2]);
    // sRGB → XYZ (D65)
    const X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * bl) / D65_Xn;
    const Y = (0.2126729 * r + 0.7151522 * g + 0.072175 * bl) / D65_Yn;
    const Z = (0.0193339 * r + 0.119192 * g + 0.9503041 * bl) / D65_Zn;
    const fx = labF(X);
    const fy = labF(Y);
    const fz = labF(Z);
    const Lf = 116 * fy - 16;          // 0..100
    const af = 500 * (fx - fy);        // ~-128..127
    const bf = 200 * (fy - fz);        // ~-128..127
    // cv2 8-bit packing: L*255/100, a+128, b+128, clamped
    const Lp = Math.round((Lf * 255) / 100);
    const ap = Math.round(af + 128);
    const bp = Math.round(bf + 128);
    L[j] = Lp < 0 ? 0 : Lp > 255 ? 255 : Lp;
    a[j] = ap < 0 ? 0 : ap > 255 ? 255 : ap;
    b[j] = bp < 0 ? 0 : bp > 255 ? 255 : bp;
  }
  return { L, a, b };
}

function labPackedToRgbU8(
  L: Uint8Array,
  a: Uint8Array,
  b: Uint8Array,
  size: number,
): Uint8ClampedArray {
  const N = size * size;
  const out = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const Lf = (L[i] * 100) / 255;
    const af = a[i] - 128;
    const bf = b[i] - 128;
    const fy = (Lf + 16) / 116;
    const fx = af / 500 + fy;
    const fz = fy - bf / 200;
    const X = labFInv(fx) * D65_Xn;
    const Y = labFInv(fy) * D65_Yn;
    const Z = labFInv(fz) * D65_Zn;
    // XYZ → linear RGB
    const r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    const g = -0.969266 * X + 1.8760108 * Y + 0.041556 * Z;
    const bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    const j = i * 4;
    out[j] = linearToSrgb(r);
    out[j + 1] = linearToSrgb(g);
    out[j + 2] = linearToSrgb(bl);
    out[j + 3] = 255;
  }
  return out;
}

// Same RGB CLAHE+gamma pipeline as src/augmentations.py:preprocess(grayscale=False).
// Returns RGBA Uint8ClampedArray ready for ImageData (also used for the
// "preprocessed view" preview).
function preprocessRgbClaheGamma(px: Uint8ClampedArray, size: number): Uint8ClampedArray {
  const { L, a, b } = rgbU8ToLabPacked(px, size);
  const Leq = clahe(L, size, size, CLAHE_PARAMS.clipLimit, CLAHE_PARAMS.tilesX, CLAHE_PARAMS.tilesY);
  const rgba = labPackedToRgbU8(Leq, a, b, size);
  // Gamma 0.8 on the final RGB
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = GAMMA_08_LUT[rgba[i]];
    rgba[i + 1] = GAMMA_08_LUT[rgba[i + 1]];
    rgba[i + 2] = GAMMA_08_LUT[rgba[i + 2]];
  }
  return rgba;
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
  toPreview: (canvas, size) => {
    const px = getImageData(canvas, size);
    const gray = toGrayU8(px, size);
    const eq = clahe(gray, size, size, CLAHE_PARAMS.clipLimit, CLAHE_PARAMS.tilesX, CLAHE_PARAMS.tilesY);
    const rgba = new Uint8ClampedArray(size * size * 4);
    for (let i = 0, j = 0; i < eq.length; i++, j += 4) {
      rgba[j] = rgba[j + 1] = rgba[j + 2] = eq[i];
      rgba[j + 3] = 255;
    }
    return new ImageData(rgba, size, size);
  },
};

// Grayscale + CLAHE + gamma 0.8 — matches src/augmentations.py:preprocess(grayscale=True).
const grayscale_clahe_gamma_01: PreprocessFn = {
  channels: 1,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const gray = toGrayU8(px, size);
    const eq = clahe(gray, size, size, CLAHE_PARAMS.clipLimit, CLAHE_PARAMS.tilesX, CLAHE_PARAMS.tilesY);
    const out = new Float32Array(size * size);
    for (let i = 0; i < eq.length; i++) out[i] = GAMMA_08_LUT[eq[i]] / 255.0;
    return out;
  },
  toPreview: (canvas, size) => {
    const px = getImageData(canvas, size);
    const gray = toGrayU8(px, size);
    const eq = clahe(gray, size, size, CLAHE_PARAMS.clipLimit, CLAHE_PARAMS.tilesX, CLAHE_PARAMS.tilesY);
    const rgba = new Uint8ClampedArray(size * size * 4);
    for (let i = 0, j = 0; i < eq.length; i++, j += 4) {
      const v = GAMMA_08_LUT[eq[i]];
      rgba[j] = rgba[j + 1] = rgba[j + 2] = v;
      rgba[j + 3] = 255;
    }
    return new ImageData(rgba, size, size);
  },
};

// RGB + CLAHE on LAB.L + gamma 0.8 — matches src/augmentations.py:preprocess(grayscale=False).
// This is the preprocess for the production ritnet_in_2x__final model.
const rgb_clahe_lab_gamma_01: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const rgba = preprocessRgbClaheGamma(px, size);
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
      out[p] = rgba[i] / 255.0;             // R plane
      out[plane + p] = rgba[i + 1] / 255.0; // G plane
      out[2 * plane + p] = rgba[i + 2] / 255.0; // B plane
    }
    return out;
  },
  toPreview: (canvas, size) => {
    const px = getImageData(canvas, size);
    const rgba = preprocessRgbClaheGamma(px, size);
    return new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, size, size);
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
  grayscale_clahe_gamma_01,
  rgb_clahe_lab_gamma_01,
  rgb_imagenet,
  rgb_minus1_1,
};
