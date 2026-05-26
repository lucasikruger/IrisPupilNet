// JS port of src/postprocess.py — cleans (H,W) argmax masks (0=bg, 1=iris, 2=pupil)
// and fits anatomical ellipses.
//
// Variants ordered by aggressiveness (same names as the python registry):
//   raw                  — passthrough
//   largest_cc           — largest connected component per class + fill holes
//   morph                — largest_cc + morph close (5×5)
//   ellipse_iris         — morph + replace iris with fitted ellipse disc
//   ellipse_iris_pupil   — + fit pupil ellipse, no anatomical constraints
//   ellipse_anatomical   — + clamp pupil inside iris + Hu 2018 radius cap (0.40)
//                          + offset cap (0.30 of iris radius)

export type PostprocessName =
  | "raw"
  | "largest_cc"
  | "morph"
  | "ellipse_iris"
  | "ellipse_iris_pupil"
  | "ellipse_anatomical"
  | "ellipse_anatomical_clean";

export const POSTPROCESS_VARIANTS: PostprocessName[] = [
  "raw",
  "largest_cc",
  "morph",
  "ellipse_iris",
  "ellipse_iris_pupil",
  "ellipse_anatomical",
  "ellipse_anatomical_clean",
];

// Anatomical priors (Hu et al. 2018) — matches src/postprocess.py defaults.
export const PUPIL_TO_IRIS_RADIUS_MAX = 0.40;
export const PUPIL_CENTER_OFFSET_MAX = 0.30;

export interface PostprocessOptions {
  /** Morph close kernel for the iris ring (default 5). */
  morphKsizeIris?: number;
  /** Morph close kernel for the pupil (default 3). */
  morphKsizePupil?: number;
  /** Drop iris-class connected component if it has fewer pixels (default 0 = off). */
  minIrisPixels?: number;
  /** Drop pupil-class connected component if it has fewer pixels (default 0 = off). */
  minPupilPixels?: number;
  /** Swap class encoding 1 ↔ 2 (iris ↔ pupil) — debug toggle. */
  swapClasses?: boolean;
  /** Confidence threshold (0..1) — set to bg below this. Requires probs. */
  probThreshold?: number;
  /** Eyelid landmark points (in mask coords, after resize to `size`).
   *  Used by `ellipse_anatomical_clean` to crop iris by polynomial eyelid fit. */
  eyelidPoints?: { x: number; y: number }[];
  /** Crop pixel data at mask resolution (RGBA Uint8ClampedArray, size×size×4).
   *  Used by `ellipse_anatomical_clean` to mask specular reflections. */
  imageData?: Uint8ClampedArray;
  /** Specular-highlight percentile threshold inside iris (default 99 = top 1%). */
  specularPct?: number;
}

/** Swap classes 1↔2 in a (size×size) Uint8 mask. */
export function swapMaskClasses(mask: Uint8Array): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    out[i] = c === 1 ? 2 : c === 2 ? 1 : 0;
  }
  return out;
}

export type Ellipse = {
  cx: number;
  cy: number;
  rxMajor: number; // semi-major
  rxMinor: number; // semi-minor
  angleDeg: number; // CCW rotation of major axis from +x, degrees
};

export interface EllipsesResult {
  iris: Ellipse | null;
  pupil: Ellipse | null;
}

// ---------------------------------------------------------------------------
// Binary helpers
// ---------------------------------------------------------------------------

function asBinary(mask: Uint8Array, value: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = mask[i] === value ? 1 : 0;
  return out;
}

function anyTrue(bin: Uint8Array): boolean {
  for (let i = 0; i < bin.length; i++) if (bin[i]) return true;
  return false;
}

function orInto(dst: Uint8Array, src: Uint8Array): Uint8Array {
  for (let i = 0; i < dst.length; i++) if (src[i]) dst[i] = 1;
  return dst;
}

function copyBinary(bin: Uint8Array): Uint8Array {
  const out = new Uint8Array(bin.length);
  out.set(bin);
  return out;
}

// 8-connected largest component (in place over a label scratch buffer).
function largestCc(bin: Uint8Array, w: number, h: number): Uint8Array {
  const visited = new Uint8Array(bin.length);
  let bestSize = 0;
  let bestStart = -1;

  // Reusable BFS queue using a flat Int32Array (size up to bin.length).
  const queue = new Int32Array(bin.length);

  for (let start = 0; start < bin.length; start++) {
    if (!bin[start] || visited[start]) continue;
    let qHead = 0;
    let qTail = 0;
    queue[qTail++] = start;
    visited[start] = 1;
    let size = 0;
    while (qHead < qTail) {
      const idx = queue[qHead++];
      size++;
      const y = (idx / w) | 0;
      const x = idx - y * w;
      // 8 neighbours
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const nIdx = ny * w + nx;
          if (bin[nIdx] && !visited[nIdx]) {
            visited[nIdx] = 1;
            queue[qTail++] = nIdx;
          }
        }
      }
    }
    if (size > bestSize) {
      bestSize = size;
      bestStart = start;
    }
  }

  if (bestStart < 0) return new Uint8Array(bin.length);

  // Second pass: keep only the best component (BFS again from bestStart).
  const out = new Uint8Array(bin.length);
  const visited2 = new Uint8Array(bin.length);
  let qHead = 0;
  let qTail = 0;
  queue[qTail++] = bestStart;
  visited2[bestStart] = 1;
  while (qHead < qTail) {
    const idx = queue[qHead++];
    out[idx] = 1;
    const y = (idx / w) | 0;
    const x = idx - y * w;
    for (let dy = -1; dy <= 1; dy++) {
      const ny = y + dy;
      if (ny < 0 || ny >= h) continue;
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const nIdx = ny * w + nx;
        if (bin[nIdx] && !visited2[nIdx]) {
          visited2[nIdx] = 1;
          queue[qTail++] = nIdx;
        }
      }
    }
  }
  return out;
}

// Fill internal holes: BFS background from the border (4-conn), then anything
// not reached and originally 0 becomes 1.
function fillHoles(bin: Uint8Array, w: number, h: number): Uint8Array {
  const reached = new Uint8Array(bin.length);
  const queue = new Int32Array(bin.length);
  let qHead = 0;
  let qTail = 0;
  const push = (idx: number) => {
    if (!reached[idx] && !bin[idx]) {
      reached[idx] = 1;
      queue[qTail++] = idx;
    }
  };
  for (let x = 0; x < w; x++) {
    push(x);
    push((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    push(y * w);
    push(y * w + (w - 1));
  }
  while (qHead < qTail) {
    const idx = queue[qHead++];
    const y = (idx / w) | 0;
    const x = idx - y * w;
    if (x > 0) push(idx - 1);
    if (x < w - 1) push(idx + 1);
    if (y > 0) push(idx - w);
    if (y < h - 1) push(idx + w);
  }
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin[i] || !reached[i] ? 1 : 0;
  return out;
}

// Morph dilation 8-connected, square kernel of odd size (ksize).
function dilate(bin: Uint8Array, w: number, h: number, ksize: number): Uint8Array {
  const r = (ksize - 1) >> 1;
  // Two-pass: horizontal then vertical for separable structuring element.
  const tmp = new Uint8Array(bin.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let dx = -r; dx <= r; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        if (bin[y * w + nx]) { v = 1; break; }
      }
      tmp[y * w + x] = v;
    }
  }
  const out = new Uint8Array(bin.length);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        if (tmp[ny * w + x]) { v = 1; break; }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function erode(bin: Uint8Array, w: number, h: number, ksize: number): Uint8Array {
  const r = (ksize - 1) >> 1;
  const tmp = new Uint8Array(bin.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 1;
      for (let dx = -r; dx <= r; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) { v = 0; break; }
        if (!bin[y * w + nx]) { v = 0; break; }
      }
      tmp[y * w + x] = v;
    }
  }
  const out = new Uint8Array(bin.length);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = 1;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) { v = 0; break; }
        if (!tmp[ny * w + x]) { v = 0; break; }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function morphClose(bin: Uint8Array, w: number, h: number, ksize = 5): Uint8Array {
  return erode(dilate(bin, w, h, ksize), w, h, ksize);
}

// ---------------------------------------------------------------------------
// Ellipse fitting via image moments (PCA on filled mask).
// For a uniformly-filled ellipse with semi-axes (a, b):
//   second-central moments along principal axes are area * a^2 / 4 and area * b^2 / 4
// So semi-axes = 2 * sqrt(eigenvalue / area). Returns semi-axes (rxMajor, rxMinor).
// Stable + closed-form (just a 2×2 eigen), no need for Fitzgibbon's algebraic LS.
// ---------------------------------------------------------------------------

function fitEllipseFromFilled(bin: Uint8Array, w: number, h: number): Ellipse | null {
  // First pass: count + centroid
  let n = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x]) {
        n++;
        sx += x;
        sy += y;
      }
    }
  }
  if (n < 8) return null;
  const cx = sx / n;
  const cy = sy / n;
  // Second-central moments
  let mxx = 0;
  let myy = 0;
  let mxy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x]) {
        const dx = x - cx;
        const dy = y - cy;
        mxx += dx * dx;
        myy += dy * dy;
        mxy += dx * dy;
      }
    }
  }
  // 2×2 eigen-decomp of [[mxx, mxy], [mxy, myy]]
  const tr = mxx + myy;
  const det = mxx * myy - mxy * mxy;
  const disc = Math.max(0, tr * tr / 4 - det);
  const sq = Math.sqrt(disc);
  const l1 = tr / 2 + sq;
  const l2 = tr / 2 - sq;
  const lMax = Math.max(l1, l2);
  const lMin = Math.max(0, Math.min(l1, l2));
  // Semi-axes
  const rxMajor = 2 * Math.sqrt(lMax / n);
  const rxMinor = 2 * Math.sqrt(lMin / n);
  // Angle of major axis (eigenvector of largest eigenvalue)
  let angleRad: number;
  if (Math.abs(mxy) < 1e-9 && Math.abs(mxx - myy) < 1e-9) {
    angleRad = 0;
  } else {
    angleRad = 0.5 * Math.atan2(2 * mxy, mxx - myy);
  }
  let angleDeg = (angleRad * 180) / Math.PI;
  while (angleDeg < 0) angleDeg += 180;
  while (angleDeg >= 180) angleDeg -= 180;

  if (!isFinite(rxMajor) || !isFinite(rxMinor) || rxMajor <= 0 || rxMinor <= 0) return null;
  return { cx, cy, rxMajor, rxMinor, angleDeg };
}

// Rasterise an ellipse into a binary mask. Returns Uint8Array (size×size).
export function rasterEllipse(
  e: Ellipse,
  w: number,
  h: number,
): Uint8Array {
  const out = new Uint8Array(w * h);
  if (!isFinite(e.cx) || !isFinite(e.cy) || !isFinite(e.rxMajor) || !isFinite(e.rxMinor)) return out;
  if (e.rxMajor <= 0 || e.rxMinor <= 0) return out;
  const cosT = Math.cos((-e.angleDeg * Math.PI) / 180);
  const sinT = Math.sin((-e.angleDeg * Math.PI) / 180);
  const rA = e.rxMajor;
  const rB = e.rxMinor;
  // Iterate bounding box only
  const r = Math.max(rA, rB) + 1;
  const y0 = Math.max(0, Math.floor(e.cy - r));
  const y1 = Math.min(h - 1, Math.ceil(e.cy + r));
  const x0 = Math.max(0, Math.floor(e.cx - r));
  const x1 = Math.min(w - 1, Math.ceil(e.cx + r));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - e.cx;
      const dy = y - e.cy;
      const xr = cosT * dx - sinT * dy;
      const yr = sinT * dx + cosT * dy;
      const t = (xr * xr) / (rA * rA) + (yr * yr) / (rB * rB);
      if (t <= 1) out[y * w + x] = 1;
    }
  }
  return out;
}

function decode(mask: Uint8Array): { iris: Uint8Array; pupil: Uint8Array } {
  return { iris: asBinary(mask, 1), pupil: asBinary(mask, 2) };
}

function encode(iris: Uint8Array, pupil: Uint8Array): Uint8Array {
  const out = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) {
    if (pupil[i]) out[i] = 2;
    else if (iris[i]) out[i] = 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

function vRaw(mask: Uint8Array): Uint8Array {
  return mask.slice();
}

function vLargestCc(mask: Uint8Array, w: number, h: number): Uint8Array {
  const { iris, pupil } = decode(mask);
  let irisFull = copyBinary(iris);
  orInto(irisFull, pupil);
  irisFull = largestCc(irisFull, w, h);
  irisFull = fillHoles(irisFull, w, h);
  const pupilC = largestCc(pupil, w, h);
  const irisRing = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) {
    irisRing[i] = irisFull[i] && !pupilC[i] ? 1 : 0;
  }
  return encode(irisRing, pupilC);
}

function vMorph(
  mask: Uint8Array,
  w: number,
  h: number,
  ksIris = 5,
  ksPupil = 3,
): Uint8Array {
  const cleaned = vLargestCc(mask, w, h);
  const { iris, pupil } = decode(cleaned);
  let irisFull = copyBinary(iris);
  orInto(irisFull, pupil);
  irisFull = ksIris > 1 ? morphClose(irisFull, w, h, ksIris) : irisFull;
  let pupilC = ksPupil > 1 ? morphClose(pupil, w, h, ksPupil) : pupil;
  // never paint pupil outside iris disc
  for (let i = 0; i < pupilC.length; i++) pupilC[i] = pupilC[i] && irisFull[i] ? 1 : 0;
  const irisRing = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) {
    irisRing[i] = irisFull[i] && !pupilC[i] ? 1 : 0;
  }
  return encode(irisRing, pupilC);
}

function vEllipseIris(mask: Uint8Array, w: number, h: number, ksIris = 5, ksPupil = 3): Uint8Array {
  const cleaned = vMorph(mask, w, h, ksIris, ksPupil);
  const { iris, pupil } = decode(cleaned);
  const irisFull = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) irisFull[i] = iris[i] || pupil[i] ? 1 : 0;
  if (!anyTrue(irisFull)) return cleaned;
  const e = fitEllipseFromFilled(irisFull, w, h);
  if (!e) return cleaned;
  const disc = rasterEllipse(e, w, h);
  const newPupil = new Uint8Array(pupil.length);
  for (let i = 0; i < pupil.length; i++) newPupil[i] = pupil[i] && disc[i] ? 1 : 0;
  const ring = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) ring[i] = disc[i] && !newPupil[i] ? 1 : 0;
  return encode(ring, newPupil);
}

function vEllipseIrisPupil(mask: Uint8Array, w: number, h: number, ksIris = 5, ksPupil = 3): Uint8Array {
  const cleaned = vEllipseIris(mask, w, h, ksIris, ksPupil);
  const { iris, pupil } = decode(cleaned);
  const disc = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) disc[i] = iris[i] || pupil[i] ? 1 : 0;
  if (!anyTrue(pupil)) return cleaned;
  const e = fitEllipseFromFilled(pupil, w, h);
  if (!e) return cleaned;
  let pupilDisc = rasterEllipse(e, w, h);
  for (let i = 0; i < pupilDisc.length; i++) pupilDisc[i] = pupilDisc[i] && disc[i] ? 1 : 0;
  const ring = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) ring[i] = disc[i] && !pupilDisc[i] ? 1 : 0;
  return encode(ring, pupilDisc);
}

// ---------------------------------------------------------------------------
// Open-iris–inspired refinements (eyelid polynomial mask + specular masking)
// ---------------------------------------------------------------------------

/**
 * Fit y = a*x² + b*x + c to a list of (x, y) points via normal equations.
 * Returns null if fewer than 3 points or singular system.
 */
function fitQuadratic(pts: { x: number; y: number }[]): { a: number; b: number; c: number } | null {
  if (pts.length < 3) return null;
  let sx0 = 0, sx1 = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  let sy0 = 0, syx1 = 0, syx2 = 0;
  for (const p of pts) {
    const x = p.x, y = p.y;
    const x2 = x * x;
    sx0 += 1;
    sx1 += x;
    sx2 += x2;
    sx3 += x2 * x;
    sx4 += x2 * x2;
    sy0 += y;
    syx1 += y * x;
    syx2 += y * x2;
  }
  // Solve [[sx4,sx3,sx2],[sx3,sx2,sx1],[sx2,sx1,sx0]] · [a,b,c]ᵀ = [syx2,syx1,sy0]ᵀ
  // 3×3 inverse via cofactors.
  const m: number[][] = [[sx4, sx3, sx2], [sx3, sx2, sx1], [sx2, sx1, sx0]];
  const v = [syx2, syx1, sy0];
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-9) return null;
  const inv00 = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
  const inv01 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det;
  const inv02 = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
  const inv10 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / det;
  const inv11 = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
  const inv12 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / det;
  const inv20 = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
  const inv21 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / det;
  const inv22 = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
  const a = inv00 * v[0] + inv01 * v[1] + inv02 * v[2];
  const b = inv10 * v[0] + inv11 * v[1] + inv12 * v[2];
  const c = inv20 * v[0] + inv21 * v[1] + inv22 * v[2];
  return { a, b, c };
}

/**
 * Open-iris–style eyelid mask. Splits the eyelid landmarks into upper/lower by
 * comparing each point to the centroid y, fits a quadratic y(x) to each, and
 * returns a binary mask of the pixels BETWEEN the two curves (=eye opening).
 */
function eyelidOpeningMask(
  eyelidPoints: { x: number; y: number }[],
  w: number,
  h: number,
): Uint8Array | null {
  if (eyelidPoints.length < 6) return null;
  let cyMean = 0;
  for (const p of eyelidPoints) cyMean += p.y;
  cyMean /= eyelidPoints.length;
  const upper = eyelidPoints.filter((p) => p.y < cyMean);
  const lower = eyelidPoints.filter((p) => p.y >= cyMean);
  const fitU = fitQuadratic(upper);
  const fitL = fitQuadratic(lower);
  if (!fitU || !fitL) return null;
  const out = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) {
    const yU = fitU.a * x * x + fitU.b * x + fitU.c;
    const yL = fitL.a * x * x + fitL.b * x + fitL.c;
    const top = Math.max(0, Math.floor(Math.min(yU, yL)));
    const bot = Math.min(h - 1, Math.ceil(Math.max(yU, yL)));
    for (let y = top; y <= bot; y++) out[y * w + x] = 1;
  }
  return out;
}

/**
 * Compute a binary mask of specular-reflection pixels INSIDE `irisMask`.
 * Implementation: for pixels in the iris, find the threshold at
 * `specularPct`-th percentile of luminance. Mark anything above as specular.
 */
function specularMaskInIris(
  imageData: Uint8ClampedArray,
  irisMask: Uint8Array,
  w: number,
  h: number,
  specularPct: number,
): Uint8Array {
  const out = new Uint8Array(w * h);
  // Collect luminance values inside iris
  const lums: number[] = [];
  for (let i = 0, p = 0; i < imageData.length; i += 4, p++) {
    if (!irisMask[p]) continue;
    const l = (0.299 * imageData[i] + 0.587 * imageData[i + 1] + 0.114 * imageData[i + 2]) | 0;
    lums.push(l);
  }
  if (lums.length === 0) return out;
  lums.sort((a, b) => a - b);
  const idx = Math.min(lums.length - 1, Math.floor((specularPct / 100) * lums.length));
  const thr = lums[idx];
  for (let i = 0, p = 0; i < imageData.length; i += 4, p++) {
    if (!irisMask[p]) continue;
    const l = (0.299 * imageData[i] + 0.587 * imageData[i + 1] + 0.114 * imageData[i + 2]) | 0;
    if (l >= thr) out[p] = 1;
  }
  return out;
}

function vEllipseAnatomicalClean(
  mask: Uint8Array,
  w: number,
  h: number,
  ksIris: number,
  ksPupil: number,
  eyelidPoints?: { x: number; y: number }[],
  imageData?: Uint8ClampedArray,
  specularPct = 99,
): Uint8Array {
  // 1. Start from anatomical ellipse output
  let m = vEllipseAnatomical(mask, w, h, ksIris, ksPupil);

  // 2. Crop by eyelid opening (if landmarks available)
  if (eyelidPoints && eyelidPoints.length >= 6) {
    const opening = eyelidOpeningMask(eyelidPoints, w, h);
    if (opening) {
      for (let i = 0; i < m.length; i++) {
        if (!opening[i] && m[i] !== 0) m[i] = 0;
      }
    }
  }

  // 3. Specular reflection masking — pixels inside the iris with luminance in
  //    the top `100 - specularPct`% are demoted from iris/pupil to bg.
  if (imageData && imageData.length === w * h * 4) {
    const irisOrPupil = new Uint8Array(m.length);
    for (let i = 0; i < m.length; i++) irisOrPupil[i] = m[i] !== 0 ? 1 : 0;
    const spec = specularMaskInIris(imageData, irisOrPupil, w, h, specularPct);
    for (let i = 0; i < m.length; i++) if (spec[i]) m[i] = 0;
  }
  return m;
}

function vEllipseAnatomical(mask: Uint8Array, w: number, h: number, ksIris = 5, ksPupil = 3): Uint8Array {
  const cleaned = vMorph(mask, w, h, ksIris, ksPupil);
  const { iris, pupil } = decode(cleaned);
  const irisFull = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) irisFull[i] = iris[i] || pupil[i] ? 1 : 0;
  if (!anyTrue(irisFull)) return cleaned;
  const eIris = fitEllipseFromFilled(irisFull, w, h);
  if (!eIris) return cleaned;
  const irisDisc = rasterEllipse(eIris, w, h);
  if (!anyTrue(pupil)) {
    return encode(irisDisc, new Uint8Array(iris.length));
  }
  const ePupilRaw = fitEllipseFromFilled(pupil, w, h);
  if (!ePupilRaw) {
    const ring = new Uint8Array(iris.length);
    for (let i = 0; i < iris.length; i++) ring[i] = irisDisc[i] && !pupil[i] ? 1 : 0;
    return encode(ring, pupil);
  }
  // Clamp pupil ellipse: center ≤ PUPIL_CENTER_OFFSET_MAX * iris_r, axes ≤ PUPIL_TO_IRIS_RADIUS_MAX * iris_r
  const irisR = Math.max(eIris.rxMajor, eIris.rxMinor);
  const maxOffset = PUPIL_CENTER_OFFSET_MAX * irisR;
  const maxPupilR = PUPIL_TO_IRIS_RADIUS_MAX * irisR;
  let dx = ePupilRaw.cx - eIris.cx;
  let dy = ePupilRaw.cy - eIris.cy;
  const dist = Math.hypot(dx, dy);
  let px = ePupilRaw.cx, py = ePupilRaw.cy;
  if (dist > maxOffset && dist > 0) {
    const s = maxOffset / dist;
    px = eIris.cx + dx * s;
    py = eIris.cy + dy * s;
  }
  let pMaj = Math.min(ePupilRaw.rxMajor, maxPupilR);
  let pMin = Math.min(ePupilRaw.rxMinor, maxPupilR);
  pMin = Math.min(pMin, pMaj);
  const ePupil: Ellipse = {
    cx: px,
    cy: py,
    rxMajor: pMaj,
    rxMinor: pMin,
    angleDeg: ePupilRaw.angleDeg,
  };
  let pupilDisc = rasterEllipse(ePupil, w, h);
  for (let i = 0; i < pupilDisc.length; i++) pupilDisc[i] = pupilDisc[i] && irisDisc[i] ? 1 : 0;
  const ring = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) ring[i] = irisDisc[i] && !pupilDisc[i] ? 1 : 0;
  return encode(ring, pupilDisc);
}

// Suppress low-confidence pixels: argmax stays but if max softmax < threshold,
// pixel becomes bg. `probs` is (numClasses * size * size), softmax-normalised.
function applyProbThreshold(
  mask: Uint8Array,
  probs: Float32Array | null,
  size: number,
  threshold: number,
): Uint8Array {
  if (!probs || threshold <= 0) return mask;
  const plane = size * size;
  const numClasses = probs.length / plane;
  const out = new Uint8Array(mask.length);
  for (let p = 0; p < plane; p++) {
    const cls = mask[p];
    if (cls === 0) { out[p] = 0; continue; }
    const conf = probs[cls * plane + p];
    out[p] = conf >= threshold ? cls : 0;
    // also re-check numClasses to silence unused warning
    if (numClasses < 1) out[p] = 0;
  }
  return out;
}

// Zero out a class (1 or 2) if its total pixel count is below `minPixels`.
function applyMinArea(mask: Uint8Array, cls: 1 | 2, minPixels: number): Uint8Array {
  if (minPixels <= 0) return mask;
  let count = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i] === cls) count++;
  if (count >= minPixels) return mask;
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = mask[i] === cls ? 0 : mask[i];
  return out;
}

export function applyPostprocess(
  mask: Uint8Array,
  size: number,
  variant: PostprocessName,
  opts: PostprocessOptions = {},
  probs: Float32Array | null = null,
): Uint8Array {
  let m = mask;

  // 1. Per-pixel confidence gate (before any morphology).
  if (opts.probThreshold && opts.probThreshold > 0) {
    m = applyProbThreshold(m, probs, size, opts.probThreshold);
  }

  // 2. Optional class swap (1↔2) before postprocess so the structural rules
  //    (pupil inside iris, anatomical clamp) still make sense.
  if (opts.swapClasses) m = swapMaskClasses(m);

  const ksI = opts.morphKsizeIris ?? 5;
  const ksP = opts.morphKsizePupil ?? 3;
  let out: Uint8Array;
  switch (variant) {
    case "raw":                out = vRaw(m); break;
    case "largest_cc":         out = vLargestCc(m, size, size); break;
    case "morph":              out = vMorph(m, size, size, ksI, ksP); break;
    case "ellipse_iris":       out = vEllipseIris(m, size, size, ksI, ksP); break;
    case "ellipse_iris_pupil": out = vEllipseIrisPupil(m, size, size, ksI, ksP); break;
    case "ellipse_anatomical": out = vEllipseAnatomical(m, size, size, ksI, ksP); break;
    case "ellipse_anatomical_clean":
      out = vEllipseAnatomicalClean(
        m, size, size, ksI, ksP,
        opts.eyelidPoints,
        opts.imageData,
        opts.specularPct ?? 99,
      );
      break;
  }

  // 3. Min area filter (drop tiny detections).
  if (opts.minIrisPixels && opts.minIrisPixels > 0) out = applyMinArea(out, 1, opts.minIrisPixels);
  if (opts.minPupilPixels && opts.minPupilPixels > 0) out = applyMinArea(out, 2, opts.minPupilPixels);

  return out;
}

// ---------------------------------------------------------------------------
// Public: fit ellipses (with anatomical constraints applied to pupil) from a
// post-processed mask. Mirrors src/postprocess.py:fit_iris_pupil_ellipses.
// ---------------------------------------------------------------------------

export function fitIrisPupilEllipses(mask: Uint8Array, size: number): EllipsesResult {
  const { iris, pupil } = decode(mask);
  const irisFull = new Uint8Array(iris.length);
  for (let i = 0; i < iris.length; i++) irisFull[i] = iris[i] || pupil[i] ? 1 : 0;
  const irisEll = anyTrue(irisFull)
    ? fitEllipseFromFilled(irisFull, size, size)
    : null;
  const pupilEll = anyTrue(pupil)
    ? fitEllipseFromFilled(pupil, size, size)
    : null;
  return { iris: irisEll, pupil: pupilEll };
}
