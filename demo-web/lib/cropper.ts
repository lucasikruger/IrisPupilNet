// Port of tools/prepare/eye_cropper.py — MediaPipe Face Landmarker → eye crops.
// Mismos índices, padding 0.4, square crop forzado. Sin rotación/alineación.

import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const LEFT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const RIGHT_EYE_IDXS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

// Iris-refined landmarks (only present if MediaPipe Face Landmarker is loaded
// with the 478-point model — face_landmarker.task does by default).
//   468 = left iris centre, 469-472 = perimeter (top, right, bottom, left)
//   473 = right iris centre, 474-477 = perimeter
export const LEFT_IRIS_IDXS = [468, 469, 470, 471, 472];
export const RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477];

export type CropMode =
  | "eye"        // 16 eyelid landmarks + uniform padding (legacy)
  | "eye_tight"  // 16 eyelid landmarks + asymmetric padding + vertical anchor down
  | "iris";      // 5 iris landmarks (478-pt model only) + asymmetric padding

export interface EyeCropperOptions {
  /** Uniform padding fraction (back-compat). If set, applied to both axes. */
  padding?: number;
  /** Horizontal padding fraction (default 0.4). */
  paddingX?: number;
  /** Vertical padding fraction (default 0.4). */
  paddingY?: number;
  /** When squaring, shift the centre down by `verticalAnchor * side` (0..1).
   *  Positive values drop the eyebrow from the top of the crop. */
  verticalAnchor?: number;
  minEyeSize?: number;    // default 20 px
  square?: boolean;       // default true
  outputSize?: number;    // default 160 (final resize)
  cropMode?: CropMode;    // default "eye"
}

export interface EyeCrop {
  side: "left" | "right";
  bbox: { x: number; y: number; w: number; h: number };  // in source coords
  canvas: HTMLCanvasElement;  // resized to outputSize x outputSize
  eyelidPoints: { x: number; y: number }[];  // 16 contour landmarks, in crop coords
}

export class EyeCropper {
  // Public + mutable so callers (e.g. sidebar controls) can tweak between frames.
  paddingX: number;
  paddingY: number;
  /** Shift the square's centre down by this fraction of the side length, to
   *  push the eyebrow out of the crop. 0 = centred on the eye, 0.2 ≈ moves
   *  centre ~20% of `side` downwards. */
  verticalAnchor: number;
  minEyeSize: number;
  square: boolean;
  outputSize: number;
  cropMode: CropMode;
  /** When true the SOURCE is treated as mirrored (we mirror the bbox X axis
   *  on the source before reading pixels). Lets the UI show a non-mirrored
   *  view but feed the model with the same chirality used at training time. */
  mirror = false;

  constructor(opts: EyeCropperOptions = {}) {
    const base = opts.padding ?? 0.4;
    this.paddingX = opts.paddingX ?? base;
    this.paddingY = opts.paddingY ?? base;
    this.verticalAnchor = opts.verticalAnchor ?? 0;
    this.minEyeSize = opts.minEyeSize ?? 20;
    this.square = opts.square ?? true;
    this.outputSize = opts.outputSize ?? 160;
    this.cropMode = opts.cropMode ?? "eye";
  }

  /** Back-compat shim for older callers that read .padding. */
  get padding(): number {
    return (this.paddingX + this.paddingY) / 2;
  }
  set padding(p: number) {
    this.paddingX = p;
    this.paddingY = p;
  }

  /** Detect faces and produce up to 2 eye crops. */
  cropEyes(
    landmarker: FaceLandmarker,
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    timestampMs?: number,
  ): EyeCrop[] {
    const W = "videoWidth" in source ? source.videoWidth : (source as HTMLImageElement).naturalWidth || source.width;
    const H = "videoHeight" in source ? source.videoHeight : (source as HTMLImageElement).naturalHeight || source.height;
    if (W === 0 || H === 0) return [];

    const result =
      source instanceof HTMLVideoElement
        ? landmarker.detectForVideo(source, timestampMs ?? performance.now())
        : landmarker.detect(source);

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) return [];
    const landmarks = result.faceLandmarks[0];

    const crops: EyeCrop[] = [];
    const eyeIdx = this.cropMode === "iris" ? LEFT_IRIS_IDXS : LEFT_EYE_IDXS;
    const eyeIdxR = this.cropMode === "iris" ? RIGHT_IRIS_IDXS : RIGHT_EYE_IDXS;
    const eyes: ReadonlyArray<readonly [number[], "left" | "right"]> = [
      [eyeIdx, "left"],
      [eyeIdxR, "right"],
    ];
    // For drawing eyelid points we always use the eyelid contour, even in iris mode
    const eyelidIndices = this.cropMode === "iris"
      ? { left: LEFT_EYE_IDXS, right: RIGHT_EYE_IDXS }
      : { left: LEFT_EYE_IDXS, right: RIGHT_EYE_IDXS };
    for (const [indices, side] of eyes) {
      const bbox = this.bboxFromLandmarks(landmarks, indices, W, H);
      if (!bbox) continue;
      const eyelidPoints = this.mapLandmarksToCrop(landmarks, eyelidIndices[side], bbox, W, H);
      crops.push({
        side,
        bbox,
        canvas: this.cropAndResize(source, bbox),
        eyelidPoints,
      });
    }
    return crops;
  }

  private mapLandmarksToCrop(
    landmarks: NormalizedLandmark[],
    indices: number[],
    bbox: { x: number; y: number; w: number; h: number },
    W: number,
    H: number,
  ): { x: number; y: number }[] {
    const sx = this.outputSize / bbox.w;
    const sy = this.outputSize / bbox.h;
    const out: { x: number; y: number }[] = [];
    for (const i of indices) {
      const lm = landmarks[i];
      if (!lm) continue;
      const cropX = (lm.x * W - bbox.x) * sx;
      out.push({
        x: this.mirror ? this.outputSize - cropX : cropX,
        y: (lm.y * H - bbox.y) * sy,
      });
    }
    return out;
  }

  private bboxFromLandmarks(
    landmarks: NormalizedLandmark[],
    indices: number[],
    W: number,
    H: number,
  ): { x: number; y: number; w: number; h: number } | null {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const i of indices) {
      const lm = landmarks[i];
      if (lm) {
        xs.push(lm.x * W);
        ys.push(lm.y * H);
      }
    }
    if (xs.length === 0) return null;

    let x0 = Math.min(...xs);
    let x1 = Math.max(...xs);
    let y0 = Math.min(...ys);
    let y1 = Math.max(...ys);
    let w = x1 - x0;
    let h = y1 - y0;

    // Iris landmarks form a near-circle; "h" from them is ~iris height, so
    // we boost the padding minimum to give the model some sclera context.
    const minSide = this.cropMode === "iris" ? Math.max(w, h) * 1.6 : 0;
    if (minSide > 0) {
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      x0 = cx - minSide / 2;
      y0 = cy - minSide / 2;
      w = minSide;
      h = minSide;
    }

    const px = w * this.paddingX;
    const py = h * this.paddingY;
    x0 -= px;
    y0 -= py;
    w += 2 * px;
    h += 2 * py;

    if (this.square) {
      const side = Math.max(w, h);
      const cx = x0 + w / 2;
      // verticalAnchor > 0 shifts the centre downwards (out of the eyebrow).
      const cy = y0 + h / 2 + this.verticalAnchor * side;
      x0 = cx - side / 2;
      y0 = cy - side / 2;
      w = side;
      h = side;
    }

    x0 = Math.max(0, Math.floor(x0));
    y0 = Math.max(0, Math.floor(y0));
    w = Math.min(W - x0, Math.floor(w));
    h = Math.min(H - y0, Math.floor(h));

    if (w < this.minEyeSize || h < this.minEyeSize) return null;
    return { x: x0, y: y0, w, h };
  }

  private cropAndResize(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    bbox: { x: number; y: number; w: number; h: number },
  ): HTMLCanvasElement {
    const out = document.createElement("canvas");
    out.width = this.outputSize;
    out.height = this.outputSize;
    const ctx = out.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    if (this.mirror) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(source, bbox.x, bbox.y, bbox.w, bbox.h, -this.outputSize, 0, this.outputSize, this.outputSize);
      ctx.restore();
    } else {
      ctx.drawImage(source, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, this.outputSize, this.outputSize);
    }
    return out;
  }
}
