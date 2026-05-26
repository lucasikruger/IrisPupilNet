// Port of tools/prepare/eye_cropper.py — MediaPipe Face Landmarker → eye crops.
// Mismos índices, padding 0.4, square crop forzado. Sin rotación/alineación.

import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const LEFT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const RIGHT_EYE_IDXS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

export interface EyeCropperOptions {
  padding?: number;       // default 0.4 (40% extra)
  minEyeSize?: number;    // default 30 px
  square?: boolean;       // default true
  outputSize?: number;    // default 160 (final resize)
}

export interface EyeCrop {
  side: "left" | "right";
  bbox: { x: number; y: number; w: number; h: number };  // in source coords
  canvas: HTMLCanvasElement;  // resized to outputSize x outputSize
  eyelidPoints: { x: number; y: number }[];  // 16 contour landmarks, in crop coords
}

export class EyeCropper {
  private padding: number;
  private minEyeSize: number;
  private square: boolean;
  private outputSize: number;

  constructor(opts: EyeCropperOptions = {}) {
    this.padding = opts.padding ?? 0.4;
    this.minEyeSize = opts.minEyeSize ?? 30;
    this.square = opts.square ?? true;
    this.outputSize = opts.outputSize ?? 160;
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
    for (const [indices, side] of [
      [LEFT_EYE_IDXS, "left" as const],
      [RIGHT_EYE_IDXS, "right" as const],
    ]) {
      const bbox = this.bboxFromLandmarks(landmarks, indices as number[], W, H);
      if (!bbox) continue;
      const eyelidPoints = this.mapLandmarksToCrop(
        landmarks,
        indices as number[],
        bbox,
        W,
        H,
      );
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
      out.push({
        x: (lm.x * W - bbox.x) * sx,
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

    const px = w * this.padding;
    const py = h * this.padding;
    x0 -= px;
    y0 -= py;
    w += 2 * px;
    h += 2 * py;

    if (this.square) {
      const side = Math.max(w, h);
      const cx = x0 + w / 2;
      const cy = y0 + h / 2;
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
    ctx.drawImage(source, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, this.outputSize, this.outputSize);
    return out;
  }
}
