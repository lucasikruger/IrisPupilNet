// MediaPipe Face Landmarker → iris-centred eye crop that matches the training
// data distribution exactly.
//
// Training (train_final.py + src/dataset.py online_crop="jitter") builds each
// crop from the iris bbox so the iris fills ~targetIrisPct of the side. We
// mirror that here by using the 5 iris landmarks from the 478-point Face
// Landmarker model (face_landmarker.task ships with iris refinement enabled):
//
//   468 = left iris centre, 469-472 = perimeter (top, right, bottom, left)
//   473 = right iris centre, 474-477 = perimeter
//
// Algorithm:
//   1. iris centre  = landmark[468 or 473]
//   2. iris radius  = max distance from centre to the 4 perimeter points
//   3. side         = (2 * radius) / targetIrisPct  →  iris fills targetIrisPct
//   4. bbox         = square centred on iris centre, clipped to image bounds

import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const LEFT_IRIS_IDXS = [468, 469, 470, 471, 472];
export const RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477];
// Eyelid contour — used only for the eyelid-points overlay, not for cropping.
export const LEFT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const RIGHT_EYE_IDXS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

const IRIS_CENTRE = { left: 468, right: 473 };
const IRIS_PERIM = { left: [469, 470, 471, 472], right: [474, 475, 476, 477] };

export interface EyeCropperOptions {
  /** Iris occupies this fraction of the crop side. Matches training's
   *  crop_target_iris_pct (default 0.35). */
  targetIrisPct?: number;
  minEyeSize?: number;    // default 20 px
  outputSize?: number;    // default 160 (final resize)
}

export interface EyeCrop {
  side: "left" | "right";
  bbox: { x: number; y: number; w: number; h: number };  // in source coords
  canvas: HTMLCanvasElement;  // resized to outputSize x outputSize
  eyelidPoints: { x: number; y: number }[];  // 16 contour landmarks, in crop coords
}

export class EyeCropper {
  // Public + mutable so callers (e.g. sidebar controls) can tweak between frames.
  targetIrisPct: number;
  minEyeSize: number;
  outputSize: number;
  /** When true the SOURCE is treated as mirrored (we mirror the bbox X axis
   *  on the source before reading pixels). Lets the UI show a non-mirrored
   *  view but feed the model with the same chirality used at training time. */
  mirror = false;

  constructor(opts: EyeCropperOptions = {}) {
    this.targetIrisPct = opts.targetIrisPct ?? 0.35;
    this.minEyeSize = opts.minEyeSize ?? 20;
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
    for (const side of ["left", "right"] as const) {
      const bbox = this.bboxFromIrisLandmarks(landmarks, side, W, H);
      if (!bbox) continue;
      const eyelidPoints = this.mapLandmarksToCrop(
        landmarks,
        side === "left" ? LEFT_EYE_IDXS : RIGHT_EYE_IDXS,
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
      const cropX = (lm.x * W - bbox.x) * sx;
      out.push({
        x: this.mirror ? this.outputSize - cropX : cropX,
        y: (lm.y * H - bbox.y) * sy,
      });
    }
    return out;
  }

  private bboxFromIrisLandmarks(
    landmarks: NormalizedLandmark[],
    side: "left" | "right",
    W: number,
    H: number,
  ): { x: number; y: number; w: number; h: number } | null {
    const centre = landmarks[IRIS_CENTRE[side]];
    if (!centre) return null;
    const cx = centre.x * W;
    const cy = centre.y * H;

    let maxR = 0;
    for (const i of IRIS_PERIM[side]) {
      const lm = landmarks[i];
      if (!lm) continue;
      const dx = lm.x * W - cx;
      const dy = lm.y * H - cy;
      const r = Math.hypot(dx, dy);
      if (r > maxR) maxR = r;
    }
    if (maxR <= 0) return null;

    const irisDiameter = 2 * maxR;
    const side_px = irisDiameter / this.targetIrisPct;

    let x0 = Math.floor(cx - side_px / 2);
    let y0 = Math.floor(cy - side_px / 2);
    let w = Math.floor(side_px);
    let h = Math.floor(side_px);

    x0 = Math.max(0, x0);
    y0 = Math.max(0, y0);
    w = Math.min(W - x0, w);
    h = Math.min(H - y0, h);

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
