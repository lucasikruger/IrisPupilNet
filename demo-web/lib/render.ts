// Render utilities for overlaying segmentation masks on eye crops.

import type { SegmentationResult } from "./onnx";
import { applyPostprocess, fitIrisPupilEllipses, type Ellipse, type PostprocessName, type PostprocessOptions } from "./postprocess";
import { PREPROCESS, type PreprocessName } from "./preprocess";

// 3-class palette: 0=bg (transparent), 1=iris (cyan), 2=pupil (magenta).
const PALETTE: Array<[number, number, number, number]> = [
  [0, 0, 0, 0],
  [55, 220, 245, 200],   // iris
  [245, 80, 200, 220],   // pupil
];

// "Hard" palette for the mask-only view (full alpha).
const PALETTE_HARD: Array<[number, number, number, number]> = [
  [12, 14, 18, 255],
  [55, 220, 245, 255],
  [245, 80, 200, 255],
];

export interface RenderOptions {
  show: "crop" | "mask" | "blend";
  blendAlpha?: number;        // 0..1, default 0.55
  bw?: boolean;               // render crop as grayscale (matches BW model input)
  postprocess?: PostprocessName;  // default "morph"
  postprocessOpts?: PostprocessOptions; // morph kernels, min area, swap, threshold
  preprocessName?: PreprocessName; // for the "preprocessed" preview
  showIris?: boolean;         // default true
  showPupil?: boolean;        // default true
  showEllipse?: boolean;      // draw fitted iris/pupil ellipses on top
  showEyelid?: boolean;       // draw eyelid landmark dots
  showPupilCenter?: boolean;  // draw pupil centre crosshair
  eyelidPoints?: { x: number; y: number }[];
  /** If true, mask view shows a saturated colour-coded mask instead of overlay-on-black. */
  hardMask?: boolean;
}

/** Probability heatmap view: per-pixel softmax for a chosen class painted as
 *  a viridis-like colour ramp (0 → dark blue, 1 → bright yellow). */
export function drawProbHeatmap(
  target: HTMLCanvasElement,
  seg: SegmentationResult,
  classIdx: number,
  alpha = 0.85,
): void {
  const size = seg.size;
  const plane = size * size;
  const offset = classIdx * plane;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let p = 0; p < plane; p++) {
    const v = Math.max(0, Math.min(1, seg.probs[offset + p]));
    const [r, g, b] = viridis(v);
    const i = p * 4;
    rgba[i] = r;
    rgba[i + 1] = g;
    rgba[i + 2] = b;
    rgba[i + 3] = Math.round(255 * alpha);
  }
  target.width = size;
  target.height = size;
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  tmp.getContext("2d")!.putImageData(
    new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, size, size),
    0,
    0,
  );
  const ctx = target.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, target.width, target.height);
  ctx.drawImage(tmp, 0, 0, target.width, target.height);
}

// Compact viridis approximation (5-stop linear interpolation).
function viridis(t: number): [number, number, number] {
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const u = (t - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + u * (c1[0] - c0[0])),
        Math.round(c0[1] + u * (c1[1] - c0[1])),
        Math.round(c0[2] + u * (c1[2] - c0[2])),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

export function maskToRgba(
  mask: Uint8Array,
  size: number,
  showIris = true,
  showPupil = true,
  hard = false,
): ImageData {
  const palette = hard ? PALETTE_HARD : PALETTE;
  const bgAlpha = hard ? palette[0][3] : 0;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < mask.length; i++) {
    let cls = mask[i];
    if (cls === 1 && !showIris) cls = 0;
    else if (cls === 2 && !showPupil) cls = 0;
    if (cls === 0) {
      if (hard) {
        out[i * 4] = palette[0][0];
        out[i * 4 + 1] = palette[0][1];
        out[i * 4 + 2] = palette[0][2];
        out[i * 4 + 3] = bgAlpha;
      } else {
        out[i * 4 + 3] = 0;
      }
      continue;
    }
    const color = palette[cls] ?? palette[0];
    out[i * 4] = color[0];
    out[i * 4 + 1] = color[1];
    out[i * 4 + 2] = color[2];
    out[i * 4 + 3] = color[3];
  }
  return new ImageData(out, size, size);
}

// Stamp a canvas with whatever the model "sees" (preprocessed crop). Falls
// back to a plain drawImage if the preprocess does not provide a preview.
export function drawPreprocessed(
  target: HTMLCanvasElement,
  crop: HTMLCanvasElement,
  preprocessName: PreprocessName | undefined,
  size: number,
): void {
  target.width = crop.width;
  target.height = crop.height;
  const ctx = target.getContext("2d");
  if (!ctx) return;
  const pre = preprocessName ? PREPROCESS[preprocessName] : null;
  if (pre?.toPreview) {
    const imgData = pre.toPreview(crop, size);
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    tmp.getContext("2d")!.putImageData(imgData, 0, 0);
    ctx.drawImage(tmp, 0, 0, target.width, target.height);
  } else {
    ctx.drawImage(crop, 0, 0);
  }
}

export interface RenderResult {
  /** Argmax after postprocess (size×size, values 0/1/2). */
  postMask: Uint8Array;
  /** Fitted ellipses (may be null). */
  ellipses: { iris: Ellipse | null; pupil: Ellipse | null };
}

export function renderCropWithMask(
  target: HTMLCanvasElement,
  crop: HTMLCanvasElement,
  seg: SegmentationResult,
  opts: RenderOptions,
): RenderResult {
  target.width = crop.width;
  target.height = crop.height;
  const ctx = target.getContext("2d");
  const variant: PostprocessName = opts.postprocess ?? "morph";
  // For the open-iris-style "clean" variant, inject the eyelid landmarks
  // (mapped from crop coords to mask coords) and the crop pixel data so the
  // postprocess can do eyelid polynomial cropping + specular masking.
  let ppOpts = opts.postprocessOpts;
  if (variant === "ellipse_anatomical_clean") {
    ppOpts = { ...(ppOpts ?? {}) };
    if (opts.eyelidPoints && !ppOpts.eyelidPoints) {
      const sx = seg.size / crop.width;
      const sy = seg.size / crop.height;
      ppOpts.eyelidPoints = opts.eyelidPoints.map((p) => ({ x: p.x * sx, y: p.y * sy }));
    }
    if (!ppOpts.imageData) {
      const ctx0 = crop.getContext("2d");
      if (ctx0) {
        // Get crop pixels at mask resolution. If crop != seg.size, downsample.
        if (crop.width === seg.size && crop.height === seg.size) {
          ppOpts.imageData = ctx0.getImageData(0, 0, seg.size, seg.size).data;
        } else {
          const tmp = document.createElement("canvas");
          tmp.width = seg.size; tmp.height = seg.size;
          const tctx = tmp.getContext("2d");
          if (tctx) {
            tctx.drawImage(crop, 0, 0, seg.size, seg.size);
            ppOpts.imageData = tctx.getImageData(0, 0, seg.size, seg.size).data;
          }
        }
      }
    }
  }
  const postMask = applyPostprocess(seg.argmax, seg.size, variant, ppOpts, seg.probs);
  const ellipses = fitIrisPupilEllipses(postMask, seg.size);

  if (!ctx) return { postMask, ellipses };
  ctx.clearRect(0, 0, target.width, target.height);
  if (opts.show !== "mask") {
    if (opts.bw) drawCropGrayscale(ctx, crop);
    else ctx.drawImage(crop, 0, 0);
  } else if (opts.hardMask) {
    ctx.fillStyle = "#0c0e12";
    ctx.fillRect(0, 0, target.width, target.height);
  }

  if (opts.show !== "crop") {
    const maskData = maskToRgba(
      postMask,
      seg.size,
      opts.showIris ?? true,
      opts.showPupil ?? true,
      opts.show === "mask" && (opts.hardMask ?? true),
    );
    const tmp = document.createElement("canvas");
    tmp.width = seg.size;
    tmp.height = seg.size;
    tmp.getContext("2d")!.putImageData(maskData, 0, 0);
    ctx.globalAlpha = opts.show === "mask" ? 1 : (opts.blendAlpha ?? 0.55);
    ctx.drawImage(tmp, 0, 0, target.width, target.height);
    ctx.globalAlpha = 1;
  }

  // Crop is size×size; target is the same. Coordinate scale = 1 when sizes match.
  const sx = target.width / seg.size;
  const sy = target.height / seg.size;

  if (opts.showEllipse) {
    if (ellipses.iris) drawEllipse(ctx, ellipses.iris, sx, sy, "rgba(40, 200, 235, 0.95)", 2);
    if (ellipses.pupil) drawEllipse(ctx, ellipses.pupil, sx, sy, "rgba(250, 90, 210, 0.95)", 2);
  }
  if (opts.showPupilCenter && ellipses.pupil) {
    drawCrosshair(ctx, ellipses.pupil.cx * sx, ellipses.pupil.cy * sy, "#ffe24a");
  }
  if (opts.showEyelid && opts.eyelidPoints) {
    drawEyelidPoints(ctx, opts.eyelidPoints, target.width / 160, target.height / 160);
  }

  return { postMask, ellipses };
}

function drawEllipse(
  ctx: CanvasRenderingContext2D,
  e: Ellipse,
  sx: number,
  sy: number,
  stroke: string,
  lineWidth = 2,
) {
  if (!isFinite(e.cx) || !isFinite(e.cy) || e.rxMajor <= 0 || e.rxMinor <= 0) return;
  ctx.save();
  ctx.translate(e.cx * sx, e.cy * sy);
  ctx.rotate((e.angleDeg * Math.PI) / 180);
  ctx.beginPath();
  ctx.ellipse(0, 0, e.rxMajor * sx, e.rxMinor * sy, 0, 0, 2 * Math.PI);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.restore();
}

function drawCrosshair(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x - 6, y);
  ctx.lineTo(x + 6, y);
  ctx.moveTo(x, y - 6);
  ctx.lineTo(x, y + 6);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawEyelidPoints(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  sx: number,
  sy: number,
) {
  // Dots + thin polyline connecting them
  ctx.save();
  ctx.fillStyle = "rgba(255, 226, 74, 0.95)";
  ctx.strokeStyle = "rgba(255, 226, 74, 0.55)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = pts[i].x * sx;
    const y = pts[i].y * sy;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x * sx, p.y * sy, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}

function drawCropGrayscale(ctx: CanvasRenderingContext2D, crop: HTMLCanvasElement) {
  ctx.drawImage(crop, 0, 0);
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    d[i] = d[i + 1] = d[i + 2] = g;
  }
  ctx.putImageData(img, 0, 0);
}
