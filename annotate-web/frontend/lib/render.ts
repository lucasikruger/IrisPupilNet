// Render utilities for overlaying segmentation masks on crops.

import type { SegmentationResult } from "./onnx";

// 3-class palette: 0=bg (transparent), 1=iris (cyan), 2=pupil (magenta).
const PALETTE: Array<[number, number, number, number]> = [
  [0, 0, 0, 0],
  [55, 220, 245, 200],   // iris
  [245, 80, 200, 220],   // pupil
];

export function maskToRgba(seg: SegmentationResult): ImageData {
  const { argmax, size, numClasses } = seg;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < argmax.length; i++) {
    const cls = argmax[i];
    const color = PALETTE[cls] ?? PALETTE[0];
    out[i * 4] = color[0];
    out[i * 4 + 1] = color[1];
    out[i * 4 + 2] = color[2];
    out[i * 4 + 3] = color[3];
    if (cls >= numClasses) {
      out[i * 4 + 3] = 0;
    }
  }
  return new ImageData(out, size, size);
}

export interface RenderOptions {
  show: "crop" | "mask" | "blend";
  blendAlpha?: number;  // 0..1, default 0.55
  bw?: boolean;         // render the crop as grayscale (matches model input)
}

export function renderCropWithMask(
  target: HTMLCanvasElement,
  crop: HTMLCanvasElement,
  seg: SegmentationResult,
  opts: RenderOptions,
): void {
  target.width = crop.width;
  target.height = crop.height;
  const ctx = target.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, target.width, target.height);
  if (opts.show !== "mask") {
    if (opts.bw) drawCropGrayscale(ctx, crop);
    else ctx.drawImage(crop, 0, 0);
  }
  if (opts.show === "crop") return;

  const maskData = maskToRgba(seg);
  const tmp = document.createElement("canvas");
  tmp.width = seg.size;
  tmp.height = seg.size;
  tmp.getContext("2d")!.putImageData(maskData, 0, 0);

  ctx.globalAlpha = opts.show === "mask" ? 1 : (opts.blendAlpha ?? 0.55);
  ctx.drawImage(tmp, 0, 0, target.width, target.height);
  ctx.globalAlpha = 1;
}

// Apply the same Rec.709 luma transform the grayscale_norm_01 preprocess uses,
// so the preview matches what the model actually sees.
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
