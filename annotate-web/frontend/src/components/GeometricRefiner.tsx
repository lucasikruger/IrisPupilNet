import { useEffect, useRef, useState } from "react";

export interface Point { x: number; y: number; }
export interface Circle { cx: number; cy: number; r: number; }
export interface Eyelid {
  canti: { left: Point; right: Point };
  upperCtl: Point;
  lowerCtl: Point;
}
export interface RefinerGeometry {
  iris: Circle;
  pupil: Circle;
  eyelid: Eyelid;
}

type HandleId =
  | "iris-center" | "iris-radius"
  | "pupil-center" | "pupil-radius"
  | "canti-left" | "canti-right"
  | "eyelid-upper" | "eyelid-lower";

const HIT_RADIUS = 12;
const HANDLE_RADIUS = 4;

export function defaultGeometry(size: number): RefinerGeometry {
  const cx = size / 2;
  const cy = size / 2;
  const irisR = size * 0.28;
  const pupilR = size * 0.10;
  return {
    iris: { cx, cy, r: irisR },
    pupil: { cx, cy, r: pupilR },
    eyelid: {
      canti: {
        left: { x: cx - size * 0.38, y: cy },
        right: { x: cx + size * 0.38, y: cy },
      },
      upperCtl: { x: cx, y: cy - size * 0.22 },
      lowerCtl: { x: cx, y: cy + size * 0.22 },
    },
  };
}

export default function GeometricRefiner({
  img,
  geometry,
  onChange,
}: {
  img: HTMLImageElement;
  geometry: RefinerGeometry;
  onChange: (g: RefinerGeometry) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState<HandleId | null>(null);
  const [hovering, setHovering] = useState<HandleId | null>(null);

  useEffect(() => {
    render(canvasRef.current, img, geometry);
  }, [img, geometry]);

  function toLocal(e: React.MouseEvent | React.TouchEvent): Point {
    const c = canvasRef.current;
    if (!c) return { x: 0, y: 0 };
    const rect = c.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : (e as React.MouseEvent).clientY;
    const x = ((clientX - rect.left) / rect.width) * c.width;
    const y = ((clientY - rect.top) / rect.height) * c.height;
    return { x, y };
  }

  function hitTest(p: Point): HandleId | null {
    const handles: Array<[HandleId, Point]> = [
      ["iris-center", { x: geometry.iris.cx, y: geometry.iris.cy }],
      ["iris-radius", { x: geometry.iris.cx + geometry.iris.r, y: geometry.iris.cy }],
      ["pupil-center", { x: geometry.pupil.cx, y: geometry.pupil.cy }],
      ["pupil-radius", { x: geometry.pupil.cx + geometry.pupil.r, y: geometry.pupil.cy }],
      ["canti-left", geometry.eyelid.canti.left],
      ["canti-right", geometry.eyelid.canti.right],
      ["eyelid-upper", geometry.eyelid.upperCtl],
      ["eyelid-lower", geometry.eyelid.lowerCtl],
    ];
    let bestId: HandleId | null = null;
    let bestD2 = HIT_RADIUS * HIT_RADIUS;
    for (const [id, h] of handles) {
      const d2 = (p.x - h.x) ** 2 + (p.y - h.y) ** 2;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestId = id;
      }
    }
    return bestId;
  }

  function applyDrag(id: HandleId, p: Point): RefinerGeometry {
    const g = JSON.parse(JSON.stringify(geometry)) as RefinerGeometry;
    switch (id) {
      case "iris-center":
        g.iris.cx = p.x; g.iris.cy = p.y; break;
      case "iris-radius":
        g.iris.r = Math.max(4, Math.hypot(p.x - g.iris.cx, p.y - g.iris.cy)); break;
      case "pupil-center":
        g.pupil.cx = p.x; g.pupil.cy = p.y; break;
      case "pupil-radius":
        g.pupil.r = Math.max(2, Math.hypot(p.x - g.pupil.cx, p.y - g.pupil.cy)); break;
      case "canti-left":
        g.eyelid.canti.left = p; break;
      case "canti-right":
        g.eyelid.canti.right = p; break;
      case "eyelid-upper":
        g.eyelid.upperCtl = p; break;
      case "eyelid-lower":
        g.eyelid.lowerCtl = p; break;
    }
    return g;
  }

  function onDown(e: React.MouseEvent | React.TouchEvent) {
    const p = toLocal(e);
    const id = hitTest(p);
    if (id) {
      setDragging(id);
      e.preventDefault();
    }
  }
  function onMove(e: React.MouseEvent | React.TouchEvent) {
    const p = toLocal(e);
    if (dragging) {
      onChange(applyDrag(dragging, p));
      e.preventDefault();
      return;
    }
    const id = hitTest(p);
    if (id !== hovering) setHovering(id);
  }
  function onUp() {
    setDragging(null);
  }
  function onLeave() {
    setDragging(null);
    setHovering(null);
  }

  return (
    <div ref={containerRef} style={{ position: "relative", userSelect: "none", touchAction: "none" }}>
      <canvas
        ref={canvasRef}
        width={img.width}
        height={img.height}
        style={{
          width: "100%",
          aspectRatio: "1 / 1",
          border: "1px solid var(--border)",
          borderRadius: 8,
          background: "#000",
          cursor: dragging ? "grabbing" : hovering ? "grab" : "default",
          display: "block",
          imageRendering: "pixelated",
        }}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onLeave}
        onTouchStart={onDown}
        onTouchMove={onMove}
        onTouchEnd={onUp}
      />
    </div>
  );
}

function render(canvas: HTMLCanvasElement | null, img: HTMLImageElement, g: RefinerGeometry) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;

  // Eyelid curves (filled band semi-transparent)
  ctx.beginPath();
  ctx.moveTo(g.eyelid.canti.left.x, g.eyelid.canti.left.y);
  ctx.quadraticCurveTo(g.eyelid.upperCtl.x, g.eyelid.upperCtl.y, g.eyelid.canti.right.x, g.eyelid.canti.right.y);
  ctx.quadraticCurveTo(g.eyelid.lowerCtl.x, g.eyelid.lowerCtl.y, g.eyelid.canti.left.x, g.eyelid.canti.left.y);
  ctx.closePath();
  ctx.fillStyle = "rgba(80, 200, 255, 0.10)";
  ctx.fill();
  ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
  ctx.stroke();

  // Iris circle
  ctx.beginPath();
  ctx.arc(g.iris.cx, g.iris.cy, g.iris.r, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255, 220, 90, 0.95)";
  ctx.stroke();

  // Pupil circle
  ctx.beginPath();
  ctx.arc(g.pupil.cx, g.pupil.cy, g.pupil.r, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255, 90, 200, 0.95)";
  ctx.stroke();

  // Handles
  const handles: Array<[number, number, string]> = [
    [g.iris.cx, g.iris.cy, "rgba(255, 220, 90, 0.95)"],
    [g.iris.cx + g.iris.r, g.iris.cy, "rgba(255, 220, 90, 0.6)"],
    [g.pupil.cx, g.pupil.cy, "rgba(255, 90, 200, 0.95)"],
    [g.pupil.cx + g.pupil.r, g.pupil.cy, "rgba(255, 90, 200, 0.6)"],
    [g.eyelid.canti.left.x, g.eyelid.canti.left.y, "rgba(80, 200, 255, 0.95)"],
    [g.eyelid.canti.right.x, g.eyelid.canti.right.y, "rgba(80, 200, 255, 0.95)"],
    [g.eyelid.upperCtl.x, g.eyelid.upperCtl.y, "rgba(80, 200, 255, 0.7)"],
    [g.eyelid.lowerCtl.x, g.eyelid.lowerCtl.y, "rgba(80, 200, 255, 0.7)"],
  ];
  for (const [x, y, color] of handles) {
    ctx.beginPath();
    ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// Rasterize the geometry to three separate masks (PNG blobs).
// Each mask is RGBA where white pixels mean ON.
export async function geometryToMasks(
  g: RefinerGeometry,
  size: number,
): Promise<{ iris: Blob; pupil: Blob; eyelid: Blob }> {
  const makeMask = (paint: (ctx: CanvasRenderingContext2D) => void): HTMLCanvasElement => {
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = "#fff";
    paint(ctx);
    return c;
  };

  const irisCanvas = makeMask((ctx) => {
    ctx.beginPath();
    ctx.arc(g.iris.cx, g.iris.cy, g.iris.r, 0, Math.PI * 2);
    ctx.fill();
  });
  const pupilCanvas = makeMask((ctx) => {
    ctx.beginPath();
    ctx.arc(g.pupil.cx, g.pupil.cy, g.pupil.r, 0, Math.PI * 2);
    ctx.fill();
  });
  const eyelidCanvas = makeMask((ctx) => {
    ctx.beginPath();
    ctx.moveTo(g.eyelid.canti.left.x, g.eyelid.canti.left.y);
    ctx.quadraticCurveTo(g.eyelid.upperCtl.x, g.eyelid.upperCtl.y, g.eyelid.canti.right.x, g.eyelid.canti.right.y);
    ctx.quadraticCurveTo(g.eyelid.lowerCtl.x, g.eyelid.lowerCtl.y, g.eyelid.canti.left.x, g.eyelid.canti.left.y);
    ctx.closePath();
    ctx.fill();
  });

  const [iris, pupil, eyelid] = await Promise.all([
    canvasToPng(irisCanvas),
    canvasToPng(pupilCanvas),
    canvasToPng(eyelidCanvas),
  ]);
  return { iris, pupil, eyelid };
}

function canvasToPng(c: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob null"))), "image/png");
  });
}
