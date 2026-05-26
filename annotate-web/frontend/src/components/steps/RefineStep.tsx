import { useEffect, useRef, useState } from "react";
import type { CaptureBundle } from "./CaptureStep";
import { OnnxSegmenter, loadManifest, type ModelSpec, type SegmentationResult } from "@lib/onnx";
import GeometricRefiner, { type RefinerGeometry, defaultGeometry, geometryToMasks } from "../GeometricRefiner";

type EyeSide = "left" | "right";

interface EyeState {
  img: HTMLImageElement;
  geometry: RefinerGeometry;
  busy: boolean;
}

export default function RefineStep({
  capture,
  submissionId,
  apiUrl,
  onDone,
}: {
  capture: CaptureBundle;
  submissionId: string;
  apiUrl: string;
  onDone: () => void;
}) {
  const segmenterRef = useRef<OnnxSegmenter | null>(null);
  const autoRanRef = useRef<{ left: boolean; right: boolean }>({ left: false, right: false });
  const [eyes, setEyes] = useState<Record<EyeSide, EyeState | null>>({ left: null, right: null });
  const [model, setModel] = useState<ModelSpec | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadImg = (url: string) =>
      new Promise<HTMLImageElement>((resolve, reject) => {
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => reject(new Error("img load"));
        im.src = url;
      });
    (async () => {
      const [li, ri] = await Promise.all([loadImg(capture.leftDataUrl), loadImg(capture.rightDataUrl)]);
      const size = li.width;
      setEyes({
        left: { img: li, geometry: defaultGeometry(size), busy: false },
        right: { img: ri, geometry: defaultGeometry(size), busy: false },
      });
    })();
  }, [capture]);

  useEffect(() => {
    (async () => {
      try {
        const manifest = await loadManifest();
        if (manifest.length === 0) return;
        const spec = manifest[0];
        setModel(spec);
        const seg = new OnnxSegmenter();
        await seg.load(spec);
        segmenterRef.current = seg;
      } catch (e) {
        console.warn("segmenter load failed for refine:", e);
      }
    })();
  }, []);

  // Auto-estimate once per side as soon as we have both the image and the
  // model loaded, so the user doesn't see floating default handles.
  useEffect(() => {
    if (!model || !segmenterRef.current) return;
    for (const side of ["left", "right"] as EyeSide[]) {
      if (eyes[side] && !autoRanRef.current[side]) {
        autoRanRef.current[side] = true;
        autoEstimate(side);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, eyes.left, eyes.right]);

  async function autoEstimate(side: EyeSide) {
    const eye = eyes[side];
    const seg = segmenterRef.current;
    if (!eye || !seg) return;
    setEyes((prev) => ({ ...prev, [side]: prev[side] ? { ...prev[side]!, busy: true } : null }));
    try {
      const canvas = document.createElement("canvas");
      canvas.width = eye.img.width;
      canvas.height = eye.img.height;
      canvas.getContext("2d")!.drawImage(eye.img, 0, 0);
      const result = await seg.run(canvas);
      const eyelidPoints = side === "left" ? capture.leftEyelid : capture.rightEyelid;
      const geom = estimateGeometry(result, eyelidPoints, eye.img.width);
      setEyes((prev) => ({ ...prev, [side]: { ...prev[side]!, geometry: geom, busy: false } }));
    } catch (e) {
      console.warn("auto-estimate failed:", e);
      setEyes((prev) => ({ ...prev, [side]: { ...prev[side]!, busy: false } }));
    }
  }

  function setGeometry(side: EyeSide, geom: RefinerGeometry) {
    setEyes((prev) => ({ ...prev, [side]: prev[side] ? { ...prev[side]!, geometry: geom } : null }));
  }

  async function submitRefinement() {
    if (!eyes.left || !eyes.right) return;
    setSubmitting(true);
    setError(null);
    try {
      const form = new FormData();
      for (const side of ["left", "right"] as EyeSide[]) {
        const eye = eyes[side]!;
        const masks = await geometryToMasks(eye.geometry, eye.img.width);
        form.append(`mask_iris_${side}`, masks.iris, `mask_iris_${side}.png`);
        form.append(`mask_pupil_${side}`, masks.pupil, `mask_pupil_${side}.png`);
        form.append(`mask_eyelid_${side}`, masks.eyelid, `mask_eyelid_${side}.png`);
      }
      const geometryJson = {
        left: eyes.left.geometry,
        right: eyes.right.geometry,
      };
      form.append(
        "geometry",
        new Blob([JSON.stringify(geometryJson)], { type: "application/json" }),
        "geometry.json",
      );
      const resp = await fetch(`${apiUrl}/api/submit/${submissionId}/refine`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  if (!eyes.left || !eyes.right) {
    return (
      <section className="panel">
        <h2>Refinador</h2>
        <p className="muted">Cargando crops…</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Refiná las anotaciones (opcional)</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        Para cada ojo: ajustá los dos círculos (iris afuera, pupila adentro) y las dos curvas
        de párpado (arriba y abajo). El botón <em>Auto-estimar</em> usa el modelo localmente
        para sembrar las primitivas — después podés ajustar.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {(["left", "right"] as EyeSide[]).map((side) => (
          <EyePanel
            key={side}
            side={side}
            eye={eyes[side]!}
            modelAvailable={!!model}
            onAutoEstimate={() => autoEstimate(side)}
            onChange={(g) => setGeometry(side, g)}
          />
        ))}
      </div>
      <div className="row" style={{ marginTop: 18 }}>
        <button onClick={submitRefinement} disabled={submitting}>
          {submitting ? "guardando…" : "Guardar refinamiento"}
        </button>
        <button className="secondary" onClick={onDone} disabled={submitting}>Saltar</button>
        {error && <span style={{ color: "var(--warn)" }}>{error}</span>}
      </div>
    </section>
  );
}

function EyePanel({
  side,
  eye,
  modelAvailable,
  onAutoEstimate,
  onChange,
}: {
  side: EyeSide;
  eye: EyeState;
  modelAvailable: boolean;
  onAutoEstimate: () => void;
  onChange: (g: RefinerGeometry) => void;
}) {
  return (
    <div>
      <div className="muted" style={{ marginBottom: 6 }}>ojo {side === "left" ? "izquierdo" : "derecho"}</div>
      <GeometricRefiner img={eye.img} geometry={eye.geometry} onChange={onChange} />
      <div className="row" style={{ marginTop: 8 }}>
        <button onClick={onAutoEstimate} disabled={!modelAvailable || eye.busy} className="secondary">
          {eye.busy ? "estimando…" : "Auto-estimar con modelo"}
        </button>
      </div>
    </div>
  );
}

// Estimate iris/pupil from the model mask; eyelid from MediaPipe contour
// landmarks when available, falling back to the iris∪pupil bbox if not.
function estimateGeometry(
  result: SegmentationResult,
  eyelidPoints: { x: number; y: number }[] | undefined,
  displaySize: number,
): RefinerGeometry {
  const { argmax, size } = result;
  const stats: Record<number, { count: number; xs: number[]; ys: number[]; minX: number; maxX: number; minY: number; maxY: number }> = {};
  for (const c of [1, 2]) {
    stats[c] = { count: 0, xs: [], ys: [], minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cls = argmax[y * size + x];
      if (cls === 1 || cls === 2) {
        const s = stats[cls];
        s.count++;
        s.xs.push(x);
        s.ys.push(y);
        if (x < s.minX) s.minX = x;
        if (x > s.maxX) s.maxX = x;
        if (y < s.minY) s.minY = y;
        if (y > s.maxY) s.maxY = y;
      }
    }
  }

  const scale = displaySize / size;
  const fallback = defaultGeometry(displaySize);

  function fitCircle(cls: number) {
    const s = stats[cls];
    if (s.count < 6) return null;
    const cx = (s.xs.reduce((a, b) => a + b, 0) / s.count) * scale;
    const cy = (s.ys.reduce((a, b) => a + b, 0) / s.count) * scale;
    const rx = ((s.maxX - s.minX) / 2) * scale;
    const ry = ((s.maxY - s.minY) / 2) * scale;
    const r = Math.max(4, (rx + ry) / 2);
    return { cx, cy, r };
  }

  const iris = fitCircle(1) ?? fallback.iris;
  const pupil = fitCircle(2) ?? { cx: iris.cx, cy: iris.cy, r: iris.r * 0.35 };

  const eyelid = eyelidPoints && eyelidPoints.length >= 4
    ? eyelidFromContour(eyelidPoints)
    : eyelidFromMaskBbox(stats, scale, displaySize, fallback.eyelid);

  return { iris, pupil, eyelid };
}

// Fit our 4-handle eyelid shape (2 cantos + 2 Bezier control points) to the
// 16 MediaPipe contour landmarks.
function eyelidFromContour(pts: { x: number; y: number }[]) {
  let leftP = pts[0], rightP = pts[0];
  for (const p of pts) {
    if (p.x < leftP.x) leftP = p;
    if (p.x > rightP.x) rightP = p;
  }
  // Split upper/lower by y relative to the canti midline.
  const yMid = (leftP.y + rightP.y) / 2;
  let upperApex = leftP, lowerApex = leftP;
  let foundUpper = false, foundLower = false;
  for (const p of pts) {
    if (p === leftP || p === rightP) continue;
    if (p.y < yMid) {
      if (!foundUpper || p.y < upperApex.y) { upperApex = p; foundUpper = true; }
    } else {
      if (!foundLower || p.y > lowerApex.y) { lowerApex = p; foundLower = true; }
    }
  }
  // For B(t) = (1-t)²P0 + 2(1-t)t·P1 + t²P2, the midpoint is (P0 + 2P1 + P2)/4.
  // To make the curve pass through `apex` at t=0.5: P1 = (4·apex - P0 - P2)/2.
  const ctlFromApex = (apex: { x: number; y: number }) => ({
    x: (4 * apex.x - leftP.x - rightP.x) / 2,
    y: (4 * apex.y - leftP.y - rightP.y) / 2,
  });
  return {
    canti: { left: { ...leftP }, right: { ...rightP } },
    upperCtl: foundUpper ? ctlFromApex(upperApex) : { x: (leftP.x + rightP.x) / 2, y: yMid - 20 },
    lowerCtl: foundLower ? ctlFromApex(lowerApex) : { x: (leftP.x + rightP.x) / 2, y: yMid + 20 },
  };
}

function eyelidFromMaskBbox(
  stats: Record<number, { count: number; minX: number; maxX: number; minY: number; maxY: number }>,
  scale: number,
  displaySize: number,
  fallback: RefinerGeometry["eyelid"],
) {
  const u = stats[1].count > 0 ? stats[1] : stats[2];
  if (u.count < 6) return fallback;
  const left = { x: u.minX * scale, y: ((u.minY + u.maxY) / 2) * scale };
  const right = { x: u.maxX * scale, y: ((u.minY + u.maxY) / 2) * scale };
  const cx = ((u.minX + u.maxX) / 2) * scale;
  const upperCtl = { x: cx, y: Math.max(0, u.minY * scale - (u.maxY - u.minY) * scale * 0.3) };
  const lowerCtl = { x: cx, y: Math.min(displaySize, u.maxY * scale + (u.maxY - u.minY) * scale * 0.3) };
  return { canti: { left, right }, upperCtl, lowerCtl };
}
