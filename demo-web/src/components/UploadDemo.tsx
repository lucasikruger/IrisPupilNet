import { useEffect, useRef, useState } from "react";
import { EyeCropper, type EyeCrop } from "@lib/cropper";
import { loadFaceLandmarker } from "@lib/mediapipe";
import { OnnxSegmenter, loadManifest, type ModelSpec } from "@lib/onnx";
import { renderCropWithMask } from "@lib/render";

type ShowMode = "crop" | "mask" | "blend";

export default function UploadDemo() {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const leftRef = useRef<HTMLCanvasElement | null>(null);
  const rightRef = useRef<HTMLCanvasElement | null>(null);
  const segmenterRef = useRef<OnnxSegmenter | null>(null);

  const [models, setModels] = useState<ModelSpec[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [status, setStatus] = useState("cargá una imagen para empezar");
  const [show, setShow] = useState<ShowMode>("blend");
  const [imgUrl, setImgUrl] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const manifest = await loadManifest();
        setModels(manifest);
        if (manifest.length > 0) setSelectedName(manifest[0].name);
      } catch (e) {
        setStatus(`error: ${(e as Error).message}`);
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedName) return;
    const spec = models.find((m) => m.name === selectedName);
    if (!spec) return;
    (async () => {
      const seg = segmenterRef.current ?? new OnnxSegmenter();
      segmenterRef.current = seg;
      await seg.load(spec);
      // If the user uploaded a photo before the model finished loading,
      // reprocess now so the canvases get the segmentation pass instead
      // of staying on the raw-crop fallback.
      if (imgRef.current?.complete) process();
    })();
  }, [selectedName, models]);

  async function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    setImgUrl(url);
    setStatus("procesando…");
  }

  async function process() {
    const img = imgRef.current;
    if (!img || !img.complete) return;
    const cropper = new EyeCropper({ outputSize: 160 });
    const landmarker = await loadFaceLandmarker("IMAGE");
    const crops = cropper.cropEyes(landmarker, img);
    if (crops.length === 0) {
      setStatus("no se detectó cara — probá otra foto");
      return;
    }
    for (const ref of [leftRef, rightRef]) {
      const c = ref.current;
      if (c) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    }
    const seg = segmenterRef.current;
    const segReady = seg?.ready ?? false;
    for (const crop of crops) {
      const target = crop.side === "left" ? leftRef.current : rightRef.current;
      if (!target) continue;
      if (!segReady) {
        // Model still loading — paint the raw crop so the user sees something.
        // The useEffect that drives seg.load() will re-invoke process() once
        // the session is ready and replace these with the segmented version.
        target.width = crop.canvas.width;
        target.height = crop.canvas.height;
        target.getContext("2d")?.drawImage(crop.canvas, 0, 0);
        continue;
      }
      try {
        const r = await seg!.run(crop.canvas);
        renderCropWithMask(target, crop.canvas, r, { show });
      } catch (err) {
        // Don't leave the canvas blank if inference dies — fall back to the
        // raw crop and surface the error in the status panel.
        target.width = crop.canvas.width;
        target.height = crop.canvas.height;
        target.getContext("2d")?.drawImage(crop.canvas, 0, 0);
        setStatus(`error en segmentación: ${(err as Error).message}`);
      }
    }
    setStatus(segReady
      ? `${crops.length} ojo(s) procesado(s)`
      : `${crops.length} ojo(s) recortado(s) · esperando modelo…`);
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 18, alignItems: "start" }}>
      <div className="panel">
        {imgUrl && (
          <img
            ref={imgRef}
            src={imgUrl}
            alt=""
            onLoad={process}
            style={{ width: "100%", borderRadius: 6, background: "#000" }}
          />
        )}
        {!imgUrl && (
          <div className="muted" style={{ textAlign: "center", padding: 40 }}>
            Elegí una imagen para empezar
          </div>
        )}
        <div style={{ display: "flex", gap: 14, marginTop: 14 }}>
          {[leftRef, rightRef].map((r, i) => (
            <div
              key={i}
              style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}
            >
              <span className="muted">{i === 0 ? "izq" : "der"}</span>
              <canvas
                ref={r}
                style={{
                  width: "100%",
                  maxWidth: 260,
                  aspectRatio: "1 / 1",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  background: "#0a0c10",
                }}
              />
            </div>
          ))}
        </div>
      </div>
      <aside className="panel" style={{ display: "grid", gap: 12 }}>
        <div>
          <strong>Estado</strong>
          <div className="muted" style={{ marginTop: 4 }}>{status}</div>
        </div>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        {models.length > 1 && (
          <label>
            Modelo
            <select
              value={selectedName ?? ""}
              onChange={(e) => {
                setSelectedName(e.target.value);
                if (imgRef.current?.complete) process();
              }}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </label>
        )}
        <fieldset style={{ border: "1px solid var(--border)", borderRadius: 6, padding: 10 }}>
          <legend className="muted">Overlay</legend>
          {(["crop", "blend", "mask"] as const).map((m) => (
            <label key={m} style={{ display: "flex", marginTop: 4 }}>
              <input
                type="radio"
                name="show"
                checked={show === m}
                onChange={() => {
                  setShow(m);
                  if (imgRef.current?.complete) process();
                }}
              />
              {m}
            </label>
          ))}
        </fieldset>
      </aside>
    </div>
  );
}
