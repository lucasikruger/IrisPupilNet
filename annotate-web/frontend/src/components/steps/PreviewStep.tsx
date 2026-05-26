import type { CaptureBundle } from "./CaptureStep";

export default function PreviewStep({
  capture,
  onRetake,
  onConfirm,
}: {
  capture: CaptureBundle;
  onRetake: () => void;
  onConfirm: () => void;
}) {
  return (
    <section className="panel">
      <h2>¿Te gusta?</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        Si los recortes muestran tus ojos completos y enfocados, segui. Si están borrosos o
        cortados, repetí la foto.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: 12 }}>
        <img src={capture.fullDataUrl} alt="" style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />
        <img src={capture.leftDataUrl} alt="izq" style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />
        <img src={capture.rightDataUrl} alt="der" style={{ width: "100%", borderRadius: 8, border: "1px solid var(--border)" }} />
      </div>
      <div className="row" style={{ marginTop: 18 }}>
        <button onClick={onConfirm}>Seguir</button>
        <button className="secondary" onClick={onRetake}>Repetir foto</button>
      </div>
    </section>
  );
}
