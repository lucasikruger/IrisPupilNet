import { useEffect, useRef, useState } from "react";
import { EyeCropper } from "@lib/cropper";
import { loadFaceLandmarker } from "@lib/mediapipe";

export interface CaptureBundle {
  fullBlob: Blob;       // jpeg of the entire frame
  fullDataUrl: string;  // for preview
  leftBlob: Blob;       // png 160x160 crop
  leftDataUrl: string;
  rightBlob: Blob;
  rightDataUrl: string;
  // 16 MediaPipe eyelid contour landmarks, mapped into 160x160 crop coords.
  // Optional: synthetic / non-MediaPipe sources may omit them.
  leftEyelid?: { x: number; y: number }[];
  rightEyelid?: { x: number; y: number }[];
}

interface CameraOption {
  deviceId: string;
  label: string;
}

export default function CaptureStep({ onDone }: { onDone: (b: CaptureBundle) => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<string>("inicializando webcam…");
  const [eyesOK, setEyesOK] = useState(false);
  const [busy, setBusy] = useState(false);
  const [cameraOn, setCameraOn] = useState(true);
  const [cameras, setCameras] = useState<CameraOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const stateRef = useRef<{ stopped: boolean }>({ stopped: false });

  // Re-enumerate cameras (labels populate after permission is granted).
  const refreshCameras = async (preferredId?: string) => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams: CameraOption[] = devices
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `cámara ${i + 1}` }));
      setCameras(cams);
      if (preferredId && cams.some((c) => c.deviceId === preferredId)) {
        setSelectedDeviceId(preferredId);
      } else if (!selectedDeviceId && cams[0]) {
        setSelectedDeviceId(cams[0].deviceId);
      }
    } catch (e) {
      console.warn("enumerateDevices failed:", e);
    }
  };

  useEffect(() => {
    const handler = () => refreshCameras(selectedDeviceId ?? undefined);
    navigator.mediaDevices?.addEventListener?.("devicechange", handler);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", handler);
  }, [selectedDeviceId]);

  useEffect(() => {
    if (!cameraOn) return;
    let raf: number | null = null;
    let landmarker: Awaited<ReturnType<typeof loadFaceLandmarker>> | null = null;
    const cropper = new EyeCropper({ outputSize: 160 });
    stateRef.current.stopped = false;

    (async () => {
      try {
        const constraints: MediaStreamConstraints = {
          video: selectedDeviceId
            ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (stateRef.current.stopped) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        const activeId = stream.getVideoTracks()[0]?.getSettings().deviceId;
        await refreshCameras(activeId);
        setStatus("cargando detector de cara…");
        landmarker = await loadFaceLandmarker();
        setStatus("posicionate frente a la cámara");

        const loop = () => {
          if (stateRef.current.stopped) return;
          const overlay = overlayRef.current;
          const v = videoRef.current;
          if (!overlay || !v || !landmarker) {
            raf = requestAnimationFrame(loop);
            return;
          }
          const W = v.videoWidth;
          const H = v.videoHeight;
          if (W === 0 || H === 0) {
            raf = requestAnimationFrame(loop);
            return;
          }
          if (overlay.width !== W || overlay.height !== H) {
            overlay.width = W;
            overlay.height = H;
          }
          const ctx = overlay.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, W, H);
            const crops = cropper.cropEyes(landmarker, v, performance.now());
            ctx.lineWidth = Math.max(2, Math.round(W / 400));
            for (const c of crops) {
              ctx.strokeStyle = c.side === "left" ? "rgba(80,180,255,0.95)" : "rgba(255,160,90,0.95)";
              ctx.strokeRect(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h);
            }
            setEyesOK(crops.length === 2);
          }
          raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      } catch (e) {
        setStatus(`no se pudo iniciar la webcam: ${(e as Error).message}`);
      }
    })();

    return () => {
      stateRef.current.stopped = true;
      if (raf != null) cancelAnimationFrame(raf);
      const v = videoRef.current;
      if (v?.srcObject instanceof MediaStream) {
        v.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOn, selectedDeviceId]);

  const stopCamera = () => {
    setEyesOK(false);
    setStatus("cámara detenida");
    setCameraOn(false);
  };
  const startCamera = () => {
    setStatus("inicializando webcam…");
    setCameraOn(true);
  };

  async function capture() {
    const v = videoRef.current;
    if (!v) return;
    setBusy(true);
    try {
      const landmarker = await loadFaceLandmarker();
      const cropper = new EyeCropper({ outputSize: 160 });
      const crops = cropper.cropEyes(landmarker, v, performance.now());
      if (crops.length < 2) {
        setStatus("no se detectaron ambos ojos — acercate un poco");
        return;
      }
      const left = crops.find((c) => c.side === "left");
      const right = crops.find((c) => c.side === "right");
      if (!left || !right) {
        setStatus("se detectó un solo ojo — acercate un poco");
        return;
      }
      const full = document.createElement("canvas");
      full.width = v.videoWidth;
      full.height = v.videoHeight;
      full.getContext("2d")?.drawImage(v, 0, 0);

      const [fullBlob, leftBlob, rightBlob] = await Promise.all([
        canvasToBlob(full, "image/jpeg", 0.92),
        canvasToBlob(left.canvas, "image/png"),
        canvasToBlob(right.canvas, "image/png"),
      ]);

      onDone({
        fullBlob,
        fullDataUrl: full.toDataURL("image/jpeg", 0.85),
        leftBlob,
        leftDataUrl: left.canvas.toDataURL("image/png"),
        rightBlob,
        rightDataUrl: right.canvas.toDataURL("image/png"),
        leftEyelid: left.eyelidPoints,
        rightEyelid: right.eyelidPoints,
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Sacate la foto</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        Posicionate frente a la cámara con buena luz. Cuando aparezcan los dos rectángulos
        sobre tus ojos, apretá <strong>Capturar</strong>.
      </p>
      <div style={{ position: "relative", maxWidth: 640, margin: "0 auto" }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ width: "100%", borderRadius: 8, background: "#000", display: "block" }}
        />
        <canvas
          ref={overlayRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
      </div>
      <div className="row" style={{ marginTop: 16, gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={capture} disabled={!eyesOK || busy || !cameraOn}>
          {busy ? "capturando…" : "Capturar"}
        </button>
        {cameraOn ? (
          <button
            type="button"
            onClick={stopCamera}
            style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)" }}
          >
            stop cámara
          </button>
        ) : (
          <button
            type="button"
            onClick={startCamera}
            style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)" }}
          >
            reanudar cámara
          </button>
        )}
        {cameras.length > 0 && (
          <select
            value={selectedDeviceId ?? ""}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={cameras.length < 2 || !cameraOn}
            title={cameras.length < 2 ? "una sola cámara detectada" : "elegir cámara"}
            style={{ minWidth: 180 }}
          >
            {cameras.map((c) => (
              <option key={c.deviceId} value={c.deviceId}>{c.label}</option>
            ))}
          </select>
        )}
        <span className="muted">{status}</span>
      </div>
    </section>
  );
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob returned null"))),
      type,
      quality,
    );
  });
}
