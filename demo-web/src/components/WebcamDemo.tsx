import { useEffect, useRef, useState } from "react";
import { EyeCropper, type CropMode } from "@lib/cropper";
import { loadFaceLandmarker } from "@lib/mediapipe";
import { OnnxSegmenter, loadManifest, type ModelSpec, type SegmentationResult } from "@lib/onnx";
import { drawPreprocessed, drawProbHeatmap, renderCropWithMask } from "@lib/render";
import type { PostprocessName, PostprocessOptions } from "@lib/postprocess";
import { POSTPROCESS_VARIANTS } from "@lib/postprocess";

type ShowMode = "crop" | "mask" | "blend";

const VIEW_OPTIONS = [
  { id: "crop", label: "crop" },
  { id: "preprocessed", label: "preproc" },
  { id: "raw", label: "raw" },
  { id: "post", label: "post" },
  { id: "ellipse", label: "ellipse" },
  { id: "landmarks", label: "eyelid" },
  { id: "heatmap", label: "heatmap" },
] as const;
type ViewId = (typeof VIEW_OPTIONS)[number]["id"];

const OUTPUT_SIZE_OPTIONS = [128, 160, 192, 224, 256] as const;
const HEATMAP_CLASSES = ["bg", "iris", "pupil"] as const;

interface CameraOption {
  deviceId: string;
  label: string;
}

export default function WebcamDemo() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Per-side, per-view canvas refs. Keyed via React state + computed ids.
  const canvasMapRef = useRef<Record<string, HTMLCanvasElement | null>>({});
  const segmenterRef = useRef<OnnxSegmenter | null>(null);
  const cropperRef = useRef<EyeCropper | null>(null);
  const rafRef = useRef<number | null>(null);

  const [models, setModels] = useState<ModelSpec[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("inicializando…");

  // Sidebar state — held in refs so render loop reads fresh values without restart.
  const [overlay, setOverlay] = useState<ShowMode>("blend");
  const overlayRef = useRef<ShowMode>("blend");
  useEffect(() => { overlayRef.current = overlay; }, [overlay]);

  const [blendAlpha, setBlendAlpha] = useState(0.55);
  const blendAlphaRef = useRef(0.55);
  useEffect(() => { blendAlphaRef.current = blendAlpha; }, [blendAlpha]);

  const [bw, setBw] = useState(false);
  const bwRef = useRef(false);
  useEffect(() => { bwRef.current = bw; }, [bw]);

  const [postprocess, setPostprocess] = useState<PostprocessName>("morph");
  const postprocessRef = useRef<PostprocessName>("morph");
  useEffect(() => { postprocessRef.current = postprocess; }, [postprocess]);

  const [showIris, setShowIris] = useState(true);
  const showIrisRef = useRef(true);
  useEffect(() => { showIrisRef.current = showIris; }, [showIris]);

  const [showPupil, setShowPupil] = useState(true);
  const showPupilRef = useRef(true);
  useEffect(() => { showPupilRef.current = showPupil; }, [showPupil]);

  const [showEllipse, setShowEllipse] = useState(true);
  const showEllipseRef = useRef(true);
  useEffect(() => { showEllipseRef.current = showEllipse; }, [showEllipse]);

  const [showPupilCenter, setShowPupilCenter] = useState(true);
  const showPupilCenterRef = useRef(true);
  useEffect(() => { showPupilCenterRef.current = showPupilCenter; }, [showPupilCenter]);

  const [showEyelid, setShowEyelid] = useState(true);
  const showEyelidRef = useRef(true);
  useEffect(() => { showEyelidRef.current = showEyelid; }, [showEyelid]);

  const [enabledViews, setEnabledViews] = useState<Record<ViewId, boolean>>({
    crop: true,
    preprocessed: true,
    raw: true,
    post: true,
    ellipse: true,
    landmarks: true,
    heatmap: false,
  });
  const enabledViewsRef = useRef(enabledViews);
  useEffect(() => { enabledViewsRef.current = enabledViews; }, [enabledViews]);

  // --- Debug controls ---
  const [swapClasses, setSwapClasses] = useState(false);
  const swapClassesRef = useRef(false);
  useEffect(() => { swapClassesRef.current = swapClasses; }, [swapClasses]);

  const [mirror, setMirror] = useState(false);
  const mirrorRef = useRef(false);
  useEffect(() => { mirrorRef.current = mirror; }, [mirror]);

  const [cropMode, setCropMode] = useState<CropMode>("eye");
  const cropModeRef = useRef<CropMode>("eye");
  useEffect(() => { cropModeRef.current = cropMode; }, [cropMode]);

  const [paddingX, setPaddingX] = useState(0.4);
  const paddingXRef = useRef(0.4);
  useEffect(() => { paddingXRef.current = paddingX; }, [paddingX]);

  const [paddingY, setPaddingY] = useState(0.4);
  const paddingYRef = useRef(0.4);
  useEffect(() => { paddingYRef.current = paddingY; }, [paddingY]);

  const [verticalAnchor, setVerticalAnchor] = useState(0);
  const verticalAnchorRef = useRef(0);
  useEffect(() => { verticalAnchorRef.current = verticalAnchor; }, [verticalAnchor]);

  const [outputSize, setOutputSize] = useState<number>(160);
  const outputSizeRef = useRef(160);
  useEffect(() => { outputSizeRef.current = outputSize; }, [outputSize]);

  const [probThreshold, setProbThreshold] = useState(0.0);
  const probThresholdRef = useRef(0.0);
  useEffect(() => { probThresholdRef.current = probThreshold; }, [probThreshold]);

  const [morphKsizeIris, setMorphKsizeIris] = useState(5);
  const morphKsizeIrisRef = useRef(5);
  useEffect(() => { morphKsizeIrisRef.current = morphKsizeIris; }, [morphKsizeIris]);

  const [morphKsizePupil, setMorphKsizePupil] = useState(3);
  const morphKsizePupilRef = useRef(3);
  useEffect(() => { morphKsizePupilRef.current = morphKsizePupil; }, [morphKsizePupil]);

  const [minIrisPixels, setMinIrisPixels] = useState(0);
  const minIrisPixelsRef = useRef(0);
  useEffect(() => { minIrisPixelsRef.current = minIrisPixels; }, [minIrisPixels]);

  const [minPupilPixels, setMinPupilPixels] = useState(0);
  const minPupilPixelsRef = useRef(0);
  useEffect(() => { minPupilPixelsRef.current = minPupilPixels; }, [minPupilPixels]);

  const [heatmapClass, setHeatmapClass] = useState<number>(1); // 0=bg, 1=iris, 2=pupil
  const heatmapClassRef = useRef(1);
  useEffect(() => { heatmapClassRef.current = heatmapClass; }, [heatmapClass]);

  // Apply cropper option changes between frames.
  useEffect(() => {
    if (cropperRef.current) {
      cropperRef.current.paddingX = paddingX;
      cropperRef.current.paddingY = paddingY;
      cropperRef.current.verticalAnchor = verticalAnchor;
      cropperRef.current.outputSize = outputSize;
      cropperRef.current.mirror = mirror;
      cropperRef.current.cropMode = cropMode;
    }
  }, [paddingX, paddingY, verticalAnchor, outputSize, mirror, cropMode]);

  const [fps, setFps] = useState(0);
  const [eyesDetected, setEyesDetected] = useState(0);
  const [pupilOffset, setPupilOffset] = useState<{ left?: number; right?: number }>({});

  const [cameras, setCameras] = useState<CameraOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(true);
  // Gallery — each capture is a GalleryItem with the full frame URL and one
  // url per (side, view) tile. All URLs are object URLs and revoked on remove.
  type GallerySnap = { id: string; ts: number; full: string; tiles: { side: "left" | "right"; view: ViewId; url: string }[]; note: string };
  const [gallery, setGallery] = useState<GallerySnap[]>([]);
  const [autoDownload, setAutoDownload] = useState(false);

  // Revoke all object URLs on unmount.
  useEffect(() => {
    return () => {
      for (const item of gallery) {
        URL.revokeObjectURL(item.full);
        for (const t of item.tiles) URL.revokeObjectURL(t.url);
      }
    };
    // intentionally only on unmount — manual revoke happens in removeGalleryItem
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canvasToBlobAsync = (c: HTMLCanvasElement, type = "image/png", quality?: number): Promise<Blob | null> =>
    new Promise((resolve) => c.toBlob((b) => resolve(b), type, quality));

  const captureSnapshot = async () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;
    const ts = Date.now();
    const stamp = new Date(ts).toISOString().replace(/[:.]/g, "-").slice(0, -5);

    const full = document.createElement("canvas");
    full.width = video.videoWidth;
    full.height = video.videoHeight;
    full.getContext("2d")?.drawImage(video, 0, 0);
    const fullBlob = await canvasToBlobAsync(full, "image/jpeg", 0.92);
    if (!fullBlob) return;
    const fullUrl = URL.createObjectURL(fullBlob);

    const tiles: GallerySnap["tiles"] = [];
    for (const side of ["left", "right"] as const) {
      for (const v of VIEW_OPTIONS) {
        const c = canvasMapRef.current[`${side}-${v.id}`];
        if (!c || c.width === 0 || c.height === 0) continue;
        const b = await canvasToBlobAsync(c, "image/png");
        if (!b) continue;
        tiles.push({ side, view: v.id, url: URL.createObjectURL(b) });
      }
    }

    const item: GallerySnap = {
      id: `snap-${ts}-${Math.random().toString(36).slice(2, 7)}`,
      ts,
      full: fullUrl,
      tiles,
      note: "",
    };
    setGallery((g) => [item, ...g]);

    if (autoDownload) {
      triggerDownload(fullUrl, `iris-seg-${stamp}.jpg`);
      for (const t of tiles) triggerDownload(t.url, `iris-seg-${stamp}-${t.side}-${t.view}.png`);
    }
  };

  const removeGalleryItem = (id: string) => {
    setGallery((g) => {
      const item = g.find((x) => x.id === id);
      if (item) {
        URL.revokeObjectURL(item.full);
        for (const t of item.tiles) URL.revokeObjectURL(t.url);
      }
      return g.filter((x) => x.id !== id);
    });
  };

  const downloadGalleryItem = (item: GallerySnap) => {
    const stamp = new Date(item.ts).toISOString().replace(/[:.]/g, "-").slice(0, -5);
    triggerDownload(item.full, `iris-seg-${stamp}.jpg`);
    for (const t of item.tiles) triggerDownload(t.url, `iris-seg-${stamp}-${t.side}-${t.view}.png`);
  };

  const clearGallery = () => {
    setGallery((g) => {
      for (const item of g) {
        URL.revokeObjectURL(item.full);
        for (const t of item.tiles) URL.revokeObjectURL(t.url);
      }
      return [];
    });
  };

  const updateNote = (id: string, note: string) => {
    setGallery((g) => g.map((x) => (x.id === id ? { ...x, note } : x)));
  };

  // Manifest
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setStatus("cargando manifest…");
        const manifest = await loadManifest();
        if (cancelled) return;
        setModels(manifest);
        if (manifest.length > 0) setSelectedName(manifest[0].name);
      } catch (e) {
        setStatus(`error cargando manifest: ${(e as Error).message}`);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Model load
  useEffect(() => {
    if (!selectedName) return;
    const spec = models.find((m) => m.name === selectedName);
    if (!spec) return;
    let cancelled = false;
    (async () => {
      setStatus(`cargando modelo ${spec.name}…`);
      const seg = segmenterRef.current ?? new OnnxSegmenter();
      segmenterRef.current = seg;
      try {
        await seg.load(spec);
        if (cancelled) return;
        setStatus(`listo — ${spec.architecture ?? spec.name}, IoU=${spec.val_iou?.toFixed(3) ?? "?"}`);
      } catch (e) {
        setStatus(`error cargando modelo: ${(e as Error).message}`);
      }
    })();
    return () => { cancelled = true; };
  }, [selectedName, models]);

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

  // Camera + render loop
  useEffect(() => {
    if (!cameraOn) {
      setFps(0);
      setEyesDetected(0);
      setStatus("cámara detenida");
      return;
    }
    let cancelled = false;
    let frameCount = 0;
    let fpsTimer = performance.now();
    cropperRef.current ??= new EyeCropper({
      outputSize: outputSizeRef.current,
      paddingX: paddingXRef.current,
      paddingY: paddingYRef.current,
      verticalAnchor: verticalAnchorRef.current,
      cropMode: cropModeRef.current,
    });
    cropperRef.current.paddingX = paddingXRef.current;
    cropperRef.current.paddingY = paddingYRef.current;
    cropperRef.current.verticalAnchor = verticalAnchorRef.current;
    cropperRef.current.outputSize = outputSizeRef.current;
    cropperRef.current.mirror = mirrorRef.current;
    cropperRef.current.cropMode = cropModeRef.current;

    (async () => {
      try {
        setStatus((s) =>
          s.startsWith("listo") || s.startsWith("inicializando") ? "pidiendo webcam…" : s,
        );
        const constraints: MediaStreamConstraints = {
          video: selectedDeviceId
            ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();

        const activeId = stream.getVideoTracks()[0]?.getSettings().deviceId;
        await refreshCameras(activeId);

        setStatus("cargando face landmarker…");
        const landmarker = await loadFaceLandmarker();
        if (cancelled) return;

        const loop = async () => {
          if (cancelled) return;
          const now = performance.now();
          frameCount++;
          if (now - fpsTimer >= 1000) {
            setFps(frameCount);
            frameCount = 0;
            fpsTimer = now;
          }

          const crops = cropperRef.current?.cropEyes(landmarker, video, now) ?? [];
          setEyesDetected(crops.length);

          const seenSides = new Set<string>();
          const sidePupilOff: { left?: number; right?: number } = {};

          for (const crop of crops) {
            seenSides.add(crop.side);
            const seg = segmenterRef.current;
            const spec = seg?.currentSpec;
            const enabled = enabledViewsRef.current;

            // 1. raw crop
            const cropCanvas = canvasMapRef.current[`${crop.side}-crop`];
            if (cropCanvas && enabled.crop) {
              cropCanvas.width = crop.canvas.width;
              cropCanvas.height = crop.canvas.height;
              cropCanvas.getContext("2d")?.drawImage(crop.canvas, 0, 0);
            }

            // 2. preprocessed (what the model sees)
            const preCanvas = canvasMapRef.current[`${crop.side}-preprocessed`];
            if (preCanvas && enabled.preprocessed) {
              drawPreprocessed(preCanvas, crop.canvas, spec?.input.preprocess, spec?.input.size ?? 160);
            }

            if (!seg || !spec) continue;

            let result: SegmentationResult;
            try {
              result = await seg.run(crop.canvas);
            } catch (e) {
              console.warn("seg failed:", e);
              continue;
            }

            const ppOpts: PostprocessOptions = {
              morphKsizeIris: morphKsizeIrisRef.current,
              morphKsizePupil: morphKsizePupilRef.current,
              minIrisPixels: minIrisPixelsRef.current,
              minPupilPixels: minPupilPixelsRef.current,
              swapClasses: swapClassesRef.current,
              probThreshold: probThresholdRef.current,
            };

            // 3. raw mask (postprocess = raw, blend, no ellipse)
            if (enabled.raw) {
              const c = canvasMapRef.current[`${crop.side}-raw`];
              if (c) {
                renderCropWithMask(c, crop.canvas, result, {
                  show: "blend",
                  blendAlpha: blendAlphaRef.current,
                  bw: bwRef.current,
                  postprocess: "raw",
                  postprocessOpts: { swapClasses: swapClassesRef.current, probThreshold: probThresholdRef.current },
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                });
              }
            }

            // 4. post-processed (chosen variant, with overlay mode)
            if (enabled.post) {
              const c = canvasMapRef.current[`${crop.side}-post`];
              if (c) {
                renderCropWithMask(c, crop.canvas, result, {
                  show: overlayRef.current,
                  blendAlpha: blendAlphaRef.current,
                  bw: bwRef.current,
                  postprocess: postprocessRef.current,
                  postprocessOpts: ppOpts,
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                  hardMask: true,
                });
              }
            }

            // 5. ellipse-anatomical with ellipses + pupil-centre overlay
            if (enabled.ellipse) {
              const c = canvasMapRef.current[`${crop.side}-ellipse`];
              if (c) {
                const r = renderCropWithMask(c, crop.canvas, result, {
                  show: "blend",
                  blendAlpha: blendAlphaRef.current,
                  bw: bwRef.current,
                  postprocess: "ellipse_anatomical",
                  postprocessOpts: ppOpts,
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                  showEllipse: showEllipseRef.current,
                  showPupilCenter: showPupilCenterRef.current,
                });
                if (r.ellipses.pupil && r.ellipses.iris) {
                  const dx = r.ellipses.pupil.cx - r.ellipses.iris.cx;
                  const dy = r.ellipses.pupil.cy - r.ellipses.iris.cy;
                  const irisR = Math.max(r.ellipses.iris.rxMajor, r.ellipses.iris.rxMinor);
                  if (irisR > 0) sidePupilOff[crop.side] = Math.hypot(dx, dy) / irisR;
                }
              }
            }

            // 6. eyelid landmarks overlay on plain crop
            if (enabled.landmarks) {
              const c = canvasMapRef.current[`${crop.side}-landmarks`];
              if (c) {
                renderCropWithMask(c, crop.canvas, result, {
                  show: "blend",
                  blendAlpha: blendAlphaRef.current,
                  bw: bwRef.current,
                  postprocess: postprocessRef.current,
                  postprocessOpts: ppOpts,
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                  showEyelid: showEyelidRef.current,
                  eyelidPoints: crop.eyelidPoints,
                });
              }
            }

            // 7. probability heatmap for chosen class
            if (enabled.heatmap) {
              const c = canvasMapRef.current[`${crop.side}-heatmap`];
              if (c) drawProbHeatmap(c, result, heatmapClassRef.current);
            }
          }

          // Clear canvases of sides with no eye this frame
          for (const side of ["left", "right"] as const) {
            if (seenSides.has(side)) continue;
            for (const v of VIEW_OPTIONS) {
              const c = canvasMapRef.current[`${side}-${v.id}`];
              if (c && c.width > 0) c.getContext("2d")?.clearRect(0, 0, c.width, c.height);
            }
          }
          setPupilOffset(sidePupilOff);

          rafRef.current = requestAnimationFrame(loop);
        };
        rafRef.current = requestAnimationFrame(loop);
      } catch (e) {
        setStatus(`error: ${(e as Error).message}`);
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      const video = videoRef.current;
      if (video?.srcObject instanceof MediaStream) {
        video.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDeviceId, cameraOn]);

  const activeViews = VIEW_OPTIONS.filter((v) => enabledViews[v.id]);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "300px minmax(0, 1fr) 320px", gap: 16, alignItems: "start" }}>
      <aside className="panel" style={{ display: "grid", gap: 16, position: "sticky", top: 80, maxHeight: "calc(100vh - 100px)", overflowY: "auto" }}>
        {cameras.length > 0 && (
          <section style={{ display: "grid", gap: 6 }}>
            <h3>Cámara</h3>
            <select
              value={selectedDeviceId ?? ""}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              disabled={cameras.length < 2}
            >
              {cameras.map((c) => (
                <option key={c.deviceId} value={c.deviceId}>{c.label}</option>
              ))}
            </select>
          </section>
        )}

        {models.length > 1 && (
          <section style={{ display: "grid", gap: 6 }}>
            <h3>Modelo</h3>
            <select
              value={selectedName ?? ""}
              onChange={(e) => setSelectedName(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </section>
        )}

        <section style={{ display: "grid", gap: 6 }}>
          <h3>Postproceso</h3>
          <select value={postprocess} onChange={(e) => setPostprocess(e.target.value as PostprocessName)}>
            {POSTPROCESS_VARIANTS.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <span className="muted mono" style={{ fontSize: 11 }}>
            morph = default · ellipse_anatomical = mejor para gaze
          </span>
        </section>

        <section style={{ display: "grid", gap: 8 }}>
          <h3>Vistas</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
            {VIEW_OPTIONS.map((v) => (
              <label key={v.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="checkbox"
                  checked={enabledViews[v.id]}
                  onChange={(e) =>
                    setEnabledViews((m) => ({ ...m, [v.id]: e.target.checked }))
                  }
                />
                <span className="mono" style={{ fontSize: 12 }}>{v.label}</span>
              </label>
            ))}
          </div>
        </section>

        <section style={{ display: "grid", gap: 10 }}>
          <h3>Overlay (vista «post»)</h3>
          <div className="segmented" role="tablist">
            {(["crop", "blend", "mask"] as const).map((m) => (
              <button
                key={m}
                type="button"
                className={overlay === m ? "active" : ""}
                onClick={() => setOverlay(m)}
                role="tab"
                aria-selected={overlay === m}
              >
                {m}
              </button>
            ))}
          </div>
          {overlay === "blend" && (
            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                opacidad {(blendAlpha * 100).toFixed(0)}%
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={blendAlpha}
                onChange={(e) => setBlendAlpha(parseFloat(e.target.value))}
              />
            </label>
          )}
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
            <input type="checkbox" checked={bw} onChange={(e) => setBw(e.target.checked)} />
            <span>como entrada del modelo (BW)</span>
          </label>
        </section>

        <section style={{ display: "grid", gap: 6 }}>
          <h3>Clases & overlays</h3>
          <label><input type="checkbox" checked={showIris} onChange={(e) => setShowIris(e.target.checked)} /><span>iris</span></label>
          <label><input type="checkbox" checked={showPupil} onChange={(e) => setShowPupil(e.target.checked)} /><span>pupila</span></label>
          <label><input type="checkbox" checked={showEllipse} onChange={(e) => setShowEllipse(e.target.checked)} /><span>elipse ajustada</span></label>
          <label><input type="checkbox" checked={showPupilCenter} onChange={(e) => setShowPupilCenter(e.target.checked)} /><span>centro pupila</span></label>
          <label><input type="checkbox" checked={showEyelid} onChange={(e) => setShowEyelid(e.target.checked)} /><span>puntos párpado</span></label>
        </section>

        <details open style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: 10 }}>
          <summary style={{ cursor: "pointer", fontFamily: "var(--mono)", fontSize: 13, fontWeight: 600 }}>Debug</summary>
          <div style={{ display: "grid", gap: 10, marginTop: 10 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="checkbox" checked={swapClasses} onChange={(e) => setSwapClasses(e.target.checked)} />
              <span>swap iris ↔ pupila</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />
              <span>flip horizontal del crop</span>
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>modo de crop</span>
              <select
                value={cropMode}
                onChange={(e) => {
                  const m = e.target.value as CropMode;
                  setCropMode(m);
                  if (m === "eye_tight") {
                    setPaddingX(0.45);
                    setPaddingY(0.10);
                    setVerticalAnchor(0.18);
                  } else if (m === "iris") {
                    setPaddingX(0.10);
                    setPaddingY(0.10);
                    setVerticalAnchor(0);
                  } else {
                    setPaddingX(0.4);
                    setPaddingY(0.4);
                    setVerticalAnchor(0);
                  }
                }}
              >
                <option value="eye">eye — 16 landmarks párpado (default)</option>
                <option value="eye_tight">eye_tight — sin cejas (anchor abajo)</option>
                <option value="iris">iris — 5 landmarks iris (sin cejas)</option>
              </select>
              <span className="muted mono" style={{ fontSize: 10 }}>
                {cropMode === "iris" && "iris-only requiere el modelo 478-pt de MediaPipe"}
              </span>
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                padding X {(paddingX * 100).toFixed(0)}%
              </span>
              <input
                type="range" min={0} max={1.5} step={0.05}
                value={paddingX}
                onChange={(e) => setPaddingX(parseFloat(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                padding Y {(paddingY * 100).toFixed(0)}%
              </span>
              <input
                type="range" min={0} max={1.5} step={0.05}
                value={paddingY}
                onChange={(e) => setPaddingY(parseFloat(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                vertical anchor {verticalAnchor >= 0 ? "+" : ""}{(verticalAnchor * 100).toFixed(0)}% (↓ deja afuera la ceja)
              </span>
              <input
                type="range" min={-0.3} max={0.5} step={0.02}
                value={verticalAnchor}
                onChange={(e) => setVerticalAnchor(parseFloat(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>tamaño del crop (px)</span>
              <select value={outputSize} onChange={(e) => setOutputSize(parseInt(e.target.value, 10))}>
                {OUTPUT_SIZE_OPTIONS.map((s) => (
                  <option key={s} value={s}>{s}×{s}</option>
                ))}
              </select>
              <span className="muted mono" style={{ fontSize: 10 }}>
                {outputSize !== 160 && "⚠ el modelo se entrenó a 160×160"}
              </span>
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                umbral confianza {(probThreshold * 100).toFixed(0)}%
              </span>
              <input
                type="range" min={0} max={0.99} step={0.01}
                value={probThreshold}
                onChange={(e) => setProbThreshold(parseFloat(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>morph kernel iris: {morphKsizeIris}</span>
              <input
                type="range" min={1} max={11} step={2}
                value={morphKsizeIris}
                onChange={(e) => setMorphKsizeIris(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>morph kernel pupila: {morphKsizePupil}</span>
              <input
                type="range" min={1} max={9} step={2}
                value={morphKsizePupil}
                onChange={(e) => setMorphKsizePupil(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                min iris px: {minIrisPixels}
              </span>
              <input
                type="range" min={0} max={2000} step={50}
                value={minIrisPixels}
                onChange={(e) => setMinIrisPixels(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                min pupila px: {minPupilPixels}
              </span>
              <input
                type="range" min={0} max={500} step={10}
                value={minPupilPixels}
                onChange={(e) => setMinPupilPixels(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>heatmap: clase</span>
              <select value={heatmapClass} onChange={(e) => setHeatmapClass(parseInt(e.target.value, 10))}>
                {HEATMAP_CLASSES.map((label, idx) => (
                  <option key={label} value={idx}>{idx} — {label}</option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={() => {
                setSwapClasses(false);
                setMirror(false);
                setCropMode("eye");
                setPaddingX(0.4);
                setPaddingY(0.4);
                setVerticalAnchor(0);
                setOutputSize(160);
                setProbThreshold(0);
                setMorphKsizeIris(5);
                setMorphKsizePupil(3);
                setMinIrisPixels(0);
                setMinPupilPixels(0);
                setHeatmapClass(1);
              }}
              style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", marginTop: 4 }}
            >
              reset defaults
            </button>
          </div>
        </details>
      </aside>

      <div className="panel" style={{ display: "grid", gap: 16 }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ width: "100%", borderRadius: 8, background: "#000", display: "block" }}
        />
        <div style={{ display: "grid", gap: 10 }}>
          <div className="muted mono" style={{ fontSize: 11, letterSpacing: 0.5 }}>
            vistas activas — {activeViews.length}/{VIEW_OPTIONS.length}
          </div>
          {(["left", "right"] as const).map((side) => (
            <SideRow
              key={side}
              side={side}
              views={activeViews}
              canvasMapRef={canvasMapRef}
              pupilOffsetFrac={pupilOffset[side]}
            />
          ))}
        </div>
      </div>

      <aside className="panel" style={{ display: "grid", gap: 16, position: "sticky", top: 80, maxHeight: "calc(100vh - 100px)", overflowY: "auto" }}>
        <section>
          <h3>Estado</h3>
          <div className="muted mono" style={{ fontSize: 12 }}>{status}</div>
          <div className="mono" style={{ fontSize: 12, marginTop: 4, color: "var(--fg)" }}>
            <span style={{ color: "var(--accent)" }}>{fps}</span>
            <span className="muted"> fps · </span>
            <span style={{ color: "var(--accent)" }}>{eyesDetected}</span>
            <span className="muted">/2 ojos</span>
          </div>
          {(pupilOffset.left !== undefined || pupilOffset.right !== undefined) && (
            <div className="mono" style={{ fontSize: 11, marginTop: 6, color: "var(--muted)" }}>
              pupil/iris offset:
              {pupilOffset.left !== undefined && (
                <span style={{ marginLeft: 6, color: "var(--accent)" }}>
                  izq {(pupilOffset.left * 100).toFixed(0)}%
                </span>
              )}
              {pupilOffset.right !== undefined && (
                <span style={{ marginLeft: 6, color: "var(--accent)" }}>
                  der {(pupilOffset.right * 100).toFixed(0)}%
                </span>
              )}
            </div>
          )}
        </section>

        <section style={{ display: "grid", gap: 8 }}>
          <h3>Captura</h3>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={captureSnapshot}
              disabled={!cameraOn || fps === 0}
            >
              📸 capturar
            </button>
            {cameraOn ? (
              <button
                type="button"
                onClick={() => setCameraOn(false)}
                style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)" }}
              >
                stop
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setCameraOn(true)}
                style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)" }}
              >
                start
              </button>
            )}
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={autoDownload} onChange={(e) => setAutoDownload(e.target.checked)} />
            <span className="muted mono" style={{ fontSize: 11 }}>auto-descargar al capturar</span>
          </label>
          <div className="muted mono" style={{ fontSize: 11 }}>
            galería: <span style={{ color: "var(--accent)" }}>{gallery.length}</span> {gallery.length === 1 ? "foto" : "fotos"}
          </div>
          {gallery.length > 0 && (
            <button
              type="button"
              onClick={clearGallery}
              style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)" }}
            >
              vaciar galería
            </button>
          )}
        </section>

        <section style={{ display: "grid", gap: 10 }}>
          <h3>Galería</h3>
          {gallery.length === 0 ? (
            <div className="muted mono" style={{ fontSize: 11 }}>
              sin capturas todavía — apretá 📸 capturar
            </div>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              {gallery.map((item) => (
                <GalleryItem
                  key={item.id}
                  item={item}
                  onRemove={() => removeGalleryItem(item.id)}
                  onDownload={() => downloadGalleryItem(item)}
                  onNote={(n) => updateNote(item.id, n)}
                />
              ))}
            </div>
          )}
        </section>
      </aside>
    </div>
  );
}

function GalleryItem({
  item,
  onRemove,
  onDownload,
  onNote,
}: {
  item: { id: string; ts: number; full: string; tiles: { side: "left" | "right"; view: ViewId; url: string }[]; note: string };
  onRemove: () => void;
  onDownload: () => void;
  onNote: (n: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const tilesBySide: Record<"left" | "right", typeof item.tiles> = { left: [], right: [] };
  for (const t of item.tiles) tilesBySide[t.side].push(t);
  const ts = new Date(item.ts);
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 10, display: "grid", gap: 8, background: "var(--panel-2)" }}>
      <a href={item.full} target="_blank" rel="noreferrer">
        <img src={item.full} alt="" style={{ width: "100%", borderRadius: 6, display: "block", border: "1px solid var(--border)" }} />
      </a>
      <div className="mono" style={{ fontSize: 11 }}>
        {ts.toLocaleTimeString()} <span className="muted">· {item.tiles.length} tiles</span>
      </div>
      <input
        type="text"
        placeholder="nota"
        value={item.note}
        onChange={(e) => onNote(e.target.value)}
        style={{ background: "var(--panel)", color: "var(--fg)", border: "1px solid var(--border)", padding: "5px 8px", borderRadius: 6, font: "inherit", fontSize: 12 }}
      />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 4 }}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", fontSize: 11, padding: "5px 8px" }}
        >
          {open ? "ocultar" : "ver tiles"}
        </button>
        <button type="button" onClick={onDownload} style={{ fontSize: 11, padding: "5px 8px" }}>↓ todo</button>
        <button
          type="button"
          onClick={onRemove}
          style={{ background: "transparent", border: "1px solid #5a2230", color: "#e08a8a", fontSize: 11, padding: "5px 8px" }}
          title="eliminar"
        >
          ✕
        </button>
      </div>
      {open && (
        <div style={{ display: "grid", gap: 6 }}>
          {(["left", "right"] as const).map((side) =>
            tilesBySide[side].length === 0 ? null : (
              <div key={side} style={{ display: "grid", gap: 4 }}>
                <span className="muted mono" style={{ fontSize: 10, letterSpacing: 0.5 }}>{side}</span>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(56px, 1fr))", gap: 4 }}>
                  {tilesBySide[side].map((t) => (
                    <a key={t.view} href={t.url} download={`iris-seg-${item.ts}-${t.side}-${t.view}.png`} title={t.view}>
                      <div style={{ display: "grid", gap: 2, justifyItems: "center" }}>
                        <span className="muted mono" style={{ fontSize: 9 }}>{t.view}</span>
                        <img src={t.url} alt="" style={{ width: "100%", borderRadius: 4, border: "1px solid var(--border)", display: "block" }} />
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

function SideRow({
  side,
  views,
  canvasMapRef,
  pupilOffsetFrac,
}: {
  side: "left" | "right";
  views: readonly { id: ViewId; label: string }[];
  canvasMapRef: React.MutableRefObject<Record<string, HTMLCanvasElement | null>>;
  pupilOffsetFrac?: number;
}) {
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span className="mono muted" style={{ fontSize: 11, letterSpacing: 0.5 }}>{side}</span>
        {pupilOffsetFrac !== undefined && (
          <span className="mono" style={{ fontSize: 11, color: "var(--accent)" }}>
            pupil/iris offset {(pupilOffsetFrac * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))",
          gap: 8,
        }}
      >
        {views.map((v) => (
          <div key={v.id} style={{ display: "grid", gap: 4, justifyItems: "center" }}>
            <span className="muted mono" style={{ fontSize: 10, letterSpacing: 0.5 }}>{v.label}</span>
            <canvas
              ref={(el) => {
                canvasMapRef.current[`${side}-${v.id}`] = el;
              }}
              style={{
                width: "100%",
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
  );
}

function triggerDownload(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
