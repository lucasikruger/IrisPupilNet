import { useEffect, useRef, useState } from "react";
import { EyeCropper } from "@lib/cropper";
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
  const bboxOverlayRef = useRef<HTMLCanvasElement | null>(null);
  // Per-side, per-view canvas refs. Keyed via React state + computed ids.
  const canvasMapRef = useRef<Record<string, HTMLCanvasElement | null>>({});
  const segmenterRef = useRef<OnnxSegmenter | null>(null);
  const cropperRef = useRef<EyeCropper | null>(null);
  const rafRef = useRef<number | null>(null);
  // Most recent per-frame crops kept around so captureSnapshot can store raw
  // pixels + logits per side without having to re-run the model on the paused
  // frame.
  const lastCropsRef = useRef<
    Array<{
      side: "left" | "right";
      bbox: { x: number; y: number; w: number; h: number };
      cropImageData: ImageData;
      result: SegmentationResult;
      eyelidPoints: { x: number; y: number }[];
    }>
  >([]);

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

  const [postprocess, setPostprocess] = useState<PostprocessName>("ellipse_anatomical");
  const postprocessRef = useRef<PostprocessName>("ellipse_anatomical");
  useEffect(() => { postprocessRef.current = postprocess; }, [postprocess]);

  const [showSclera, setShowSclera] = useState(true);
  const showScleraRef = useRef(true);
  useEffect(() => { showScleraRef.current = showSclera; }, [showSclera]);

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

  const [targetIrisPct, setTargetIrisPct] = useState(0.35);
  const targetIrisPctRef = useRef(0.35);
  useEffect(() => { targetIrisPctRef.current = targetIrisPct; }, [targetIrisPct]);

  // Freeze-frame mode: while paused the video element is paused and the RAF
  // loop skips inference. Set true by capturar(), cleared by reanudar().
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  useEffect(() => { pausedRef.current = paused; }, [paused]);

  const [showBboxes, setShowBboxes] = useState(true);
  const showBboxesRef = useRef(true);
  useEffect(() => { showBboxesRef.current = showBboxes; }, [showBboxes]);

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
      cropperRef.current.targetIrisPct = targetIrisPct;
      cropperRef.current.outputSize = outputSize;
      cropperRef.current.mirror = mirror;
    }
  }, [targetIrisPct, outputSize, mirror]);

  const [fps, setFps] = useState(0);
  const [eyesDetected, setEyesDetected] = useState(0);
  const [pupilOffset, setPupilOffset] = useState<{ left?: number; right?: number }>({});

  const [cameras, setCameras] = useState<CameraOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(true);
  // Gallery — each capture is a GalleryItem with the full frame URL and one
  // url per (side, view) tile. All URLs are object URLs and revoked on remove.
  // `crops` keeps the raw crop pixels + logits per side so the inspector can
  // re-render any postprocess variant offline without re-running the model.
  type GallerySnap = {
    id: string;
    ts: number;
    full: string;
    tiles: { side: "left" | "right"; view: ViewId; url: string }[];
    crops: Array<{
      side: "left" | "right";
      bbox: { x: number; y: number; w: number; h: number };
      cropImageData: ImageData;
      result: SegmentationResult;
      eyelidPoints: { x: number; y: number }[];
      modelSpec: ModelSpec;
    }>;
    note: string;
  };
  const [gallery, setGallery] = useState<GallerySnap[]>([]);
  const [autoDownload, setAutoDownload] = useState(false);
  const [inspectingId, setInspectingId] = useState<string | null>(null);
  const inspectingItem = inspectingId
    ? gallery.find((g) => g.id === inspectingId) ?? null
    : null;

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

    // Freeze the live feed so the user can inspect what was captured.
    video.pause();
    setPaused(true);

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

    // Snapshot raw crops + logits from the most-recent live frame so the
    // inspector can re-render any postprocess variant offline.
    const spec = segmenterRef.current?.currentSpec ?? null;
    const savedCrops: GallerySnap["crops"] = spec
      ? lastCropsRef.current.map((c) => ({ ...c, modelSpec: spec }))
      : [];

    const item: GallerySnap = {
      id: `snap-${ts}-${Math.random().toString(36).slice(2, 7)}`,
      ts,
      full: fullUrl,
      tiles,
      crops: savedCrops,
      note: "",
    };
    setGallery((g) => [item, ...g]);

    if (autoDownload) {
      triggerDownload(fullUrl, `iris-seg-${stamp}.jpg`);
      for (const t of tiles) triggerDownload(t.url, `iris-seg-${stamp}-${t.side}-${t.view}.png`);
    }
  };

  const resumeLive = () => {
    setPaused(false);
    const video = videoRef.current;
    if (video) void video.play().catch(() => {});
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
      targetIrisPct: targetIrisPctRef.current,
    });
    cropperRef.current.targetIrisPct = targetIrisPctRef.current;
    cropperRef.current.outputSize = outputSizeRef.current;
    cropperRef.current.mirror = mirrorRef.current;

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
          const isPaused = pausedRef.current;
          const now = performance.now();
          if (!isPaused) {
            frameCount++;
            if (now - fpsTimer >= 1000) {
              setFps(frameCount);
              frameCount = 0;
              fpsTimer = now;
            }
          }

          // When paused, replay the frozen crops + logits from the last live
          // frame so the user can tweak sliders and see the tiles update —
          // no face detection, no model inference. cropEyes() is only called
          // in the live branch.
          type LoopCrop = {
            side: "left" | "right";
            bbox: { x: number; y: number; w: number; h: number };
            canvas: HTMLCanvasElement;
            eyelidPoints: { x: number; y: number }[];
            savedResult?: SegmentationResult;
          };
          let crops: LoopCrop[];
          if (isPaused) {
            crops = lastCropsRef.current.map((sav) => {
              const c = document.createElement("canvas");
              c.width = sav.cropImageData.width;
              c.height = sav.cropImageData.height;
              c.getContext("2d")?.putImageData(sav.cropImageData, 0, 0);
              return {
                side: sav.side,
                bbox: sav.bbox,
                canvas: c,
                eyelidPoints: sav.eyelidPoints,
                savedResult: sav.result,
              };
            });
          } else {
            crops = cropperRef.current?.cropEyes(landmarker, video, now) ?? [];
          }
          setEyesDetected(crops.length);

          // Draw eye bboxes on overlay canvas (in source-video coordinates).
          const overlay = bboxOverlayRef.current;
          if (overlay) {
            if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
              overlay.width = video.videoWidth;
              overlay.height = video.videoHeight;
            }
            const octx = overlay.getContext("2d");
            if (octx) {
              octx.clearRect(0, 0, overlay.width, overlay.height);
              if (showBboxesRef.current) {
                octx.lineWidth = Math.max(2, Math.round(video.videoWidth / 400));
                octx.strokeStyle = "rgba(80, 220, 245, 0.95)";
                octx.font = `${Math.max(14, Math.round(video.videoWidth / 80))}px sans-serif`;
                octx.fillStyle = "rgba(80, 220, 245, 0.95)";
                for (const c of crops) {
                  octx.strokeRect(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h);
                  octx.fillText(c.side, c.bbox.x + 4, c.bbox.y - 4);
                }
              }
            }
          }

          // Reset per-frame snapshot store; we'll fill it as we process each crop.
          const frameSnapshot: typeof lastCropsRef.current = [];

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
            if (crop.savedResult) {
              // Paused: reuse the logits captured on the last live frame.
              result = crop.savedResult;
            } else {
              try {
                result = await seg.run(crop.canvas);
              } catch (e) {
                console.warn("seg failed:", e);
                continue;
              }
              // Snapshot this crop's raw pixels + logits so captureSnapshot can
              // freeze the data without needing a second model invocation.
              const cctx = crop.canvas.getContext("2d");
              if (cctx) {
                frameSnapshot.push({
                  side: crop.side,
                  bbox: crop.bbox,
                  cropImageData: cctx.getImageData(0, 0, crop.canvas.width, crop.canvas.height),
                  result,
                  eyelidPoints: crop.eyelidPoints,
                });
              }
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
                  showSclera: showScleraRef.current,
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
                  showSclera: showScleraRef.current,
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                  hardMask: true,
                  eyelidPoints: crop.eyelidPoints,
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
                  showSclera: showScleraRef.current,
                  showIris: showIrisRef.current,
                  showPupil: showPupilRef.current,
                  showEllipse: showEllipseRef.current,
                  showPupilCenter: showPupilCenterRef.current,
                  eyelidPoints: crop.eyelidPoints,
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
                  showSclera: showScleraRef.current,
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
          // Only refresh the snapshot store from a LIVE frame. While paused,
          // frameSnapshot stays empty (we reused the saved logits instead of
          // running the model) and we must not overwrite the captured data.
          if (!isPaused) lastCropsRef.current = frameSnapshot;

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
    <div style={{ display: "grid", gridTemplateColumns: "340px minmax(0, 1fr) 320px", gap: 16, alignItems: "start" }}>
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
            ellipse_anatomical = default · morph = raw morph close
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
          <label><input type="checkbox" checked={showSclera} onChange={(e) => setShowSclera(e.target.checked)} /><span>sclera <span className="muted" style={{ fontSize: 10 }}>(solo modelos 4-class)</span></span></label>
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
              <span className="muted mono" style={{ fontSize: 11 }}>
                iris ocupa {(targetIrisPct * 100).toFixed(0)}% del crop (training = 35%)
              </span>
              <input
                type="range" min={0.15} max={0.65} step={0.01}
                value={targetIrisPct}
                onChange={(e) => setTargetIrisPct(parseFloat(e.target.value))}
              />
              <span className="muted mono" style={{ fontSize: 10 }}>
                side = iris_diameter / pct, usando los 5 landmarks de iris
              </span>
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
                umbral confianza {(probThreshold * 100).toFixed(0)}% (default 0)
              </span>
              <input
                type="range" min={0} max={0.99} step={0.01}
                value={probThreshold}
                onChange={(e) => setProbThreshold(parseFloat(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>morph k iris: {morphKsizeIris} (default 5)</span>
              <input
                type="range" min={1} max={11} step={2}
                value={morphKsizeIris}
                onChange={(e) => setMorphKsizeIris(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>morph k pupila: {morphKsizePupil} (default 3)</span>
              <input
                type="range" min={1} max={9} step={2}
                value={morphKsizePupil}
                onChange={(e) => setMorphKsizePupil(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                min iris px: {minIrisPixels} (default 0)
              </span>
              <input
                type="range" min={0} max={2000} step={50}
                value={minIrisPixels}
                onChange={(e) => setMinIrisPixels(parseInt(e.target.value, 10))}
              />
            </label>

            <label style={{ display: "grid", gap: 4 }}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                min pupila px: {minPupilPixels} (default 0)
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
                setTargetIrisPct(0.35);
                setOutputSize(160);
                setProbThreshold(0);
                setMorphKsizeIris(5);
                setMorphKsizePupil(3);
                setMinIrisPixels(0);
                setMinPupilPixels(0);
                setHeatmapClass(1);
                setBlendAlpha(0.55);
              }}
              style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", marginTop: 4 }}
            >
              reset defaults
            </button>
          </div>
        </details>
      </aside>

      <div className="panel" style={{ display: "grid", gap: 16 }}>
        <div style={{ position: "relative", width: "100%" }}>
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ width: "100%", borderRadius: 8, background: "#000", display: "block" }}
          />
          <canvas
            ref={bboxOverlayRef}
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              borderRadius: 8,
            }}
          />
          {paused && (
            <div
              style={{
                position: "absolute",
                top: 8,
                right: 8,
                background: "rgba(0,0,0,0.75)",
                color: "#fff",
                padding: "4px 10px",
                borderRadius: 6,
                fontFamily: "var(--mono)",
                fontSize: 11,
              }}
            >
              ⏸ pausado
            </div>
          )}
        </div>
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
              disabled={!cameraOn || paused || (fps === 0 && !paused)}
            >
              📸 capturar
            </button>
            {paused && (
              <button
                type="button"
                onClick={resumeLive}
              >
                ▶ reanudar
              </button>
            )}
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
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showBboxes} onChange={(e) => setShowBboxes(e.target.checked)} />
            <span className="muted mono" style={{ fontSize: 11 }}>mostrar bbox de los ojos</span>
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
                  onInspect={() => setInspectingId(item.id)}
                />
              ))}
            </div>
          )}
        </section>


        <section style={{ display: "grid", gap: 6 }}>
          <h3>Modelo activo</h3>
          {(() => {
            const spec = models.find((m) => m.name === selectedName);
            if (!spec) return <div className="muted mono" style={{ fontSize: 11 }}>—</div>;
            return (
              <div style={{ display: "grid", gap: 6, fontSize: 12 }}>
                <div className="mono" style={{ color: "var(--accent)", wordBreak: "break-all" }}>{spec.name}</div>
                <div className="muted mono" style={{ fontSize: 11, lineHeight: 1.4 }}>
                  {spec.architecture ?? "—"}
                </div>
                <div style={{ borderTop: "1px solid var(--border)", paddingTop: 6, display: "grid", gap: 4 }}>
                  <div className="mono" style={{ fontSize: 11 }}>
                    <span className="muted">input:</span> {spec.input.channels}ch · {spec.input.size}×{spec.input.size}
                  </div>
                  <div className="mono" style={{ fontSize: 11 }}>
                    <span className="muted">preproc:</span> {spec.input.preprocess}
                  </div>
                  <div className="mono" style={{ fontSize: 11 }}>
                    <span className="muted">val_iou:</span>{" "}
                    <span style={{ color: "var(--accent)" }}>{spec.val_iou?.toFixed(4) ?? "?"}</span>
                  </div>
                  {spec.dataset && (
                    <div className="mono" style={{ fontSize: 10, lineHeight: 1.5 }}>
                      <span className="muted">train:</span> {spec.dataset}
                    </div>
                  )}
                </div>
              </div>
            );
          })()}
        </section>

        <section style={{ display: "grid", gap: 6 }}>
          <h3>Postproc activo</h3>
          <div style={{ display: "grid", gap: 4 }}>
            <div className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>{postprocess}</div>
            <div className="muted mono" style={{ fontSize: 10, lineHeight: 1.5 }}>
              {postprocess === "raw" && "argmax sin tocar"}
              {postprocess === "largest_cc" && "componente conexo más grande + fill holes"}
              {postprocess === "morph" && "largest_cc + morph close (k=5/3)"}
              {postprocess === "ellipse_iris" && "morph + reemplaza iris por disco de elipse fiteada"}
              {postprocess === "ellipse_iris_pupil" && "+ elipse para pupila sin restricciones"}
              {postprocess === "ellipse_anatomical" && "+ Hu 2018: pupila ⊂ iris, rₚ ≤ 0.40·rᵢ, offset ≤ 0.30·rᵢ"}
              {postprocess === "ellipse_anatomical_clean" && "+ open-iris style: crop por párpado (poly quad) + máscara specular (top 1% luminancia en iris)"}
            </div>
          </div>
        </section>

        <section style={{ display: "grid", gap: 6 }}>
          <h3>Anatomía</h3>
          {eyesDetected === 0 ? (
            <div className="muted mono" style={{ fontSize: 11 }}>
              sin ojos detectados
            </div>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              {(["left", "right"] as const).map((side) => {
                const off = pupilOffset[side];
                if (off === undefined) return null;
                const offPct = (off * 100).toFixed(0);
                const cap = (0.30 * 100).toFixed(0);  // Hu 2018 cap
                const overCap = off > 0.30;
                return (
                  <div key={side} style={{ display: "grid", gap: 4, padding: 8, background: "var(--panel-2)", borderRadius: 6, border: "1px solid var(--border)" }}>
                    <div className="mono muted" style={{ fontSize: 10, letterSpacing: 0.5 }}>{side}</div>
                    <div className="mono" style={{ fontSize: 11 }}>
                      pupil/iris offset:{" "}
                      <span style={{ color: overCap ? "#e08a8a" : "var(--accent)" }}>{offPct}%</span>
                      <span className="muted" style={{ fontSize: 10 }}> (cap {cap}%)</span>
                    </div>
                    {overCap && (
                      <div className="mono" style={{ fontSize: 10, color: "#e08a8a" }}>
                        ⚠ supera el cap Hu 2018 — gaze poco confiable
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section style={{ display: "grid", gap: 6 }}>
          <h3>Atajos</h3>
          <div className="muted mono" style={{ fontSize: 11, lineHeight: 1.6 }}>
            <div><kbd style={{ fontFamily: "var(--mono)", padding: "1px 4px", background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 3 }}>Esc</kbd> cierra el inspector</div>
            <div>📸 pausa video al capturar</div>
            <div>▶ reanudar reactiva el live</div>
            <div>click en foto → inspector grande</div>
          </div>
        </section>
      </aside>

      {inspectingItem && (
        <GalleryInspector
          item={inspectingItem}
          onClose={() => setInspectingId(null)}
          views={VIEW_OPTIONS}
          renderOpts={{
            overlay,
            blendAlpha,
            bw,
            postprocess,
            showSclera,
            showIris,
            showPupil,
            showEllipse,
            showPupilCenter,
            showEyelid,
            swapClasses,
            probThreshold,
            morphKsizeIris,
            morphKsizePupil,
            minIrisPixels,
            minPupilPixels,
            heatmapClass,
            showBboxes,
          }}
          setRenderOpts={{
            setBlendAlpha,
            setPostprocess,
            setProbThreshold,
            setMorphKsizeIris,
            setMorphKsizePupil,
            setMinIrisPixels,
            setMinPupilPixels,
            setSwapClasses,
            setShowEllipse,
            setShowPupilCenter,
          }}
        />
      )}
    </div>
  );
}

function GalleryItem({
  item,
  onRemove,
  onDownload,
  onNote,
  onInspect,
}: {
  item: { id: string; ts: number; full: string; tiles: { side: "left" | "right"; view: ViewId; url: string }[]; note: string; crops?: unknown };
  onRemove: () => void;
  onDownload: () => void;
  onNote: (n: string) => void;
  onInspect: () => void;
}) {
  const [open, setOpen] = useState(false);
  const tilesBySide: Record<"left" | "right", typeof item.tiles> = { left: [], right: [] };
  for (const t of item.tiles) tilesBySide[t.side].push(t);
  const ts = new Date(item.ts);
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 10, display: "grid", gap: 8, background: "var(--panel-2)" }}>
      <button
        type="button"
        onClick={onInspect}
        title="inspeccionar en grande con sliders en vivo"
        style={{ padding: 0, border: 0, background: "transparent", cursor: "zoom-in" }}
      >
        <img src={item.full} alt="" style={{ width: "100%", borderRadius: 6, display: "block", border: "1px solid var(--border)" }} />
      </button>
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
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr auto", gap: 4 }}>
        <button
          type="button"
          onClick={onInspect}
          style={{ fontSize: 11, padding: "5px 8px" }}
        >
          🔍 inspeccionar
        </button>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", fontSize: 11, padding: "5px 8px" }}
        >
          {open ? "ocultar" : "tiles"}
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

// Full-screen modal for an existing gallery snapshot. Re-renders all the
// per-side view tiles from the captured (cropImageData, SegmentationResult)
// pair every time the inspector sliders change — no model re-inference needed.
function GalleryInspector({
  item,
  onClose,
  views,
  renderOpts,
  setRenderOpts,
}: {
  item: {
    id: string;
    ts: number;
    full: string;
    crops: Array<{
      side: "left" | "right";
      bbox: { x: number; y: number; w: number; h: number };
      cropImageData: ImageData;
      result: SegmentationResult;
      eyelidPoints: { x: number; y: number }[];
      modelSpec: ModelSpec;
    }>;
    note: string;
  };
  onClose: () => void;
  views: readonly { id: ViewId; label: string }[];
  renderOpts: {
    overlay: ShowMode;
    blendAlpha: number;
    bw: boolean;
    postprocess: PostprocessName;
    showSclera: boolean;
    showIris: boolean;
    showPupil: boolean;
    showEllipse: boolean;
    showPupilCenter: boolean;
    showEyelid: boolean;
    swapClasses: boolean;
    probThreshold: number;
    morphKsizeIris: number;
    morphKsizePupil: number;
    minIrisPixels: number;
    minPupilPixels: number;
    heatmapClass: number;
    showBboxes: boolean;
  };
  setRenderOpts: {
    setBlendAlpha: (v: number) => void;
    setPostprocess: (v: PostprocessName) => void;
    setProbThreshold: (v: number) => void;
    setMorphKsizeIris: (v: number) => void;
    setMorphKsizePupil: (v: number) => void;
    setMinIrisPixels: (v: number) => void;
    setMinPupilPixels: (v: number) => void;
    setSwapClasses: (v: boolean) => void;
    setShowEllipse: (v: boolean) => void;
    setShowPupilCenter: (v: boolean) => void;
  };
}) {
  // Map per (side, view) → canvas, then re-render whenever renderOpts changes.
  const canvasMap = useRef<Record<string, HTMLCanvasElement | null>>({});
  const fullImgRef = useRef<HTMLImageElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // Draw bbox overlay on the big preview image.
  useEffect(() => {
    const img = fullImgRef.current;
    const ov = overlayRef.current;
    if (!img || !ov) return;
    const draw = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      if (W === 0) return;
      ov.width = W; ov.height = H;
      const ctx = ov.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);
      if (!renderOpts.showBboxes) return;
      ctx.lineWidth = Math.max(2, Math.round(W / 400));
      ctx.strokeStyle = "rgba(80, 220, 245, 0.95)";
      ctx.font = `${Math.max(14, Math.round(W / 80))}px sans-serif`;
      ctx.fillStyle = "rgba(80, 220, 245, 0.95)";
      for (const c of item.crops) {
        ctx.strokeRect(c.bbox.x, c.bbox.y, c.bbox.w, c.bbox.h);
        ctx.fillText(c.side, c.bbox.x + 4, c.bbox.y - 4);
      }
    };
    if (img.complete) draw();
    else img.addEventListener("load", draw, { once: true });
  }, [item, renderOpts.showBboxes]);

  // Re-render all tile canvases on slider/options change.
  useEffect(() => {
    for (const c of item.crops) {
      const spec = c.modelSpec;
      const srcCanvas = document.createElement("canvas");
      srcCanvas.width = c.cropImageData.width;
      srcCanvas.height = c.cropImageData.height;
      const sctx = srcCanvas.getContext("2d");
      if (!sctx) continue;
      sctx.putImageData(c.cropImageData, 0, 0);

      const ppOpts: PostprocessOptions = {
        morphKsizeIris: renderOpts.morphKsizeIris,
        morphKsizePupil: renderOpts.morphKsizePupil,
        minIrisPixels: renderOpts.minIrisPixels,
        minPupilPixels: renderOpts.minPupilPixels,
        swapClasses: renderOpts.swapClasses,
        probThreshold: renderOpts.probThreshold,
      };

      const crop = canvasMap.current[`${c.side}-crop`];
      if (crop) {
        crop.width = srcCanvas.width;
        crop.height = srcCanvas.height;
        crop.getContext("2d")?.drawImage(srcCanvas, 0, 0);
      }
      const pre = canvasMap.current[`${c.side}-preprocessed`];
      if (pre) drawPreprocessed(pre, srcCanvas, spec.input.preprocess, spec.input.size ?? 160);

      const rawC = canvasMap.current[`${c.side}-raw`];
      if (rawC) {
        renderCropWithMask(rawC, srcCanvas, c.result, {
          show: "blend",
          blendAlpha: renderOpts.blendAlpha,
          bw: renderOpts.bw,
          postprocess: "raw",
          postprocessOpts: { swapClasses: renderOpts.swapClasses, probThreshold: renderOpts.probThreshold },
          showSclera: renderOpts.showSclera,
          showIris: renderOpts.showIris,
          showPupil: renderOpts.showPupil,
        });
      }

      const postC = canvasMap.current[`${c.side}-post`];
      if (postC) {
        renderCropWithMask(postC, srcCanvas, c.result, {
          show: renderOpts.overlay,
          blendAlpha: renderOpts.blendAlpha,
          bw: renderOpts.bw,
          postprocess: renderOpts.postprocess,
          postprocessOpts: ppOpts,
          showSclera: renderOpts.showSclera,
          showIris: renderOpts.showIris,
          showPupil: renderOpts.showPupil,
          hardMask: true,
          eyelidPoints: c.eyelidPoints,
        });
      }

      const ellC = canvasMap.current[`${c.side}-ellipse`];
      if (ellC) {
        renderCropWithMask(ellC, srcCanvas, c.result, {
          show: "blend",
          blendAlpha: renderOpts.blendAlpha,
          bw: renderOpts.bw,
          postprocess: "ellipse_anatomical",
          postprocessOpts: ppOpts,
          showSclera: renderOpts.showSclera,
          showIris: renderOpts.showIris,
          showPupil: renderOpts.showPupil,
          showEllipse: renderOpts.showEllipse,
          showPupilCenter: renderOpts.showPupilCenter,
          eyelidPoints: c.eyelidPoints,
        });
      }

      const lmC = canvasMap.current[`${c.side}-landmarks`];
      if (lmC) {
        renderCropWithMask(lmC, srcCanvas, c.result, {
          show: "blend",
          blendAlpha: renderOpts.blendAlpha,
          bw: renderOpts.bw,
          postprocess: renderOpts.postprocess,
          postprocessOpts: ppOpts,
          showSclera: renderOpts.showSclera,
          showIris: renderOpts.showIris,
          showPupil: renderOpts.showPupil,
          showEyelid: renderOpts.showEyelid,
          eyelidPoints: c.eyelidPoints,
        });
      }

      const hmC = canvasMap.current[`${c.side}-heatmap`];
      if (hmC) drawProbHeatmap(hmC, c.result, renderOpts.heatmapClass);
    }
  }, [item, renderOpts]);

  // Lock body scroll while modal open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, []);

  // Close on Esc.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const ts = new Date(item.ts);
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.85)",
        zIndex: 1000,
        overflowY: "auto",
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: 1400,
          margin: "0 auto",
          background: "var(--panel)",
          borderRadius: 12,
          border: "1px solid var(--border)",
          padding: 20,
          display: "grid",
          gap: 16,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div className="mono" style={{ fontSize: 13 }}>
            <span style={{ color: "var(--accent)" }}>inspector</span>{" "}
            <span className="muted">· {ts.toLocaleString()}</span>
            {item.note && <span className="muted"> · {item.note}</span>}
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", fontSize: 13, padding: "6px 12px" }}
          >
            cerrar ✕
          </button>
        </div>

        <div style={{ position: "relative" }}>
          <img
            ref={fullImgRef}
            src={item.full}
            alt=""
            style={{ width: "100%", display: "block", borderRadius: 8, border: "1px solid var(--border)" }}
          />
          <canvas
            ref={overlayRef}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", borderRadius: 8 }}
          />
        </div>

        <div className="muted mono" style={{ fontSize: 11 }}>
          tuning en vivo — sin re-inferencia (logits guardados). los cambios persisten al panel left/live.
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 14,
            padding: 12,
            background: "var(--panel-2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
          }}
        >
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>postproceso</span>
            <select value={renderOpts.postprocess} onChange={(e) => setRenderOpts.setPostprocess(e.target.value as PostprocessName)}>
              {POSTPROCESS_VARIANTS.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>
              opacidad overlay {(renderOpts.blendAlpha * 100).toFixed(0)}%
            </span>
            <input
              type="range" min={0} max={1} step={0.05}
              value={renderOpts.blendAlpha}
              onChange={(e) => setRenderOpts.setBlendAlpha(parseFloat(e.target.value))}
            />
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>
              umbral confianza {(renderOpts.probThreshold * 100).toFixed(0)}%
            </span>
            <input
              type="range" min={0} max={0.99} step={0.01}
              value={renderOpts.probThreshold}
              onChange={(e) => setRenderOpts.setProbThreshold(parseFloat(e.target.value))}
            />
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>morph k iris: {renderOpts.morphKsizeIris}</span>
            <input
              type="range" min={1} max={11} step={2}
              value={renderOpts.morphKsizeIris}
              onChange={(e) => setRenderOpts.setMorphKsizeIris(parseInt(e.target.value, 10))}
            />
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>morph k pupila: {renderOpts.morphKsizePupil}</span>
            <input
              type="range" min={1} max={9} step={2}
              value={renderOpts.morphKsizePupil}
              onChange={(e) => setRenderOpts.setMorphKsizePupil(parseInt(e.target.value, 10))}
            />
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>min iris px: {renderOpts.minIrisPixels}</span>
            <input
              type="range" min={0} max={2000} step={50}
              value={renderOpts.minIrisPixels}
              onChange={(e) => setRenderOpts.setMinIrisPixels(parseInt(e.target.value, 10))}
            />
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="mono" style={{ fontSize: 11 }}>min pupila px: {renderOpts.minPupilPixels}</span>
            <input
              type="range" min={0} max={500} step={10}
              value={renderOpts.minPupilPixels}
              onChange={(e) => setRenderOpts.setMinPupilPixels(parseInt(e.target.value, 10))}
            />
          </label>
          <div style={{ display: "grid", gap: 6 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={renderOpts.showEllipse} onChange={(e) => setRenderOpts.setShowEllipse(e.target.checked)} />
              <span className="mono" style={{ fontSize: 11 }}>dibujar elipses</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={renderOpts.showPupilCenter} onChange={(e) => setRenderOpts.setShowPupilCenter(e.target.checked)} />
              <span className="mono" style={{ fontSize: 11 }}>centro pupila</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={renderOpts.swapClasses} onChange={(e) => setRenderOpts.setSwapClasses(e.target.checked)} />
              <span className="mono" style={{ fontSize: 11 }}>swap iris↔pupila</span>
            </label>
          </div>
        </div>

        {(["left", "right"] as const).map((side) => {
          const has = item.crops.find((c) => c.side === side);
          if (!has) return null;
          return (
            <div key={side} style={{ display: "grid", gap: 8 }}>
              <span className="mono muted" style={{ fontSize: 12, letterSpacing: 0.5 }}>{side}</span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                {views.map((v) => (
                  <div key={v.id} style={{ display: "grid", gap: 4, justifyItems: "center" }}>
                    <span className="muted mono" style={{ fontSize: 11, letterSpacing: 0.5 }}>{v.label}</span>
                    <canvas
                      ref={(el) => { canvasMap.current[`${side}-${v.id}`] = el; }}
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
        })}
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
