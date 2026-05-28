// Lazy-load MediaPipe Face Landmarker, self-hosted from /mediapipe/.
//
// MediaPipe requires the running mode to match the call: `detect()` only
// works with "IMAGE", `detectForVideo()` only with "VIDEO". We cache one
// instance per mode so callers (UploadDemo vs WebcamDemo) can ask for the
// flavour they need without paying a model-load each time.

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

export type RunningMode = "IMAGE" | "VIDEO";

const landmarkerPromises: Partial<Record<RunningMode, Promise<FaceLandmarker>>> = {};

export function loadFaceLandmarker(mode: RunningMode = "IMAGE"): Promise<FaceLandmarker> {
  const cached = landmarkerPromises[mode];
  if (cached) return cached;
  const p = (async () => {
    const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    return FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/mediapipe/face_landmarker.task",
        // CPU is the most portable choice; GPU often fails to create a WebGL
        // context in containers, headless browsers, or with hardware
        // acceleration disabled (kGpuService error).
        delegate: "CPU",
      },
      numFaces: 1,
      minFaceDetectionConfidence: 0.2,
      minFacePresenceConfidence: 0.2,
      runningMode: mode,
    });
  })();
  landmarkerPromises[mode] = p;
  return p;
}
