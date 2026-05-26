// Lazy-load MediaPipe Face Landmarker, self-hosted from /mediapipe/.

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

export function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
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
      runningMode: "VIDEO",
    });
  })();
  return landmarkerPromise;
}
