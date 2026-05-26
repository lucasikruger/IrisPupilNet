# demo-web — iris/pupil segmentation webcam demo

Astro + React + onnxruntime-web + MediaPipe FaceLandmarker. Runs the trained
RITnet iris/pupil segmenter on a live webcam feed, fully in the browser.

## Quick start

```bash
npm install
npm run dev     # http://localhost:4323
```

`predev` automatically copies the MediaPipe and onnxruntime-web WASM bundles
from `node_modules/` into `public/onnx/` and `public/mediapipe/wasm/` (both
git-ignored). The trained ONNX models and `face_landmarker.task` are tracked
in git.

## Docker

```bash
docker build -t iris-demo .
docker run --rm -p 4323:4323 iris-demo
```

## What's in here

```
lib/
  cropper.ts        MediaPipe → eye crops (eye / eye_tight / iris modes)
  preprocess.ts     CLAHE-on-LAB + gamma 0.8 (matches training) + others
  postprocess.ts    6 variants (raw → ellipse_anatomical), ellipse fit, class swap,
                    confidence threshold, min-area filter
  render.ts         multi-view renderer, ellipse + eyelid + heatmap overlays
  onnx.ts           onnxruntime-web wrapper
  mediapipe.ts      FaceLandmarker loader (478-point model)
src/
  components/WebcamDemo.tsx   main 3-column UI: controls | video+tiles | stats+gallery
  components/UploadDemo.tsx   single-image upload variant
public/
  models/                     three ONNX models (RGB prod + 2 BW legacy)
  mediapipe/face_landmarker.task
  mediapipe/wasm/   (auto-generated, gitignored)
  onnx/             (auto-generated, gitignored)
```

## Models (`public/models/models.json`)

- `ritnet_in_2x__final` — production RGB, 0.336 M params, val mIoU 0.860
- `ritnet_in__ubiris` — grayscale legacy (UBIRIS only)
- `mobilenet_lraspp_large__mobius` — grayscale legacy (MOBIUS only)

All expect 160×160 input, output 3-class logits (bg / iris / pupil).
Preprocess for the RGB model: CLAHE on LAB.L (clipLimit 1.5, 8×8 tiles) +
gamma 0.8, tensor in [0, 1] — no ImageNet normalize.

## Controls (sidebar)

- camera + model selectors
- postprocess variant (`morph` default, `ellipse_anatomical` best for gaze)
- 7 toggleable views per eye: crop · preproc · raw · post · ellipse · eyelid · heatmap
- overlay mode (crop / blend / mask) + blend alpha
- class & overlay toggles (iris, pupil, ellipse, pupil-centre, eyelid points)
- **debug**: swap iris↔pupil, mirror, crop mode (eye / eye_tight / iris),
  padding X/Y, vertical anchor, output size, confidence threshold,
  morph kernels, min-area filters, heatmap class
- gallery: 📸 capture saves full video frame + every active tile per eye into
  an in-memory list with notes, expand/download/delete per entry
