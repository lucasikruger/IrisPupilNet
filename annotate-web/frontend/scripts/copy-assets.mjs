// Copy MediaPipe + ONNX runtime assets from node_modules to public/.
// Idempotent — safe to run on every dev/build.

import { cp, mkdir, readdir, copyFile, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const REPO_ROOT = resolve(ROOT, "../..");

const tasks = [
  {
    label: "mediapipe wasm",
    src: join(ROOT, "node_modules/@mediapipe/tasks-vision/wasm"),
    dst: join(ROOT, "public/mediapipe/wasm"),
  },
  {
    label: "onnxruntime-web wasm",
    src: join(ROOT, "node_modules/onnxruntime-web/dist"),
    dst: join(ROOT, "public/onnx"),
    filter: (name) =>
      name.endsWith(".wasm") || name.endsWith(".mjs") || name.endsWith(".js") || name.endsWith(".jsep.wasm"),
  },
];

async function copyDir(src, dst, filter) {
  if (!existsSync(src)) {
    console.warn(`[copy-assets] source missing: ${src} — run npm install first`);
    return 0;
  }
  await mkdir(dst, { recursive: true });
  const entries = await readdir(src);
  let count = 0;
  for (const name of entries) {
    if (filter && !filter(name)) continue;
    const s = join(src, name);
    const d = join(dst, name);
    const st = await stat(s);
    if (st.isDirectory()) {
      count += await copyDir(s, d, filter);
    } else {
      await copyFile(s, d);
      count++;
    }
  }
  return count;
}

async function copyFaceLandmarker() {
  const dst = join(ROOT, "public/mediapipe/face_landmarker.task");
  if (existsSync(dst)) {
    console.log(`[copy-assets] face_landmarker.task already present, skipping`);
    return;
  }
  // Fallback paths: try the parent repo's tools/prepare/ for dev mode.
  const candidates = [
    join(REPO_ROOT, "tools/prepare/face_landmarker_v2_with_blendshapes.task"),
    join(REPO_ROOT, "tools/prepare/face_landmarker.task"),
  ];
  for (const src of candidates) {
    if (existsSync(src)) {
      await mkdir(dirname(dst), { recursive: true });
      await copyFile(src, dst);
      console.log(`[copy-assets] face_landmarker.task ← ${src}`);
      return;
    }
  }
  console.warn(
    `[copy-assets] face_landmarker.task missing — download from ` +
    `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task ` +
    `and place it at public/mediapipe/face_landmarker.task`,
  );
}

for (const t of tasks) {
  const n = await copyDir(t.src, t.dst, t.filter);
  console.log(`[copy-assets] ${t.label}: ${n} files → ${t.dst.replace(ROOT, ".")}`);
}
await copyFaceLandmarker();
