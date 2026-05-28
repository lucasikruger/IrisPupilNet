// onnxruntime-web wrapper. Carga el modelo desde URL y corre inferencia
// 1×C×H×W → tensor 1×num_classes×H×W (logits).

import * as ort from "onnxruntime-web";
import { PREPROCESS, type PreprocessName } from "./preprocess";

// Let onnxruntime-web auto-detect its sibling .wasm/.mjs files relative to
// its own bundle. Overriding wasmPaths to "/onnx/" makes ORT do dynamic
// import() of glue .mjs from /public/, which Vite blocks.

export interface ModelSpec {
  name: string;
  url: string;
  input: { channels: number; size: number; preprocess: PreprocessName };
  output: { classes: string[] };
  dataset?: string;
  val_iou?: number;
  architecture?: string;
}

export interface SegmentationResult {
  argmax: Uint8Array;   // size*size, valor en [0, num_classes)
  probs: Float32Array;  // num_classes * size * size, softmax aplicado
  size: number;
  numClasses: number;
}

export class OnnxSegmenter {
  private session: ort.InferenceSession | null = null;
  private spec: ModelSpec | null = null;
  private loading: Promise<void> | null = null;

  async load(spec: ModelSpec): Promise<void> {
    if (this.spec?.url === spec.url && this.session) return;
    if (this.loading) await this.loading;
    this.spec = spec;
    this.loading = (async () => {
      // WebGPU primero, fallback WASM
      try {
        this.session = await ort.InferenceSession.create(spec.url, {
          executionProviders: ["webgpu", "wasm"],
          graphOptimizationLevel: "all",
        });
      } catch (e) {
        console.warn("[onnx] webgpu failed, falling back to wasm:", e);
        this.session = await ort.InferenceSession.create(spec.url, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      }
    })();
    await this.loading;
    this.loading = null;
  }

  async run(canvas: HTMLCanvasElement): Promise<SegmentationResult> {
    if (!this.session || !this.spec) throw new Error("load() first");
    const { input, output } = this.spec;
    const size = input.size;
    const numClasses = output.classes.length;

    const pre = PREPROCESS[input.preprocess];
    if (!pre) throw new Error(`unknown preprocess ${input.preprocess}`);
    const data = pre.toTensor(canvas, size);

    const tensor = new ort.Tensor("float32", data, [1, pre.channels, size, size]);
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const results = await this.session.run({ [inputName]: tensor });
    const logits = results[outputName].data as Float32Array;

    // Softmax + argmax over channel dim
    const plane = size * size;
    const probs = new Float32Array(numClasses * plane);
    const argmax = new Uint8Array(plane);
    for (let p = 0; p < plane; p++) {
      let maxLogit = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const v = logits[c * plane + p];
        if (v > maxLogit) maxLogit = v;
      }
      let sum = 0;
      for (let c = 0; c < numClasses; c++) {
        const e = Math.exp(logits[c * plane + p] - maxLogit);
        probs[c * plane + p] = e;
        sum += e;
      }
      let best = 0;
      let bestProb = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        const v = probs[c * plane + p] / sum;
        probs[c * plane + p] = v;
        if (v > bestProb) {
          bestProb = v;
          best = c;
        }
      }
      argmax[p] = best;
    }

    return { argmax, probs, size, numClasses };
  }

  get currentSpec(): ModelSpec | null {
    return this.spec;
  }

  // True only when the session is fully created and run() will succeed.
  // `spec` is set synchronously at the start of load(), so it can't be used
  // as a readiness signal on its own.
  get ready(): boolean {
    return this.session !== null;
  }
}

export async function loadManifest(url = "/models/models.json"): Promise<ModelSpec[]> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`failed to load ${url}`);
  return resp.json();
}
