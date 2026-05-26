import { useReducer } from "react";
import LandingStep from "./steps/LandingStep";
import CaptureStep, { type CaptureBundle } from "./steps/CaptureStep";
import PreviewStep from "./steps/PreviewStep";
import MetadataStep, { type Metadata } from "./steps/MetadataStep";
import SubmittingStep from "./steps/SubmittingStep";
import ThanksStep from "./steps/ThanksStep";
import RefineStep from "./steps/RefineStep";
import DoneStep from "./steps/DoneStep";

export type Step =
  | "landing"
  | "capture"
  | "preview"
  | "metadata"
  | "submitting"
  | "thanks"
  | "refine"
  | "done";

export interface FlowState {
  step: Step;
  capture: CaptureBundle | null;
  metadata: Metadata | null;
  submissionId: string | null;
  error: string | null;
}

type Action =
  | { type: "GO"; step: Step }
  | { type: "SET_CAPTURE"; capture: CaptureBundle }
  | { type: "SET_METADATA"; metadata: Metadata }
  | { type: "SET_ID"; id: string }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "RESET" };

const initial: FlowState = {
  step: "landing",
  capture: null,
  metadata: null,
  submissionId: null,
  error: null,
};

function reducer(state: FlowState, action: Action): FlowState {
  switch (action.type) {
    case "GO":
      return { ...state, step: action.step, error: null };
    case "SET_CAPTURE":
      return { ...state, capture: action.capture };
    case "SET_METADATA":
      return { ...state, metadata: action.metadata };
    case "SET_ID":
      return { ...state, submissionId: action.id };
    case "SET_ERROR":
      return { ...state, error: action.error };
    case "RESET":
      return initial;
  }
}

export default function AnnotateFlow({ apiUrl }: { apiUrl: string }) {
  const [state, dispatch] = useReducer(reducer, initial);

  return (
    <div>
      <Stepper step={state.step} />
      {state.error && (
        <div className="panel" style={{ borderColor: "var(--warn)", color: "var(--warn)", marginBottom: 14 }}>
          {state.error}
        </div>
      )}
      {state.step === "landing" && <LandingStep onAccept={() => dispatch({ type: "GO", step: "capture" })} />}
      {state.step === "capture" && (
        <CaptureStep
          onDone={(capture) => {
            dispatch({ type: "SET_CAPTURE", capture });
            dispatch({ type: "GO", step: "preview" });
          }}
        />
      )}
      {state.step === "preview" && state.capture && (
        <PreviewStep
          capture={state.capture}
          onRetake={() => dispatch({ type: "GO", step: "capture" })}
          onConfirm={() => dispatch({ type: "GO", step: "metadata" })}
        />
      )}
      {state.step === "metadata" && (
        <MetadataStep
          initial={state.metadata}
          onSubmit={async (md) => {
            dispatch({ type: "SET_METADATA", metadata: md });
            dispatch({ type: "GO", step: "submitting" });
            try {
              if (!state.capture) throw new Error("capture missing");
              const id = await submitBase(apiUrl, state.capture, md);
              dispatch({ type: "SET_ID", id });
              dispatch({ type: "GO", step: "thanks" });
            } catch (e) {
              dispatch({ type: "SET_ERROR", error: `submit fallo: ${(e as Error).message}` });
              dispatch({ type: "GO", step: "metadata" });
            }
          }}
        />
      )}
      {state.step === "submitting" && <SubmittingStep />}
      {state.step === "thanks" && (
        <ThanksStep
          onRefine={() => dispatch({ type: "GO", step: "refine" })}
          onDone={() => dispatch({ type: "GO", step: "done" })}
        />
      )}
      {state.step === "refine" && state.capture && state.submissionId && (
        <RefineStep
          capture={state.capture}
          submissionId={state.submissionId}
          apiUrl={apiUrl}
          onDone={() => dispatch({ type: "GO", step: "done" })}
        />
      )}
      {state.step === "done" && <DoneStep onAnother={() => dispatch({ type: "RESET" })} />}
    </div>
  );
}

function Stepper({ step }: { step: Step }) {
  const labels: Array<{ key: Step; label: string }> = [
    { key: "landing", label: "1. inicio" },
    { key: "capture", label: "2. foto" },
    { key: "metadata", label: "3. datos" },
    { key: "thanks", label: "4. enviado" },
    { key: "refine", label: "5. refinar (opc)" },
  ];
  const stepIdx = (s: Step) => labels.findIndex((l) => l.key === s);
  const currentIdx = Math.max(0, [
    "landing", "capture", "preview", "metadata", "submitting", "thanks", "refine", "done",
  ].indexOf(step));
  const labelKey = (() => {
    if (step === "preview") return "capture";
    if (step === "submitting") return "metadata";
    if (step === "done") return "refine";
    return step;
  })();
  return (
    <nav className="steps">
      {labels.map((l) => (
        <span key={l.key} className={l.key === labelKey ? "active" : ""}>{l.label}</span>
      ))}
    </nav>
  );
}

async function submitBase(apiUrl: string, capture: CaptureBundle, md: Metadata): Promise<string> {
  const form = new FormData();
  form.append("full", capture.fullBlob, "full.jpg");
  form.append("crop_left", capture.leftBlob, "crop_left.png");
  form.append("crop_right", capture.rightBlob, "crop_right.png");
  form.append("metadata", new Blob([JSON.stringify(md)], { type: "application/json" }), "metadata.json");
  const resp = await fetch(`${apiUrl}/api/submit`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  if (!data.ok || !data.id) throw new Error("respuesta inesperada");
  return data.id;
}
