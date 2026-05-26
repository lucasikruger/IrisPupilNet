import { useState } from "react";

export interface Metadata {
  age_range: string;
  eye_color: string;
  wears_glasses: string;
  lighting: string;
  contact_email: string;
  country: string;
}

const AGE = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+", "prefiero no decir"];
const EYE_COLOR = ["marrón", "verde", "azul", "avellana", "gris", "otro / no sé"];
const GLASSES = ["no", "lentes", "lentes de contacto"];
const LIGHTING = ["luz natural", "luz cálida", "luz fría", "poca luz"];

function defaultCountry() {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return tz?.split("/")[0] ?? "";
  } catch {
    return "";
  }
}

export default function MetadataStep({
  initial,
  onSubmit,
}: {
  initial: Metadata | null;
  onSubmit: (md: Metadata) => void;
}) {
  const [md, setMd] = useState<Metadata>(
    initial ?? {
      age_range: "",
      eye_color: "",
      wears_glasses: "",
      lighting: "",
      contact_email: "",
      country: defaultCountry(),
    },
  );

  const valid = md.age_range && md.eye_color && md.wears_glasses && md.lighting;

  return (
    <section className="panel">
      <h2>Datos sobre vos y la foto</h2>
      <p className="muted" style={{ marginTop: 0 }}>Todo es opcional excepto los selectores marcados.</p>
      <Field label="Rango de edad *">
        <select value={md.age_range} onChange={(e) => setMd({ ...md, age_range: e.target.value })}>
          <option value="">—</option>
          {AGE.map((v) => <option key={v}>{v}</option>)}
        </select>
      </Field>
      <Field label="Color de ojos declarado *">
        <select value={md.eye_color} onChange={(e) => setMd({ ...md, eye_color: e.target.value })}>
          <option value="">—</option>
          {EYE_COLOR.map((v) => <option key={v}>{v}</option>)}
        </select>
      </Field>
      <Field label="Lentes / contacto *">
        <select value={md.wears_glasses} onChange={(e) => setMd({ ...md, wears_glasses: e.target.value })}>
          <option value="">—</option>
          {GLASSES.map((v) => <option key={v}>{v}</option>)}
        </select>
      </Field>
      <Field label="Iluminación de la foto *">
        <select value={md.lighting} onChange={(e) => setMd({ ...md, lighting: e.target.value })}>
          <option value="">—</option>
          {LIGHTING.map((v) => <option key={v}>{v}</option>)}
        </select>
      </Field>
      <Field label="Email (opcional, sirve para pedir borrar tu envío)">
        <input
          type="email"
          value={md.contact_email}
          onChange={(e) => setMd({ ...md, contact_email: e.target.value })}
          placeholder="opcional"
        />
      </Field>
      <Field label="Región / país (auto-detectado)">
        <input
          type="text"
          value={md.country}
          onChange={(e) => setMd({ ...md, country: e.target.value })}
        />
      </Field>
      <div className="row" style={{ marginTop: 14 }}>
        <button onClick={() => onSubmit(md)} disabled={!valid}>Enviar</button>
        <span className="muted">{!valid ? "completá los campos con *" : "se sube la foto + crops + estos datos"}</span>
      </div>
      <div className="row" style={{ marginTop: 14, paddingTop: 12, borderTop: "1px dashed var(--border)", opacity: 0.85 }}>
        <button
          type="button"
          onClick={() => onSubmit(DEFAULT_METADATA)}
          style={{ background: "transparent", border: "1px dashed var(--border)", color: "var(--muted)" }}
        >
          dev: saltar con defaults
        </button>
        <span className="muted" style={{ fontSize: 12 }}>
          envía con metadata genérica para testear el refine
        </span>
      </div>
    </section>
  );
}

const DEFAULT_METADATA: Metadata = {
  age_range: "26-35",
  eye_color: "marrón",
  wears_glasses: "no",
  lighting: "luz natural",
  contact_email: "",
  country: defaultCountry(),
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="form-row">
      <label>{label}</label>
      {children}
    </div>
  );
}
