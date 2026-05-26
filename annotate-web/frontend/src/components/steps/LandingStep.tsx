export default function LandingStep({ onAccept }: { onAccept: () => void }) {
  return (
    <section className="panel">
      <h2>Aportá una foto para mejorar el modelo</h2>
      <p>
        Para mi tesis estoy juntando muestras de ojos en condiciones variadas (luz natural,
        gafas, distintos colores de iris). Si querés colaborar, en 30 segundos podés mandar
        una foto desde tu celu o webcam y completar 4-5 datos.
      </p>
      <p className="muted">
        <strong>Qué se guarda:</strong> la foto de tu cara y los dos recortes de ojos.
        <br />
        <strong>Cómo se usa:</strong> entrenamiento y evaluación de modelos de segmentación
        de iris. No se publica ninguna foto sin tu permiso explícito.
        <br />
        <strong>Privacidad:</strong> podés borrar tu envío si dejás un email opcional.
      </p>
      <div className="row" style={{ marginTop: 18 }}>
        <button onClick={onAccept}>Empezar</button>
        <span className="muted">No pedimos cuenta ni login.</span>
      </div>
    </section>
  );
}
