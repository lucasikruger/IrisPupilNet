export default function ThanksStep({
  onRefine,
  onDone,
}: {
  onRefine: () => void;
  onDone: () => void;
}) {
  return (
    <section className="panel">
      <h2>¡Gracias! Tu muestra ya quedó guardada.</h2>
      <p>
        ¿Querés tomarte un minuto más para mejorar las anotaciones manualmente? Es opcional
        y suma muchísimo a la calidad del dataset.
      </p>
      <p className="muted">
        Si decís que no, ya está todo guardado: la foto, los recortes y los datos.
      </p>
      <div className="row" style={{ marginTop: 18 }}>
        <button onClick={onRefine}>Sí, refinar anotaciones</button>
        <button className="secondary" onClick={onDone}>No, gracias</button>
      </div>
    </section>
  );
}
