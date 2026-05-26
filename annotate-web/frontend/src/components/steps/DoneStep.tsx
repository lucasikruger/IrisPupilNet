export default function DoneStep({ onAnother }: { onAnother: () => void }) {
  return (
    <section className="panel" style={{ textAlign: "center" }}>
      <h2>¡Listo, gracias por colaborar!</h2>
      <p>Si querés aportar otra muestra (más luz, sin lentes, etc), apretá abajo.</p>
      <button onClick={onAnother} style={{ marginTop: 12 }}>Aportar otra</button>
    </section>
  );
}
