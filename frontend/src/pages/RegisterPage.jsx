import { useContext, useEffect, useState } from "react";
import { UserPlus } from "lucide-react";
import StreamImage from "../components/StreamImage";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./register.css";

function RegisterPage() {
  const { apiBase, status, enrollStatus, startEnroll, refreshEnroll, stopEnroll } = useContext(SurveillanceContext);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  const active = Boolean(enrollStatus?.active && !enrollStatus?.is_finished);

  useEffect(() => {
    if (!active) {
      return;
    }
    const id = setInterval(() => {
      refreshEnroll().catch(() => {});
    }, 1000);
    return () => clearInterval(id);
  }, [active, refreshEnroll]);

  const onStart = async (e) => {
    e.preventDefault();
    const cleaned = name.trim();
    if (!cleaned || busy) {
      return;
    }
    try {
      setBusy(true);
      await startEnroll(cleaned);
    } finally {
      setBusy(false);
    }
  };

  const onStop = async () => {
    try {
      setBusy(true);
      await stopEnroll();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Face Enrollment</h1>
        <p className="page-subtitle">Capture samples into face DB through backend enrollment mode.</p>
      </div>

      <div className="register-grid">
        <section className="glass-panel">
          <h3>Enrollment Session</h3>
          <form onSubmit={onStart} className="register-form">
            <div className="field">
              <label>Name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Hemanth" />
            </div>
            <div className="register-actions">
              <button className="btn-primary" type="submit" disabled={busy || !name.trim() || active}>
                <UserPlus size={16} />
                Start Enroll
              </button>
              <button className="btn-danger" type="button" disabled={busy || !active} onClick={onStop}>
                Stop
              </button>
            </div>
          </form>
          <pre className="json-box">{JSON.stringify(enrollStatus || {}, null, 2)}</pre>
        </section>

        <section className="glass-panel">
          <h3>Enrollment Feed</h3>
          <div className="register-stream">
            <StreamImage
              src={`${apiBase}/api/v1/stream/video`}
              alt="Enrollment stream"
              stalled={false}
              running={Boolean(status?.running && status?.mode === "enroll")}
              startupPhase={String(status?.startup_phase || "idle")}
              sequence={Number(status?.frame?.sequence || 0)}
            />
          </div>
          <div className="tiny">
            Progress: {enrollStatus?.samples_captured || 0} / {enrollStatus?.target_samples || 0}
          </div>
        </section>
      </div>
    </div>
  );
}

export default RegisterPage;
