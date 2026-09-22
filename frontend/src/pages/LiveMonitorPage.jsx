import { useContext, useMemo, useState } from "react";
import { Camera, Pause, Play, ToggleLeft, ToggleRight } from "lucide-react";
import StreamImage from "../components/StreamImage";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./live.css";

function LiveMonitorPage() {
  const {
    apiBase,
    status,
    detections,
    wsState,
    streamStalled,
    startupFailure,
    startMonitoring,
    stopMonitoring,
    setToggles,
    captureSnapshot
  } = useContext(SurveillanceContext);

  const [busy, setBusy] = useState(false);
  const isRunning = Boolean(status?.running && status?.mode === "monitor");
  const yolo = status?.yolo || {};
  const toggles = {
    general: Boolean(yolo?.general?.enabled),
    custom: Boolean(yolo?.custom?.enabled),
    gaze: Boolean(status?.config && !status?.config?.disable_gaze)
  };

  const activeSummary = useMemo(() => {
    const faces = detections?.faces?.length || 0;
    const objects = detections?.objects?.length || 0;
    return { faces, objects };
  }, [detections]);
  // `wsState.reconnecting` is only true while a retry is pending, so an idle page
  // always showed "no". Report the actual socket state instead.
  const connectionState = wsState.connected
    ? { label: "Live", tone: "ok" }
    : wsState.reconnecting
      ? { label: "Retrying", tone: "warn" }
      : { label: "Offline", tone: "bad" };

  const liveSequence = Number(detections?.sequence || status?.frame?.sequence || 0);
  const warmingUp = Boolean(isRunning && liveSequence <= 0);
  const startupPhase = String(status?.startup_phase || "idle");
  const startupPhaseText =
    startupPhase === "camera_opening"
      ? "Opening camera..."
      : startupPhase === "warming_up"
        ? "Warming up inference pipeline..."
        : startupPhase === "starting"
          ? "Starting pipeline..."
          : startupPhase === "failed"
            ? "Startup failed."
            : "Initializing camera and inference pipeline...";

  const submit = async (fn) => {
    try {
      setBusy(true);
      await fn();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Live Monitor</h1>
        <p className="page-subtitle">MJPEG stream + realtime detections + pipeline controls.</p>
      </div>

      <div className="live-grid">
        <section className="glass-panel live-stream-card">
          <div className="live-stream-header">
            <div>
              <h3>Camera Feed</h3>
              <div className="tiny">
                Sequence {liveSequence} · Phase {startupPhase} · WS {wsState.connected ? "connected" : "disconnected"}
              </div>
            </div>
            {streamStalled ? (
              <span className="badge danger">Stalled</span>
            ) : (
              <span className="badge ok">Live</span>
            )}
          </div>
          <div className="live-viewport">
            <StreamImage
              src={`${apiBase}/api/v1/stream/video`}
              alt="OpenCV live stream"
              stalled={streamStalled}
              running={isRunning}
              startupPhase={startupPhase}
              sequence={liveSequence}
            />
            <div className="scanline"></div>
            {warmingUp && (
              <div className="live-warmup-overlay">
                <div className="live-warmup-card">
                  <div className="live-warmup-spinner" />
                  <span>{startupPhaseText}</span>
                </div>
              </div>
            )}
          </div>
          <div className="controls-row">
            {!isRunning ? (
              <button className="btn-primary" disabled={busy} onClick={() => submit(() => startMonitoring())}>
                <Play size={16} />
                Start Monitoring
              </button>
            ) : (
              <button className="btn-danger" disabled={busy} onClick={() => submit(() => stopMonitoring())}>
                <Pause size={16} />
                Stop Monitoring
              </button>
            )}
            <button className="btn-secondary" disabled={!isRunning || busy} onClick={() => submit(() => captureSnapshot())}>
              <Camera size={16} />
              Snapshot
            </button>
          </div>
          {startupFailure && (
            <div className="startup-failure-box">
              <div className="tiny">Startup failed</div>
              <div>{startupFailure}</div>
              <button className="btn-secondary" disabled={busy} onClick={() => submit(() => startMonitoring())}>
                Retry Startup
              </button>
            </div>
          )}
        </section>

        <section className="glass-panel">
          <h3>Runtime Controls</h3>
          <p className="tiny">
            Applied to the running pipeline within one frame. Switching a detector off frees CPU for the ones you
            keep.
          </p>
          <div className="toggle-list">
            <div className="toggle-row">
              <button
                className="btn-secondary toggle-btn"
                disabled={!isRunning || busy}
                aria-pressed={toggles.general}
                onClick={() => submit(() => setToggles({ general_yolo: !toggles.general }))}
              >
                {toggles.general ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
                General YOLO
                <span className="toggle-chip">{toggles.general ? "On" : "Off"}</span>
              </button>
              <span className="toggle-hint">Everyday classes — person, laptop, chair, phone, bottle.</span>
            </div>
            <div className="toggle-row">
              <button
                className="btn-secondary toggle-btn"
                disabled={!isRunning || busy}
                aria-pressed={toggles.custom}
                onClick={() => submit(() => setToggles({ custom_yolo: !toggles.custom }))}
              >
                {toggles.custom ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
                Custom YOLO
                <span className="toggle-chip">{toggles.custom ? "On" : "Off"}</span>
              </button>
              <span className="toggle-hint">Your own trained classes. Off until a custom model is configured.</span>
            </div>
            <div className="toggle-row">
              <button
                className="btn-secondary toggle-btn"
                disabled={!isRunning || busy}
                aria-pressed={toggles.gaze}
                onClick={() => submit(() => setToggles({ gaze: !toggles.gaze }))}
              >
                {toggles.gaze ? <ToggleRight size={16} /> : <ToggleLeft size={16} />}
                Gaze
                <span className="toggle-chip">{toggles.gaze ? "On" : "Off"}</span>
              </button>
              <span className="toggle-hint">Where each person looks. The heaviest per-frame step.</span>
            </div>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3>Live Counts</h3>
            <div className="counts-grid">
              <div className="count-box">
                <span>Faces</span>
                <strong>{activeSummary.faces}</strong>
              </div>
              <div className="count-box">
                <span>Objects</span>
                <strong>{activeSummary.objects}</strong>
              </div>
              <div className="count-box">
                <span>FPS Cap</span>
                <strong>{status?.fps_cap || 12}</strong>
              </div>
              <div className="count-box">
                <span>Snapshot</span>
                <strong>{status?.snapshot_interval ? `${status.snapshot_interval}s` : "8s"}</strong>
              </div>
              <div className="count-box">
                <span>Connection</span>
                <strong className={`conn-${connectionState.tone}`}>{connectionState.label}</strong>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default LiveMonitorPage;
