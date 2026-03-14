import { useCallback, useEffect, useRef, useState } from "react";

const STARTUP_PHASES = new Set(["starting", "camera_opening", "warming_up"]);
const RECONNECT_COOLDOWN_MS = 1500;

function StreamImage({
  src,
  alt,
  className,
  stalled,
  running = false,
  startupPhase = "idle",
  sequence = 0
}) {
  const [cacheBust, setCacheBust] = useState(Date.now());
  const [errored, setErrored] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const timer = useRef(null);
  const mountedAtMsRef = useRef(Date.now());
  const lastLoadMsRef = useRef(0);
  const lastReconnectMsRef = useRef(0);
  const lastSequenceRef = useRef(Number(sequence || 0));
  const lastSequenceChangeMsRef = useRef(Date.now());

  useEffect(
    () => () => {
      if (timer.current) {
        clearTimeout(timer.current);
      }
    },
    []
  );

  useEffect(() => {
    const seq = Number(sequence || 0);
    if (seq !== lastSequenceRef.current) {
      lastSequenceRef.current = seq;
      lastSequenceChangeMsRef.current = Date.now();
    }
  }, [sequence]);

  useEffect(() => {
    setLoaded(false);
    setErrored(false);
    mountedAtMsRef.current = Date.now();
  }, [src, cacheBust]);

  const fullSrc = `${src}${src.includes("?") ? "&" : "?"}t=${cacheBust}`;

  const reconnectNow = useCallback(() => {
    const now = Date.now();
    if (now - lastReconnectMsRef.current < RECONNECT_COOLDOWN_MS) {
      return;
    }
    lastReconnectMsRef.current = now;
    setCacheBust(now);
    setErrored(false);
  }, []);

  const scheduleRetry = () => {
    if (timer.current) {
      clearTimeout(timer.current);
    }
    timer.current = setTimeout(() => {
      reconnectNow();
    }, 1200);
  };

  useEffect(() => {
    const id = setInterval(() => {
      if (!running) {
        return;
      }
      const now = Date.now();
      const phase = String(startupPhase || "idle");
      const inStartup = STARTUP_PHASES.has(phase);
      const seq = Number(sequence || 0);
      const loadAgeMs = lastLoadMsRef.current > 0 ? now - lastLoadMsRef.current : now - mountedAtMsRef.current;
      const sequenceAdvancedButImageOld =
        seq > 0 && lastLoadMsRef.current > 0 && lastLoadMsRef.current + 2500 < lastSequenceChangeMsRef.current;
      const startupNoImageYet = inStartup && seq <= 0 && loadAgeMs > 7000;
      const stalledTooLong = Boolean(stalled) && loadAgeMs > 3500;
      const hardErrorTooLong = Boolean(errored) && loadAgeMs > 1500;

      if (startupNoImageYet || stalledTooLong || hardErrorTooLong || sequenceAdvancedButImageOld) {
        reconnectNow();
      }
    }, 1200);
    return () => clearInterval(id);
  }, [errored, reconnectNow, running, sequence, stalled, startupPhase]);

  const showConnecting = !loaded || (running && STARTUP_PHASES.has(String(startupPhase || "idle")) && Number(sequence || 0) <= 0);
  const overlayText = stalled
    ? "Stream stalled (sequence not changing). Reconnecting..."
    : showConnecting
      ? "Connecting to live stream..."
      : "Reconnecting stream...";

  return (
    <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
      <img
        src={fullSrc}
        alt={alt}
        onError={() => {
          setErrored(true);
          scheduleRetry();
        }}
        onLoad={() => {
          setErrored(false);
          setLoaded(true);
          lastLoadMsRef.current = Date.now();
          if (timer.current) {
            clearTimeout(timer.current);
            timer.current = null;
          }
        }}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          opacity: errored ? 0.4 : 1
        }}
      />
      {(errored || stalled || showConnecting) && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.35)",
            color: "white",
            fontWeight: 600
          }}
        >
          {overlayText}
        </div>
      )}
    </div>
  );
}

export default StreamImage;
