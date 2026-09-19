import { createContext, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { createEventSocket } from "../realtime/socket";
import { isStreamStalledByMetadata } from "../realtime/stall";

export const SurveillanceContext = createContext(null);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function SurveillanceProvider({ children }) {
  const [initializing, setInitializing] = useState(true);
  const [status, setStatus] = useState(null);
  const [detections, setDetections] = useState(null);
  const [behavior, setBehavior] = useState({ summary: {}, recent_events: [] });
  const [memoryStats, setMemoryStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [lastSummary, setLastSummary] = useState(null);
  const [enrollStatus, setEnrollStatus] = useState(null);
  const [wsState, setWsState] = useState({ connected: false, reconnecting: false });
  const [streamStalled, setStreamStalled] = useState(false);
  const [errors, setErrors] = useState([]);
  const suppressedErrorsRef = useRef({});
  const [operation, setOperation] = useState({ active: false, message: "", since: 0 });
  const [startupFailure, setStartupFailure] = useState(null);
  const [chatSessionId, setChatSessionId] = useState(null);

  const lastDetectionTsRef = useRef(0);
  const lastSeqRef = useRef(0);
  const lastSeqChangeTsRef = useRef(0);
  const lastFrameUtcTsRef = useRef(0);
  const monitorStartedMsRef = useRef(0);
  const operationTokenRef = useRef(0);

  const updateFrameHealth = useCallback((input = {}) => {
    const now = Date.now();
    const seq = Number(input.sequence || 0);
    if (seq > 0) {
      if (seq !== lastSeqRef.current) {
        lastSeqRef.current = seq;
        lastSeqChangeTsRef.current = now;
      }
      lastDetectionTsRef.current = now;
    }
    const parsedFrameTs = Date.parse(String(input.timestamp_utc || ""));
    if (!Number.isNaN(parsedFrameTs)) {
      lastFrameUtcTsRef.current = parsedFrameTs;
    }
  }, []);

  const pushError = useCallback((err) => {
    const text = err instanceof Error ? err.message : String(err);
    const now = Date.now();
    const suppressedUntil = suppressedErrorsRef.current[text] || 0;
    if (now < suppressedUntil) {
      return;
    }
    setErrors((prev) => {
      // The heartbeat poll re-fires the same failure every 2s; without dedupe
      // the stack fills with identical toasts faster than they can be dismissed.
      if (prev.some((row) => row.message === text && now - row.ts < 10_000)) {
        return prev;
      }
      return [{ id: now, ts: now, message: text }, ...prev].slice(0, 15);
    });
  }, []);

  const removeError = useCallback((id) => {
    const target = Number(id);
    setErrors((prev) => {
      // Suppress this exact message for a while so the heartbeat poll doesn't
      // instantly re-add what the user just dismissed.
      const suppressed = prev.find((row) => Number(row?.id) === target);
      if (suppressed) {
        suppressedErrorsRef.current = {
          ...suppressedErrorsRef.current,
          [suppressed.message]: Date.now() + 60_000
        };
      }
      return prev.filter((row) => Number(row?.id) !== target);
    });
  }, []);

  const beginOperation = useCallback((message) => {
    const token = operationTokenRef.current + 1;
    operationTokenRef.current = token;
    setOperation({ active: true, message: String(message || "Working..."), since: Date.now() });
    return token;
  }, []);

  const endOperation = useCallback((token) => {
    if (token !== operationTokenRef.current) {
      return;
    }
    setOperation({ active: false, message: "", since: 0 });
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const row = await api.status();
      setStatus(row);
      setEnrollStatus(row.enroll || null);
      if (row?.startup_phase === "failed") {
        setStartupFailure(row?.startup_failure_reason || "Pipeline startup failed.");
      } else if (row?.startup_phase === "ready" || row?.startup_phase === "idle") {
        setStartupFailure(null);
      }
      updateFrameHealth(row?.frame || {});
      const parsedStartTs = Date.parse(String(row?.started_utc || ""));
      if (!Number.isNaN(parsedStartTs)) {
        monitorStartedMsRef.current = parsedStartTs;
      }
    } catch (err) {
      pushError(err);
    }
  }, [pushError, updateFrameHealth]);

  const pollStatusUntil = useCallback(
    async (predicate, { timeoutMs = 15000, intervalMs = 350, onTick } = {}) => {
      const deadline = Date.now() + Number(timeoutMs);
      let latest = null;
      while (Date.now() < deadline) {
        latest = await api.status();
        setStatus(latest);
        setEnrollStatus(latest?.enroll || null);
        updateFrameHealth(latest?.frame || {});
        const parsedStartTs = Date.parse(String(latest?.started_utc || ""));
        if (!Number.isNaN(parsedStartTs)) {
          monitorStartedMsRef.current = parsedStartTs;
        }
        if (typeof onTick === "function") {
          onTick(latest);
        }
        if (predicate(latest)) {
          return latest;
        }
        await sleep(intervalMs);
      }
      return latest;
    },
    [updateFrameHealth]
  );

  const refreshMemoryStats = useCallback(async () => {
    try {
      const row = await api.memoryStats();
      setMemoryStats(row);
    } catch (err) {
      pushError(err);
    }
  }, [pushError]);

  const refreshLogs = useCallback(async () => {
    try {
      const row = await api.logs({ limit: 400 });
      setLogs(row.items || []);
    } catch (err) {
      pushError(err);
    }
  }, [pushError]);

  const refreshDetections = useCallback(async () => {
    try {
      const row = await api.detectionsLatest();
      if (row && typeof row === "object" && Object.keys(row).length > 0) {
        setDetections(row);
        updateFrameHealth(row);
      } else {
        setDetections(null);
      }
    } catch (err) {
      pushError(err);
    }
  }, [pushError, updateFrameHealth]);

  const refreshBehavior = useCallback(async () => {
    try {
      const row = await api.behaviorLatest();
      setBehavior(row);
    } catch (err) {
      pushError(err);
    }
  }, [pushError]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setInitializing(true);
      await Promise.allSettled([refreshStatus(), refreshDetections(), refreshMemoryStats(), refreshLogs(), refreshBehavior()]);
      if (!cancelled) {
        setInitializing(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshBehavior, refreshDetections, refreshLogs, refreshMemoryStats, refreshStatus]);

  useEffect(() => {
    const socket = createEventSocket({
      onState: setWsState,
      onEvent: (event) => {
        if (!event || typeof event !== "object") {
          return;
        }
        const { type, payload } = event;
        if (type === "pipeline_state") {
          if (payload && typeof payload === "object") {
            if (payload.running !== undefined) {
              setStatus(payload);
              setEnrollStatus(payload.enroll || null);
              updateFrameHealth(payload?.frame || {});
              const parsedStartTs = Date.parse(String(payload?.started_utc || ""));
              if (!Number.isNaN(parsedStartTs)) {
                monitorStartedMsRef.current = parsedStartTs;
              }
            }
          }
          return;
        }
        if (type === "detections") {
          updateFrameHealth(payload || {});
          setStreamStalled(false);
          setDetections(payload || null);
          return;
        }
        if (type === "behavior_event") {
          setBehavior((prev) => ({
            ...prev,
            recent_events: [payload, ...(prev?.recent_events || [])].slice(0, 60)
          }));
          return;
        }
        if (type === "memory_event") {
          refreshMemoryStats();
          return;
        }
        if (type === "chat_result") {
          // Chat API call already appends assistant rows; avoid duplicate inserts from websocket fan-out.
          return;
        }
        if (type === "summary_result") {
          setLastSummary(payload);
        }
      }
    });

    return () => socket.close();
  }, [refreshMemoryStats, updateFrameHealth]);

  // Heartbeat poll: keeps sequence/state fresh even if WS detections are delayed or dropped.
  useEffect(() => {
    const id = setInterval(() => {
      refreshStatus();
      refreshDetections();
    }, 2000);
    return () => clearInterval(id);
  }, [refreshDetections, refreshStatus]);

  useEffect(() => {
    const id = setInterval(() => {
      const running = Boolean(status?.running && status?.mode === "monitor");
      if (!running) {
        setStreamStalled(false);
        lastDetectionTsRef.current = 0;
        lastSeqRef.current = 0;
        lastSeqChangeTsRef.current = 0;
        lastFrameUtcTsRef.current = 0;
        monitorStartedMsRef.current = 0;
        return;
      }
      const now = Date.now();
      const warmup = monitorStartedMsRef.current > 0 && now - monitorStartedMsRef.current < 15000;
      if (warmup) {
        setStreamStalled(false);
        return;
      }
      if (
        isStreamStalledByMetadata({
          running: true,
          nowMs: now,
          lastEventTsMs: lastDetectionTsRef.current,
          lastSequence: lastSeqRef.current,
          lastSequenceChangeTsMs: lastSeqChangeTsRef.current,
          lastFrameUtcTsMs: lastFrameUtcTsRef.current,
          eventStallMs: 5000,
          sequenceStallMs: 5000,
          frameTimestampStallMs: 7000
        })
      ) {
        setStreamStalled(true);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [status]);

  const startMonitoring = useCallback(async (payload = {}) => {
    const token = beginOperation("Starting pipeline...");
    try {
      setStartupFailure(null);
      const row = await api.startMonitor(payload);
      const ready = await pollStatusUntil(
        (current) => {
          if (!current) {
            return false;
          }
          const phase = String(current?.startup_phase || "");
          if (phase === "failed") {
            return true;
          }
          return Boolean(
            current?.running &&
              current?.mode === "monitor" &&
              phase === "ready" &&
              Number(current?.frame?.sequence || 0) > 0
          );
        },
        {
          timeoutMs: api.monitorStartupWaitMs || 70_000,
          intervalMs: 450,
          onTick: (current) => {
            const phase = String(current?.startup_phase || "");
            if (phase === "camera_opening") {
              setOperation((prev) => ({ ...prev, message: "Opening camera..." }));
            } else if (phase === "warming_up") {
              setOperation((prev) => ({ ...prev, message: "Warming up inference pipeline..." }));
            } else if (phase === "starting") {
              setOperation((prev) => ({ ...prev, message: "Starting pipeline..." }));
            } else if (phase === "failed") {
              setOperation((prev) => ({ ...prev, message: "Startup failed." }));
            } else if (phase === "ready") {
              setOperation((prev) => ({ ...prev, message: "Pipeline ready." }));
            }
          }
        }
      );
      if (String(ready?.startup_phase || "") === "failed") {
        const reason = ready?.startup_failure_reason || "Pipeline startup failed.";
        setStartupFailure(reason);
        throw new Error(reason);
      }
      if (!ready || !ready.running || ready.mode !== "monitor" || String(ready?.startup_phase || "") !== "ready") {
        throw new Error("Failed to start monitor pipeline within startup timeout.");
      }
      if (Number(ready?.frame?.sequence || 0) <= 0) {
        throw new Error("Monitoring started, but no video frames are arriving yet. Check camera availability.");
      }
      setStartupFailure(null);
      return row;
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      setStartupFailure(reason || "Pipeline startup failed.");
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pollStatusUntil, pushError]);

  const stopMonitoring = useCallback(async () => {
    const token = beginOperation("Stopping monitoring pipeline...");
    try {
      const row = await api.stopMonitor();
      await pollStatusUntil((current) => !Boolean(current?.running), { timeoutMs: 8000, intervalMs: 250 });
      setStartupFailure(null);
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pollStatusUntil, pushError]);

  const setToggles = useCallback(async (payload) => {
    const token = beginOperation("Applying runtime toggles...");
    try {
      const row = await api.setToggles(payload);
      await refreshStatus();
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pushError, refreshStatus]);

  const captureSnapshot = useCallback(async () => {
    const token = beginOperation("Capturing snapshot...");
    try {
      const row = await api.snapshot();
      await refreshMemoryStats();
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pushError, refreshMemoryStats]);

  const askChat = useCallback(async (question) => {
    const token = beginOperation("Assistant is analyzing your question...");
    try {
      const cleaned = (question || "").trim();
      if (!cleaned) {
        return null;
      }
      setChatHistory((prev) => [...prev, { role: "user", question: cleaned, ts: Date.now() }]);
      const row = await api.chatQuery({ message: cleaned, question: cleaned, session_id: chatSessionId || undefined });
      if (row?.session_id) {
        setChatSessionId(row.session_id);
      }
      setChatHistory((prev) => [...prev, { role: "assistant", ...row, ts: Date.now() }]);
      if (row?.summary) {
        setLastSummary(row.summary);
      }
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, chatSessionId, endOperation, pushError]);

  const confirmChatAction = useCallback(
    async (confirmActionId) => {
      const token = beginOperation("Executing confirmed action...");
      try {
        const row = await api.chatQuery({
          session_id: chatSessionId || undefined,
          confirm_action_id: String(confirmActionId || "")
        });
        if (row?.session_id) {
          setChatSessionId(row.session_id);
        }
        setChatHistory((prev) => [...prev, { role: "assistant", ...row, ts: Date.now() }]);
        if (row?.summary) {
          setLastSummary(row.summary);
        }
        return row;
      } catch (err) {
        pushError(err);
        throw err;
      } finally {
        endOperation(token);
      }
    },
    [beginOperation, chatSessionId, endOperation, pushError]
  );

  const dismissChatProposal = useCallback((confirmActionId) => {
    setChatHistory((prev) =>
      prev.map((row) => {
        if (!row || row.role !== "assistant") {
          return row;
        }
        const rowConfirmId = row?.proposed_action?.confirm_action_id;
        if (String(rowConfirmId || "") !== String(confirmActionId || "")) {
          return row;
        }
        return {
          ...row,
          proposed_action: null,
          reply: `${row.reply || row.answer || "Action proposal"} (dismissed)`,
          answer: `${row.reply || row.answer || "Action proposal"} (dismissed)`
        };
      })
    );
  }, []);

  const runSessionSummary = useCallback(async (minutes = 5) => {
    const token = beginOperation("Generating situation summary...");
    try {
      const row = await api.sessionSummary(minutes);
      setLastSummary(row.json || null);
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pushError]);

  const startEnroll = useCallback(async (name) => {
    const token = beginOperation("Starting enrollment pipeline...");
    try {
      const row = await api.startEnroll(name);
      const ready = await pollStatusUntil(
        (current) =>
          Boolean(current?.running && current?.mode === "enroll" && (current?.enroll?.active || current?.enroll?.is_finished)),
        { timeoutMs: 14000, intervalMs: 350 }
      );
      if (!ready || ready.mode !== "enroll") {
        throw new Error("Failed to enter enrollment mode.");
      }
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pollStatusUntil, pushError]);

  const refreshEnroll = useCallback(async () => {
    try {
      const row = await api.enrollStatus();
      setEnrollStatus(row);
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    }
  }, [pushError]);

  const stopEnroll = useCallback(async () => {
    const token = beginOperation("Stopping enrollment...");
    try {
      const row = await api.stopEnroll();
      await pollStatusUntil((current) => !Boolean(current?.running), { timeoutMs: 8000, intervalMs: 250 });
      return row;
    } catch (err) {
      pushError(err);
      throw err;
    } finally {
      endOperation(token);
    }
  }, [beginOperation, endOperation, pollStatusUntil, pushError]);

  const value = useMemo(
    () => ({
      apiBase: api.base,
      initializing,
      operation,
      startupFailure,
      status,
      detections,
      behavior,
      memoryStats,
      logs,
      chatHistory,
      chatSessionId,
      lastSummary,
      enrollStatus,
      wsState,
      streamStalled,
      errors,
      removeError,
      refreshStatus,
      refreshDetections,
      refreshMemoryStats,
      refreshBehavior,
      refreshLogs,
      startMonitoring,
      stopMonitoring,
      setToggles,
      captureSnapshot,
      askChat,
      confirmChatAction,
      dismissChatProposal,
      runSessionSummary,
      startEnroll,
      refreshEnroll,
      stopEnroll,
      lastSequence: lastSeqRef.current
    }),
    [
      askChat,
      behavior,
      captureSnapshot,
      chatHistory,
      chatSessionId,
      confirmChatAction,
      detections,
      dismissChatProposal,
      enrollStatus,
      errors,
      initializing,
      lastSummary,
      logs,
      memoryStats,
      operation,
      refreshBehavior,
      refreshEnroll,
      refreshLogs,
      refreshMemoryStats,
      refreshStatus,
      refreshDetections,
      runSessionSummary,
      removeError,
      setToggles,
      startEnroll,
      startMonitoring,
      startupFailure,
      status,
      stopEnroll,
      stopMonitoring,
      streamStalled,
      wsState
    ]
  );

  return <SurveillanceContext.Provider value={value}>{children}</SurveillanceContext.Provider>;
}
