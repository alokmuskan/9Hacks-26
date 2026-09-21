from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Literal
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import common

LOGGER = logging.getLogger("monitoring-backend")
logging.basicConfig(level=logging.INFO)


class _ShutdownNoiseFilter(logging.Filter):
    """Drop uvicorn ERROR records whose exception is a benign shutdown cancel.

    When the server stops while browsers still hold the WebSocket event feed or
    the MJPEG stream open, uvicorn/starlette/anyio surface the resulting
    cancellation as ``CancelledError`` (or a re-raised ``KeyboardInterrupt``)
    traceback logged at ERROR by uvicorn's error logger. That is normal
    shutdown mechanics — the server still exits cleanly ("Finished server
    process") — and logging it at ERROR hides real errors. Real failures (any
    other exception type, or a benign signal wrapped around one) still log at
    ERROR untouched.
    """

    _BENIGN_FINALS = ("CancelledError", "KeyboardInterrupt")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        if record.exc_info and record.exc_info[0] is not None:
            formatted = "\n".join(traceback.format_exception(*record.exc_info))
        else:
            formatted = record.getMessage()
        final_line = formatted.rstrip().rsplit("\n", 1)[-1].strip()
        # Final traceback line is "module.Exception: optional message"; match on
        # the type part so a CancelledError carrying a message still matches.
        exc_type = final_line.split(":", 1)[0].strip()
        return not exc_type.endswith(self._BENIGN_FINALS)


def _install_shutdown_noise_filter() -> None:
    # uvicorn's protocol handlers log via the "uvicorn.error" logger directly,
    # so a filter attached here sees every "Exception in ASGI application" record.
    logging.getLogger("uvicorn.error").addFilter(_ShutdownNoiseFilter())


_install_shutdown_noise_filter()

# Live pacing & capture cadence, env-tunable — see common.py for the rationale.
FPS_CAP_DEFAULT = common.FPS_CAP_DEFAULT
ENROLL_FPS_CAP_DEFAULT = common.ENROLL_FPS_CAP_DEFAULT
SNAPSHOT_INTERVAL_DEFAULT = common.SNAPSHOT_INTERVAL_DEFAULT
MJPEG_QUALITY = 75
WS_QUEUE_MAX = 256
CAMERA_FAILURE_THRESHOLD = 20
CAMERA_RECOVERY_BACKOFF_START_SEC = 0.5
CAMERA_RECOVERY_BACKOFF_MAX_SEC = 5.0
FRAME_WAIT_IDLE_SEC = 0.02
MJPEG_KEEPALIVE_SEC = 1.0
STARTUP_TIMEOUT_SEC = 60.0
STARTUP_CAMERA_FAILURE_THRESHOLD = 6
CHAT_SESSION_TTL_SEC = 30 * 60
CHAT_CONFIRM_TTL_SEC = 2 * 60
CHAT_HISTORY_MAX_TURNS = 10
CHAT_CONTEXT_LOOKBACK_MIN = 30
# Backbones L2CS-Net can instantiate. The CLI restricts `--gaze-arch`, the API used
# to silently rewrite whatever the caller asked for.
GazeArch = Literal["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
GAZE_ARCH_DEFAULT: GazeArch = "ResNet50"
# Gaze scheduling defaults come from `common` so the CLI and the API schedule gaze
# identically; the interval only adapts when the caller opts in.
GAZE_INTERVAL_DEFAULT = common.GAZE_INTERVAL_DEFAULT
GAZE_TARGET_FPS_DROP_DEFAULT = common.GAZE_TARGET_FPS_DROP_DEFAULT

METRICS_PATH = Path(common.METRICS_FILENAME)
MEMORY_DIR = Path("memory")
SNAPSHOTS_DIR = MEMORY_DIR / "snapshots"
INCIDENTS_DIR = Path("unknown_incidents")

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)


def _core() -> Any:
    import main as core

    return core


# Thin aliases over `common` so both entry points behave identically and there is
# one implementation to fix. The CV stack stays unimported here on purpose.
def _iso(dt: datetime | None = None) -> str:
    return common.iso(dt)


def _parse_iso(value: str | None) -> datetime | None:
    return common.parse_iso(value)


def _safe_int(value: Any, default: int = 0) -> int:
    return common.safe_int(value, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return common.safe_float(value, default)


def _session_id(prefix: str) -> str:
    return common.session_id(prefix)


def _encode_jpeg(frame: np.ndarray, quality: int = MJPEG_QUALITY) -> bytes | None:
    try:
        ok, buff = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None
        return bytes(buff)
    except Exception:
        return None


def _mjpeg_chunk(frame_bytes: bytes) -> bytes:
    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii")
    )
    return header + frame_bytes + b"\r\n"


def _make_placeholder(primary: str, secondary: str | None = None) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        primary,
        (22, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if secondary:
        cv2.putText(
            frame,
            secondary,
            (22, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )
    return frame


@dataclass
class FramePacket:
    frame_bytes: bytes | None = None
    timestamp_utc: str | None = None
    sequence: int = 0
    raw_frame: np.ndarray | None = None


class LatestFrameStore:
    """Thread-safe frame slot with sequence metadata."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._packet = FramePacket()

    def update(self, frame_bytes: bytes, raw_frame: np.ndarray | None = None) -> dict[str, Any]:
        with self._lock:
            self._packet.sequence += 1
            self._packet.frame_bytes = frame_bytes
            self._packet.timestamp_utc = _iso()
            self._packet.raw_frame = raw_frame.copy() if raw_frame is not None else None
            return {
                "timestamp_utc": self._packet.timestamp_utc,
                "sequence": self._packet.sequence,
            }

    def clear(self) -> None:
        with self._lock:
            self._packet = FramePacket()

    def get(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame_bytes": self._packet.frame_bytes,
                "timestamp_utc": self._packet.timestamp_utc,
                "sequence": self._packet.sequence,
            }

    def get_raw_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._packet.raw_frame is None:
                return None
            return self._packet.raw_frame.copy()


class EventHub:
    """Async pub/sub hub for websocket clients."""

    def __init__(self, max_queue_size: int = WS_QUEUE_MAX) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue_size)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            if q.full():
                with suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            # Dropping oldest already attempted; skip if still full.
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)


class MonitorStartRequest(BaseModel):
    model: str = "buffalo_sc"
    general_model: str | None = None
    custom_model: str | None = None
    disable_general: bool = False
    disable_custom: bool = False
    snapshot_interval: float = Field(default=SNAPSHOT_INTERVAL_DEFAULT, ge=1.0, le=3600.0)
    disable_gaze: bool = False
    gaze_arch: GazeArch = GAZE_ARCH_DEFAULT
    gaze_weights: str = "models/L2CSNet_gaze360.pkl"
    gaze_weights_source: str = (
        "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing"
    )
    disable_gaze_auto_download: bool = False
    gaze_max_interval: int = Field(default=GAZE_INTERVAL_DEFAULT, ge=1, le=60)
    gaze_target_fps_drop: float = Field(default=GAZE_TARGET_FPS_DROP_DEFAULT, ge=0.0, le=0.9)
    fps_cap: int = Field(default=FPS_CAP_DEFAULT, ge=1, le=60)


class ToggleRequest(BaseModel):
    general_yolo: bool | None = None
    custom_yolo: bool | None = None
    gaze: bool | None = None


class ChatRequest(BaseModel):
    question: str | None = None
    message: str | None = None
    session_id: str | None = None
    confirm_action_id: str | None = None


class EnrollStartRequest(BaseModel):
    name: str
    model: str = "buffalo_sc"
    # Enrollment samples at a faster cap than monitoring: more samples per
    # wall-clock second, and enrollment is brief by design.
    fps_cap: int = Field(default=ENROLL_FPS_CAP_DEFAULT, ge=1, le=60)


@dataclass
class EnrollStatus:
    active: bool = False
    name: str | None = None
    target_samples: int = 0
    samples_captured: int = 0
    is_finished: bool = False
    started_utc: str | None = None
    finished_utc: str | None = None


class PipelineManager:
    def __init__(self) -> None:
        self.frame_store = LatestFrameStore()
        self.event_hub = EventHub()
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._placeholder_idle = _encode_jpeg(_make_placeholder("STREAM INACTIVE")) or b""
        self._placeholder_stalled = _encode_jpeg(_make_placeholder("WAITING FOR FRAMES")) or b""

        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._control_q: Queue[dict[str, Any]] = Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._mode = "idle"
        self._session_id: str | None = None
        self._started_utc: str | None = None
        self._fps_cap = FPS_CAP_DEFAULT
        self._snapshot_interval = SNAPSHOT_INTERVAL_DEFAULT
        self._degraded = False
        self._degraded_reason: str | None = None
        self._last_error: str | None = None
        self._startup_phase = "idle"
        self._startup_started_utc: str | None = None
        self._startup_deadline_utc: str | None = None
        self._startup_failure_reason: str | None = None

        self._latest_detections: dict[str, Any] = {}
        self._latest_behavior: dict[str, Any] = {
            "recent_events": [],
            "summary": {},
        }
        self._latest_runtime_context: dict[str, Any] = {}
        self._recent_behavior_events: deque[dict[str, Any]] = deque(maxlen=200)
        self._latest_session_summary: dict[str, Any] | None = None
        self._latest_yolo_state: dict[str, Any] = {}
        self._latest_pipeline_state: dict[str, Any] = {}
        self._latest_config: dict[str, Any] = {}

        self._enroll = EnrollStatus()
        self._manual_snapshots = 0

    def _set_startup_phase(self, phase: str, failure_reason: str | None = None, publish: bool = True) -> None:
        with self._lock:
            self._startup_phase = str(phase)
            if phase == "starting":
                now = _iso()
                self._startup_started_utc = now
                self._startup_deadline_utc = _iso(datetime.now(UTC) + timedelta(seconds=STARTUP_TIMEOUT_SEC))
                self._startup_failure_reason = None
            elif phase == "failed":
                self._startup_failure_reason = failure_reason or "startup_failed"
            elif phase == "ready":
                self._startup_failure_reason = None
            elif phase == "idle":
                self._startup_started_utc = None
                self._startup_deadline_utc = None
                self._startup_failure_reason = None
        if publish:
            self._set_pipeline_state()

    def _mark_startup_failed(self, reason: str) -> None:
        with self._lock:
            self._last_error = str(reason)
            self._mode = "idle"
            self._degraded = True
            self._degraded_reason = str(reason)
            self._thread = None
        self._set_startup_phase("failed", failure_reason=str(reason), publish=False)
        self._set_pipeline_state(mode="idle", running=False, degraded=True, reason=str(reason))

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _event(self, event_type: str, payload: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        return {
            "type": event_type,
            "timestamp": _iso(),
            "session_id": session_id or self._session_id,
            "payload": payload,
        }

    @staticmethod
    def _compute_sleep_duration(frame_interval_sec: float, processing_sec: float) -> float:
        return max(0.0, float(frame_interval_sec) - float(processing_sec))

    def _publish_event(self, event_type: str, payload: dict[str, Any], session_id: str | None = None) -> None:
        event = self._event(event_type, payload, session_id=session_id)
        self._recent_events.append(event)
        if self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self.event_hub.publish(event), self._loop)
            fut.result(timeout=0.25)
        except Exception:
            # Non-fatal: websocket fan-out should never crash pipeline.
            pass

    def _set_pipeline_state(
        self,
        *,
        mode: str | None = None,
        running: bool | None = None,
        degraded: bool | None = None,
        reason: str | None = None,
    ) -> None:
        with self._lock:
            if mode is not None:
                self._mode = mode
            if degraded is not None:
                self._degraded = degraded
            if reason is not None or degraded is False:
                self._degraded_reason = reason if degraded else None
            state = self.status(running_override=running)
            self._latest_pipeline_state = state
        self._publish_event("pipeline_state", state)

    def status(self, running_override: bool | None = None) -> dict[str, Any]:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            if running_override is not None:
                running = running_override
            frame = self.frame_store.get()
            startup_elapsed_ms = None
            if self._startup_started_utc:
                started = _parse_iso(self._startup_started_utc)
                if started is not None:
                    startup_elapsed_ms = max(
                        int((datetime.now(UTC) - started).total_seconds() * 1000.0),
                        0,
                    )
            last_seq = _safe_int(frame.get("sequence"), 0)
            return {
                "running": running,
                "mode": self._mode,
                "session_id": self._session_id,
                "started_utc": self._started_utc,
                "fps_cap": self._fps_cap,
                "snapshot_interval": self._snapshot_interval,
                "degraded": self._degraded,
                "degraded_reason": self._degraded_reason,
                "last_error": self._last_error,
                "startup_phase": self._startup_phase,
                "startup_started_utc": self._startup_started_utc,
                "startup_deadline_utc": self._startup_deadline_utc,
                "startup_elapsed_ms": startup_elapsed_ms,
                "startup_failure_reason": self._startup_failure_reason,
                "last_frame_sequence": last_seq,
                "frame": {
                    "timestamp_utc": frame.get("timestamp_utc"),
                    "sequence": last_seq,
                },
                "yolo": self._latest_yolo_state,
                "config": self._latest_config,
                "enroll": {
                    "active": self._enroll.active,
                    "name": self._enroll.name,
                    "samples_captured": self._enroll.samples_captured,
                    "target_samples": self._enroll.target_samples,
                    "is_finished": self._enroll.is_finished,
                    "started_utc": self._enroll.started_utc,
                    "finished_utc": self._enroll.finished_utc,
                },
            }

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        lim = max(1, int(limit))
        return list(self._recent_events)[-lim:]

    def latest_detections(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_detections)

    def latest_behavior(self) -> dict[str, Any]:
        with self._lock:
            return {
                "summary": dict(self._latest_behavior.get("summary", {})),
                "recent_events": list(self._latest_behavior.get("recent_events", [])),
            }

    def latest_session_summary(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest_session_summary) if self._latest_session_summary else None

    def _validate_can_start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise HTTPException(status_code=409, detail=f"Pipeline already running in '{self._mode}' mode")

    def note_manual_snapshot(self) -> int:
        """Count a snapshot saved outside the worker (API request or chat action)."""
        with self._lock:
            self._manual_snapshots += 1
            return int(self._manual_snapshots)

    def start_monitor(self, req: MonitorStartRequest) -> dict[str, Any]:
        with self._lock:
            self._validate_can_start()
            self._stop_event.clear()
            self.frame_store.clear()
            self._mode = "monitor"
            self._session_id = _session_id("monitor")
            self._started_utc = _iso()
            self._fps_cap = int(req.fps_cap)
            self._snapshot_interval = float(req.snapshot_interval)
            self._degraded = False
            self._degraded_reason = None
            self._last_error = None
            self._startup_phase = "starting"
            self._startup_started_utc = _iso()
            self._startup_deadline_utc = _iso(datetime.now(UTC) + timedelta(seconds=STARTUP_TIMEOUT_SEC))
            self._startup_failure_reason = None
            self._enroll = EnrollStatus()
            self._latest_session_summary = None
            self._manual_snapshots = 0

            cfg = req.model_dump()
            self._latest_config = dict(cfg)
            self._thread = threading.Thread(
                target=self._run_monitor_worker,
                kwargs=cfg,
                name="monitor-worker",
                daemon=True,
            )
            self._thread.start()

        self._set_pipeline_state(mode="monitor", running=True, degraded=False)
        status = self.status(running_override=True)
        return {
            "status": "starting",
            "session_id": self._session_id,
            "mode": "monitor",
            "startup_phase": status.get("startup_phase"),
            "startup_started_utc": status.get("startup_started_utc"),
            "startup_deadline_utc": status.get("startup_deadline_utc"),
        }

    def start_enroll(self, req: EnrollStartRequest) -> dict[str, Any]:
        with self._lock:
            self._validate_can_start()
            if not req.name.strip():
                raise HTTPException(status_code=400, detail="Enrollment name cannot be empty")
            self._stop_event.clear()
            self.frame_store.clear()
            self._mode = "enroll"
            self._session_id = _session_id("enroll")
            self._started_utc = _iso()
            self._fps_cap = int(req.fps_cap)
            self._degraded = False
            self._degraded_reason = None
            self._last_error = None
            self._startup_phase = "idle"
            self._startup_started_utc = None
            self._startup_deadline_utc = None
            self._startup_failure_reason = None
            core = _core()
            self._enroll = EnrollStatus(
                active=True,
                name=req.name.strip(),
                target_samples=core.ENROLL_SAMPLES,
                samples_captured=0,
                is_finished=False,
                started_utc=_iso(),
                finished_utc=None,
            )
            self._latest_config = req.model_dump()

            self._thread = threading.Thread(
                target=self._run_enroll_worker,
                kwargs=req.model_dump(),
                name="enroll-worker",
                daemon=True,
            )
            self._thread.start()

        self._set_pipeline_state(mode="enroll", running=True, degraded=False)
        return {"status": "started", "session_id": self._session_id, "mode": "enroll"}

    def stop(self) -> dict[str, Any]:
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            self._stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._mode = "idle"
            self._thread = None
            self._degraded = False
            self._degraded_reason = None
            self._startup_phase = "idle"
            self._startup_started_utc = None
            self._startup_deadline_utc = None
            self._startup_failure_reason = None
            if self._enroll.active:
                self._enroll.active = False
                self._enroll.finished_utc = _iso()
        self._set_pipeline_state(mode="idle", running=False, degraded=False)
        return {"status": "stopped"}

    def enqueue_toggle(self, req: ToggleRequest) -> dict[str, Any]:
        if not (self._thread and self._thread.is_alive() and self._mode == "monitor"):
            raise HTTPException(status_code=409, detail="Monitoring pipeline is not running")

        payload = req.model_dump()
        self._control_q.put({"type": "toggle", **payload})
        return {"status": "queued", "requested": payload}

    def _drain_controls(self, detector: Any, state: dict[str, bool]) -> None:
        while True:
            try:
                cmd = self._control_q.get_nowait()
            except Empty:
                break
            if cmd.get("type") != "toggle":
                continue

            if cmd.get("general_yolo") is not None:
                desired = bool(cmd["general_yolo"])
                while bool(state["general"]) != desired:
                    state["general"] = detector.toggle_general()
            if cmd.get("custom_yolo") is not None:
                desired = bool(cmd["custom_yolo"])
                while bool(state["custom"]) != desired:
                    state["custom"] = detector.toggle_custom()
            if cmd.get("gaze") is not None:
                state["gaze"] = bool(cmd["gaze"])

            self._publish_event(
                "pipeline_state",
                {"toggle_update": dict(state), "session_id": self._session_id},
            )

    def _attempt_camera_recovery(
        self,
        *,
        startup: bool = False,
        deadline_monotonic: float | None = None,
    ) -> tuple[Any, Any] | None:
        core = _core()
        attempts = 0
        backoff = CAMERA_RECOVERY_BACKOFF_START_SEC
        while not self._stop_event.is_set():
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                return None
            attempts += 1
            if startup:
                self._set_startup_phase("camera_opening", publish=False)
            self._set_pipeline_state(
                degraded=True,
                reason=f"camera_unavailable_attempt_{attempts}",
            )
            try:
                cap = core._open_camera()
                reader = core._AsyncCameraReader(cap)
                if startup:
                    self._set_startup_phase("warming_up", publish=False)
                self._set_pipeline_state(degraded=False)
                return cap, reader
            except Exception as exc:
                self._last_error = str(exc)
                self._publish_event(
                    "pipeline_state",
                    {
                        "camera_recovery_attempt": attempts,
                        "error": str(exc),
                        "startup": bool(startup),
                    },
                )
                sleep_for = backoff
                if deadline_monotonic is not None:
                    sleep_for = min(sleep_for, max(deadline_monotonic - time.monotonic(), 0.0))
                if sleep_for > 0:
                    time.sleep(sleep_for)
                backoff = min(backoff * 2.0, CAMERA_RECOVERY_BACKOFF_MAX_SEC)
        return None

    def _run_enroll_worker(self, name: str, model: str, fps_cap: int, **_: Any) -> None:
        core = _core()
        capture = self._attempt_camera_recovery()
        if capture is None:
            return
        cap, reader = capture
        db = core.FaceDB.load()
        app = core._build_app(model)
        samples: list[np.ndarray] = []
        interval = 1.0 / max(int(fps_cap), 1)
        last_capture = -1e9
        start_utc = _iso()
        started = time.time()
        frames_total = 0
        frames_dropped = 0
        capture_done = False

        try:
            while not self._stop_event.is_set():
                loop_t0 = time.perf_counter()
                ok, frame = reader.read(timeout_sec=1.0)
                if not ok or frame is None:
                    frames_dropped += 1
                    if frames_dropped >= CAMERA_FAILURE_THRESHOLD:
                        reader.close()
                        cap.release()
                        capture = self._attempt_camera_recovery()
                        if capture is None:
                            return
                        cap, reader = capture
                        frames_dropped = 0
                    continue

                frames_total += 1
                faces = core._detect(app, frame)
                picked = core._best_face(faces)

                status_text = "No face detected"
                color = (0, 140, 255)
                if len(faces) > 1:
                    status_text = "Multiple faces detected"
                elif picked is not None:
                    bbox, emb, _, _landmarks = picked
                    core._bracket_box(frame, bbox, (200, 220, 0))
                    now_ts = time.time()
                    if (now_ts - last_capture) >= core.ENROLL_CAPTURE_INTERVAL_SEC:
                        samples.append(emb)
                        last_capture = now_ts
                        status_text = f"Captured {len(samples)} / {core.ENROLL_SAMPLES}"
                        color = (0, 210, 80)
                    else:
                        status_text = "Hold still..."
                        color = (200, 220, 0)

                core._hud(
                    frame,
                    [
                        (f"Enrolling: {name}", (255, 255, 255)),
                        (status_text, color),
                        ("API mode: no keyboard shortcuts", (140, 140, 140)),
                    ],
                )
                core._progress_bar(frame, len(samples), core.ENROLL_SAMPLES)

                jpeg = _encode_jpeg(frame)
                if jpeg:
                    meta = self.frame_store.update(jpeg, raw_frame=frame)
                    self._publish_event(
                        "detections",
                        {
                            "sequence": meta["sequence"],
                            "timestamp_utc": meta["timestamp_utc"],
                            "mode": "enroll",
                            "enroll": {
                                "name": name,
                                "samples_captured": len(samples),
                                "target_samples": core.ENROLL_SAMPLES,
                                "is_finished": len(samples) >= core.ENROLL_SAMPLES,
                            },
                        },
                    )

                with self._lock:
                    self._enroll.samples_captured = len(samples)
                    self._enroll.is_finished = len(samples) >= core.ENROLL_SAMPLES

                if len(samples) >= core.ENROLL_SAMPLES:
                    capture_done = True
                    break

                processing = time.perf_counter() - loop_t0
                time.sleep(self._compute_sleep_duration(interval, processing))

            if capture_done and samples:
                db.upsert(name, samples)
                db.save()
                with self._lock:
                    self._enroll.active = False
                    self._enroll.is_finished = True
                    self._enroll.finished_utc = _iso()

                core._append_metric(
                    "enroll",
                    {
                        "schema_version": common.ENROLL_SCHEMA_VERSION,
                        "session_id": self._session_id,
                        "name": name,
                        "model": model,
                        "camera_source": str(core.CAMERA_SOURCE),
                        "start_utc": start_utc,
                        "end_utc": _iso(),
                        "duration_sec": round(max(time.time() - started, 0.0), 3),
                        "samples_captured": len(samples),
                        "total_samples_for_name": int(db.counts[db.names.index(name)]),
                        "frame_metrics": {
                            "frames_total": frames_total,
                            "frames_dropped": frames_dropped,
                        },
                    },
                )

                self._publish_event(
                    "pipeline_state",
                    {
                        "enroll_complete": True,
                        "name": name,
                        "samples_captured": len(samples),
                    },
                )
        except Exception as exc:
            self._last_error = str(exc)
            # logging.exception already includes the exception and traceback.
            LOGGER.exception("Enrollment worker failed")
            self._publish_event("pipeline_state", {"error": str(exc)})
        finally:
            with suppress(Exception):
                reader.close()
            with suppress(Exception):
                cap.release()
            with self._lock:
                if self._mode == "enroll":
                    self._mode = "idle"
                    self._thread = None
                self._startup_phase = "idle"
                self._startup_started_utc = None
                self._startup_deadline_utc = None
                self._startup_failure_reason = None
                if self._enroll.active and not self._enroll.finished_utc:
                    self._enroll.active = False
                    self._enroll.finished_utc = _iso()
            self._set_pipeline_state(mode="idle", running=False, degraded=False)

    def _run_monitor_worker(
        self,
        model: str,
        general_model: str | None,
        custom_model: str | None,
        disable_general: bool,
        disable_custom: bool,
        snapshot_interval: float,
        disable_gaze: bool,
        gaze_arch: str,
        gaze_weights: str,
        gaze_weights_source: str,
        disable_gaze_auto_download: bool,
        fps_cap: int,
        gaze_max_interval: int = GAZE_INTERVAL_DEFAULT,
        gaze_target_fps_drop: float = GAZE_TARGET_FPS_DROP_DEFAULT,
        **_: Any,
    ) -> None:
        core = _core()

        startup_ready = False
        startup_failed_reason: str | None = None
        try:
            db = core.FaceDB.load()
            if not db.names:
                LOGGER.warning("No enrolled identities found. Monitoring will run with Unknown-only matches.")

            detector = core.DualYoloDetector(
                general_model_path=core._resolve_general_model_path(general_model),
                custom_model_path=custom_model or core._load_default_custom_model_path(),
                enable_general=not disable_general,
                enable_custom=(not disable_custom) and bool(custom_model or core._load_default_custom_model_path()),
            )
            memory = core.SceneMemoryManager(
                snapshot_interval_sec=float(snapshot_interval),
                base_dir=core.MEMORY_DIR,
                enable_vectors=False,
            )

            # Face recognition is degradable: object detection, gaze, memory and the
            # dashboard all keep working without it, so a failure here becomes a
            # reported reason rather than a dead monitor.
            app, face_reason = core._try_build_face_app(model)
            gaze_enabled = not disable_gaze
            gaze_runtime = (
                core._load_gaze_runtime(
                    gaze_arch=gaze_arch,
                    gaze_weights=gaze_weights,
                    gaze_weights_source=gaze_weights_source,
                    gaze_auto_download=not disable_gaze_auto_download,
                )
                if gaze_enabled
                else None
            )
            gaze_state = {
                "general": detector.get_state()["general"]["enabled"],
                "custom": detector.get_state()["custom"]["enabled"],
                "gaze": bool(gaze_runtime is not None and gaze_enabled),
            }
            gaze_scheduler = core.GazeScheduler(
                base_interval=GAZE_INTERVAL_DEFAULT,
                max_interval=gaze_max_interval,
                target_fps_drop=gaze_target_fps_drop,
            )
        except Exception as exc:
            startup_failed_reason = f"startup_init_error:{exc}"
            self._mark_startup_failed(startup_failed_reason)
            LOGGER.exception("Monitor startup failed before frame loop")
            return

        behavior_tracker = core._BehaviorTracker()
        startup_deadline_monotonic = time.monotonic() + STARTUP_TIMEOUT_SEC
        self._set_startup_phase("camera_opening")
        capture = self._attempt_camera_recovery(startup=True, deadline_monotonic=startup_deadline_monotonic)
        if capture is None:
            startup_failed_reason = "startup_timeout_camera_opening"
            self._mark_startup_failed(startup_failed_reason)
            return
        cap, reader = capture

        interval = 1.0 / max(int(fps_cap), 1)
        # Monotonic, not wall clock: a backwards wall-clock correction mid-session
        # turns one interval into microseconds and reports an absurd frame rate.
        t_prev = time.monotonic()

        frames_total = 0
        frames_with_faces = 0
        frames_empty = 0
        frames_dropped = 0
        detections_total = 0
        known_detections = 0
        unknown_detections = 0
        peak_simultaneous_faces = 0
        confidence_sum = 0.0
        object_detections_total = 0
        object_general_detections = 0
        object_custom_detections = 0
        object_conf_sum = 0.0
        object_general_conf_sum = 0.0
        object_custom_conf_sum = 0.0
        object_class_counts_total: Counter[str] = Counter()
        object_class_counts_general: Counter[str] = Counter()
        object_class_counts_custom: Counter[str] = Counter()
        label_counter: Counter[str] = Counter()
        detection_timeline: dict[str, int] = {}
        object_detection_timeline: dict[str, int] = {}
        events: list[dict[str, Any]] = []
        events_total = 0
        memory_auto_snapshots = 0
        memory_query_counts: Counter[str] = Counter()
        memory_query_hits: Counter[str] = Counter()
        memory_query_misses: Counter[str] = Counter()
        chat_queries_total = 0
        chat_queries_hit = 0
        chat_queries_llm = 0
        detection_latency_sum_ms = 0.0
        detection_calls = 0
        object_detection_calls = 0
        object_detection_latency_sum_ms = 0.0
        object_detection_latency_max_ms = 0.0
        frame_periods_ms: list[float] = []
        detection_latency_min_ms = float("inf")
        detection_latency_max_ms = 0.0
        gaze_inference_calls = 0
        gaze_inference_sum_ms = 0.0
        gaze_inference_min_ms = float("inf")
        gaze_inference_max_ms = 0.0
        last_gaze_rows: list[tuple[int, int, float, float] | None] | None = None
        fps_ema = 0.0
        fps_min = float("inf")
        fps_max = 0.0
        unknown_alert_count = 0
        last_unknown_alert_ts = -1e9
        people: dict[str, dict[str, Any]] = {}
        visible_prev: set[str] = set()
        latest_active_subjects: list[dict[str, Any]] = []
        latest_object_labels: list[str] = []
        start_dt = datetime.now(UTC)
        start_ts = time.time()
        camera_failures = 0
        detector_error_seen = False

        def add_event(
            event_type: str,
            message: str,
            severity: str = "info",
            extra: dict[str, Any] | None = None,
        ) -> None:
            nonlocal events_total
            row: dict[str, Any] = {
                "timestamp_utc": _iso(),
                "type": event_type,
                "severity": severity,
                "message": message,
            }
            if extra:
                row.update(extra)
            events.append(row)
            events_total += 1
            if len(events) > core.EVENTS_TIMELINE_CAP:
                del events[0]

        try:
            while not self._stop_event.is_set():
                if not startup_ready and time.monotonic() >= startup_deadline_monotonic:
                    startup_failed_reason = "startup_timeout_no_frames"
                    self._mark_startup_failed(startup_failed_reason)
                    break
                loop_t0 = time.perf_counter()
                self._drain_controls(detector, gaze_state)
                ok, frame = reader.read(timeout_sec=1.0)
                if not ok or frame is None:
                    frames_dropped += 1
                    camera_failures += 1
                    fail_threshold = STARTUP_CAMERA_FAILURE_THRESHOLD if not startup_ready else CAMERA_FAILURE_THRESHOLD
                    if camera_failures >= fail_threshold:
                        with suppress(Exception):
                            reader.close()
                        with suppress(Exception):
                            cap.release()
                        capture = self._attempt_camera_recovery(
                            startup=not startup_ready,
                            deadline_monotonic=startup_deadline_monotonic if not startup_ready else None,
                        )
                        if capture is None:
                            if not startup_ready:
                                startup_failed_reason = "startup_timeout_camera_recovery"
                                self._mark_startup_failed(startup_failed_reason)
                            break
                        cap, reader = capture
                        camera_failures = 0
                    time.sleep(0.01)
                    continue

                camera_failures = 0
                if not startup_ready:
                    startup_ready = True
                    self._set_startup_phase("ready")
                    if face_reason is not None:
                        # Marked once the camera is up, so camera recovery cannot
                        # clear the flag by setting degraded=False on its success path.
                        self._set_pipeline_state(
                            degraded=True,
                            reason=f"face_recognition_unavailable:{face_reason}",
                        )
                        add_event(
                            "face_recognition_unavailable",
                            str(face_reason),
                            severity="alert",
                        )
                frames_total += 1
                now_ts = time.time()

                mono_ts = time.monotonic()
                dt = max(mono_ts - t_prev, 1e-6)
                inst_fps = 1.0 / dt
                t_prev = mono_ts
                fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
                fps_min = min(fps_min, inst_fps)
                fps_max = max(fps_max, inst_fps)
                frame_periods_ms.append(dt * 1000.0)

                detect_t0 = time.perf_counter()
                face_rows = core._detect(app, frame)
                det_ms = (time.perf_counter() - detect_t0) * 1000.0
                detection_calls += 1
                detection_latency_sum_ms += det_ms
                detection_latency_min_ms = min(detection_latency_min_ms, det_ms)
                detection_latency_max_ms = max(detection_latency_max_ms, det_ms)

                # Timed separately from the face recognition call above, which is
                # what `avg_detection_latency_ms` has always measured. Object
                # detection was previously in no record at all.
                object_t0 = time.perf_counter()
                try:
                    object_rows = detector.detect(frame)
                except Exception as exc:
                    # A detector failure must not tear down the whole session; the
                    # CLI path already tolerated this, the API path did not.
                    object_rows = []
                    if not detector_error_seen:
                        detector_error_seen = True
                        add_event("object_detect_error", str(exc), severity="alert")
                        LOGGER.warning("Object detection error: %s", exc)
                object_latency_ms = (time.perf_counter() - object_t0) * 1000.0
                object_detection_calls += 1
                object_detection_latency_sum_ms += object_latency_ms
                object_detection_latency_max_ms = max(
                    object_detection_latency_max_ms, object_latency_ms
                )

                face_count = len(face_rows)
                object_count = len(object_rows)
                peak_simultaneous_faces = max(peak_simultaneous_faces, face_count)
                timeline_key = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
                detection_timeline[timeline_key] = detection_timeline.get(timeline_key, 0) + face_count
                object_detection_timeline[timeline_key] = object_detection_timeline.get(timeline_key, 0) + object_count

                if face_count:
                    frames_with_faces += 1
                else:
                    frames_empty += 1

                detections_total += face_count
                object_detections_total += object_count

                visible_now: set[str] = set()
                unknown_in_frame = False
                unknown_bboxes: list[np.ndarray] = []
                active_conf: dict[str, float] = {}
                render_rows: list[tuple[np.ndarray, np.ndarray | None, str, float]] = []

                for bbox, emb, _det_score, landmarks in face_rows:
                    label, score = core._match(emb, db)
                    render_rows.append((bbox, landmarks, label, score))
                    confidence_sum += score

                    if label == core.UNKNOWN_LABEL:
                        unknown_detections += 1
                        unknown_in_frame = True
                        unknown_bboxes.append(bbox)
                        continue

                    known_detections += 1
                    visible_now.add(label)
                    label_counter[label] += 1
                    active_conf[label] = max(active_conf.get(label, 0.0), score)

                    info = people.setdefault(
                        label,
                        {
                            "detections": 0,
                            "confidence_sum": 0.0,
                            "first_seen_utc": _iso(),
                            "last_seen_utc": _iso(),
                            "presence_sec": 0.0,
                            "present": False,
                            "last_enter_ts": None,
                        },
                    )
                    info["detections"] += 1
                    info["confidence_sum"] += score
                    info["last_seen_utc"] = _iso()

                entered = visible_now - visible_prev
                left = visible_prev - visible_now
                for name in entered:
                    info = people[name]
                    if not info["present"]:
                        info["present"] = True
                        info["last_enter_ts"] = now_ts
                        add_event("enter", f"{name} entered view")
                for name in left:
                    info = people[name]
                    if info["present"] and info["last_enter_ts"] is not None:
                        info["presence_sec"] += max(now_ts - float(info["last_enter_ts"]), 0.0)
                        info["last_enter_ts"] = None
                        info["present"] = False
                        add_event("exit", f"{name} left view")
                visible_prev = visible_now

                if unknown_in_frame and (now_ts - last_unknown_alert_ts) >= core.UNKNOWN_ALERT_COOLDOWN_SEC:
                    unknown_alert_count += 1
                    last_unknown_alert_ts = now_ts
                    snap = core._save_unknown_snapshot(frame, unknown_bboxes, datetime.now(UTC))
                    add_event(
                        "unknown_alert",
                        "Unknown face detected",
                        severity="alert",
                        extra={
                            "image_path": snap,
                            # None when the capture could not be written: the event is
                            # still reported, it just has no image to point at.
                            "image_name": Path(snap).name if snap else None,
                        },
                    )

                gaze_rows: list[tuple[int, int, float, float] | None] = [None for _ in render_rows]
                if gaze_state["gaze"] and gaze_runtime and render_rows:
                    if gaze_scheduler.should_run():
                        gaze_t0 = time.perf_counter()
                        gaze_rows = core._estimate_gaze_points(
                            frame,
                            [row[0] for row in render_rows],
                            [row[1] for row in render_rows],
                            gaze_runtime,
                        )
                        gaze_latency_ms = (time.perf_counter() - gaze_t0) * 1000.0
                        gaze_inference_calls += 1
                        gaze_inference_sum_ms += gaze_latency_ms
                        gaze_inference_min_ms = min(gaze_inference_min_ms, gaze_latency_ms)
                        gaze_inference_max_ms = max(gaze_inference_max_ms, gaze_latency_ms)
                        gaze_scheduler.observe(gaze_latency_ms, dt * 1000.0)
                        last_gaze_rows = gaze_rows
                    elif last_gaze_rows is not None:
                        # Skipped by the adaptive schedule: reuse the last estimate so
                        # attention tracking stays continuous between inferences.
                        gaze_rows = [
                            last_gaze_rows[i] if i < len(last_gaze_rows) else None
                            for i in range(len(render_rows))
                        ]

                gaze_observations: dict[str, str | None] = {}
                attention_rows: list[dict[str, Any]] = []
                face_snapshot_rows: list[dict[str, Any]] = []
                for i, (bbox, _landmarks, label, score) in enumerate(render_rows):
                    color = (0, 210, 80) if label != core.UNKNOWN_LABEL else (0, 140, 255)
                    core._bracket_box(frame, bbox, color)
                    core._label_tag(frame, f"{label} {score:.2f}", int(bbox[0]), int(bbox[1]) - 6, color)

                    target_info: dict[str, Any] | None = None
                    gaze_payload: dict[str, Any] | None = None
                    gaze_info = gaze_rows[i] if i < len(gaze_rows) else None
                    if gaze_info is not None:
                        gx, gy, pitch, yaw = gaze_info
                        cx = int((bbox[0] + bbox[2]) * 0.5)
                        cy = int((bbox[1] + bbox[3]) * 0.5)
                        cv2.line(frame, (cx, cy), (gx, gy), (200, 220, 0), 2, cv2.LINE_AA)
                        cv2.circle(frame, (gx, gy), 6, (200, 220, 0), -1)
                        target_info = core._infer_gaze_target(gaze_info, object_rows)
                        if target_info is not None:
                            core._label_tag(
                                frame,
                                f"target:{target_info['label']}",
                                gx + 8,
                                min(frame.shape[0] - 10, gy + 22),
                                (255, 255, 0),
                            )
                        gaze_payload = {
                            "endpoint": [int(gx), int(gy)],
                            "pitch": round(float(pitch), 6),
                            "yaw": round(float(yaw), 6),
                        }

                    if label != core.UNKNOWN_LABEL:
                        gaze_observations[label] = (
                            str(target_info.get("label")) if isinstance(target_info, dict) else None
                        )
                        if isinstance(target_info, dict):
                            attention_rows.append(
                                {
                                    "name": label,
                                    "target_object": str(target_info.get("label")),
                                    "method": target_info.get("method"),
                                    "distance_px": round(_safe_float(target_info.get("distance_px"), 0.0), 3),
                                }
                            )

                    face_snapshot_rows.append(
                        {
                            "name": label,
                            "confidence": round(float(score), 6),
                            "bbox": core._bbox_to_list(bbox),
                            "gaze": gaze_payload,
                            "target_object": str(target_info.get("label")) if isinstance(target_info, dict) else None,
                        }
                    )

                behavior_events = behavior_tracker.update(gaze_observations, now_ts)
                for evt in behavior_events:
                    core._append_metric("behavior_event", {"session_id": self._session_id, **evt})
                    self._recent_behavior_events.append(evt)
                    self._publish_event("behavior_event", evt)

                object_labels = set()
                for row in object_rows:
                    label = str(row.get("label", "object"))
                    conf = _safe_float(row.get("confidence"), 0.0)
                    bbox = np.asarray(row.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
                    source = str(row.get("source", "general"))
                    object_labels.add(label)
                    object_conf_sum += conf
                    object_class_counts_total[label] += 1
                    if source == "custom":
                        object_custom_detections += 1
                        object_custom_conf_sum += conf
                        object_class_counts_custom[label] += 1
                        color = (255, 0, 255)
                    else:
                        object_general_detections += 1
                        object_general_conf_sum += conf
                        object_class_counts_general[label] += 1
                        color = (255, 255, 0)
                    core._bracket_box(frame, bbox, color, thickness=2)
                    core._label_tag(
                        frame,
                        f"{source}:{label} {conf:.2f}",
                        int(bbox[0]),
                        max(int(bbox[1]) - 6, 20),
                        color,
                    )

                latest_object_labels = sorted(object_labels)
                latest_active_subjects = [
                    {"name": n, "confidence": round(c, 4)}
                    for n, c in sorted(active_conf.items(), key=lambda kv: kv[1], reverse=True)
                ]

                if memory.should_take_snapshot(now_ts):
                    snap = memory.save_snapshot(
                        frame,
                        object_rows,
                        current_time=now_ts,
                        manual=False,
                        faces=face_snapshot_rows,
                        object_detections=object_rows,
                        people=sorted(visible_now),
                        attention=attention_rows,
                    )
                    memory_auto_snapshots += 1
                    add_event(
                        "memory_snapshot_auto",
                        "Auto snapshot saved",
                        extra={
                            "image_path": snap.get("snapshot_path"),
                            "image_name": Path(str(snap.get("snapshot", ""))).name,
                        },
                    )
                    self._publish_event(
                        "memory_event",
                        {"action": "snapshot_auto", "snapshot": snap},
                    )

                yolo_state = detector.get_state()
                self._latest_yolo_state = yolo_state
                object_params = yolo_state.get("params") or {}
                gaze_status = "ON" if gaze_state["gaze"] else "OFF"
                if gaze_state["gaze"] and gaze_scheduler.adaptive:
                    gaze_status = f"ON 1/{gaze_scheduler.interval}"
                core._hud(
                    frame,
                    [
                        (f"Faces:{face_count} Objects:{object_count} FPS:{fps_ema:.1f}/{int(fps_cap)}", (255, 255, 255)),
                        (
                            (
                                f"Known:{known_detections} Unknown:{unknown_detections} "
                                f"G:{'ON' if yolo_state['general']['enabled'] else 'OFF'} "
                                f"C:{'ON' if yolo_state['custom']['enabled'] else 'OFF'}"
                            ),
                            (180, 180, 180),
                        ),
                        (f"Gaze:{gaze_status} Seq:{self.frame_store.get().get('sequence', 0)}", (170, 170, 170)),
                        (
                            # What object inference is actually using: the values used
                            # to be unreachable, so they were worth showing. Tolerant of
                            # a detector state that omits them (the HUD is not critical).
                            (
                                f"Objects: {object_params.get('imgsz', '?')}px "
                                f"conf {object_params.get('conf', '?')}"
                            ),
                            (160, 160, 160),
                        ),
                        (
                            f"Snapshots(auto/manual): {memory_auto_snapshots}/{self._manual_snapshots}",
                            (160, 160, 160),
                        ),
                    ],
                )

                jpeg = _encode_jpeg(frame)
                if jpeg:
                    meta = self.frame_store.update(jpeg, raw_frame=frame)
                    detections_payload = {
                        "sequence": meta["sequence"],
                        "timestamp_utc": meta["timestamp_utc"],
                        "mode": "monitor",
                        "faces": core._normalize_face_rows_for_json(face_snapshot_rows),
                        "objects": core._normalize_object_rows_for_json(object_rows),
                        "attention": attention_rows,
                        "active_subjects": latest_active_subjects,
                        "active_objects": latest_object_labels,
                        "counts": {
                            "face_count": face_count,
                            "object_count": object_count,
                            # Cumulative per-frame match counters (a face seen in
                            # N frames adds N). Unique people and cooldown-gated
                            # alert events are published alongside so the UI can
                            # show honest "how many faces" numbers.
                            "known_detections": known_detections,
                            "unknown_detections": unknown_detections,
                            "known_unique": len(people),
                            "unknown_alerts": unknown_alert_count,
                        },
                    }
                    with self._lock:
                        self._latest_detections = detections_payload
                        self._latest_behavior = {
                            "summary": behavior_tracker.summary(),
                            "recent_events": list(self._recent_behavior_events)[-30:],
                        }
                        self._latest_runtime_context = {
                            "frame": frame.copy(),
                            "object_rows": core._normalize_object_rows_for_json(object_rows),
                            "face_rows": core._normalize_face_rows_for_json(face_snapshot_rows),
                            "people": sorted(visible_now),
                            "attention_rows": list(attention_rows),
                        }
                    self._publish_event("detections", detections_payload)

                processing = time.perf_counter() - loop_t0
                # FPS throttling to keep browser MJPEG load under control.
                time.sleep(self._compute_sleep_duration(interval, processing))
        except Exception as exc:
            self._last_error = str(exc)
            # logging.exception already includes the exception and traceback.
            LOGGER.exception("Monitor worker crashed")
            if not startup_ready:
                startup_failed_reason = f"startup_exception:{exc}"
                self._mark_startup_failed(startup_failed_reason)
            else:
                self._publish_event("pipeline_state", {"error": str(exc)})
        finally:
            with suppress(Exception):
                reader.close()
            with suppress(Exception):
                cap.release()
            with suppress(Exception):
                memory.save_all_memory()

            if startup_failed_reason is not None and not startup_ready:
                with self._lock:
                    if self._mode == "monitor":
                        self._mode = "idle"
                        self._thread = None
                # This early exit is intentional and must stay inside `finally`: the
                # cleanup above has to run before the worker releases its slot, and the
                # session summary below must not be written for a session that never
                # started. (Ruff B012 flags it because a `return` in `finally` can also
                # swallow a BaseException; the practical exposure here is nil, since
                # this runs on a worker thread and no signal is delivered to it.)
                return  # noqa: B012

            end_dt = datetime.now(UTC)
            end_ts = time.time()
            duration_sec = max(end_ts - start_ts, 0.0)

            for evt in behavior_tracker.finalize(end_ts):
                core._append_metric("behavior_event", {"session_id": self._session_id, **evt})
                self._recent_behavior_events.append(evt)
                self._publish_event("behavior_event", evt)

            for name in visible_prev:
                presence = people.get(name)
                if presence and presence["present"] and presence["last_enter_ts"] is not None:
                    presence["presence_sec"] += max(end_ts - float(presence["last_enter_ts"]), 0.0)
                    presence["last_enter_ts"] = None
                    info["present"] = False

            people_clean: dict[str, dict[str, Any]] = {}
            for name, info in people.items():
                detections = _safe_int(info.get("detections"))
                conf_sum = _safe_float(info.get("confidence_sum"))
                people_clean[name] = {
                    "detections": detections,
                    "avg_confidence": (conf_sum / detections) if detections > 0 else 0.0,
                    "first_seen_utc": info.get("first_seen_utc"),
                    "last_seen_utc": info.get("last_seen_utc"),
                    "presence_sec": round(_safe_float(info.get("presence_sec")), 3),
                }

            behavior_summary = behavior_tracker.summary()
            with self._lock:
                manual_snapshots = int(self._manual_snapshots)
            aggregate = core.build_session_aggregate(
                session_id=self._session_id,
                duration_sec=duration_sec,
                frames_total=frames_total,
                frames_with_faces=frames_with_faces,
                frames_empty=frames_empty,
                frames_dropped=frames_dropped,
                detections_total=detections_total,
                known_detections=known_detections,
                unknown_detections=unknown_detections,
                peak_simultaneous_faces=peak_simultaneous_faces,
                confidence_sum=confidence_sum,
                detection_calls=detection_calls,
                detection_latency_sum_ms=detection_latency_sum_ms,
                detection_latency_min_ms=detection_latency_min_ms,
                detection_latency_max_ms=detection_latency_max_ms,
                object_detection_calls=object_detection_calls,
                object_detection_latency_sum_ms=object_detection_latency_sum_ms,
                object_detection_latency_max_ms=object_detection_latency_max_ms,
                fps_ema=fps_ema,
                fps_min=fps_min,
                fps_max=fps_max,
                fps_cap=self._fps_cap,
                frame_periods_ms=frame_periods_ms,
                object_detections_total=object_detections_total,
                object_general_detections=object_general_detections,
                object_custom_detections=object_custom_detections,
                object_conf_sum=object_conf_sum,
                object_general_conf_sum=object_general_conf_sum,
                object_custom_conf_sum=object_custom_conf_sum,
                unknown_alert_count=unknown_alert_count,
                memory_auto_snapshots=memory_auto_snapshots,
                memory_manual_snapshots=manual_snapshots,
                memory_total_store=_safe_int(memory.get_memory_stats().get("total_snapshots"), 0),
                memory_query_counts=memory_query_counts,
                memory_query_hits=memory_query_hits,
                memory_query_misses=memory_query_misses,
                chat_queries_total=chat_queries_total,
                chat_queries_hit=chat_queries_hit,
                chat_queries_llm=chat_queries_llm,
                unique_individuals_seen=len(people_clean),
                current_people_visible=len(visible_prev),
                active_subjects=latest_active_subjects,
                active_objects=latest_object_labels,
                detection_timeline=detection_timeline,
                object_detection_timeline=object_detection_timeline,
                object_class_counts_total=object_class_counts_total,
                object_class_counts_general=object_class_counts_general,
                object_class_counts_custom=object_class_counts_custom,
                behavior_summary=behavior_summary,
                gaze_enabled=gaze_enabled,
                gaze_model_loaded=bool(gaze_runtime is not None),
                **gaze_scheduler.metrics(),
                gaze_inference_calls=gaze_inference_calls,
                gaze_inference_sum_ms=gaze_inference_sum_ms,
                gaze_inference_min_ms=gaze_inference_min_ms,
                gaze_inference_max_ms=gaze_inference_max_ms,
                detector_state=detector.get_state(),
                model_paths={
                    "general": core._resolve_general_model_path(general_model),
                    "custom": custom_model or core._load_default_custom_model_path(),
                },
                face_recognition_enabled=app is not None,
            )

            core._append_metric(
                "recognize_session",
                {
                    "schema_version": common.SESSION_SCHEMA_VERSION,
                    "session_id": self._session_id,
                    "model": model,
                    "camera_source": str(core.CAMERA_SOURCE),
                    "start_utc": _iso(start_dt),
                    "end_utc": _iso(end_dt),
                    "duration_sec": round(duration_sec, 3),
                    "aggregate": aggregate,
                    "people": people_clean,
                    "label_counts": dict(label_counter),
                    "events": events,
                    "events_total_count": events_total,
                },
            )

            with self._lock:
                self._latest_session_summary = {
                    "session_id": self._session_id,
                    "start_utc": _iso(start_dt),
                    "end_utc": _iso(end_dt),
                    "duration_sec": round(duration_sec, 3),
                    "aggregate": aggregate,
                    "people": people_clean,
                    "label_counts": dict(label_counter),
                    "events": events,
                    "events_total_count": events_total,
                }
                if self._mode == "monitor":
                    self._mode = "idle"
                    self._thread = None
                self._startup_phase = "idle"
                self._startup_started_utc = None
                self._startup_deadline_utc = None
                self._startup_failure_reason = None

            self._set_pipeline_state(mode="idle", running=False, degraded=False)

    def get_stream_frame(self) -> dict[str, Any]:
        frame = self.frame_store.get()
        if frame["frame_bytes"]:
            return frame
        return {
            "frame_bytes": self._placeholder_idle if self._mode == "idle" else self._placeholder_stalled,
            "timestamp_utc": _iso(),
            "sequence": 0,
        }

    def build_chat_runtime_context(self) -> dict[str, Any]:
        with self._lock:
            context = dict(self._latest_runtime_context)
        frame = self.frame_store.get_raw_frame()
        if frame is not None:
            context["frame"] = frame
        if "object_rows" not in context:
            context["object_rows"] = []
        if "face_rows" not in context:
            context["face_rows"] = []
        if "people" not in context:
            context["people"] = []
        if "attention_rows" not in context:
            context["attention_rows"] = []
        context["memory"] = _core().SceneMemoryManager(base_dir=_core().MEMORY_DIR, enable_vectors=False)
        return context


MANAGER = PipelineManager()


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hooks. `@app.on_event` is deprecated in current FastAPI."""
    MANAGER.set_event_loop(asyncio.get_running_loop())
    MANAGER._set_pipeline_state(mode="idle", running=False, degraded=False)
    try:
        yield
    finally:
        # Stop any worker still running so a shutdown does not leave a camera open.
        with suppress(Exception):
            MANAGER.stop()


app = FastAPI(title="Monitoring Backend", version="1.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/incidents", StaticFiles(directory=str(INCIDENTS_DIR)), name="incidents")
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")


def _load_metric_events() -> list[dict[str, Any]]:
    return common.read_jsonl(METRICS_PATH)


def _filter_metric_events(
    *,
    event_type: str | None,
    limit: int,
    from_ts: str | None,
    to_ts: str | None,
) -> list[dict[str, Any]]:
    return common.filter_events(
        _load_metric_events(),
        event_type=event_type,
        limit=limit,
        from_ts=from_ts,
        to_ts=to_ts,
    )


@dataclass
class ChatSessionState:
    created_monotonic: float
    last_seen_monotonic: float
    history: list[dict[str, Any]]
    pending_actions: dict[str, dict[str, Any]]


class ChatSessionStore:
    def __init__(self, ttl_sec: int = CHAT_SESSION_TTL_SEC, history_max_turns: int = CHAT_HISTORY_MAX_TURNS) -> None:
        self.ttl_sec = int(ttl_sec)
        self.history_max_turns = int(history_max_turns)
        self._lock = threading.RLock()
        self._sessions: dict[str, ChatSessionState] = {}

    def _prune_locked(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, state in self._sessions.items()
            if (now - state.last_seen_monotonic) > float(self.ttl_sec)
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def get_or_create(self, session_id: str | None = None) -> str:
        with self._lock:
            self._prune_locked()
            now = time.monotonic()
            sid = (session_id or "").strip()
            if sid and sid in self._sessions:
                self._sessions[sid].last_seen_monotonic = now
                return sid
            sid = f"chat-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
            self._sessions[sid] = ChatSessionState(
                created_monotonic=now,
                last_seen_monotonic=now,
                history=[],
                pending_actions={},
            )
            return sid

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.last_seen_monotonic = time.monotonic()
            state.history.append(
                {
                    "role": role,
                    "content": str(content),
                    "timestamp_utc": _iso(),
                }
            )
            if len(state.history) > (self.history_max_turns * 2):
                state.history = state.history[-(self.history_max_turns * 2) :]

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            state.last_seen_monotonic = time.monotonic()
            return list(state.history)

    def propose_action(self, session_id: str, action: dict[str, Any], ttl_sec: int = CHAT_CONFIRM_TTL_SEC) -> str:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return ""
            confirm_id = uuid4().hex
            state.pending_actions[confirm_id] = {
                "created_monotonic": time.monotonic(),
                "expires_sec": int(ttl_sec),
                "action": dict(action),
            }
            return confirm_id

    def consume_action(self, session_id: str, confirm_action_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            row = state.pending_actions.pop(confirm_action_id, None)
            if not row:
                return None
            created = _safe_float(row.get("created_monotonic"), 0.0)
            expires = _safe_int(row.get("expires_sec"), CHAT_CONFIRM_TTL_SEC)
            if (time.monotonic() - created) > float(expires):
                return None
            action = row.get("action")
            return dict(action) if isinstance(action, dict) else None


def _compact_citation(row: dict[str, Any], idx: int) -> dict[str, Any]:
    source = str(row.get("source", "metrics"))
    ts = row.get("timestamp_utc") or row.get("timestamp_local")
    detail = str(row.get("detail", "")).strip()
    return {
        "id": f"C{idx}",
        "source": source,
        "timestamp": ts,
        "detail": detail,
    }


def _build_chat_grounding(question: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=CHAT_CONTEXT_LOOKBACK_MIN)
    metrics = _load_metric_events()
    recent_metric_rows: list[dict[str, Any]] = []
    for row in reversed(metrics):
        dt = _parse_iso(row.get("timestamp_utc"))
        if dt is None or dt < cutoff:
            continue
        et = str(row.get("event_type", ""))
        if et not in {"behavior_event", "recognize_session", "summary_query", "chat_query"}:
            continue
        recent_metric_rows.append(row)
        if len(recent_metric_rows) >= 20:
            break
    recent_metric_rows.reverse()

    memory = runtime_context.get("memory")
    if memory is None:
        memory = _core().SceneMemoryManager(base_dir=_core().MEMORY_DIR, enable_vectors=False)
    memory_hits = memory.search_similar_scene(question, top_k=5)
    recent_snaps = memory.get_recent_snapshots(minutes=15, limit=6)

    citations_raw: list[dict[str, Any]] = []
    latest_session = None
    for row in reversed(metrics):
        if row.get("event_type") == "recognize_session":
            latest_session = row
            break
    if isinstance(latest_session, dict):
        # Bind once so the isinstance check actually narrows the value used below.
        raw_agg = latest_session.get("aggregate")
        agg = raw_agg if isinstance(raw_agg, dict) else {}
        citations_raw.append(
            {
                "source": "recognize_session",
                "timestamp_utc": latest_session.get("end_utc") or latest_session.get("timestamp_utc"),
                "detail": (
                    f"known={_safe_int(agg.get('known_detections'))}, "
                    f"unknown={_safe_int(agg.get('unknown_detections'))}, "
                    f"active_people={_safe_int(agg.get('current_people_visible'))}"
                ),
            }
        )
    for row in recent_metric_rows[-4:]:
        citations_raw.append(
            {
                "source": str(row.get("event_type", "metrics")),
                "timestamp_utc": row.get("timestamp_utc"),
                "detail": (
                    str(row.get("message"))
                    if row.get("message")
                    else f"event={row.get('event')} person={row.get('person')} object={row.get('target_object')}"
                ),
            }
        )
    for row in memory_hits[:3]:
        citations_raw.append(
            {
                "source": "memory_hit",
                "timestamp_utc": row.get("timestamp_utc"),
                "detail": f"snapshot={row.get('snapshot')} objects={row.get('objects')}",
            }
        )
    for row in recent_snaps[-2:]:
        citations_raw.append(
            {
                "source": "recent_snapshot",
                "timestamp_utc": row.get("timestamp_utc"),
                "detail": f"snapshot={row.get('snapshot')} people={row.get('people', [])}",
            }
        )
    citations = [_compact_citation(row, i + 1) for i, row in enumerate(citations_raw[:8])]
    return {
        "runtime": {
            "people": runtime_context.get("people", []),
            "attention_rows": runtime_context.get("attention_rows", []),
            "object_rows": runtime_context.get("object_rows", [])[:20],
            "face_rows": runtime_context.get("face_rows", [])[:20],
        },
        "latest_session_summary": MANAGER.latest_session_summary(),
        "recent_metric_events": recent_metric_rows[-12:],
        "memory_hits": memory_hits[:5],
        "recent_snapshots": recent_snaps[-5:],
        "citations": citations,
    }


def _query_groq_grounded(
    *,
    question: str,
    session_history: list[dict[str, Any]],
    grounding: dict[str, Any],
) -> tuple[str | None, bool]:
    # Both spellings are supported on purpose: `.env.example` documents the lowercase
    # form, so dropping it would break existing setups.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")  # noqa: SIM112
    if not api_key:
        return None, False
    try:
        from groq import Groq
    except Exception:
        return None, False

    history_lines = []
    for row in session_history[-(CHAT_HISTORY_MAX_TURNS * 2) :]:
        role = str(row.get("role", "user"))
        content = str(row.get("content", ""))
        history_lines.append(f"{role}: {content}")

    system_prompt = (
        "You are a surveillance monitoring assistant. "
        "Use ONLY the provided grounding context and never invent facts. "
        "If evidence is missing, say 'Insufficient evidence in current logs/memory.' "
        "When making factual claims, include citation IDs like [C1], [C2] based on provided citations."
    )
    user_prompt = (
        "Grounding context JSON:\n"
        f"{json.dumps(grounding, ensure_ascii=True)}\n\n"
        "Recent chat history:\n"
        f"{chr(10).join(history_lines[-12:]) if history_lines else '(none)'}\n\n"
        f"User question: {question}\n\n"
        "Answer concisely with citations. If unsupported, return insufficient evidence message."
    )
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        LOGGER.warning(
            "Groq chat call failed (%s); falling back to deterministic answer.", exc
        )
        return None, False
    if not response.choices:
        return None, False
    msg = response.choices[0].message
    raw_content = getattr(msg, "content", None)
    if not raw_content:
        return None, False
    return str(raw_content).strip(), True


def _extract_minutes_from_query(question: str, default: int = 5) -> int:
    core = _core()
    base = max(int(default), 1)
    try:
        parsed = max(_safe_int(core._parse_minutes_from_text(question, default=base), base), 1)
    except Exception:
        parsed = base
    q = str(question or "").lower()
    fuzzy = re.search(r"(\d+)\s*(?:m+in(?:ute)?s?|mins?|min(?:ute)?s?)\b", q)
    if fuzzy:
        parsed = max(_safe_int(fuzzy.group(1), parsed), 1)
    return parsed


def _collect_recent_people_signal(
    minutes: int,
    runtime_context: dict[str, Any],
    memory: Any,
) -> tuple[int, list[str]]:
    core = _core()
    lookback = max(int(minutes), 1)
    cutoff = datetime.now(UTC) - timedelta(minutes=lookback)
    events = _load_metric_events()
    names: set[str] = set()
    max_unique = 0
    unknown_tokens = {"unknown", "unknown person", str(getattr(core, "UNKNOWN_LABEL", "unknown")).lower()}

    for event in events:
        dt = _parse_iso(event.get("timestamp_utc"))
        if dt is None or dt < cutoff:
            continue
        et = str(event.get("event_type", ""))
        if et == "recognize_session":
            raw_agg = event.get("aggregate")
            agg = raw_agg if isinstance(raw_agg, dict) else {}
            max_unique = max(max_unique, _safe_int(agg.get("unique_individuals_seen"), 0))
            people_map = event.get("people")
            if isinstance(people_map, dict):
                for name in people_map:
                    label = str(name).strip()
                    if label and label.lower() not in unknown_tokens:
                        names.add(label)
            active = agg.get("active_subjects")
            if isinstance(active, list):
                for row in active:
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get("name", "")).strip()
                    if label and label.lower() not in unknown_tokens:
                        names.add(label)
        elif et == "behavior_event":
            person = str(event.get("person", "")).strip()
            if person and person.lower() not in unknown_tokens:
                names.add(person)

    runtime_people = runtime_context.get("people")
    if isinstance(runtime_people, list):
        for person in runtime_people:
            label = str(person).strip()
            if label and label.lower() not in unknown_tokens:
                names.add(label)

    if memory is not None and hasattr(memory, "get_recent_snapshots"):
        try:
            rows = memory.get_recent_snapshots(minutes=lookback, limit=20)
        except Exception:
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for person in row.get("people", []) or []:
                    label = str(person).strip()
                    if label and label.lower() not in unknown_tokens:
                        names.add(label)

    names_sorted = sorted(names)
    count = max(max_unique, len(names_sorted))
    return count, names_sorted


def _parse_requested_action(message: str) -> dict[str, Any] | None:
    q = message.strip().lower()
    if not q:
        return None
    if "capture snapshot" in q or "take snapshot" in q or "save snapshot" in q:
        return {"type": "snapshot"}
    if "start monitoring" in q or "start monitor" in q:
        return {"type": "monitor_start"}
    if "stop monitoring" in q or "stop monitor" in q:
        return {"type": "monitor_stop"}
    if "run summary" in q or "generate summary" in q:
        minutes = 5
        m = re.search(r"(\d+)\s*(?:minute|min|mins)", q)
        if m:
            minutes = max(_safe_int(m.group(1), 5), 1)
        return {"type": "run_summary", "minutes": minutes}
    if "general yolo" in q:
        if any(tok in q for tok in ["turn on", "enable"]):
            return {"type": "toggle_general_yolo", "value": True}
        if any(tok in q for tok in ["turn off", "disable"]):
            return {"type": "toggle_general_yolo", "value": False}
    if "custom yolo" in q:
        if any(tok in q for tok in ["turn on", "enable"]):
            return {"type": "toggle_custom_yolo", "value": True}
        if any(tok in q for tok in ["turn off", "disable"]):
            return {"type": "toggle_custom_yolo", "value": False}
    if "gaze" in q:
        if any(tok in q for tok in ["turn on", "enable"]):
            return {"type": "toggle_gaze", "value": True}
        if any(tok in q for tok in ["turn off", "disable"]):
            return {"type": "toggle_gaze", "value": False}
    return None


def _proposal_text(action: dict[str, Any]) -> str:
    t = str(action.get("type", ""))
    if t == "snapshot":
        return "capture a manual snapshot"
    if t == "monitor_start":
        return "start the monitoring pipeline"
    if t == "monitor_stop":
        return "stop the monitoring pipeline"
    if t == "run_summary":
        return f"generate a situation summary for the last {_safe_int(action.get('minutes'), 5)} minutes"
    if t == "toggle_general_yolo":
        return f"set General YOLO to {'ON' if bool(action.get('value')) else 'OFF'}"
    if t == "toggle_custom_yolo":
        return f"set Custom YOLO to {'ON' if bool(action.get('value')) else 'OFF'}"
    if t == "toggle_gaze":
        return f"set gaze to {'ON' if bool(action.get('value')) else 'OFF'}"
    return "run an operation"


def _capture_snapshot_for_action() -> dict[str, Any]:
    core = _core()
    context = MANAGER.build_chat_runtime_context()
    frame = context.get("frame")
    if frame is None:
        raise HTTPException(status_code=409, detail="No live frame available for snapshot")
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    snap = memory.save_snapshot(
        frame,
        context.get("object_rows", []),
        current_time=time.time(),
        manual=True,
        faces=context.get("face_rows"),
        object_detections=context.get("object_rows"),
        people=context.get("people"),
        attention=context.get("attention_rows"),
    )
    MANAGER.note_manual_snapshot()
    MANAGER._publish_event("memory_event", {"action": "snapshot_manual", "snapshot": snap})
    return snap


def _execute_confirmed_action(action: dict[str, Any]) -> dict[str, Any]:
    action_type = str(action.get("type", ""))
    executed: dict[str, Any] = {"type": action_type, "status": "ok"}
    if action_type == "snapshot":
        snap = _capture_snapshot_for_action()
        executed["snapshot"] = snap
        return executed
    if action_type == "run_summary":
        minutes = max(_safe_int(action.get("minutes"), 5), 1)
        core = _core()
        summary = core._build_situation_summary(minutes=minutes)
        MANAGER._publish_event("summary_result", summary)
        executed["summary"] = summary
        executed["minutes"] = minutes
        return executed
    if action_type == "monitor_start":
        MANAGER.start_monitor(MonitorStartRequest())
        return executed
    if action_type == "monitor_stop":
        MANAGER.stop()
        return executed
    if action_type == "toggle_general_yolo":
        MANAGER.enqueue_toggle(ToggleRequest(general_yolo=bool(action.get("value"))))
        executed["value"] = bool(action.get("value"))
        return executed
    if action_type == "toggle_custom_yolo":
        MANAGER.enqueue_toggle(ToggleRequest(custom_yolo=bool(action.get("value"))))
        executed["value"] = bool(action.get("value"))
        return executed
    if action_type == "toggle_gaze":
        MANAGER.enqueue_toggle(ToggleRequest(gaze=bool(action.get("value"))))
        executed["value"] = bool(action.get("value"))
        return executed
    raise HTTPException(status_code=400, detail=f"Unsupported action '{action_type}'")


def _deterministic_chat_response(question: str, runtime_context: dict[str, Any]) -> dict[str, Any] | None:
    core = _core()
    q = question.strip().lower()
    if not q:
        return None
    memory = runtime_context.get("memory")
    if memory is None:
        memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    if re.fullmatch(r"(hi+|hello+|hey+|yo+|namaste|good\s+(?:morning|afternoon|evening))(?:[!.?,\s]*)", q):
        status = MANAGER.status()
        mode = str(status.get("mode", "idle"))
        phase = str(status.get("startup_phase", "idle"))
        return {
            "reply": (
                f"Hello. Pipeline mode is '{mode}' (startup phase: {phase}). "
                "I can answer summaries, person/object last-seen, memory status, recent snapshots, "
                "who is present, attention events, and person counts over a time window."
            ),
            "intent": "greeting",
            "hit": True,
            "include_citations": False,
        }
    if ("what happened" in q and "minute" in q) or "recent activity" in q or "situation summary" in q:
        minutes = _extract_minutes_from_query(q, default=5)
        summary = core._build_situation_summary(minutes=minutes)
        rendered = core._render_situation_summary(summary)
        return {
            "reply": rendered,
            "intent": "session_summary",
            "hit": True,
            "summary": summary,
        }
    if (
        ("how many" in q or "number of" in q or "count" in q)
        and any(tok in q for tok in ["person", "people", "face", "faces", "individual"])
    ):
        minutes = _extract_minutes_from_query(q, default=5)
        people_count, people_names = _collect_recent_people_signal(minutes=minutes, runtime_context=runtime_context, memory=memory)
        if people_count <= 0:
            return {
                "reply": f"I could not find person detections in the last {minutes} minutes.",
                "intent": "person_count",
                "hit": False,
            }
        names_suffix = ""
        if people_names:
            names_suffix = f" ({', '.join(people_names[:8])})"
        return {
            "reply": (
                f"I detected {people_count} distinct person{'s' if people_count != 1 else ''} "
                f"in the last {minutes} minutes{names_suffix}."
            ),
            "intent": "person_count",
            "hit": True,
        }
    if "memory status" in q or "memory stats" in q:
        return {
            "reply": json.dumps(memory.get_memory_stats(), ensure_ascii=True, indent=2),
            "intent": "memory_stats",
            "hit": True,
        }
    if "recent snapshot" in q:
        minutes = max(_safe_int(core._parse_minutes_from_text(q, default=5), 5), 1)
        rows = memory.get_recent_snapshots(minutes=minutes, limit=8)
        if rows:
            lines = [
                f"{row.get('timestamp_local')} | objects={row.get('objects')} | snapshot={row.get('snapshot')}"
                for row in rows
            ]
            return {
                "reply": f"Recent snapshots in last {minutes} minutes:\n" + "\n".join(lines),
                "intent": "memory_recent",
                "hit": True,
            }
        return {"reply": f"No recent snapshots found in the last {minutes} minutes.", "intent": "memory_recent", "hit": False}
    if "last see" in q or "last seen" in q:
        target = core._extract_last_seen_target(question)
        if not target:
            return {"reply": "Please specify who or what you want to look up.", "intent": "last_seen", "hit": False}
        db = core.FaceDB.load()
        known = {n.lower() for n in db.names}
        if target.lower() in known:
            row = memory.find_person_last_seen(target)
            if row:
                return {
                    "reply": (
                        f"Last seen '{target}' at {row.get('timestamp_local')} | "
                        f"people={row.get('people', [])} | snapshot={row.get('snapshot')}"
                    ),
                    "intent": "person_last_seen",
                    "hit": True,
                }
            return {"reply": f"I could not find recent sightings for '{target}'.", "intent": "person_last_seen", "hit": False}
        row = memory.find_object_last_seen(target)
        if row:
            return {
                "reply": (
                    f"Last seen '{target}' at {row.get('timestamp_local')} | "
                    f"objects={row.get('objects', [])} | snapshot={row.get('snapshot')}"
                ),
                "intent": "object_last_seen",
                "hit": True,
            }
        return {"reply": f"I could not find object '{target}' in memory.", "intent": "object_last_seen", "hit": False}
    if "who is present" in q or "who was present" in q:
        people = runtime_context.get("people", [])
        if people:
            names = sorted({str(p) for p in people if str(p).strip()})
            if names:
                return {"reply": "Currently visible: " + ", ".join(names), "intent": "presence", "hit": True}
        return {"reply": core._answer_current_presence(), "intent": "presence", "hit": True}
    if "looking at" in q or "look at" in q:
        db = core.FaceDB.load()
        attention_reply, attention_found = core._answer_attention_query(question, db.names)
        return {
            "reply": attention_reply,
            "intent": "attention_lookup",
            "hit": bool(attention_found),
        }
    return None


CHAT_SESSIONS = ChatSessionStore()


def _queue_chat_summary_events(result: dict[str, Any]) -> None:
    payload = {
        "intent": result.get("intent"),
        "hit": bool(result.get("hit", True)),
        "used_llm": bool(result.get("used_llm", False)),
        "action": result.get("action"),
        "session_id": result.get("session_id"),
        "reply": result.get("reply") or result.get("answer"),
        "citations": result.get("citations", []),
        "proposed_action": result.get("proposed_action"),
        "executed_action": result.get("executed_action"),
        "grounded": bool(result.get("grounded", True)),
    }
    MANAGER._publish_event("chat_result", payload)
    summary_payload = None
    if isinstance(result.get("summary"), dict):
        summary_payload = result.get("summary")
    executed = result.get("executed_action")
    if summary_payload is None and isinstance(executed, dict) and isinstance(executed.get("summary"), dict):
        summary_payload = executed.get("summary")
    if isinstance(summary_payload, dict):
        MANAGER._publish_event("summary_result", summary_payload)
    snapshot_payload = None
    if isinstance(result.get("snapshot"), dict):
        snapshot_payload = result.get("snapshot")
    if snapshot_payload is None and isinstance(executed, dict) and isinstance(executed.get("snapshot"), dict):
        snapshot_payload = executed.get("snapshot")
    if isinstance(snapshot_payload, dict):
        MANAGER._publish_event(
            "memory_event",
            {
                "action": "snapshot_manual",
                "snapshot": snapshot_payload,
            },
        )


@app.post("/api/v1/monitor/start")
def api_monitor_start(req: MonitorStartRequest) -> dict[str, Any]:
    return MANAGER.start_monitor(req)


@app.post("/api/v1/monitor/stop")
def api_monitor_stop() -> dict[str, Any]:
    return MANAGER.stop()


@app.get("/api/v1/monitor/status")
def api_monitor_status() -> dict[str, Any]:
    return MANAGER.status()


@app.patch("/api/v1/monitor/toggles")
def api_monitor_toggles(req: ToggleRequest) -> dict[str, Any]:
    return MANAGER.enqueue_toggle(req)


@app.post("/api/v1/monitor/snapshot")
def api_monitor_snapshot() -> dict[str, Any]:
    core = _core()
    context = MANAGER.build_chat_runtime_context()
    frame = context.get("frame")
    if frame is None:
        raise HTTPException(status_code=409, detail="No live frame available for snapshot")

    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    snap = memory.save_snapshot(
        frame,
        context.get("object_rows", []),
        current_time=time.time(),
        manual=True,
        faces=context.get("face_rows"),
        object_detections=context.get("object_rows"),
        people=context.get("people"),
        attention=context.get("attention_rows"),
    )
    MANAGER.note_manual_snapshot()
    MANAGER._publish_event("memory_event", {"action": "snapshot_manual", "snapshot": snap})
    return {"status": "ok", "snapshot": snap}


async def build_video_stream_generator() -> AsyncIterator[bytes]:
    last_seq = -1
    last_emit_mono = 0.0
    while True:
        packet = MANAGER.get_stream_frame()
        seq = _safe_int(packet.get("sequence"), 0)
        frame_bytes = packet.get("frame_bytes")
        now_mono = time.monotonic()
        should_emit = False
        if isinstance(frame_bytes, (bytes, bytearray)):
            if seq != last_seq:
                should_emit = True
            elif (now_mono - last_emit_mono) >= MJPEG_KEEPALIVE_SEC:
                # Keepalive chunk to prevent clients from hanging indefinitely on quiet sequences.
                should_emit = True
        # Re-narrow here: `should_emit` is only ever set inside the isinstance check
        # above, but type checkers cannot follow that across the boolean.
        if should_emit and isinstance(frame_bytes, (bytes, bytearray)):
            last_seq = seq
            last_emit_mono = now_mono
            yield _mjpeg_chunk(bytes(frame_bytes))
        # Async sleep keeps this generator on the event loop, so shutdown/client
        # disconnect cancels it at the await instead of blocking a threadpool
        # thread that never observes the stop event (the old Ctrl+C hang).
        await asyncio.sleep(FRAME_WAIT_IDLE_SEC)


@app.get("/api/v1/stream/video")
def api_stream_video() -> StreamingResponse:
    return StreamingResponse(
        build_video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.websocket("/api/v1/stream/events/ws")
async def api_stream_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = await MANAGER.event_hub.subscribe()
    try:
        await websocket.send_json(
            MANAGER._event(
                "pipeline_state",
                MANAGER.status(),
            )
        )
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        await MANAGER.event_hub.unsubscribe(queue)


@app.get("/api/v1/detections/latest")
def api_detections_latest() -> dict[str, Any]:
    return MANAGER.latest_detections()


@app.get("/api/v1/behavior/latest")
def api_behavior_latest() -> dict[str, Any]:
    return MANAGER.latest_behavior()


@app.get("/api/v1/memory/stats")
def api_memory_stats() -> dict[str, Any]:
    core = _core()
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    return memory.get_memory_stats()


@app.get("/api/v1/memory/recent")
def api_memory_recent(
    minutes: int = Query(default=5, ge=1, le=24 * 60),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    core = _core()
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    rows = memory.get_recent_snapshots(minutes=minutes, limit=limit)
    return {"items": rows, "count": len(rows)}


@app.get("/api/v1/memory/find/object")
def api_memory_find_object(name: str = Query(..., min_length=1)) -> dict[str, Any]:
    core = _core()
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    row = memory.find_object_last_seen(name)
    return {"item": row, "found": bool(row)}


@app.get("/api/v1/memory/find/person")
def api_memory_find_person(name: str = Query(..., min_length=1)) -> dict[str, Any]:
    core = _core()
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    row = memory.find_person_last_seen(name)
    return {"item": row, "found": bool(row)}


@app.get("/api/v1/memory/search")
def api_memory_search(
    text: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    core = _core()
    memory = core.SceneMemoryManager(base_dir=core.MEMORY_DIR, enable_vectors=False)
    rows = memory.search_similar_scene(text, top_k=top_k)
    return {"items": rows, "count": len(rows)}


@app.get("/api/v1/logs")
def api_logs(
    event_type: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    from_ts: str | None = Query(default=None, alias="from"),
    to_ts: str | None = Query(default=None, alias="to"),
) -> dict[str, Any]:
    rows = _filter_metric_events(
        event_type=event_type,
        limit=limit,
        from_ts=from_ts,
        to_ts=to_ts,
    )
    return {"items": rows, "count": len(rows)}


@app.get("/api/v1/summaries/session")
def api_session_summary(
    minutes: int = Query(default=5, ge=1, le=24 * 60),
    as_json: bool = Query(default=False, alias="json"),
) -> dict[str, Any]:
    core = _core()
    summary = core._build_situation_summary(minutes=minutes)
    rendered = core._render_situation_summary(summary)
    core._append_metric(
        "summary_query",
        {
            "source": "api",
            "minutes": int(minutes),
            "result_lines": rendered.count("\n") + 1,
            "hit": bool(summary.get("top_attention_pairs") or summary.get("snapshots_total")),
        },
    )
    MANAGER._publish_event("summary_result", summary)
    return {
        "summary": summary if as_json else rendered,
        "rendered": rendered,
        "json": summary,
    }


@app.post("/api/v1/chat/query")
def api_chat_query(req: ChatRequest) -> dict[str, Any]:
    core = _core()
    q = (req.message or req.question or "").strip()
    if not q and not (req.confirm_action_id or "").strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    runtime_context = MANAGER.build_chat_runtime_context()
    session_id = CHAT_SESSIONS.get_or_create(req.session_id)

    started = time.perf_counter()
    proposed_action: dict[str, Any] | None = None
    executed_action: dict[str, Any] | None = None
    summary_payload: dict[str, Any] | None = None
    snapshot_payload: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = []
    intent = "open_ended"
    action = "none"
    hit = True
    used_llm = False
    grounded = True
    reply = ""

    if q:
        CHAT_SESSIONS.append_turn(session_id, "user", q)

    confirm_id = (req.confirm_action_id or "").strip()
    if confirm_id:
        confirmed = CHAT_SESSIONS.consume_action(session_id, confirm_id)
        if not confirmed:
            intent = "action_confirm"
            action = "confirm_invalid"
            hit = False
            reply = "Confirmation is invalid or expired. Please request the action again."
        else:
            intent = "action_execute"
            action = str(confirmed.get("type", "none"))
            try:
                executed_action = _execute_confirmed_action(confirmed)
                if isinstance(executed_action.get("summary"), dict):
                    summary_payload = executed_action["summary"]
                if isinstance(executed_action.get("snapshot"), dict):
                    snapshot_payload = executed_action["snapshot"]
                reply = f"Executed action: {_proposal_text(confirmed)}."
                hit = True
                core._append_metric(
                    "chat_action_executed",
                    {
                        "session_id": session_id,
                        "action": action,
                        "confirm_action_id": confirm_id,
                    },
                )
            except HTTPException as exc:
                hit = False
                reply = str(exc.detail)
            except Exception as exc:
                hit = False
                reply = f"Failed to execute action: {exc}"
    else:
        requested_action = _parse_requested_action(q)
        if requested_action is not None:
            confirm_action_id = CHAT_SESSIONS.propose_action(session_id, requested_action, ttl_sec=CHAT_CONFIRM_TTL_SEC)
            proposed_action = {
                **requested_action,
                "confirm_action_id": confirm_action_id,
                "expires_in_sec": CHAT_CONFIRM_TTL_SEC,
            }
            intent = "action_proposal"
            action = str(requested_action.get("type", "none"))
            reply = (
                f"I can {_proposal_text(requested_action)}. "
                f"Please confirm to proceed."
            )
            core._append_metric(
                "chat_action_proposed",
                {
                    "session_id": session_id,
                    "action": action,
                    "confirm_action_id": confirm_action_id,
                },
            )
        else:
            deterministic = _deterministic_chat_response(q, runtime_context)
            if deterministic is not None:
                reply = str(deterministic.get("reply", ""))
                intent = str(deterministic.get("intent", "deterministic"))
                hit = bool(deterministic.get("hit", True))
                if isinstance(deterministic.get("summary"), dict):
                    summary_payload = deterministic["summary"]
                include_citations = bool(deterministic.get("include_citations", True))
                if include_citations:
                    grounding = _build_chat_grounding(q, runtime_context)
                    citations = list(grounding.get("citations", []))
                else:
                    citations = []
            else:
                grounding = _build_chat_grounding(q, runtime_context)
                citations = list(grounding.get("citations", []))
                if not citations:
                    hit = False
                    reply = "Insufficient evidence in current logs/memory."
                else:
                    llm_reply, llm_ok = _query_groq_grounded(
                        question=q,
                        session_history=CHAT_SESSIONS.get_history(session_id),
                        grounding=grounding,
                    )
                    if llm_ok and llm_reply:
                        used_llm = True
                        reply = llm_reply
                    else:
                        hit = False
                        reply = (
                            "Insufficient evidence in current logs/memory, or Groq response unavailable. "
                            "Ask about summary, last-seen, memory stats, or attention."
                        )

    # "Grounded" means the reply is backed by retrieved evidence. A running
    # pipeline or a successfully executed action is not evidence.
    grounded = bool(citations)
    CHAT_SESSIONS.append_turn(session_id, "assistant", reply)
    duration_ms = (time.perf_counter() - started) * 1000.0
    core._append_metric(
        "chat_query",
        {
            "session_id": session_id,
            "question": q,
            "intent": intent,
            "action": action,
            "hit": bool(hit),
            "used_llm": bool(used_llm),
            "grounded": bool(grounded),
            "duration_ms": round(duration_ms, 2),
        },
    )
    if isinstance(summary_payload, dict):
        core._append_metric(
            "summary_query",
            {
                "source": "chat",
                "minutes": _safe_int(summary_payload.get("minutes"), 5),
                "result_lines": str(reply).count("\n") + 1,
                "hit": bool(summary_payload.get("top_attention_pairs") or summary_payload.get("snapshots_total")),
            },
        )
    result = {
        "session_id": session_id,
        "reply": reply,
        "citations": citations,
        "proposed_action": proposed_action,
        "executed_action": executed_action,
        "grounded": bool(grounded),
        "answer": reply,
        "intent": intent,
        "action": action,
        "hit": bool(hit),
        "used_llm": bool(used_llm),
        "summary": summary_payload,
        "snapshot": snapshot_payload,
    }
    _queue_chat_summary_events(result)
    return result


@app.post("/api/v1/enroll/start")
def api_enroll_start(req: EnrollStartRequest) -> dict[str, Any]:
    return MANAGER.start_enroll(req)


@app.get("/api/v1/enroll/status")
def api_enroll_status() -> dict[str, Any]:
    status = MANAGER.status().get("enroll", {})
    return status


@app.post("/api/v1/enroll/stop")
def api_enroll_stop() -> dict[str, Any]:
    if MANAGER.status().get("mode") != "enroll":
        return {"status": "idle"}
    return MANAGER.stop()


@app.get("/api/v1/events/recent")
def api_events_recent(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    rows = MANAGER.recent_events(limit=limit)
    return {"items": rows, "count": len(rows)}


# Legacy aliases
@app.post("/start")
def legacy_start() -> dict[str, Any]:
    return api_monitor_start(MonitorStartRequest())


@app.post("/stop")
def legacy_stop() -> dict[str, Any]:
    return api_monitor_stop()


@app.get("/status")
def legacy_status() -> dict[str, Any]:
    status = MANAGER.status()
    return {
        "isMonitoring": status.get("running") and status.get("mode") == "monitor",
        **status,
    }


@app.get("/video_feed")
def legacy_video_feed() -> StreamingResponse:
    return api_stream_video()


@app.get("/logs")
def legacy_logs(limit: int = Query(default=500, ge=1, le=5000)) -> list[dict[str, Any]]:
    return _filter_metric_events(
        event_type=None,
        limit=limit,
        from_ts=None,
        to_ts=None,
    )


@app.post("/enroll")
def legacy_enroll(req: EnrollStartRequest) -> dict[str, Any]:
    return api_enroll_start(req)


@app.get("/enroll/status")
def legacy_enroll_status() -> dict[str, Any]:
    return api_enroll_status()


if __name__ == "__main__":
    import uvicorn

    # Bound the graceful-shutdown wait: an open MJPEG/WS stream otherwise keeps
    # uvicorn in "Waiting for connection to close" for a very long time.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        timeout_graceful_shutdown=int(os.getenv("UVICORN_GRACEFUL_SHUTDOWN_SEC", "5")),
    )
