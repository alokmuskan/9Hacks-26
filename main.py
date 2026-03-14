from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from object_detection import DualYoloDetector, find_latest_custom_model
from scene_memory import SceneMemoryManager

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_SOURCE = int(os.getenv("AI_STUDIO_CAM_CAMERA_INDEX", "42"))
DB_PATH = Path("face_db.npz")
METRICS_LOG_PATH = Path("metrics_log.jsonl")
REPORT_TXT_PATH = Path("report.txt")
UNKNOWN_INCIDENTS_DIR = Path("unknown_incidents")
MEMORY_DIR = Path("memory")
CUSTOM_MODEL_POINTER_PATH = Path("custom_model_path.txt")
DEFAULT_GENERAL_MODEL = os.getenv(
    "AI_STUDIO_GENERAL_YOLO_MODEL",
    ".references/AI-Studio-Cam-(On-Hold)/models/yolov8n.pt",
)

ENROLL_SAMPLES = 25
ENROLL_CAPTURE_INTERVAL_SEC = 0.25
MATCH_THRESHOLD = 0.40
DET_SIZE = 320
INFER_MAX_SIDE = 640
UNKNOWN_LABEL = "Unknown"

EVENTS_TIMELINE_CAP = 500
UNKNOWN_ALERT_COOLDOWN_SEC = 3.0

WHITE = (255, 255, 255)
GREEN = (0, 210, 80)
AMBER = (0, 140, 255)
TEAL = (200, 220, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)


# ── Database ──────────────────────────────────────────────────────────────────
@dataclass
class FaceDB:
    names: list[str]
    centroids: np.ndarray
    counts: np.ndarray

    @classmethod
    def empty(cls) -> "FaceDB":
        return cls([], np.empty((0, 0), np.float32), np.empty((0,), np.int32))

    @classmethod
    def load(cls) -> "FaceDB":
        if not DB_PATH.exists():
            return cls.empty()
        d = np.load(DB_PATH, allow_pickle=False)
        return cls(
            d["names"].astype(str).tolist(),
            d["centroids"].astype(np.float32),
            d["counts"].astype(np.int32),
        )

    def save(self) -> None:
        np.savez_compressed(
            DB_PATH,
            names=np.array(self.names),
            centroids=self.centroids.astype(np.float32),
            counts=self.counts.astype(np.int32),
        )

    def upsert(self, name: str, embeddings: list[np.ndarray]) -> None:
        stacked = np.stack(embeddings).astype(np.float32)
        n_new = len(stacked)
        centroid = _l2(np.mean(stacked, axis=0))
        if name in self.names:
            i = self.names.index(name)
            n_old = int(self.counts[i])
            self.centroids[i] = _l2(self.centroids[i] * n_old + centroid * n_new)
            self.counts[i] = n_old + n_new
        else:
            self.names.append(name)
            self.centroids = (
                np.vstack([self.centroids, centroid[None]])
                if len(self.names) > 1
                else centroid[None]
            )
            self.counts = np.append(self.counts, n_new).astype(np.int32)


# ── Core helpers ──────────────────────────────────────────────────────────────
def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= 1e-12 else v / n


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now_utc()).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _local_hms(iso_ts: str | None) -> str:
    dt = _parse_iso(iso_ts)
    if dt is None:
        return "--:--:--"
    return dt.astimezone().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    s = max(int(seconds), 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    return f"{m}m {sec}s"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _session_id(prefix: str) -> str:
    # Correlation-friendly ID shown in dashboards/logs.
    return _now_utc().astimezone().strftime("%Y%m%d-%H%M%S")


def _timeline_bar(count: int, max_count: int, width: int = 12) -> str:
    if count <= 0 or max_count <= 0:
        return ""
    n = max(1, int(round((count / max_count) * width)))
    return "█" * n


def _save_unknown_snapshot(
    frame: np.ndarray, unknown_bboxes: list[np.ndarray], ts_utc: datetime
) -> str:
    UNKNOWN_INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_local = ts_utc.astimezone()
    base = ts_local.strftime("unknown_%Y-%m-%d_%H-%M-%S")

    candidate = UNKNOWN_INCIDENTS_DIR / f"{base}.jpg"
    suffix = 1
    while candidate.exists():
        candidate = UNKNOWN_INCIDENTS_DIR / f"{base}_{suffix:02d}.jpg"
        suffix += 1

    snap = frame.copy()
    for bbox in unknown_bboxes:
        _bracket_box(snap, bbox, AMBER, thickness=2)
    cv2.imwrite(str(candidate), snap)

    return str(candidate)


def _resolve_general_model_path(path: str | None) -> str:
    if path:
        return path

    configured = Path(DEFAULT_GENERAL_MODEL)
    if configured.exists():
        return str(configured)
    return "yolov8n.pt"


def _load_default_custom_model_path() -> str | None:
    if CUSTOM_MODEL_POINTER_PATH.exists():
        raw = CUSTOM_MODEL_POINTER_PATH.read_text(encoding="utf-8").strip()
        if raw and Path(raw).exists():
            return raw

    latest = find_latest_custom_model()
    if latest and Path(latest).exists():
        return latest
    return None


def _save_default_custom_model_path(path: str) -> None:
    CUSTOM_MODEL_POINTER_PATH.write_text(str(Path(path)), encoding="utf-8")


def _print_runtime_help() -> None:
    print("\nRuntime controls:")
    print("  q  quit")
    print("  g  toggle general YOLO")
    print("  o  toggle custom YOLO")
    print("  t  manual memory snapshot")
    print("  m  memory statistics")
    print("  r  recent snapshots (5 minutes)")
    print("  f  find when object was last seen")
    print("  h  print this help")


def _make_face_analysis(model: str, providers: list[str]) -> FaceAnalysis:
    try:
        return FaceAnalysis(
            name=model,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
    except TypeError:
        return FaceAnalysis(name=model, providers=providers)


def _repair_insightface_model_layout(model: str) -> bool:
    """
    Repair model packs extracted as ~/.insightface/models/<model>/<model>/*.onnx.
    InsightFace expects ONNX files directly in ~/.insightface/models/<model>.
    """
    model_dir = Path("~/.insightface/models").expanduser() / model
    nested_dir = model_dir / model

    if not nested_dir.exists() or not nested_dir.is_dir():
        return False
    if any(model_dir.glob("*.onnx")):
        return False
    if not any(nested_dir.glob("*.onnx")):
        return False

    moved_any = False
    for item in nested_dir.iterdir():
        target = model_dir / item.name
        if target.exists():
            continue
        item.replace(target)
        moved_any = True

    try:
        nested_dir.rmdir()
    except OSError:
        pass

    if moved_any:
        print(f"Repaired InsightFace model layout for '{model}' in {model_dir}")
    return moved_any


def _build_app(model: str) -> FaceAnalysis:
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    providers = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ] or ["CPUExecutionProvider"]
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

    _repair_insightface_model_layout(model)
    try:
        app = _make_face_analysis(model, providers)
    except AssertionError as exc:
        repaired = _repair_insightface_model_layout(model)
        if repaired:
            app = _make_face_analysis(model, providers)
        else:
            model_dir = Path("~/.insightface/models").expanduser() / model
            raise RuntimeError(
                f"InsightFace model '{model}' is missing detection ONNX files in {model_dir}. "
                "Delete that model directory and rerun to re-download cleanly."
            ) from exc

    app.prepare(ctx_id=ctx_id, det_size=(DET_SIZE, DET_SIZE))
    print(f"Model: {model}  |  Providers: {providers}")
    return app


def _open_camera() -> cv2.VideoCapture:
    def camera_readable(cap: cv2.VideoCapture, warmup_reads: int = 10) -> bool:
        if not cap or not cap.isOpened():
            return False
        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
            time.sleep(0.03)
        return False

    candidates: list[int] = [CAMERA_SOURCE]
    if sys.platform.startswith("linux"):
        for path in Path("/dev").glob("video*"):
            suffix = path.name.replace("video", "", 1)
            if suffix.isdigit():
                candidates.append(int(suffix))
    candidates = list(dict.fromkeys(candidates))

    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]
    tried: list[str] = []

    for index in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                tried.append(f"{index}@{backend}:open_failed")
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if camera_readable(cap):
                return cap

            cap.release()
            tried.append(f"{index}@{backend}:no_frames")

    tried_msg = ", ".join(tried[:18])
    if len(tried) > 18:
        tried_msg += ", ..."
    raise RuntimeError(f"Failed to open readable camera stream. Tried: {tried_msg}")


def _detect(app: FaceAnalysis, frame: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    h, w = frame.shape[:2]
    scale = INFER_MAX_SIDE / max(h, w)
    inp = (
        cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
    )
    inv = 1.0 / scale if scale < 1.0 else 1.0
    return [
        (
            np.asarray(f.bbox, np.float32) * inv,
            _l2(np.asarray(f.normed_embedding, np.float32)),
            float(getattr(f, "det_score", 1.0)),
        )
        for f in app.get(inp)
    ]


def _best_face(faces: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, float] | None:
    return max(
        faces,
        key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]),
        default=None,
    )


def _match(emb: np.ndarray, db: FaceDB) -> tuple[str, float]:
    if not db.names:
        return UNKNOWN_LABEL, 0.0
    scores = db.centroids @ emb
    best_idx = int(np.argmax(scores))
    best = float(scores[best_idx])
    return (db.names[best_idx], best) if best >= MATCH_THRESHOLD else (UNKNOWN_LABEL, best)


def _append_metric(event_type: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp_utc": _iso(),
        "event_type": event_type,
        **payload,
    }
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_metric_events() -> list[dict[str, Any]]:
    if not METRICS_LOG_PATH.exists():
        return []
    events: list[dict[str, Any]] = []
    with METRICS_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# ── UI helpers ────────────────────────────────────────────────────────────────
def _bracket_box(frame: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    arm = max(12, int((x2 - x1) * 0.18))
    for pts in [
        ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
        ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
        ((x2, y2 - arm), (x2, y2), (x2 - arm, y2)),
        ((x1 + arm, y2), (x1, y2), (x1, y2 - arm)),
    ]:
        cv2.polylines(frame, [np.array(pts)], False, color, thickness, cv2.LINE_AA)


def _label_tag(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 5
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad - bl),
        (x + tw + pad, y + pad - bl),
        BLACK,
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    h, w = frame.shape[:2]
    lh = 28
    bar_h = len(lines) * lh + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, h - bar_h + 22 + i * lh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )


def _progress_bar(frame: np.ndarray, value: int, total: int) -> None:
    w = frame.shape[1]
    x1, x2, y, bh = 12, w - 12, 18, 10
    fill = int((x2 - x1) * value / max(total, 1))
    cv2.rectangle(frame, (x1, y - bh), (x2, y), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(frame, (x1, y - bh), (x1 + fill, y), GREEN, -1)
    cv2.rectangle(frame, (x1, y - bh), (x2, y), (130, 130, 130), 1)


# ── Commands ──────────────────────────────────────────────────────────────────
def cmd_enroll(name: str, model: str) -> None:
    session_id = _session_id("enroll")
    start_dt = _now_utc()
    started = time.time()

    app, db, cap = _build_app(model), FaceDB.load(), _open_camera()
    cv2.namedWindow(win := f"Enroll — {name}", cv2.WINDOW_NORMAL)

    samples: list[np.ndarray] = []
    frames_total = 0
    frames_dropped = 0
    last_capture_ts = -1e9

    print(f"Enrolling '{name}'. Keep one face visible. Press q to finish early.")

    try:
        while len(samples) < ENROLL_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                frames_dropped += 1
                continue
            frames_total += 1

            faces = _detect(app, frame)
            picked = _best_face(faces)

            if len(faces) > 1:
                status, sc = "Multiple faces — keep only one", AMBER
            elif picked is None:
                status, sc = "No face detected", AMBER
            else:
                bbox, emb, _ = picked
                _bracket_box(frame, bbox, TEAL)
                now = time.time()
                elapsed = now - last_capture_ts
                if elapsed >= ENROLL_CAPTURE_INTERVAL_SEC:
                    samples.append(emb)
                    last_capture_ts = now
                    status, sc = f"Captured {len(samples)} / {ENROLL_SAMPLES}", GREEN
                else:
                    wait_left = max(ENROLL_CAPTURE_INTERVAL_SEC - elapsed, 0.0)
                    status, sc = f"Hold still... {wait_left:.2f}s", TEAL

            _progress_bar(frame, len(samples), ENROLL_SAMPLES)
            _hud(frame, [(f"Enrolling: {name}", WHITE), (status, sc), ("Q  quit early", (160, 160, 160))])
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not samples:
        raise RuntimeError("No samples captured.")

    db.upsert(name, samples)
    db.save()
    total_for_name = int(db.counts[db.names.index(name)])

    end_dt = _now_utc()
    duration_sec = max(time.time() - started, 0.0)
    _append_metric(
        "enroll",
        {
            "schema_version": 2,
            "session_id": session_id,
            "name": name,
            "model": model,
            "camera_source": CAMERA_SOURCE,
            "start_utc": _iso(start_dt),
            "end_utc": _iso(end_dt),
            "duration_sec": round(duration_sec, 3),
            "samples_captured": len(samples),
            "total_samples_for_name": total_for_name,
            "frame_metrics": {
                "frames_total": frames_total,
                "frames_dropped": frames_dropped,
            },
        },
    )

    print(f"Saved '{name}'  +{len(samples)} samples  (total={total_for_name})")


def cmd_recognize(
    model: str,
    general_model: str | None,
    custom_model: str | None,
    disable_general: bool,
    disable_custom: bool,
    snapshot_interval: float,
) -> None:
    db = FaceDB.load()
    if not db.names:
        raise RuntimeError("No enrolled identities. Run:  enroll --name <n>")

    session_id = _session_id("recognize")
    start_dt = _now_utc()
    session_start = time.time()

    general_model_path = _resolve_general_model_path(general_model)
    custom_model_path = custom_model or _load_default_custom_model_path()

    detector = DualYoloDetector(
        general_model_path=general_model_path,
        custom_model_path=custom_model_path,
        enable_general=not disable_general,
        enable_custom=(not disable_custom) and bool(custom_model_path),
    )
    memory = SceneMemoryManager(snapshot_interval_sec=snapshot_interval, base_dir=MEMORY_DIR)

    app, cap = _build_app(model), _open_camera()
    cv2.namedWindow(win := "Recognize", cv2.WINDOW_NORMAL)

    print("Running unified stream: InsightFace + YOLO + memory.")
    _print_runtime_help()
    if custom_model_path:
        print(f"Custom YOLO model: {custom_model_path}")
    else:
        print("Custom YOLO model: not configured")

    t_prev = time.time()
    fps_ema = 0.0
    fps_min = float("inf")
    fps_max = 0.0

    frames_total = 0
    frames_with_faces = 0
    frames_empty = 0
    frames_dropped = 0

    detections_total = 0
    known_detections = 0
    unknown_detections = 0
    peak_simultaneous_faces = 0
    confidence_sum = 0.0
    detection_calls = 0
    detection_latency_sum_ms = 0.0
    detection_latency_min_ms = float("inf")
    detection_latency_max_ms = 0.0
    detection_timeline: dict[str, int] = defaultdict(int)

    object_detections_total = 0
    object_general_detections = 0
    object_custom_detections = 0
    object_conf_sum = 0.0
    object_general_conf_sum = 0.0
    object_custom_conf_sum = 0.0
    object_detection_timeline: dict[str, int] = defaultdict(int)
    object_class_counts_total: Counter[str] = Counter()
    object_class_counts_general: Counter[str] = Counter()
    object_class_counts_custom: Counter[str] = Counter()

    memory_auto_snapshots = 0
    memory_manual_snapshots = 0
    memory_query_counts: Counter[str] = Counter()
    memory_query_hits: Counter[str] = Counter()
    memory_query_misses: Counter[str] = Counter()

    unknown_alert_count = 0
    last_unknown_alert_ts = -1e9

    label_counter: Counter[str] = Counter()
    latest_active_subjects: list[dict[str, Any]] = []
    latest_object_labels: list[str] = []

    people: dict[str, dict[str, Any]] = {}
    visible_prev: set[str] = set()

    events: list[dict[str, Any]] = []
    events_total_count = 0
    detector_error_seen = False

    def add_event(
        event_type: str,
        message: str,
        severity: str = "info",
        extra: dict[str, Any] | None = None,
    ) -> None:
        nonlocal events_total_count
        e = {
            "timestamp_utc": _iso(),
            "type": event_type,
            "severity": severity,
            "message": message,
        }
        if extra:
            e.update(extra)
        events.append(e)
        events_total_count += 1
        if len(events) > EVENTS_TIMELINE_CAP:
            del events[0]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                frames_dropped += 1
                continue

            frames_total += 1
            now_ts = time.time()

            dt = max(now_ts - t_prev, 1e-6)
            inst_fps = 1.0 / dt
            t_prev = now_ts
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
            fps_min = min(fps_min, inst_fps)
            fps_max = max(fps_max, inst_fps)

            detect_t0 = time.perf_counter()
            face_rows = _detect(app, frame)
            latency_ms = (time.perf_counter() - detect_t0) * 1000.0
            detection_calls += 1
            detection_latency_sum_ms += latency_ms
            detection_latency_min_ms = min(detection_latency_min_ms, latency_ms)
            detection_latency_max_ms = max(detection_latency_max_ms, latency_ms)

            try:
                object_rows = detector.detect(frame)
            except Exception as exc:
                object_rows = []
                if not detector_error_seen:
                    detector_error_seen = True
                    add_event("object_detect_error", str(exc), severity="alert")
                    print(f"Object detection error: {exc}")

            face_count = len(face_rows)
            object_count = len(object_rows)
            peak_simultaneous_faces = max(peak_simultaneous_faces, face_count)
            timeline_key = _now_utc().astimezone().strftime("%H:%M:%S")
            detection_timeline[timeline_key] += face_count
            object_detection_timeline[timeline_key] += object_count

            if face_count > 0:
                frames_with_faces += 1
            else:
                frames_empty += 1

            detections_total += face_count
            object_detections_total += object_count

            visible_now: set[str] = set()
            unknown_in_frame = False
            unknown_bboxes: list[np.ndarray] = []
            render_rows: list[tuple[np.ndarray, str, float]] = []
            active_conf: dict[str, float] = {}

            for bbox, emb, _ in face_rows:
                label, score = _match(emb, db)
                render_rows.append((bbox, label, score))
                confidence_sum += score

                if label == UNKNOWN_LABEL:
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

            if unknown_in_frame and (now_ts - last_unknown_alert_ts) >= UNKNOWN_ALERT_COOLDOWN_SEC:
                unknown_alert_count += 1
                last_unknown_alert_ts = now_ts
                snapshot_path = _save_unknown_snapshot(frame, unknown_bboxes, _now_utc())
                add_event(
                    "unknown_alert",
                    "Unknown face detected",
                    severity="alert",
                    extra={
                        "image_path": snapshot_path,
                        "image_name": Path(snapshot_path).name,
                    },
                )

            visible_prev = visible_now
            latest_active_subjects = [
                {"name": n, "confidence": round(c, 4)}
                for n, c in sorted(active_conf.items(), key=lambda kv: kv[1], reverse=True)
            ]

            for bbox, label, score in render_rows:
                color = GREEN if label != UNKNOWN_LABEL else AMBER
                _bracket_box(frame, bbox, color)
                _label_tag(frame, f"{label}  {score:.2f}", int(bbox[0]), int(bbox[1]) - 6, color)

            object_labels_in_frame: set[str] = set()
            for row in object_rows:
                label = str(row.get("label", "object"))
                conf = _safe_float(row.get("confidence"), 0.0)
                bbox = np.asarray(row.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
                source = str(row.get("source", "general"))
                object_labels_in_frame.add(label)
                object_conf_sum += conf
                object_class_counts_total[label] += 1

                if source == "custom":
                    object_custom_detections += 1
                    object_custom_conf_sum += conf
                    object_class_counts_custom[label] += 1
                    color = MAGENTA
                else:
                    object_general_detections += 1
                    object_general_conf_sum += conf
                    object_class_counts_general[label] += 1
                    color = CYAN

                _bracket_box(frame, bbox, color, thickness=2)
                _label_tag(
                    frame,
                    f"{source}:{label} {conf:.2f}",
                    int(bbox[0]),
                    max(int(bbox[1]) - 6, 20),
                    color,
                )

            latest_object_labels = sorted(object_labels_in_frame)

            if memory.should_take_snapshot(now_ts):
                snap = memory.save_snapshot(frame, object_rows, current_time=now_ts, manual=False)
                memory_auto_snapshots += 1
                add_event(
                    "memory_snapshot_auto",
                    "Auto snapshot saved",
                    extra={
                        "image_path": snap.get("snapshot_path"),
                        "image_name": Path(str(snap.get("snapshot", ""))).name,
                    },
                )

            state = detector.get_state()
            _hud(
                frame,
                [
                    (
                        f"Faces:{face_count} Objects:{object_count} FPS:{fps_ema:.1f}",
                        WHITE,
                    ),
                    (
                        f"Known:{known_detections} Unknown:{unknown_detections} "
                        f"G:{'ON' if state['general']['enabled'] else 'OFF'} "
                        f"C:{'ON' if state['custom']['enabled'] else 'OFF'}",
                        (180, 180, 180),
                    ),
                    (
                        f"Snapshots(auto/manual): {memory_auto_snapshots}/{memory_manual_snapshots}",
                        (160, 160, 160),
                    ),
                    ("Q quit | G/O toggle YOLO | T/M/R/F/H", (140, 140, 140)),
                ],
            )
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g"):
                enabled = detector.toggle_general()
                print(f"General YOLO: {'ON' if enabled else 'OFF'}")
                add_event("toggle_general_yolo", f"General YOLO {'enabled' if enabled else 'disabled'}")
            elif key == ord("o"):
                enabled = detector.toggle_custom()
                print(f"Custom YOLO: {'ON' if enabled else 'OFF'}")
                add_event("toggle_custom_yolo", f"Custom YOLO {'enabled' if enabled else 'disabled'}")
            elif key == ord("t"):
                snap = memory.save_snapshot(frame, object_rows, current_time=now_ts, manual=True)
                memory_manual_snapshots += 1
                print(f"Manual snapshot saved: {snap.get('snapshot')}")
                add_event(
                    "memory_snapshot_manual",
                    "Manual snapshot saved",
                    extra={
                        "image_path": snap.get("snapshot_path"),
                        "image_name": Path(str(snap.get("snapshot", ""))).name,
                    },
                )
            elif key == ord("m"):
                stats = memory.get_memory_stats()
                memory_query_counts["stats"] += 1
                memory_query_hits["stats"] += 1
                print(f"Memory stats: {json.dumps(stats, ensure_ascii=True)}")
                add_event("memory_stats", "Memory statistics requested")
            elif key == ord("r"):
                rows = memory.get_recent_snapshots(minutes=5)
                memory_query_counts["recent"] += 1
                if rows:
                    memory_query_hits["recent"] += 1
                    print(f"Recent snapshots (last 5 min): {len(rows)}")
                    for row in rows[-8:]:
                        print(f"  {row.get('timestamp_local')} | {row.get('objects')} | {row.get('snapshot')}")
                else:
                    memory_query_misses["recent"] += 1
                    print("No recent snapshots in the last 5 minutes.")
                add_event("memory_recent", "Recent snapshots requested")
            elif key == ord("f"):
                memory_query_counts["find"] += 1
                query = input("Object to find: ").strip()
                if query:
                    row = memory.find_object_last_seen(query)
                    if row:
                        memory_query_hits["find"] += 1
                        print(
                            f"Last seen '{query}' at {row.get('timestamp_local')} "
                            f"objects={row.get('objects')} snapshot={row.get('snapshot')}"
                        )
                    else:
                        memory_query_misses["find"] += 1
                        print(f"'{query}' not found in memory.")
                    add_event("memory_find", f"Find object query: {query}")
            elif key == ord("h"):
                _print_runtime_help()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        memory.save_all_memory()
        cv2.destroyAllWindows()

    end_dt = _now_utc()
    end_ts = time.time()
    duration_sec = max(end_ts - session_start, 0.0)

    # Close out presence timing for those still visible at end.
    for name in visible_prev:
        info = people.get(name)
        if info and info["present"] and info["last_enter_ts"] is not None:
            info["presence_sec"] += max(end_ts - float(info["last_enter_ts"]), 0.0)
            info["last_enter_ts"] = None
            info["present"] = False

    current_people_visible = len(visible_prev)
    avg_fps = (frames_total / duration_sec) if duration_sec > 0 else 0.0
    faces_per_sec = (detections_total / duration_sec) if duration_sec > 0 else 0.0
    avg_conf = (confidence_sum / detections_total) if detections_total > 0 else 0.0
    avg_detection_latency_ms = (
        detection_latency_sum_ms / detection_calls if detection_calls > 0 else 0.0
    )
    avg_faces_per_frame = (detections_total / frames_total) if frames_total > 0 else 0.0

    recognition_rate = (known_detections / detections_total) if detections_total > 0 else 0.0
    unknown_rate = (unknown_detections / detections_total) if detections_total > 0 else 0.0
    unknown_alert_density_per_min = (
        unknown_alert_count / (duration_sec / 60.0) if duration_sec > 0 else 0.0
    )

    object_avg_conf = (
        object_conf_sum / object_detections_total if object_detections_total > 0 else 0.0
    )
    object_avg_conf_general = (
        object_general_conf_sum / object_general_detections
        if object_general_detections > 0
        else 0.0
    )
    object_avg_conf_custom = (
        object_custom_conf_sum / object_custom_detections
        if object_custom_detections > 0
        else 0.0
    )

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

    memory_stats = memory.get_memory_stats()
    aggregate = {
        "session_id": session_id,
        "frames_total": frames_total,
        "frames_with_faces": frames_with_faces,
        "frames_empty": frames_empty,
        "frames_dropped": frames_dropped,
        "average_faces_per_frame": round(avg_faces_per_frame, 4),
        "detections_total": detections_total,
        "known_detections": known_detections,
        "unknown_detections": unknown_detections,
        "peak_simultaneous_faces": peak_simultaneous_faces,
        "avg_fps": round(avg_fps, 3),
        "moving_avg_fps": round(fps_ema, 3),
        "min_fps": round(0.0 if fps_min == float("inf") else fps_min, 3),
        "max_fps": round(fps_max, 3),
        "faces_per_sec": round(faces_per_sec, 3),
        "avg_detection_latency_ms": round(avg_detection_latency_ms, 2),
        "min_detection_latency_ms": round(
            0.0 if detection_latency_min_ms == float("inf") else detection_latency_min_ms, 2
        ),
        "max_detection_latency_ms": round(detection_latency_max_ms, 2),
        "detection_calls": detection_calls,
        "avg_confidence": round(avg_conf, 4),
        "recognition_rate": round(recognition_rate, 6),
        "unknown_rate": round(unknown_rate, 6),
        "unknown_alert_events": unknown_alert_count,
        "unknown_alert_density_per_min": round(unknown_alert_density_per_min, 3),
        "unique_individuals_seen": len(people_clean),
        "current_people_visible": current_people_visible,
        "active_subjects": latest_active_subjects,
        "active_objects": latest_object_labels,
        "detection_timeline": [
            {"time_local": t, "detections": int(c)}
            for t, c in list(detection_timeline.items())[-20:]
        ],
        "object_detections_total": object_detections_total,
        "object_general_detections": object_general_detections,
        "object_custom_detections": object_custom_detections,
        "object_avg_confidence": round(object_avg_conf, 4),
        "object_avg_confidence_general": round(object_avg_conf_general, 4),
        "object_avg_confidence_custom": round(object_avg_conf_custom, 4),
        "object_class_counts_total": dict(object_class_counts_total),
        "object_class_counts_general": dict(object_class_counts_general),
        "object_class_counts_custom": dict(object_class_counts_custom),
        "object_detection_timeline": [
            {"time_local": t, "detections": int(c)}
            for t, c in list(object_detection_timeline.items())[-20:]
        ],
        "yolo_state_final": detector.get_state(),
        "yolo_model_paths": {
            "general": general_model_path,
            "custom": custom_model_path,
        },
        "memory_snapshots_auto": memory_auto_snapshots,
        "memory_snapshots_manual": memory_manual_snapshots,
        "memory_snapshots_total_session": memory_auto_snapshots + memory_manual_snapshots,
        "memory_snapshot_total_store": _safe_int(memory_stats.get("total_snapshots"), 0),
        "memory_query_counts": dict(memory_query_counts),
        "memory_query_hits": dict(memory_query_hits),
        "memory_query_misses": dict(memory_query_misses),
    }

    _append_metric(
        "recognize_session",
        {
            "schema_version": 3,
            "session_id": session_id,
            "model": model,
            "camera_source": CAMERA_SOURCE,
            "start_utc": _iso(start_dt),
            "end_utc": _iso(end_dt),
            "duration_sec": round(duration_sec, 3),
            "aggregate": aggregate,
            "people": people_clean,
            "label_counts": dict(label_counter),
            "events": events,
            "events_total_count": events_total_count,
        },
    )

    print(
        f"Session summary | frames={frames_total} avg_fps={avg_fps:.2f} "
        f"faces={detections_total} objects={object_detections_total} "
        f"known={known_detections} unknown={unknown_detections}"
    )


def cmd_list() -> None:
    db = FaceDB.load()
    if not db.names:
        print("No enrolled identities found.")
        return
    print(f"Database: {DB_PATH}")
    for name, count in zip(db.names, db.counts):
        print(f"  {name}: {int(count)} samples")


def cmd_train_objects(
    data: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    set_default: bool,
) -> None:
    data_path = Path(data)
    if not data_path.exists():
        raise RuntimeError(f"Dataset YAML does not exist: {data_path}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not available. Install dependency 'ultralytics'."
        ) from exc

    start_dt = _now_utc()
    started = time.time()

    print(f"Training custom YOLO model from: {data_path}")
    print(
        f"Config | base={base_model} epochs={epochs} imgsz={imgsz} batch={batch} "
        f"project={project} name={name}"
    )

    yolo = YOLO(base_model)
    train_result = yolo.train(
        data=str(data_path),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        project=project,
        name=name,
        exist_ok=True,
    )

    save_dir = (
        Path(getattr(train_result, "save_dir"))
        if getattr(train_result, "save_dir", None)
        else Path(getattr(getattr(yolo, "trainer", None), "save_dir", project))
    )
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    resolved_model = best_path if best_path.exists() else last_path
    if not resolved_model.exists():
        raise RuntimeError(f"Training completed but no weights found in {save_dir / 'weights'}")

    if set_default:
        _save_default_custom_model_path(str(resolved_model))
        print(f"Updated default custom model pointer: {CUSTOM_MODEL_POINTER_PATH}")

    end_dt = _now_utc()
    duration_sec = max(time.time() - started, 0.0)
    _append_metric(
        "object_train",
        {
            "schema_version": 1,
            "start_utc": _iso(start_dt),
            "end_utc": _iso(end_dt),
            "duration_sec": round(duration_sec, 3),
            "data_yaml": str(data_path),
            "base_model": base_model,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "project": project,
            "name": name,
            "save_dir": str(save_dir),
            "output_model": str(resolved_model),
            "set_default_model": bool(set_default),
        },
    )

    print(f"Training complete. Model weights: {resolved_model}")


def cmd_memory_stats() -> None:
    memory = SceneMemoryManager(base_dir=MEMORY_DIR)
    stats = memory.get_memory_stats()
    _append_metric(
        "memory_query",
        {
            "query_type": "stats",
            "hit": True,
            "result_count": 1,
        },
    )
    print(json.dumps(stats, indent=2, ensure_ascii=True))


def cmd_memory_recent(minutes: int) -> None:
    memory = SceneMemoryManager(base_dir=MEMORY_DIR)
    rows = memory.get_recent_snapshots(minutes=minutes)
    _append_metric(
        "memory_query",
        {
            "query_type": "recent",
            "minutes": int(minutes),
            "hit": bool(rows),
            "result_count": len(rows),
        },
    )

    if not rows:
        print(f"No snapshots found in the last {minutes} minutes.")
        return

    print(f"Recent snapshots in the last {minutes} minutes: {len(rows)}")
    for row in rows:
        print(
            f"  {row.get('timestamp_local')} | {row.get('objects')} | "
            f"{row.get('snapshot')}"
        )


def cmd_memory_find(object_name: str) -> None:
    memory = SceneMemoryManager(base_dir=MEMORY_DIR)
    row = memory.find_object_last_seen(object_name)
    _append_metric(
        "memory_query",
        {
            "query_type": "find",
            "object": object_name,
            "hit": bool(row),
            "result_count": 1 if row else 0,
        },
    )

    if not row:
        print(f"Object not found: {object_name}")
        return

    print(
        f"Last seen '{object_name}' at {row.get('timestamp_local')} | "
        f"objects={row.get('objects')} | snapshot={row.get('snapshot')}"
    )


def cmd_memory_search(text: str) -> None:
    memory = SceneMemoryManager(base_dir=MEMORY_DIR)
    rows = memory.search_similar_scene(text, top_k=5)
    _append_metric(
        "memory_query",
        {
            "query_type": "search",
            "text": text,
            "hit": bool(rows),
            "result_count": len(rows),
        },
    )

    if not rows:
        print("No matching scenes found.")
        return

    print(f"Search results for '{text}':")
    for row in rows:
        extra = (
            f" distance={_safe_float(row.get('distance'), 0.0):.4f}"
            if row.get("distance") is not None
            else ""
        )
        print(
            f"  {row.get('timestamp_local')} | {row.get('objects')} | "
            f"{row.get('snapshot')}{extra}"
        )


# ── Reporting helpers ─────────────────────────────────────────────────────────
def _compute_risk(unknown_events: int, unknown_rate: float) -> str:
    if unknown_events >= 15 or unknown_rate >= 0.30:
        return "HIGH"
    if unknown_events >= 6 or unknown_rate >= 0.10:
        return "MEDIUM"
    return "LOW"


def _normalize_recognize_event(event: dict[str, Any], idx: int) -> dict[str, Any] | None:
    if event.get("event_type") != "recognize_session":
        return None

    duration_sec = _safe_float(event.get("duration_sec"), 0.0)
    end_utc = event.get("end_utc") or event.get("timestamp_utc")
    start_utc = event.get("start_utc")

    if not start_utc and end_utc:
        dt = _parse_iso(end_utc)
        if dt is not None:
            start_utc = _iso(dt - timedelta(seconds=duration_sec))

    raw_aggregate = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}

    frames_total = _safe_int(raw_aggregate.get("frames_total", event.get("frames", 0)), 0)
    frames_with_faces = _safe_int(
        raw_aggregate.get("frames_with_faces", event.get("frames_with_faces", 0)),
        0,
    )
    detections_total = _safe_int(
        raw_aggregate.get("detections_total", event.get("detections_total", 0)),
        0,
    )
    known_detections = _safe_int(
        raw_aggregate.get("known_detections", event.get("known_detections", 0)),
        0,
    )
    unknown_detections = _safe_int(
        raw_aggregate.get("unknown_detections", event.get("unknown_detections", 0)),
        0,
    )
    avg_fps = _safe_float(raw_aggregate.get("avg_fps", event.get("avg_fps", 0.0)), 0.0)
    avg_conf = _safe_float(
        raw_aggregate.get("avg_confidence", event.get("avg_confidence", 0.0)),
        0.0,
    )
    faces_per_sec = _safe_float(raw_aggregate.get("faces_per_sec"), 0.0)
    if faces_per_sec == 0.0 and duration_sec > 0:
        faces_per_sec = detections_total / duration_sec

    moving_avg_fps = _safe_float(raw_aggregate.get("moving_avg_fps"), avg_fps)
    min_fps = _safe_float(raw_aggregate.get("min_fps"), avg_fps)
    max_fps = _safe_float(raw_aggregate.get("max_fps"), avg_fps)
    frames_empty = _safe_int(raw_aggregate.get("frames_empty"), max(frames_total - frames_with_faces, 0))

    aggregate = {
        "session_id": raw_aggregate.get("session_id") or event.get("session_id") or f"legacy-recognize-{idx}",
        "frames_total": frames_total,
        "frames_with_faces": frames_with_faces,
        "frames_empty": frames_empty,
        "frames_dropped": _safe_int(raw_aggregate.get("frames_dropped", event.get("frames_dropped", 0)), 0),
        "average_faces_per_frame": _safe_float(
            raw_aggregate.get("average_faces_per_frame"),
            (detections_total / frames_total) if frames_total > 0 else 0.0,
        ),
        "detections_total": detections_total,
        "known_detections": known_detections,
        "unknown_detections": unknown_detections,
        "peak_simultaneous_faces": _safe_int(
            raw_aggregate.get("peak_simultaneous_faces", event.get("peak_simultaneous_faces", 0)),
            0,
        ),
        "avg_fps": avg_fps,
        "moving_avg_fps": moving_avg_fps,
        "min_fps": min_fps,
        "max_fps": max_fps,
        "faces_per_sec": faces_per_sec,
        "avg_detection_latency_ms": _safe_float(
            raw_aggregate.get("avg_detection_latency_ms", event.get("avg_detection_latency_ms", 0.0)),
            0.0,
        ),
        "min_detection_latency_ms": _safe_float(
            raw_aggregate.get("min_detection_latency_ms", event.get("min_detection_latency_ms", 0.0)),
            0.0,
        ),
        "max_detection_latency_ms": _safe_float(
            raw_aggregate.get("max_detection_latency_ms", event.get("max_detection_latency_ms", 0.0)),
            0.0,
        ),
        "detection_calls": _safe_int(raw_aggregate.get("detection_calls", event.get("detection_calls", 0)), 0),
        "avg_confidence": avg_conf,
        "recognition_rate": _safe_float(
            raw_aggregate.get("recognition_rate"),
            (known_detections / detections_total) if detections_total > 0 else 0.0,
        ),
        "unknown_rate": _safe_float(
            raw_aggregate.get("unknown_rate"),
            (unknown_detections / detections_total) if detections_total > 0 else 0.0,
        ),
        "unknown_alert_events": _safe_int(
            raw_aggregate.get("unknown_alert_events", event.get("unknown_alert_events", 0)),
            0,
        ),
        "unknown_alert_density_per_min": _safe_float(
            raw_aggregate.get("unknown_alert_density_per_min"),
            0.0,
        ),
        "unique_individuals_seen": _safe_int(
            raw_aggregate.get("unique_individuals_seen"),
            len(event.get("label_counts", {}))
            if isinstance(event.get("label_counts"), dict)
            else 0,
        ),
        "current_people_visible": _safe_int(
            raw_aggregate.get("current_people_visible", event.get("current_people_visible", 0)),
            0,
        ),
        "active_subjects": raw_aggregate.get("active_subjects", event.get("active_subjects", [])),
        "detection_timeline": raw_aggregate.get("detection_timeline", event.get("detection_timeline", [])),
        "active_objects": raw_aggregate.get("active_objects", event.get("active_objects", [])),
        "object_detections_total": _safe_int(
            raw_aggregate.get("object_detections_total", event.get("object_detections_total", 0)),
            0,
        ),
        "object_general_detections": _safe_int(
            raw_aggregate.get("object_general_detections", event.get("object_general_detections", 0)),
            0,
        ),
        "object_custom_detections": _safe_int(
            raw_aggregate.get("object_custom_detections", event.get("object_custom_detections", 0)),
            0,
        ),
        "object_avg_confidence": _safe_float(
            raw_aggregate.get("object_avg_confidence", event.get("object_avg_confidence", 0.0)),
            0.0,
        ),
        "object_avg_confidence_general": _safe_float(
            raw_aggregate.get(
                "object_avg_confidence_general", event.get("object_avg_confidence_general", 0.0)
            ),
            0.0,
        ),
        "object_avg_confidence_custom": _safe_float(
            raw_aggregate.get(
                "object_avg_confidence_custom", event.get("object_avg_confidence_custom", 0.0)
            ),
            0.0,
        ),
        "object_class_counts_total": raw_aggregate.get(
            "object_class_counts_total", event.get("object_class_counts_total", {})
        ),
        "object_class_counts_general": raw_aggregate.get(
            "object_class_counts_general", event.get("object_class_counts_general", {})
        ),
        "object_class_counts_custom": raw_aggregate.get(
            "object_class_counts_custom", event.get("object_class_counts_custom", {})
        ),
        "object_detection_timeline": raw_aggregate.get(
            "object_detection_timeline", event.get("object_detection_timeline", [])
        ),
        "yolo_state_final": raw_aggregate.get("yolo_state_final", event.get("yolo_state_final", {})),
        "yolo_model_paths": raw_aggregate.get("yolo_model_paths", event.get("yolo_model_paths", {})),
        "memory_snapshots_auto": _safe_int(
            raw_aggregate.get("memory_snapshots_auto", event.get("memory_snapshots_auto", 0)),
            0,
        ),
        "memory_snapshots_manual": _safe_int(
            raw_aggregate.get("memory_snapshots_manual", event.get("memory_snapshots_manual", 0)),
            0,
        ),
        "memory_snapshots_total_session": _safe_int(
            raw_aggregate.get(
                "memory_snapshots_total_session", event.get("memory_snapshots_total_session", 0)
            ),
            0,
        ),
        "memory_snapshot_total_store": _safe_int(
            raw_aggregate.get("memory_snapshot_total_store", event.get("memory_snapshot_total_store", 0)),
            0,
        ),
        "memory_query_counts": raw_aggregate.get("memory_query_counts", event.get("memory_query_counts", {})),
        "memory_query_hits": raw_aggregate.get("memory_query_hits", event.get("memory_query_hits", {})),
        "memory_query_misses": raw_aggregate.get("memory_query_misses", event.get("memory_query_misses", {})),
    }

    people = event.get("people") if isinstance(event.get("people"), dict) else {}
    if not people and isinstance(event.get("label_counts"), dict):
        for label, cnt in event["label_counts"].items():
            people[str(label)] = {
                "detections": _safe_int(cnt),
                "avg_confidence": 0.0,
                "first_seen_utc": None,
                "last_seen_utc": None,
                "presence_sec": 0.0,
            }

    label_counts = event.get("label_counts") if isinstance(event.get("label_counts"), dict) else {}
    events = event.get("events") if isinstance(event.get("events"), list) else []

    return {
        "session_id": event.get("session_id") or f"legacy-recognize-{idx}",
        "model": event.get("model"),
        "camera_source": event.get("camera_source"),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_sec": duration_sec,
        "aggregate": aggregate,
        "people": people,
        "label_counts": label_counts,
        "events": events,
        "events_total_count": _safe_int(event.get("events_total_count"), len(events)),
        "raw_timestamp_utc": event.get("timestamp_utc"),
    }


def _normalize_enroll_event(event: dict[str, Any], idx: int) -> dict[str, Any] | None:
    if event.get("event_type") != "enroll":
        return None

    duration_sec = _safe_float(event.get("duration_sec"), 0.0)
    end_utc = event.get("end_utc") or event.get("timestamp_utc")
    start_utc = event.get("start_utc")

    if not start_utc and end_utc:
        dt = _parse_iso(end_utc)
        if dt is not None:
            start_utc = _iso(dt - timedelta(seconds=duration_sec))

    return {
        "session_id": event.get("session_id") or f"legacy-enroll-{idx}",
        "name": event.get("name"),
        "model": event.get("model"),
        "camera_source": event.get("camera_source"),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_sec": duration_sec,
        "samples_captured": _safe_int(event.get("samples_captured"), 0),
        "total_samples_for_name": _safe_int(event.get("total_samples_for_name"), 0),
        "frame_metrics": event.get("frame_metrics") if isinstance(event.get("frame_metrics"), dict) else {},
        "raw_timestamp_utc": event.get("timestamp_utc"),
    }


def _aggregate_report_data(events: list[dict[str, Any]], db: FaceDB) -> dict[str, Any]:
    enroll_events: list[dict[str, Any]] = []
    recognize_events: list[dict[str, Any]] = []

    for i, event in enumerate(events):
        e = _normalize_enroll_event(event, i)
        if e is not None:
            enroll_events.append(e)
            continue
        r = _normalize_recognize_event(event, i)
        if r is not None:
            recognize_events.append(r)

    recognize_events.sort(key=lambda r: r.get("end_utc") or r.get("raw_timestamp_utc") or "")
    latest = recognize_events[-1] if recognize_events else None

    hist_frames = 0
    hist_frames_with_faces = 0
    hist_frames_empty = 0
    hist_frames_dropped = 0
    hist_detections = 0
    hist_known = 0
    hist_unknown = 0
    hist_unknown_alerts = 0
    hist_runtime = 0.0
    hist_peak_faces = 0
    hist_conf_weighted = 0.0
    hist_conf_weight = 0
    hist_event_count = 0
    hist_object_detections = 0
    hist_object_general = 0
    hist_object_custom = 0
    hist_object_conf_weighted = 0.0
    hist_object_conf_weight = 0
    hist_memory_auto = 0
    hist_memory_manual = 0
    hist_memory_queries_total = 0

    label_counter: Counter[str] = Counter()
    object_label_counter: Counter[str] = Counter()
    people_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "detections": 0,
            "presence_sec": 0.0,
            "conf_sum": 0.0,
            "conf_weight": 0,
            "first_seen_utc": None,
            "last_seen_utc": None,
        }
    )

    recent_events: list[dict[str, Any]] = []
    unknown_incidents: list[dict[str, Any]] = []

    for r in recognize_events:
        a = r["aggregate"]
        hist_frames += _safe_int(a.get("frames_total"), 0)
        hist_frames_with_faces += _safe_int(a.get("frames_with_faces"), 0)
        hist_frames_empty += _safe_int(a.get("frames_empty"), 0)
        hist_frames_dropped += _safe_int(a.get("frames_dropped"), 0)
        hist_detections += _safe_int(a.get("detections_total"), 0)
        hist_known += _safe_int(a.get("known_detections"), 0)
        hist_unknown += _safe_int(a.get("unknown_detections"), 0)
        hist_unknown_alerts += _safe_int(a.get("unknown_alert_events"), 0)
        hist_runtime += _safe_float(r.get("duration_sec"), 0.0)
        hist_peak_faces = max(hist_peak_faces, _safe_int(a.get("peak_simultaneous_faces"), 0))

        avg_conf = _safe_float(a.get("avg_confidence"), 0.0)
        det_weight = _safe_int(a.get("detections_total"), 0)
        hist_conf_weighted += avg_conf * det_weight
        hist_conf_weight += det_weight
        hist_object_detections += _safe_int(a.get("object_detections_total"), 0)
        hist_object_general += _safe_int(a.get("object_general_detections"), 0)
        hist_object_custom += _safe_int(a.get("object_custom_detections"), 0)
        object_avg_conf = _safe_float(a.get("object_avg_confidence"), 0.0)
        object_weight = _safe_int(a.get("object_detections_total"), 0)
        hist_object_conf_weighted += object_avg_conf * object_weight
        hist_object_conf_weight += object_weight
        hist_memory_auto += _safe_int(a.get("memory_snapshots_auto"), 0)
        hist_memory_manual += _safe_int(a.get("memory_snapshots_manual"), 0)
        query_counts = a.get("memory_query_counts", {})
        if isinstance(query_counts, dict):
            hist_memory_queries_total += sum(_safe_int(v) for v in query_counts.values())
        class_counts = a.get("object_class_counts_total", {})
        if isinstance(class_counts, dict):
            object_label_counter.update({str(k): _safe_int(v) for k, v in class_counts.items()})

        hist_event_count += _safe_int(r.get("events_total_count"), len(r.get("events", [])))
        label_counter.update(r.get("label_counts", {}))

        for name, p in r.get("people", {}).items():
            d = _safe_int(p.get("detections"), 0)
            ps = _safe_float(p.get("presence_sec"), 0.0)
            pc = _safe_float(p.get("avg_confidence"), 0.0)
            first_seen = p.get("first_seen_utc")
            last_seen = p.get("last_seen_utc")

            acc = people_acc[name]
            acc["detections"] += d
            acc["presence_sec"] += ps
            acc["conf_sum"] += pc * d
            acc["conf_weight"] += d

            if first_seen:
                cur_first = _parse_iso(acc["first_seen_utc"]) if acc["first_seen_utc"] else None
                new_first = _parse_iso(first_seen)
                if new_first and (cur_first is None or new_first < cur_first):
                    acc["first_seen_utc"] = first_seen

            if last_seen:
                cur_last = _parse_iso(acc["last_seen_utc"]) if acc["last_seen_utc"] else None
                new_last = _parse_iso(last_seen)
                if new_last and (cur_last is None or new_last > cur_last):
                    acc["last_seen_utc"] = last_seen

    if latest:
        recent_events = latest.get("events", [])[-10:]
        for e in latest.get("events", []):
            if e.get("type") != "unknown_alert":
                continue
            unknown_incidents.append(
                {
                    "timestamp_utc": e.get("timestamp_utc"),
                    "time_local": _local_hms(e.get("timestamp_utc")),
                    "image_path": e.get("image_path"),
                    "image_name": e.get("image_name")
                    or (Path(e["image_path"]).name if e.get("image_path") else None),
                    "message": e.get("message", "Unknown face detected"),
                }
            )

    member_activity: list[dict[str, Any]] = []
    for name, acc in people_acc.items():
        cw = max(_safe_int(acc["conf_weight"]), 0)
        member_activity.append(
            {
                "name": name,
                "detections": _safe_int(acc["detections"]),
                "presence_sec": round(_safe_float(acc["presence_sec"]), 3),
                "avg_confidence": (acc["conf_sum"] / cw) if cw > 0 else 0.0,
                "first_seen_utc": acc["first_seen_utc"],
                "last_seen_utc": acc["last_seen_utc"],
            }
        )

    member_activity.sort(key=lambda x: x["detections"], reverse=True)

    hist_avg_fps = (hist_frames / hist_runtime) if hist_runtime > 0 else 0.0
    hist_recognition_rate = (hist_known / hist_detections) if hist_detections > 0 else 0.0
    hist_unknown_rate = (hist_unknown / hist_detections) if hist_detections > 0 else 0.0
    hist_faces_per_sec = (hist_detections / hist_runtime) if hist_runtime > 0 else 0.0
    hist_alert_density = (hist_unknown_alerts / (hist_runtime / 60.0)) if hist_runtime > 0 else 0.0
    hist_avg_conf = (hist_conf_weighted / hist_conf_weight) if hist_conf_weight > 0 else 0.0
    hist_object_avg_conf = (
        hist_object_conf_weighted / hist_object_conf_weight if hist_object_conf_weight > 0 else 0.0
    )

    latest_agg = latest["aggregate"] if latest else {}
    latest_unknown_events = _safe_int(latest_agg.get("unknown_alert_events"), 0)
    if latest_unknown_events == 0 and unknown_incidents:
        latest_unknown_events = len(unknown_incidents)
    latest_unknown_rate = _safe_float(latest_agg.get("unknown_rate"), 0.0)
    risk_level = _compute_risk(latest_unknown_events, latest_unknown_rate)

    summary = {
        "generated_utc": _iso(),
        "db": {
            "path": str(DB_PATH),
            "identity_count": len(db.names),
            "total_samples": int(np.sum(db.counts)) if len(db.counts) else 0,
            "identities": [
                {"name": name, "samples": int(count)}
                for name, count in zip(db.names, db.counts)
            ],
        },
        "metrics": {
            "path": str(METRICS_LOG_PATH),
            "events_total": len(events),
            "enroll_events": len(enroll_events),
            "recognize_sessions": len(recognize_events),
        },
        "latest_session": latest,
        "historical": {
            "sessions": len(recognize_events),
            "runtime_sec": hist_runtime,
            "frames_total": hist_frames,
            "frames_with_faces": hist_frames_with_faces,
            "frames_empty": hist_frames_empty,
            "frames_dropped": hist_frames_dropped,
            "detections_total": hist_detections,
            "known_detections": hist_known,
            "unknown_detections": hist_unknown,
            "unknown_alert_events": hist_unknown_alerts,
            "recognition_rate": hist_recognition_rate,
            "unknown_rate": hist_unknown_rate,
            "avg_fps": hist_avg_fps,
            "faces_per_sec": hist_faces_per_sec,
            "peak_simultaneous_faces": hist_peak_faces,
            "avg_confidence": hist_avg_conf,
            "alert_density_per_min": hist_alert_density,
            "event_timeline_total_count": hist_event_count,
            "top_labels": label_counter.most_common(15),
            "object_detections_total": hist_object_detections,
            "object_general_detections": hist_object_general,
            "object_custom_detections": hist_object_custom,
            "object_avg_confidence": hist_object_avg_conf,
            "top_object_labels": object_label_counter.most_common(15),
            "memory_auto_snapshots": hist_memory_auto,
            "memory_manual_snapshots": hist_memory_manual,
            "memory_total_snapshots": hist_memory_auto + hist_memory_manual,
            "memory_queries_total": hist_memory_queries_total,
            "unique_individuals_seen": len(member_activity),
        },
        "member_activity": member_activity,
        "recent_security_events": recent_events,
        "unknown_incident_log": unknown_incidents,
        "security_risk": {
            "unknown_face_frequency": latest_unknown_events,
            "unknown_rate": latest_unknown_rate,
            "risk_level": risk_level,
            "rule": {
                "HIGH": "unknown_events >= 15 OR unknown_rate >= 0.30",
                "MEDIUM": "unknown_events >= 6 OR unknown_rate >= 0.10",
                "LOW": "otherwise",
            },
        },
    }

    return summary


def _build_ascii_dashboard(summary: dict[str, Any]) -> str:
    latest = summary.get("latest_session") or {}
    latest_agg = latest.get("aggregate") if isinstance(latest, dict) else {}
    hist = summary.get("historical", {})
    members = summary.get("member_activity", [])
    events = summary.get("recent_security_events", [])
    incidents = summary.get("unknown_incident_log", [])
    risk = summary.get("security_risk", {})

    session_id = (
        (latest.get("session_id") if isinstance(latest, dict) else None)
        or (latest_agg.get("session_id") if isinstance(latest_agg, dict) else None)
        or "N/A"
    )
    if isinstance(session_id, str) and (
        session_id.startswith("legacy-") or session_id.startswith("recognize-")
    ):
        fallback_dt = _parse_iso(
            (latest.get("start_utc") if isinstance(latest, dict) else None)
            or (latest.get("end_utc") if isinstance(latest, dict) else None)
        )
        if fallback_dt is not None:
            session_id = fallback_dt.astimezone().strftime("%Y%m%d-%H%M%S")
    camera_status = "Active" if latest else "No recent session"
    processing_speed = _safe_float(latest_agg.get("avg_fps"), 0.0)
    total_faces_detected = _safe_int(latest_agg.get("detections_total"), 0)
    total_objects_detected = _safe_int(latest_agg.get("object_detections_total"), 0)
    objects_general = _safe_int(latest_agg.get("object_general_detections"), 0)
    objects_custom = _safe_int(latest_agg.get("object_custom_detections"), 0)
    object_avg_conf = _safe_float(latest_agg.get("object_avg_confidence"), 0.0)
    avg_latency_ms = _safe_float(latest_agg.get("avg_detection_latency_ms"), 0.0)

    recognized_members = _safe_int(latest_agg.get("known_detections"), 0)
    unknown_alerts = _safe_int(latest_agg.get("unknown_alert_events"), 0)
    unique_seen = _safe_int(latest_agg.get("unique_individuals_seen"), 0)
    current_visible = _safe_int(latest_agg.get("current_people_visible"), 0)
    memory_snapshots_auto = _safe_int(latest_agg.get("memory_snapshots_auto"), 0)
    memory_snapshots_manual = _safe_int(latest_agg.get("memory_snapshots_manual"), 0)
    memory_query_counts = (
        latest_agg.get("memory_query_counts", {})
        if isinstance(latest_agg.get("memory_query_counts"), dict)
        else {}
    )
    memory_queries_total = sum(_safe_int(v) for v in memory_query_counts.values())
    active_subjects = (
        latest_agg.get("active_subjects", [])
        if isinstance(latest_agg.get("active_subjects"), list)
        else []
    )
    active_objects = (
        latest_agg.get("active_objects", [])
        if isinstance(latest_agg.get("active_objects"), list)
        else []
    )
    detection_timeline = (
        latest_agg.get("detection_timeline", [])
        if isinstance(latest_agg.get("detection_timeline"), list)
        else []
    )

    lines = [
        "╔════════════════════════════════════════════════════╗",
        "║        FACE + OBJECT MONITORING CONSOLE           ║",
        "╚════════════════════════════════════════════════════╝",
        "",
        "SYSTEM STATUS",
        "--------------------------------------------------------",
        f"Session ID               : {session_id}",
        f"Camera Status           : {camera_status}",
        f"Processing Speed        : {processing_speed:.2f} FPS",
        f"Average Detection Latency: {avg_latency_ms:.2f} ms",
        f"Total Faces Detected    : {total_faces_detected}",
        f"Total Objects Detected  : {total_objects_detected}",
        "",
        "--------------------------------------------------------",
        "ACTIVITY OVERVIEW",
        "--------------------------------------------------------",
        f"Recognized Members      : {recognized_members} detections",
        f"Unknown Face Alerts     : {unknown_alerts} alerts",
        f"Objects (Gen/Custom)    : {objects_general}/{objects_custom}",
        f"Object Avg Confidence   : {object_avg_conf:.4f}",
        f"Unique Individuals Seen : {unique_seen} members",
        f"Current People Visible  : {current_visible}",
        f"Memory Snapshots A/M    : {memory_snapshots_auto}/{memory_snapshots_manual}",
        f"Memory Queries (session): {memory_queries_total}",
        "",
        "--------------------------------------------------------",
        "SESSION STATISTICS",
        "--------------------------------------------------------",
        f"Average Faces per Frame : {_safe_float(latest_agg.get('average_faces_per_frame'), 0.0):.2f}",
        f"Frames with Faces       : {_safe_int(latest_agg.get('frames_with_faces'), 0)}",
        f"Empty Frames            : {_safe_int(latest_agg.get('frames_empty'), 0)}",
        f"Dropped Frames          : {_safe_int(latest_agg.get('frames_dropped'), 0)}",
        "",
        "--------------------------------------------------------",
        "ACTIVE SUBJECTS",
        "--------------------------------------------------------",
    ]

    if active_subjects:
        for subj in active_subjects[:10]:
            name = str(subj.get("name", "Unknown"))
            conf = _safe_float(subj.get("confidence"), 0.0)
            lines.append(f"{name} (confidence {conf:.2f})")
    else:
        lines.append("No known members currently visible")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "ACTIVE OBJECTS",
            "--------------------------------------------------------",
        ]
    )

    if active_objects:
        lines.append(", ".join(str(x) for x in active_objects[:16]))
    else:
        lines.append("No active objects")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "DETECTION TIMELINE",
            "--------------------------------------------------------",
        ]
    )

    if detection_timeline:
        tail = detection_timeline[-10:]
        max_count = max(_safe_int(point.get("detections"), 0) for point in tail) if tail else 0
        for point in tail:
            time_local = str(point.get("time_local", "--:--:--"))
            count = _safe_int(point.get("detections"), 0)
            bar = _timeline_bar(count, max_count, width=12)
            lines.append(f"{time_local}  {bar}")
    else:
        lines.append("--:--:--")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "MEMBER ACTIVITY SUMMARY",
            "--------------------------------------------------------",
            "Name      | Detections | Presence Time | Last Seen",
            "--------------------------------------------------------",
        ]
    )

    if members:
        for m in members[:12]:
            lines.append(
                f"{m['name'][:9]:<9} | {int(m['detections']):>10} | "
                f"{_fmt_duration(_safe_float(m['presence_sec'])):>12} | {_local_hms(m.get('last_seen_utc'))}"
            )
    else:
        lines.append("No member activity captured yet")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "RECENT SECURITY EVENTS",
            "--------------------------------------------------------",
            "Time       | Event",
            "--------------------------------------------------------",
        ]
    )

    if events:
        for e in events[-10:]:
            lines.append(f"{_local_hms(e.get('timestamp_utc')):<10} | {e.get('message', 'Unknown event')}")
    else:
        lines.append("--:--:--   | No recent events")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "UNKNOWN INCIDENT LOG",
            "--------------------------------------------------------",
            "Preview  | Time     | Image",
            "--------------------------------------------------------",
        ]
    )

    if incidents:
        for incident in incidents[-10:]:
            image_name = incident.get("image_name") or "unknown.jpg"
            lines.append(
                f"{'[image]':<8} | {incident.get('time_local', '--:--:--'):<8} | {image_name}"
            )
    else:
        lines.append(f"{'[image]':<8} | {'--:--:--':<8} | --")

    lines.extend(
        [
            "",
            "--------------------------------------------------------",
            "SECURITY RISK INDICATOR",
            "--------------------------------------------------------",
            f"Unknown Face Frequency  : {_safe_int(risk.get('unknown_face_frequency'), 0)} events",
            f"Risk Level              : {risk.get('risk_level', 'LOW')}",
            "",
            "--------------------------------------------------------",
            "SYSTEM LOG SUMMARY",
            "--------------------------------------------------------",
            f"Monitoring Duration     : {_fmt_duration(_safe_float(hist.get('runtime_sec'), 0.0))}",
            f"Total Frames Processed  : {_safe_int(hist.get('frames_total'), 0):,}",
            f"Face Recognition Rate   : {_safe_float(hist.get('recognition_rate'), 0.0) * 100:.2f}%",
            f"Unknown Rate            : {_safe_float(hist.get('unknown_rate'), 0.0) * 100:.2f}%",
            f"Average FPS             : {_safe_float(hist.get('avg_fps'), 0.0):.2f}",
            f"Peak Simultaneous Faces : {_safe_int(hist.get('peak_simultaneous_faces'), 0)}",
            f"Face Throughput         : {_safe_float(hist.get('faces_per_sec'), 0.0):.2f} faces/s",
            f"Object Detections Total : {_safe_int(hist.get('object_detections_total'), 0)}",
            f"Object Gen/Custom       : {_safe_int(hist.get('object_general_detections'), 0)}/"
            f"{_safe_int(hist.get('object_custom_detections'), 0)}",
            f"Object Avg Confidence   : {_safe_float(hist.get('object_avg_confidence'), 0.0):.4f}",
            f"Memory Snapshots A/M    : {_safe_int(hist.get('memory_auto_snapshots'), 0)}/"
            f"{_safe_int(hist.get('memory_manual_snapshots'), 0)}",
            f"Memory Queries Total    : {_safe_int(hist.get('memory_queries_total'), 0)}",
            f"Alert Density           : {_safe_float(hist.get('alert_density_per_min'), 0.0):.2f} alerts/min",
            f"Average Confidence      : {_safe_float(hist.get('avg_confidence'), 0.0):.4f}",
            "",
            "╔════════════════════════════════════════════════════╗",
            "║               END OF SECURITY REPORT               ║",
            "╚════════════════════════════════════════════════════╝",
        ]
    )

    return "\n".join(lines)


def cmd_report() -> None:
    db = FaceDB.load()
    events = _load_metric_events()
    summary = _aggregate_report_data(events, db)

    ascii_text = _build_ascii_dashboard(summary)
    REPORT_TXT_PATH.write_text(ascii_text + "\n", encoding="utf-8")

    print(ascii_text)
    print("Report files generated:")
    print(f"  Metrics JSONL: {METRICS_LOG_PATH}")
    print(f"  ASCII: {REPORT_TXT_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(description="Live face enroll/recognize tool.")
    parser.add_argument(
        "--model",
        default="buffalo_sc",
        choices=["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"],
        help="InsightFace model pack  (default: buffalo_l)",
    )

    sub = parser.add_subparsers(dest="cmd")
    p_e = sub.add_parser("enroll", help="Enroll a person from live camera")
    p_e.add_argument("--name", required=True, help="Identity label")
    p_r = sub.add_parser("recognize", help="Live face + object recognition")
    p_r.add_argument(
        "--general-model",
        default=None,
        help="Path to general YOLO model (default: local reference or yolov8n.pt)",
    )
    p_r.add_argument(
        "--custom-model",
        default=None,
        help="Path to custom YOLO model (default: pointer/latest trained model)",
    )
    p_r.add_argument("--disable-general", action="store_true", help="Disable general YOLO stream")
    p_r.add_argument("--disable-custom", action="store_true", help="Disable custom YOLO stream")
    p_r.add_argument(
        "--snapshot-interval",
        type=float,
        default=15.0,
        help="Automatic snapshot interval in seconds (default: 15)",
    )

    p_t = sub.add_parser("train-objects", help="Fine-tune YOLO on a custom dataset YAML")
    p_t.add_argument("--data", required=True, help="Path to dataset YAML")
    p_t.add_argument("--base-model", default="yolov8n.pt", help="Base YOLO checkpoint")
    p_t.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p_t.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p_t.add_argument("--batch", type=int, default=16, help="Training batch size")
    p_t.add_argument("--project", default="runs/detect", help="Ultralytics project directory")
    p_t.add_argument("--name", default="custom-objects", help="Training run name")
    p_t.add_argument(
        "--set-default",
        action="store_true",
        help="Update custom model pointer after successful training",
    )

    sub.add_parser("memory-stats", help="Show memory storage statistics")
    p_mr = sub.add_parser("memory-recent", help="List recent snapshots")
    p_mr.add_argument("--minutes", type=int, default=5, help="Lookback window in minutes")
    p_mf = sub.add_parser("memory-find", help="Find when an object was last seen")
    p_mf.add_argument("--object", required=True, help="Object label or text")
    p_ms = sub.add_parser("memory-search", help="Search similar scenes")
    p_ms.add_argument("--text", required=True, help="Natural language scene query")

    sub.add_parser("list", help="List enrolled identities")
    sub.add_parser("report", help="Generate metrics/report files")

    args = parser.parse_args(sys.argv[1:] or ["recognize"])
    {
        "enroll": lambda: cmd_enroll(args.name, args.model),
        "recognize": lambda: cmd_recognize(
            args.model,
            args.general_model,
            args.custom_model,
            args.disable_general,
            args.disable_custom,
            args.snapshot_interval,
        ),
        "train-objects": lambda: cmd_train_objects(
            args.data,
            args.base_model,
            args.epochs,
            args.imgsz,
            args.batch,
            args.project,
            args.name,
            args.set_default,
        ),
        "memory-stats": cmd_memory_stats,
        "memory-recent": lambda: cmd_memory_recent(args.minutes),
        "memory-find": lambda: cmd_memory_find(args.object),
        "memory-search": lambda: cmd_memory_search(args.text),
        "list": cmd_list,
        "report": cmd_report,
    }.get(args.cmd, parser.print_help)()


if __name__ == "__main__":
    main()
