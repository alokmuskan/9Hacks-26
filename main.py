from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import common
from object_detection import DualYoloDetector, find_latest_custom_model
from scene_memory import SceneMemoryManager

# Face recognition is a degradable capability, so its import must not be fatal: a
# machine without a usable insightface still runs object detection, gaze, memory
# and the dashboard. `_try_build_face_app` turns a failure into a reason.
try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - depends on the environment
    FaceAnalysis = None  # type: ignore[assignment,misc]


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)


_load_env_file()

# ── Config ────────────────────────────────────────────────────────────────────
# /dev/video42 is a v4l2loopback virtual device: it only exists when something
# (OBS, ffmpeg) is feeding it. Prefer it when present so an existing virtual-camera
# setup keeps working, otherwise fall back to the first real device.
_LINUX_VIRTUAL_CAMERA = "/dev/video42"
_DEFAULT_CAMERA_SOURCE = (
    _LINUX_VIRTUAL_CAMERA
    if sys.platform.startswith("linux") and Path(_LINUX_VIRTUAL_CAMERA).exists()
    else "0"
)
CAMERA_SOURCE = os.getenv("AI_STUDIO_CAM_CAMERA_INDEX", _DEFAULT_CAMERA_SOURCE).strip()
DB_PATH = Path("face_db.npz")
METRICS_LOG_PATH = Path(common.METRICS_FILENAME)
REPORT_TXT_PATH = Path("report.txt")
UNKNOWN_INCIDENTS_DIR = Path("unknown_incidents")
MEMORY_DIR = Path("memory")
CUSTOM_MODEL_POINTER_PATH = Path("custom_model_path.txt")
# Ultralytics downloads `yolov8n.pt` on first use, so this default works on a clean
# checkout. The previous default pointed into `.references/`, a git-ignored path
# belonging to an older project, which made it a dead reference for every user.
DEFAULT_GENERAL_MODEL = os.getenv("AI_STUDIO_GENERAL_YOLO_MODEL", "yolov8n.pt")

ENROLL_SAMPLES = 25
ENROLL_CAPTURE_INTERVAL_SEC = 0.25
MATCH_THRESHOLD = 0.40
DET_SIZE = 320
INFER_MAX_SIDE = 640
UNKNOWN_LABEL = "Unknown"

EVENTS_TIMELINE_CAP = 500
UNKNOWN_ALERT_COOLDOWN_SEC = 3.0
# Shared with server.py via common so both entry points schedule gaze identically.
GAZE_INTERVAL_DEFAULT = common.GAZE_INTERVAL_DEFAULT
GAZE_MAX_INTERVAL_DEFAULT = common.GAZE_MAX_INTERVAL_DEFAULT
GAZE_TARGET_FPS_DROP_DEFAULT = common.GAZE_TARGET_FPS_DROP_DEFAULT
GAZE_ARCH_DEFAULT = "ResNet50"
GAZE_WEIGHTS_DEFAULT = "models/L2CSNet_gaze360.pkl"
GAZE_WEIGHTS_SOURCE_DEFAULT = (
    "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing"
)
GAZE_EMA_ALPHA = 0.10
GAZE_RECOVERY_STREAK_MIN = common.GAZE_RECOVERY_STREAK_MIN
GAZE_OBJECT_HIT_PADDING_PX = 8.0
GAZE_OBJECT_MAX_DIST_PX = 120.0
GAZE_SMOOTHING_WINDOW = 5
GAZE_SWITCH_CONFIRMATION = 3
BEHAVIOR_LOST_TIMEOUT_SEC = 1.0

GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_SYSTEM_PROMPT = (
    "You are an assistant for an AI monitoring system. Prefer concise factual answers "
    "based on provided logs/memory context, avoid speculation, and include timestamps when relevant."
)

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
    def empty(cls) -> FaceDB:
        return cls([], np.empty((0, 0), np.float32), np.empty((0,), np.int32))

    @classmethod
    def load(cls) -> FaceDB:
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
    return common.now_utc()


def _iso(dt: datetime | None = None) -> str:
    return common.iso(dt)


def _parse_iso(s: str | None) -> datetime | None:
    return common.parse_iso(s)


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
    return common.safe_int(v, default)


def _safe_float(v: Any, default: float = 0.0) -> float:
    return common.safe_float(v, default)


def _session_id(prefix: str) -> str:
    # Correlation-friendly ID shown in dashboards/logs. The prefix used to be
    # dropped here while the API kept it, so the two modes disagreed.
    return common.session_id(prefix)


def _timeline_bar(count: int, max_count: int, width: int = 12) -> str:
    if count <= 0 or max_count <= 0:
        return ""
    n = max(1, round((count / max_count) * width))
    return "█" * n


def _save_unknown_snapshot(
    frame: np.ndarray, unknown_bboxes: list[np.ndarray], ts_utc: datetime
) -> str:
    UNKNOWN_INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_local = ts_utc.astimezone()
    base = ts_local.strftime("unknown_%Y-%m-%d_%H-%M-%S")

    snap = frame.copy()
    for bbox in unknown_bboxes:
        _bracket_box(snap, bbox, AMBER, thickness=2)

    # Reserving the name and writing the image must be one step. Checking
    # `exists()` and writing afterwards let two concurrent writers pick the same
    # filename, so one capture silently overwrote the other.
    with common.process_lock(UNKNOWN_INCIDENTS_DIR / "incidents"):
        candidate: Path | None = None
        for suffix in range(1000):
            name = f"{base}.jpg" if suffix == 0 else f"{base}_{suffix:02d}.jpg"
            path = UNKNOWN_INCIDENTS_DIR / name
            try:
                handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            except OSError:
                break
            os.close(handle)
            candidate = path
            break

        if candidate is None:
            candidate = UNKNOWN_INCIDENTS_DIR / f"{base}_overflow.jpg"

        cv2.imwrite(str(candidate), snap)

    pruned = _prune_unknown_incidents()
    if pruned:
        print(
            f"Incident retention: pruned {pruned} capture(s); "
            f"keeping the newest {common.UNKNOWN_INCIDENT_MAX_FILES}."
        )

    return str(candidate)


def _prune_unknown_incidents(keep: int | None = None) -> int:
    """Delete the oldest unknown-face captures once the directory exceeds its cap.

    Unlike the snapshot store there is no index to keep in step, so this prunes by
    file modification time. 0 disables pruning.
    """
    limit = common.UNKNOWN_INCIDENT_MAX_FILES if keep is None else max(int(keep), 0)
    if limit <= 0 or not UNKNOWN_INCIDENTS_DIR.exists():
        return 0

    try:
        captures = sorted(
            (path for path in UNKNOWN_INCIDENTS_DIR.iterdir() if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
    except OSError:
        return 0

    excess = len(captures) - limit
    if excess <= 0:
        return 0

    removed = 0
    for path in captures[:excess]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


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
    print("  c  chatbot query")
    print("  t  manual memory snapshot")
    print("  m  memory statistics")
    print("  r  recent snapshots (5 minutes)")
    print("  f  find when object was last seen")
    print("  h  print this help")


INSIGHTFACE_MIN_VERSION = (0, 7, 3)


def _insightface_version() -> str:
    try:
        import insightface

        return str(getattr(insightface, "__version__", "") or "").strip()
    except Exception:
        return ""


def _insightface_unavailable_message(installed: str | None = None) -> str:
    required = ".".join(str(part) for part in INSIGHTFACE_MIN_VERSION)
    version = installed if installed is not None else (_insightface_version() or "not installed")
    return (
        f"insightface {version} cannot be used: FaceAnalysis(providers=...) requires "
        f"insightface >= {required}. Install it with `pixi install` (pixi.toml pins "
        f"insightface>=0.7.3,<0.8) or `pip install -U 'insightface>=0.7.3,<3'`."
    )


def _make_face_analysis(model: str, providers: list[str]) -> FaceAnalysis:
    if FaceAnalysis is None:
        raise RuntimeError(_insightface_unavailable_message())
    try:
        return FaceAnalysis(
            name=model,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
    except TypeError:
        pass
    try:
        return FaceAnalysis(name=model, providers=providers)
    except TypeError as exc:
        # InsightFace 0.2.x had `FaceAnalysis(name, root=...)` and no `providers`
        # argument, so both attempts above fail. Without this the user sees a bare
        # TypeError from deep inside the stack and has nothing to act on.
        raise RuntimeError(_insightface_unavailable_message()) from exc


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

    with suppress(OSError):
        nested_dir.rmdir()

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


def _try_build_face_app(model: str) -> tuple[FaceAnalysis | None, str | None]:
    """Build the InsightFace app, or explain why face recognition is unavailable.

    Returning a reason instead of raising lets both entry points keep running and
    report *why* the capability is off, through `degraded_reason`, `doctor`, and the
    `face_recognition_enabled` session metric. A session that silently reported zero
    faces would be indistinguishable from one where nobody was in frame.
    """
    try:
        return _build_app(model), None
    except Exception as exc:
        return None, (str(exc).strip() or type(exc).__name__)


def _open_camera() -> cv2.VideoCapture:
    def suppress_opencv_warnings() -> None:
        try:
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
                return
            if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                cvlog = cv2.utils.logging
                if hasattr(cvlog, "setLogLevel"):
                    cvlog.setLogLevel(getattr(cvlog, "LOG_LEVEL_ERROR", 0))
        except Exception:
            pass

    def camera_readable(cap: cv2.VideoCapture, warmup_reads: int = 10) -> bool:
        if not cap or not cap.isOpened():
            return False
        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
            time.sleep(0.03)
        return False

    source_value = os.getenv("AI_STUDIO_CAM_CAMERA_INDEX", CAMERA_SOURCE).strip()

    suppress_opencv_warnings()

    candidates: list[int | str] = []
    if source_value.startswith("/dev/"):
        candidates.append(source_value)
        # Optional fallback for OpenCV builds that only work with numeric indices.
        if os.getenv("AI_STUDIO_CAM_INCLUDE_INDEX_FALLBACK", "0") == "1":
            suffix = source_value.replace("/dev/video", "", 1)
            if source_value.startswith("/dev/video") and suffix.isdigit():
                candidates.append(int(suffix))
    elif source_value.isdigit():
        # On some systems, path-based open works while index-based open does not.
        candidates.append(f"/dev/video{source_value}")
        candidates.append(int(source_value))
    elif source_value:
        candidates.append(source_value)

    if sys.platform.startswith("linux") and os.getenv("AI_STUDIO_CAM_SCAN_ALL_DEVICES", "0") == "1":
        for path in sorted(Path("/dev").glob("video*"), key=lambda p: p.name, reverse=True):
            suffix = path.name.replace("video", "", 1)
            candidates.append(str(path))
            if suffix.isdigit():
                candidates.append(int(suffix))
    deduped: list[int | str] = []
    seen: set[tuple[type, str]] = set()
    for candidate in candidates:
        key = (type(candidate), str(candidate))
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    candidates = deduped

    tried: list[str] = []

    for source in candidates:
        if sys.platform.startswith("linux"):
            if isinstance(source, str) and source.startswith("/dev/video"):
                # Many Linux builds read v4l2loopback more reliably via FFmpeg path.
                backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY, cv2.CAP_V4L2]
            elif isinstance(source, int):
                backends = [cv2.CAP_V4L2]
            elif isinstance(source, str):
                backends = [cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                tried.append(f"{source}@{backend}:open_failed")
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Require readable frames for non-device-path sources to fail-fast on broken streams.
            if camera_readable(cap):
                return cap

            cap.release()
            tried.append(f"{source}@{backend}:no_frames")

    tried_msg = ", ".join(tried[:18])
    if len(tried) > 18:
        tried_msg += ", ..."
    raise RuntimeError(f"Failed to open readable camera stream. Tried: {tried_msg}")


class _AsyncCameraReader:
    """Background frame reader to keep UI responsive when camera reads block."""

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            else:
                time.sleep(0.01)

    def read(self, timeout_sec: float = 1.0) -> tuple[bool, np.ndarray | None]:
        deadline = time.monotonic() + max(timeout_sec, 0.01)
        with self._lock:
            start_seq = self._seq

        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                if self._seq > start_seq and self._frame is not None:
                    return True, self._frame.copy()
            time.sleep(0.005)

        # Fallback to last frame so the app can keep rendering instead of freezing.
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.2)


def _face_score(face: object) -> float:
    # InsightFace 2.x's Face.__getattr__ returns None for absent keys, which
    # bypasses getattr()'s default. A missing score must degrade to a neutral
    # value, not crash the frame loop with float(None).
    raw = getattr(face, "det_score", None)
    return 1.0 if raw is None else float(raw)


def _detect(
    app: FaceAnalysis | None, frame: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None]]:
    if app is None:
        # Face recognition is a degradable capability: without it the frame simply
        # contributes no face rows and the session keeps running.
        return []
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
            _face_score(f),
            (
                np.asarray(f.kps, np.float32) * inv
                if getattr(f, "kps", None) is not None
                else (
                    np.asarray(f.landmark, np.float32) * inv
                    if getattr(f, "landmark", None) is not None
                    else None
                )
            ),
        )
        for f in app.get(inp)
    ]


def _arch_name_tokens(arch: str) -> list[str]:
    s = arch.lower().replace("-", "").replace("_", "")
    if s.endswith("18"):
        return ["resnet18", "res18", "r18"]
    if s.endswith("34"):
        return ["resnet34", "res34", "r34"]
    if s.endswith("50"):
        return ["resnet50", "res50", "r50"]
    if s.endswith("101"):
        return ["resnet101", "res101", "r101"]
    if s.endswith("152"):
        return ["resnet152", "res152", "r152"]
    return [s]


def _select_gaze360_weight_path(
    candidates: list[str | Path], preferred_arch: str | None = None
) -> Path | None:
    filtered: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        name = path.name.lower()
        if path.suffix.lower() == ".pkl" and "gaze360" in name:
            filtered.append(path)
    if not filtered:
        return None

    if preferred_arch:
        arch_filtered: list[Path] = []
        tokens = _arch_name_tokens(preferred_arch)
        for path in filtered:
            name = path.name.lower().replace("-", "").replace("_", "")
            if any(tok in name for tok in tokens):
                arch_filtered.append(path)
        if arch_filtered:
            return min(arch_filtered, key=lambda p: p.name.lower())

    return min(filtered, key=lambda p: p.name.lower())


def _resolve_l2cs_weights_path(
    weights_path: Path,
    weights_source: str,
    auto_download: bool,
    gdown_module: Any | None,
    cached_resolved_path: Path | None = None,
    preferred_arch: str | None = None,
    force_download: bool = False,
) -> Path | None:
    if not force_download and cached_resolved_path is not None and cached_resolved_path.exists():
        return cached_resolved_path
    if not force_download and weights_path.exists():
        return weights_path

    # Fast local fallback: discover nested gaze360 checkpoints under the same base directory.
    if not force_download:
        local_candidates = list(weights_path.parent.rglob("*.pkl")) if weights_path.parent.exists() else []
        local_selected = _select_gaze360_weight_path(
            [str(p) for p in local_candidates], preferred_arch=preferred_arch
        )
        if local_selected is not None and local_selected.exists():
            return local_selected.resolve()

    if not auto_download or gdown_module is None:
        return None

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []
    try:
        rows = gdown_module.download_folder(
            url=weights_source,
            output=str(weights_path.parent),
            quiet=True,
            use_cookies=False,
        )
        if isinstance(rows, list):
            downloaded_files = [str(x) for x in rows if x]
    except Exception:
        return None

    candidate_paths: list[Path] = []
    for row in downloaded_files:
        path = Path(row)
        if path.is_dir():
            candidate_paths.extend(path.rglob("*.pkl"))
        else:
            candidate_paths.append(path)

    if not candidate_paths:
        candidate_paths = list(weights_path.parent.rglob("*.pkl"))
    selected = _select_gaze360_weight_path([str(p) for p in candidate_paths], preferred_arch=preferred_arch)
    return selected if selected is None else selected.resolve()


def _expand_bbox_by_ratio(
    bbox: np.ndarray,
    frame_w: int,
    frame_h: int,
    ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad_x = bw * ratio
    pad_y = bh * ratio
    ex1 = max(0, round(x1 - pad_x))
    ey1 = max(0, round(y1 - pad_y))
    ex2 = min(max(frame_w - 1, 0), round(x2 + pad_x))
    ey2 = min(max(frame_h - 1, 0), round(y2 + pad_y))
    return ex1, ey1, ex2, ey2


def _align_face_from_landmarks(face_crop: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
    if face_crop is None or face_crop.size == 0:
        return face_crop

    h, w = face_crop.shape[:2]
    if w < 32:
        return face_crop

    if landmarks is None:
        return face_crop
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 2 or lm.shape[0] < 2 or lm.shape[1] < 2:
        return face_crop

    left_eye = lm[0]
    right_eye = lm[1]
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    eye_distance = float(np.hypot(dx, dy))
    if eye_distance < 20.0:
        return face_crop

    angle = float(np.degrees(np.arctan2(dy, dx)))
    center = (w * 0.5, h * 0.5)
    try:
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_mode = int(getattr(cv2, "BORDER_REPLICATE", 1))
        return cv2.warpAffine(face_crop, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
    except Exception:
        return face_crop


def _decode_l2cs_pitch_yaw_rad(
    pitch_logits: Any,
    yaw_logits: Any,
    softmax: Any,
    idx_tensor_deg: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    # Convert to radians exactly once after expected-value decoding in degrees.
    pitch_rad = torch_module.sum(softmax(pitch_logits) * idx_tensor_deg, dim=1) * (np.pi / 180.0)
    yaw_rad = torch_module.sum(softmax(yaw_logits) * idx_tensor_deg, dim=1) * (np.pi / 180.0)
    return pitch_rad, yaw_rad


def _normalize_l2cs_state_dict(payload: Any) -> dict[str, Any]:
    state_dict = payload
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("L2CS checkpoint did not contain a state_dict mapping")

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        k = str(key)
        k = k.removeprefix("module.")
        normalized[k] = value
    return normalized


def _infer_l2cs_arch_from_state_dict(state_dict: dict[str, Any]) -> str | None:
    fc_w = state_dict.get("fc_yaw_gaze.weight")
    fc_shape = getattr(fc_w, "shape", None)
    if not fc_shape or len(fc_shape) < 2:
        return None

    width = int(fc_shape[1])
    keyset = {str(k) for k in state_dict}
    if width == 512:
        # Distinguish 18 vs 34 by existence of deeper stage block indexes.
        if any(k.startswith(("layer1.2.", "layer2.3.", "layer3.5.")) for k in keyset):
            return "ResNet34"
        return "ResNet18"

    if width == 2048:
        # Distinguish 50/101/152 using deepest layer3 block index.
        if any(k.startswith("layer3.35.") for k in keyset):
            return "ResNet152"
        if any(k.startswith("layer3.23.") for k in keyset):
            return "ResNet101"
        return "ResNet50"

    return None


def _gaze_endpoint_from_pitch_yaw(
    cx: float,
    cy: float,
    length: float,
    pitch: float,
    yaw: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    # Match L2CS draw math: dx uses pitch, dy uses yaw.
    dx = -length * float(np.sin(pitch) * np.cos(yaw))
    dy = -length * float(np.sin(yaw))
    gx = round(cx + dx)
    gy = round(cy + dy)
    gx = int(np.clip(gx, 0, max(frame_w - 1, 0)))
    gy = int(np.clip(gy, 0, max(frame_h - 1, 0)))
    return gx, gy


def _load_gaze_runtime(
    gaze_arch: str,
    gaze_weights: str,
    gaze_weights_source: str,
    gaze_auto_download: bool,
) -> dict[str, Any] | None:
    try:
        import torch
        from l2cs.utils import getArch, prep_input_numpy
    except Exception as ex:
        print(f"[GAZE] L2CS dependencies unavailable ({ex})")
        return None

    gdown_module: Any | None = None
    if gaze_auto_download:
        try:
            import gdown  # type: ignore

            gdown_module = gdown
        except Exception:
            gdown_module = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = Path(gaze_weights)
    resolved_path = _resolve_l2cs_weights_path(
        weights_path=weights_path,
        weights_source=gaze_weights_source,
        auto_download=gaze_auto_download,
        gdown_module=gdown_module,
        cached_resolved_path=None,
        preferred_arch=gaze_arch,
        force_download=False,
    )
    if resolved_path is None or not resolved_path.exists():
        print("[GAZE] L2CS weights unavailable — gaze estimation disabled")
        return None

    try:
        state_dict = _normalize_l2cs_state_dict(torch.load(str(resolved_path), map_location=device))
        inferred_arch = _infer_l2cs_arch_from_state_dict(state_dict)

        # If local/default weights don't match requested arch, try a forced arch-specific download once.
        if inferred_arch is not None and inferred_arch != gaze_arch and gaze_auto_download and gdown_module is not None:
            alt_path = _resolve_l2cs_weights_path(
                weights_path=weights_path,
                weights_source=gaze_weights_source,
                auto_download=True,
                gdown_module=gdown_module,
                cached_resolved_path=None,
                preferred_arch=gaze_arch,
                force_download=True,
            )
            if alt_path is not None and alt_path.exists() and alt_path.resolve() != resolved_path.resolve():
                resolved_path = alt_path.resolve()
                state_dict = _normalize_l2cs_state_dict(torch.load(str(resolved_path), map_location=device))
                inferred_arch = _infer_l2cs_arch_from_state_dict(state_dict)

        if inferred_arch is not None and inferred_arch != gaze_arch:
            print(
                f"[GAZE] Requested arch {gaze_arch} but checkpoint arch is {inferred_arch}. "
                "Gaze estimation disabled."
            )
            return None

        model = getArch(gaze_arch, 90)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        softmax = torch.nn.Softmax(dim=1)
        idx_tensor_deg = (torch.arange(90, dtype=torch.float32, device=device) * 4.0) - 180.0
    except Exception as ex:
        msg = str(ex).splitlines()[0] if str(ex) else type(ex).__name__
        print(f"[GAZE] L2CS model load failed ({msg})")
        return None

    print(f"Gaze model loaded: L2CS-Net {gaze_arch} ({device.type})")
    return {
        "model": model,
        "device": device,
        "torch": torch,
        "softmax": softmax,
        "idx_tensor_deg": idx_tensor_deg,
        "prep_input_numpy": prep_input_numpy,
        "weights_path": str(resolved_path),
        "weights_source": gaze_weights_source,
        "arch": gaze_arch,
    }


def _estimate_gaze_points(
    frame: np.ndarray,
    bboxes: list[np.ndarray],
    landmarks_list: list[np.ndarray | None],
    gaze_runtime: dict[str, Any] | None,
) -> list[tuple[int, int, float, float] | None]:
    if gaze_runtime is None or not bboxes:
        return [None for _ in bboxes]

    model = gaze_runtime.get("model")
    device = gaze_runtime.get("device")
    torch = gaze_runtime.get("torch")
    softmax = gaze_runtime.get("softmax")
    idx_tensor_deg = gaze_runtime.get("idx_tensor_deg")
    prep_input_numpy = gaze_runtime.get("prep_input_numpy")
    if (
        model is None
        or device is None
        or torch is None
        or softmax is None
        or idx_tensor_deg is None
        or prep_input_numpy is None
    ):
        return [None for _ in bboxes]

    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return [None for _ in bboxes]

    valid_idx: list[int] = []
    crops_rgb: list[np.ndarray] = []
    for i, bbox in enumerate(bboxes):
        landmarks = landmarks_list[i] if i < len(landmarks_list) else None
        ex1, ey1, ex2, ey2 = _expand_bbox_by_ratio(bbox, frame_w, frame_h, ratio=0.10)
        if ex2 <= ex1 or ey2 <= ey1:
            continue
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        crop_landmarks: np.ndarray | None = None
        if landmarks is not None:
            lm = np.asarray(landmarks, dtype=np.float32)
            if lm.ndim == 2 and lm.shape[0] >= 2 and lm.shape[1] >= 2:
                crop_landmarks = lm.copy()
                crop_landmarks[:, 0] -= float(ex1)
                crop_landmarks[:, 1] -= float(ey1)

        aligned = _align_face_from_landmarks(crop, crop_landmarks)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = cv2.resize(aligned, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops_rgb.append(aligned)
        valid_idx.append(i)

    results: list[tuple[int, int, float, float] | None] = [None for _ in bboxes]
    if not crops_rgb:
        return results

    inp = prep_input_numpy(np.stack(crops_rgb), device)
    with torch.no_grad():
        pitch_logits, yaw_logits = model(inp)
        pitch_rad, yaw_rad = _decode_l2cs_pitch_yaw_rad(
            pitch_logits=pitch_logits,
            yaw_logits=yaw_logits,
            softmax=softmax,
            idx_tensor_deg=idx_tensor_deg,
            torch_module=torch,
        )
        pitch_np = pitch_rad.detach().cpu().numpy()
        yaw_np = yaw_rad.detach().cpu().numpy()

    for j, i in enumerate(valid_idx):
        bbox = bboxes[i]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bbox_width = max(x2 - x1, 1.0)
        length = bbox_width * 0.7

        pitch = float(pitch_np[j])
        yaw = float(yaw_np[j])
        gx, gy = _gaze_endpoint_from_pitch_yaw(
            cx=cx,
            cy=cy,
            length=length,
            pitch=pitch,
            yaw=yaw,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        results[i] = (gx, gy, pitch, yaw)

    return results


def _adapt_gaze_interval(
    current_interval: int,
    base_interval: int,
    max_interval: int,
    overhead_ratio: float,
    target_drop: float,
    recovery_streak: int,
) -> tuple[int, int]:
    current_interval = max(current_interval, 1)
    base_interval = max(base_interval, 1)
    max_interval = max(max_interval, base_interval)

    if overhead_ratio > target_drop:
        return min(current_interval + 1, max_interval), 0

    if overhead_ratio < (target_drop * 0.5):
        next_streak = recovery_streak + 1
        if next_streak >= GAZE_RECOVERY_STREAK_MIN and current_interval > base_interval:
            return max(current_interval - 1, base_interval), 0
        return current_interval, next_streak

    return current_interval, 0


class GazeScheduler:
    """Decides which processed frames run gaze inference.

    Adaptation is opt-in: it is only active when ``max_interval`` exceeds
    ``base_interval``. At the default (both 1) every frame with faces runs gaze,
    so attention/behaviour metrics stay per-frame rather than sampled.

    When enabled, the interval grows by one whenever gaze inference costs more
    than ``target_fps_drop`` of the frame budget, and shrinks back one step at a
    time after ``GAZE_RECOVERY_STREAK_MIN`` consecutive cheap frames. Skipped
    frames reuse the previous estimate, so tracking stays continuous.
    """

    def __init__(
        self,
        base_interval: int = GAZE_INTERVAL_DEFAULT,
        max_interval: int = GAZE_INTERVAL_DEFAULT,
        target_fps_drop: float = GAZE_TARGET_FPS_DROP_DEFAULT,
    ) -> None:
        self.base_interval = max(_safe_int(base_interval, GAZE_INTERVAL_DEFAULT), 1)
        self.max_interval = max(_safe_int(max_interval, GAZE_INTERVAL_DEFAULT), self.base_interval)
        self.target_fps_drop = _safe_float(target_fps_drop, GAZE_TARGET_FPS_DROP_DEFAULT)
        self.interval = self.base_interval
        self.recovery_streak = 0
        self.frames_seen = 0

    @property
    def adaptive(self) -> bool:
        return self.max_interval > self.base_interval

    def should_run(self) -> bool:
        """Call once per processed frame that has faces. True => run inference."""
        self.frames_seen += 1
        if not self.adaptive:
            return True
        return (self.frames_seen - 1) % self.interval == 0

    def mode_label(self) -> str:
        if not self.adaptive:
            return "full-rate"
        return (
            f"adaptive(base={self.base_interval}, max={self.max_interval}, "
            f"target-drop={self.target_fps_drop:g})"
        )

    def metrics(self) -> dict[str, float]:
        """The aggregate fields describing what actually ran, in one place."""
        return {
            "gaze_base_interval_frames": self.base_interval,
            "gaze_interval_frames_final": self.interval,
            "gaze_target_fps_drop": self.target_fps_drop if self.adaptive else 0.0,
        }

    def observe(self, latency_ms: float, frame_ms: float) -> None:
        """Feed back the cost of a frame that ran inference."""
        if not self.adaptive:
            return
        frame_ms = _safe_float(frame_ms, 0.0)
        overhead_ratio = (_safe_float(latency_ms, 0.0) / frame_ms) if frame_ms > 0 else 0.0
        self.interval, self.recovery_streak = _adapt_gaze_interval(
            current_interval=self.interval,
            base_interval=self.base_interval,
            max_interval=self.max_interval,
            overhead_ratio=overhead_ratio,
            target_drop=self.target_fps_drop,
            recovery_streak=self.recovery_streak,
        )


def _best_face(
    faces: list[tuple[np.ndarray, np.ndarray, float, np.ndarray | None]]
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None] | None:
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
    common.append_jsonl(
        METRICS_LOG_PATH,
        {
            "timestamp_utc": _iso(),
            "event_type": event_type,
            **payload,
        },
    )


def _metrics_parse_errors() -> int:
    return common.metrics_parse_errors()


def _load_metric_events() -> list[dict[str, Any]]:
    return common.read_jsonl(METRICS_LOG_PATH)


def _bbox_to_list(bbox: Any) -> list[float]:
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def _normalize_object_rows_for_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "label": str(row.get("label", "object")),
                "confidence": round(_safe_float(row.get("confidence"), 0.0), 6),
                "bbox": _bbox_to_list(row.get("bbox", [0, 0, 0, 0])),
                "source": str(row.get("source", "general")),
                "class_id": _safe_int(row.get("class_id"), -1),
            }
        )
    return out


def _normalize_face_rows_for_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "name": str(row.get("name", UNKNOWN_LABEL)),
                "confidence": round(_safe_float(row.get("confidence"), 0.0), 6),
                "bbox": _bbox_to_list(row.get("bbox", [0, 0, 0, 0])),
                "gaze": row.get("gaze") if isinstance(row.get("gaze"), dict) else None,
                "target_object": row.get("target_object"),
            }
        )
    return out


def _point_to_rect_distance(
    x: float,
    y: float,
    bbox: np.ndarray | list[float] | tuple[float, float, float, float],
    pad: float = 0.0,
) -> float:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(4)]
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.hypot(dx, dy))


def _point_in_rect(
    x: float,
    y: float,
    bbox: np.ndarray | list[float] | tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(4)]
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _infer_gaze_target(
    gaze_row: tuple[int, int, float, float] | None,
    object_rows: list[dict[str, Any]],
    hit_padding_px: float = GAZE_OBJECT_HIT_PADDING_PX,
    max_dist_px: float = GAZE_OBJECT_MAX_DIST_PX,
) -> dict[str, Any] | None:
    if gaze_row is None or not object_rows:
        return None

    gx, gy, _pitch, _yaw = gaze_row

    inside: list[dict[str, Any]] = []
    nearest: dict[str, Any] | None = None
    nearest_dist = float("inf")

    for row in object_rows:
        bbox = np.asarray(row.get("bbox", [0, 0, 0, 0]), dtype=np.float32).reshape(4)
        if _point_in_rect(gx, gy, bbox, pad=hit_padding_px):
            inside.append(row)
        dist = _point_to_rect_distance(float(gx), float(gy), bbox, pad=0.0)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = row

    if inside:
        picked = max(inside, key=lambda r: _safe_float(r.get("confidence"), 0.0))
        return {
            "label": str(picked.get("label", "object")),
            "source": str(picked.get("source", "general")),
            "confidence": _safe_float(picked.get("confidence"), 0.0),
            "distance_px": 0.0,
            "method": "inside",
        }

    if nearest is not None and nearest_dist <= max_dist_px:
        return {
            "label": str(nearest.get("label", "object")),
            "source": str(nearest.get("source", "general")),
            "confidence": _safe_float(nearest.get("confidence"), 0.0),
            "distance_px": float(nearest_dist),
            "method": "nearest",
        }
    return None


@dataclass
class _BehaviorPersonState:
    history: deque[str | None]
    stable_target: str | None = None
    stable_start_ts: float | None = None
    pending_target: str | None = None
    pending_count: int = 0
    present: bool = False
    last_seen_ts: float | None = None
    last_update_ts: float | None = None


class _BehaviorTracker:
    def __init__(
        self,
        smoothing_window: int = GAZE_SMOOTHING_WINDOW,
        switch_confirmation: int = GAZE_SWITCH_CONFIRMATION,
        lost_timeout_sec: float = BEHAVIOR_LOST_TIMEOUT_SEC,
    ) -> None:
        self.smoothing_window = max(int(smoothing_window), 1)
        self.switch_confirmation = max(int(switch_confirmation), 1)
        self.lost_timeout_sec = max(float(lost_timeout_sec), 0.1)

        self._states: dict[str, _BehaviorPersonState] = {}
        self.attention_sec: defaultdict[tuple[str, str], float] = defaultdict(float)
        self.object_attention_sec: defaultdict[str, float] = defaultdict(float)
        self.interactions: Counter[tuple[str, str]] = Counter()
        self.events_count = 0

    def _state_for(self, person: str) -> _BehaviorPersonState:
        state = self._states.get(person)
        if state is None:
            state = _BehaviorPersonState(history=deque(maxlen=self.smoothing_window))
            self._states[person] = state
        return state

    def _majority_target(self, history: deque[str | None]) -> tuple[str | None, int]:
        counter: Counter[str] = Counter(v for v in history if v)
        if not counter:
            return None, 0
        target, count = counter.most_common(1)[0]
        return target, int(count)

    def _accumulate(self, person: str, state: _BehaviorPersonState, now_ts: float) -> None:
        if state.stable_target is None or state.last_update_ts is None:
            return
        dt = max(float(now_ts) - float(state.last_update_ts), 0.0)
        if dt <= 0.0:
            return
        key = (person, state.stable_target)
        self.attention_sec[key] += dt
        self.object_attention_sec[state.stable_target] += dt

    def _build_event(
        self,
        event: str,
        person: str,
        target_object: str | None,
        now_ts: float,
        previous_target: str | None = None,
        duration_sec: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "event": event,
            "person": person,
            "target_object": target_object,
            "previous_target": previous_target,
            "duration_sec": round(max(duration_sec, 0.0), 3),
            "event_time_utc": _iso(datetime.fromtimestamp(now_ts, tz=UTC)),
        }
        self.events_count += 1
        return payload

    def _close_target(
        self,
        person: str,
        state: _BehaviorPersonState,
        now_ts: float,
        reason: str,
    ) -> dict[str, Any] | None:
        if state.stable_target is None:
            return None
        duration = (
            max(float(now_ts) - float(state.stable_start_ts), 0.0)
            if state.stable_start_ts is not None
            else 0.0
        )
        payload = self._build_event(
            event=reason,
            person=person,
            target_object=state.stable_target,
            previous_target=None,
            now_ts=now_ts,
            duration_sec=duration,
        )
        state.stable_target = None
        state.stable_start_ts = None
        state.pending_target = None
        state.pending_count = 0
        state.history.clear()
        return payload

    def update(self, observations: dict[str, str | None], now_ts: float) -> list[dict[str, Any]]:
        now_ts = float(now_ts)
        emitted: list[dict[str, Any]] = []
        visible = set(observations.keys())

        for person in visible:
            state = self._state_for(person)
            self._accumulate(person, state, now_ts)

            target = observations.get(person)
            state.history.append(target or None)
            state.present = True
            state.last_seen_ts = now_ts
            state.last_update_ts = now_ts

            majority_target, majority_count = self._majority_target(state.history)
            if majority_target is None:
                state.pending_target = None
                state.pending_count = 0
                continue

            if state.stable_target is None:
                if majority_count >= self.switch_confirmation:
                    state.stable_target = majority_target
                    state.stable_start_ts = now_ts
                    self.interactions[(person, majority_target)] += 1
                    emitted.append(
                        self._build_event(
                            event="start",
                            person=person,
                            target_object=majority_target,
                            now_ts=now_ts,
                            duration_sec=0.0,
                        )
                    )
                continue

            if majority_target == state.stable_target:
                state.pending_target = None
                state.pending_count = 0
                continue

            if state.pending_target == majority_target:
                state.pending_count += 1
            else:
                state.pending_target = majority_target
                state.pending_count = 1

            if state.pending_count >= self.switch_confirmation:
                previous = state.stable_target
                previous_duration = (
                    max(now_ts - float(state.stable_start_ts), 0.0)
                    if state.stable_start_ts is not None
                    else 0.0
                )
                emitted.append(
                    self._build_event(
                        event="switch",
                        person=person,
                        target_object=majority_target,
                        previous_target=previous,
                        now_ts=now_ts,
                        duration_sec=previous_duration,
                    )
                )
                state.stable_target = majority_target
                state.stable_start_ts = now_ts
                state.pending_target = None
                state.pending_count = 0
                self.interactions[(person, majority_target)] += 1

        for person, state in self._states.items():
            if person in visible:
                continue
            if state.present:
                self._accumulate(person, state, now_ts)
                state.present = False
                state.last_update_ts = None
            if (
                state.stable_target is not None
                and state.last_seen_ts is not None
                and (now_ts - float(state.last_seen_ts)) > self.lost_timeout_sec
            ):
                closed = self._close_target(person, state, now_ts, reason="end")
                if closed:
                    emitted.append(closed)

        return emitted

    def finalize(self, now_ts: float) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        now_ts = float(now_ts)
        for person, state in self._states.items():
            self._accumulate(person, state, now_ts)
            state.present = False
            state.last_update_ts = None
            closed = self._close_target(person, state, now_ts, reason="end")
            if closed:
                emitted.append(closed)
        return emitted

    def current_targets(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for person, state in self._states.items():
            if state.stable_target:
                out[person] = state.stable_target
        return out

    def summary(self) -> dict[str, Any]:
        person_map: dict[str, dict[str, float]] = defaultdict(dict)
        for (person, obj), seconds in self.attention_sec.items():
            person_map[person][obj] = round(float(seconds), 3)
        top_objects = sorted(
            self.object_attention_sec.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {
            "attention_map": dict(person_map),
            "top_objects": [[obj, round(float(sec), 3)] for obj, sec in top_objects[:10]],
            "interactions_total": int(sum(self.interactions.values())),
            "events_count": int(self.events_count),
            "attention_total_sec": round(float(sum(self.attention_sec.values())), 3),
            "interaction_counts": {
                f"{person}|{obj}": int(count)
                for (person, obj), count in self.interactions.items()
            },
        }


def _parse_minutes_from_text(text: str, default: int = 5) -> int:
    m = re.search(r"(\d+)\s*(?:minute|min|mins)", text.lower())
    if not m:
        return max(int(default), 1)
    return max(_safe_int(m.group(1), default), 1)


def _phrase_for_attention(person: str, obj: str, seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 30.0:
        return f"{person} briefly checked a {obj}."
    if seconds < 120.0:
        return f"{person} looked at a {obj} for about {round(seconds)} seconds."
    minutes = max(round(seconds / 60.0), 1)
    return f"{person} looked at a {obj} for {minutes} minutes."


def _build_situation_summary(minutes: int = 5, now_utc: datetime | None = None) -> dict[str, Any]:
    lookback = max(int(minutes), 1)
    now_dt = now_utc or _now_utc()
    cutoff = now_dt - timedelta(minutes=lookback)

    events = _load_metric_events()
    attention_sec: defaultdict[tuple[str, str], float] = defaultdict(float)
    interaction_counts: Counter[tuple[str, str]] = Counter()

    for event in events:
        dt = _parse_iso(event.get("timestamp_utc"))
        if dt is None or dt < cutoff:
            continue

        et = str(event.get("event_type", ""))
        if et == "behavior_event":
            person = str(event.get("person") or "Unknown person")
            obj = str(event.get("target_object") or "").strip()
            duration = _safe_float(event.get("duration_sec"), 0.0)
            if obj and duration > 0:
                attention_sec[(person, obj)] += duration
            if obj and str(event.get("event")) in {"start", "switch"}:
                interaction_counts[(person, obj)] += 1
            continue

        if et == "recognize_session":
            agg = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}
            attn_map = agg.get("behavior_attention_map")
            if isinstance(attn_map, dict):
                for person, obj_map in attn_map.items():
                    if not isinstance(obj_map, dict):
                        continue
                    for obj, sec in obj_map.items():
                        obj_name = str(obj).strip()
                        if not obj_name:
                            continue
                        attention_sec[(str(person), obj_name)] += _safe_float(sec, 0.0)

    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
    recent_rows: list[dict[str, Any]] = []
    for row in memory.metadata:
        dt = _parse_iso(row.get("datetime") or row.get("timestamp_utc"))
        if dt is None or dt < cutoff:
            continue
        recent_rows.append(row)

    snapshots_total = len(recent_rows)
    snapshots_manual = sum(1 for row in recent_rows if bool(row.get("manual")))
    snapshots_auto = max(snapshots_total - snapshots_manual, 0)

    top_pairs = sorted(attention_sec.items(), key=lambda kv: kv[1], reverse=True)
    most_viewed_object = None
    if top_pairs:
        object_totals: defaultdict[str, float] = defaultdict(float)
        for (_, obj), sec in top_pairs:
            object_totals[obj] += sec
        if object_totals:
            most_viewed_object = max(object_totals.items(), key=lambda kv: kv[1])[0]
    else:
        object_freq: Counter[str] = Counter()
        for row in recent_rows:
            for obj in row.get("objects", []):
                object_freq[str(obj)] += 1
        if object_freq:
            most_viewed_object = object_freq.most_common(1)[0][0]

    return {
        "minutes": lookback,
        "generated_utc": _iso(now_dt),
        "window_start_utc": _iso(cutoff),
        "window_end_utc": _iso(now_dt),
        "top_attention_pairs": [
            {"person": person, "object": obj, "seconds": round(float(sec), 3)}
            for (person, obj), sec in top_pairs[:10]
        ],
        "interaction_counts": {
            f"{person}|{obj}": int(count)
            for (person, obj), count in interaction_counts.items()
        },
        "snapshots_total": snapshots_total,
        "snapshots_manual": snapshots_manual,
        "snapshots_auto": snapshots_auto,
        "most_viewed_object": most_viewed_object,
    }


def _render_situation_summary(summary: dict[str, Any], max_lines: int = 4) -> str:
    minutes = _safe_int(summary.get("minutes"), 5)
    pairs = summary.get("top_attention_pairs", [])
    snapshots_total = _safe_int(summary.get("snapshots_total"), 0)
    most_viewed_object = summary.get("most_viewed_object")

    bullets: list[str] = []
    if isinstance(pairs, list):
        for row in pairs[:3]:
            person = str(row.get("person", "Unknown person"))
            obj = str(row.get("object", "object"))
            sec = _safe_float(row.get("seconds"), 0.0)
            bullets.append(_phrase_for_attention(person, obj, sec))

    if snapshots_total > 0:
        verb = "were" if snapshots_total != 1 else "was"
        bullets.append(
            f"{snapshots_total} snapshot{'s' if snapshots_total != 1 else ''} {verb} captured."
        )
    if most_viewed_object:
        bullets.append(f"The most viewed object was a {most_viewed_object}.")

    if not bullets:
        return f"No notable activity was recorded in the last {minutes} minutes."

    lines = [f"In the last {minutes} minutes:"]
    for line in bullets[: max(int(max_lines), 1)]:
        lines.append(f"- {line}")
    return "\n".join(lines)


def _build_llm_context(question: str, lookback_minutes: int = 30) -> dict[str, Any]:
    events = _load_metric_events()
    cutoff = _now_utc() - timedelta(minutes=max(int(lookback_minutes), 1))
    recent_behavior: list[dict[str, Any]] = []
    latest_recognize: dict[str, Any] | None = None
    for event in reversed(events):
        if latest_recognize is None and event.get("event_type") == "recognize_session":
            agg = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}
            latest_recognize = {
                "session_id": event.get("session_id"),
                "start_utc": event.get("start_utc"),
                "end_utc": event.get("end_utc"),
                "duration_sec": event.get("duration_sec"),
                "aggregate": {
                    "known_detections": agg.get("known_detections"),
                    "unknown_detections": agg.get("unknown_detections"),
                    "active_subjects": agg.get("active_subjects"),
                    "active_objects": agg.get("active_objects"),
                    "behavior_top_objects": agg.get("behavior_top_objects"),
                    "behavior_interactions_total": agg.get("behavior_interactions_total"),
                },
            }
        dt = _parse_iso(event.get("timestamp_utc"))
        if dt is None or dt < cutoff:
            continue
        if event.get("event_type") == "behavior_event":
            recent_behavior.append(
                {
                    "ts": event.get("timestamp_utc"),
                    "event": event.get("event"),
                    "person": event.get("person"),
                    "target_object": event.get("target_object"),
                    "duration_sec": event.get("duration_sec"),
                }
            )
    recent_behavior.reverse()

    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
    memory_hits = memory.search_similar_scene(question, top_k=5)
    compact_hits = [
        {
            "timestamp_local": row.get("timestamp_local"),
            "objects": row.get("objects"),
            "snapshot": row.get("snapshot"),
        }
        for row in memory_hits
    ]

    return {
        "question": question,
        "latest_session": latest_recognize,
        "recent_behavior": recent_behavior[-30:],
        "memory_hits": compact_hits,
    }


def _query_groq(question: str, context: dict[str, Any]) -> str | None:
    # Both spellings are supported on purpose: `.env.example` documents the lowercase
    # form, so dropping it would break existing setups.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")  # noqa: SIM112
    if not api_key:
        return None
    try:
        from groq import Groq
    except Exception:
        return None

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL_DEFAULT,
            temperature=0.2,
            max_tokens=350,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Use this context to answer the question.\n"
                        f"Context JSON:\n{json.dumps(context, ensure_ascii=True)}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )
    except Exception:
        return None
    if not response.choices:
        return None
    msg = response.choices[0].message
    content = getattr(msg, "content", None)
    if not content:
        return None
    return str(content).strip()


def _extract_last_seen_target(question: str) -> str | None:
    q = question.strip()
    patterns = [
        r"last\s+see\s+(.+)\??$",
        r"last\s+seen\s+(.+)\??$",
        r"when\s+did\s+you\s+last\s+see\s+(.+)\??$",
    ]
    ql = q.lower()
    for pattern in patterns:
        m = re.search(pattern, ql)
        if not m:
            continue
        target = m.group(1).strip()
        target = re.sub(r"^(a|an|the)\s+", "", target)
        return target.strip(" ?.")
    return None


def _answer_current_presence() -> str:
    events = _load_metric_events()
    for event in reversed(events):
        if event.get("event_type") != "recognize_session":
            continue
        agg = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}
        active = agg.get("active_subjects")
        if isinstance(active, list) and active:
            names = [str(row.get("name", UNKNOWN_LABEL)) for row in active[:8] if isinstance(row, dict)]
            if names:
                return "Currently visible: " + ", ".join(names)
        return "No known people are currently visible in the latest session state."
    return "No recognition session data is available yet."


def _answer_attention_query(question: str, known_people: list[str]) -> tuple[str, bool]:
    """Answer an attention question, reporting whether any event backed it."""
    target_person = None
    ql = question.lower()
    for name in known_people:
        if name.lower() in ql:
            target_person = name
            break

    events = _load_metric_events()
    for event in reversed(events):
        if event.get("event_type") != "behavior_event":
            continue
        if str(event.get("event")) not in {"start", "switch", "end"}:
            continue
        person = str(event.get("person", "Unknown"))
        if target_person and person.lower() != target_person.lower():
            continue
        obj = event.get("target_object")
        if not obj:
            continue
        ts = event.get("timestamp_utc")
        if str(event.get("event")) == "end":
            return f"{person} stopped attending {obj} around {_local_hms(ts)}.", True
        return f"{person} was looking at {obj} around {_local_hms(ts)}.", True

    if target_person:
        return f"No recent attention events were found for {target_person}.", False
    return "No recent gaze attention events were found.", False


def _handle_chat_query(
    question: str,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = question.strip()
    started = time.perf_counter()
    answer = ""
    intent = "open_ended"
    action = "none"
    hit = True
    used_llm = False
    summary_minutes = None
    summary_payload: dict[str, Any] | None = None
    snapshot_payload: dict[str, Any] | None = None

    q = question.lower()
    memory = runtime_context.get("memory") if isinstance(runtime_context, dict) else None
    if memory is None:
        memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
    db = FaceDB.load()
    known_people = {n.lower() for n in db.names}

    if ("what happened" in q and "minute" in q) or "recent activity" in q or "situation summary" in q:
        intent = "session_summary"
        summary_minutes = _parse_minutes_from_text(q, default=5)
        summary_payload = _build_situation_summary(summary_minutes)
        answer = _render_situation_summary(summary_payload)
        _append_metric(
            "summary_query",
            {
                "source": "chat",
                "minutes": summary_minutes,
                "result_lines": answer.count("\n") + 1,
                "hit": bool(summary_payload.get("top_attention_pairs") or summary_payload.get("snapshots_total")),
            },
        )
    elif "memory status" in q or "memory stats" in q:
        intent = "memory_stats"
        answer = json.dumps(memory.get_memory_stats(), ensure_ascii=True, indent=2)
    elif "recent snapshot" in q:
        intent = "memory_recent"
        minutes = _parse_minutes_from_text(q, default=5)
        rows = memory.get_recent_snapshots(minutes=minutes)
        if rows:
            snippets = [
                f"{row.get('timestamp_local')} | {row.get('objects')} | {row.get('snapshot')}"
                for row in rows[-5:]
            ]
            answer = f"Recent snapshots in the last {minutes} minutes:\n" + "\n".join(snippets)
        else:
            hit = False
            answer = f"No recent snapshots were found in the last {minutes} minutes."
    elif "take snapshot" in q or "capture snapshot" in q:
        intent = "snapshot"
        if isinstance(runtime_context, dict) and runtime_context.get("frame") is not None:
            snap = memory.save_snapshot(
                runtime_context["frame"],
                runtime_context.get("object_rows", []),
                current_time=time.time(),
                manual=True,
                faces=runtime_context.get("face_rows"),
                object_detections=runtime_context.get("object_rows"),
                people=runtime_context.get("people"),
                attention=runtime_context.get("attention_rows"),
            )
            snapshot_payload = snap
            answer = f"Snapshot captured: {snap.get('snapshot')}"
            action = "snapshot"
        else:
            hit = False
            answer = "Snapshot capture is only available while live recognition is running."
    elif "who is present" in q or "who was present" in q:
        intent = "presence"
        if isinstance(runtime_context, dict) and runtime_context.get("people"):
            names = [str(x) for x in runtime_context.get("people", []) if str(x).strip()]
            if names:
                answer = "Currently visible: " + ", ".join(sorted(set(names)))
            else:
                answer = _answer_current_presence()
        else:
            answer = _answer_current_presence()
    elif "looking at" in q or "look at" in q:
        intent = "attention_lookup"
        answer, hit = _answer_attention_query(question, db.names)
    elif "most viewed object" in q:
        intent = "session_summary"
        summary_minutes = _parse_minutes_from_text(q, default=5)
        summary_payload = _build_situation_summary(summary_minutes)
        mvo = summary_payload.get("most_viewed_object")
        if mvo:
            answer = f"In the last {summary_minutes} minutes, the most viewed object was a {mvo}."
        else:
            hit = False
            answer = f"No viewed-object data is available in the last {summary_minutes} minutes."
        _append_metric(
            "summary_query",
            {
                "source": "chat",
                "minutes": summary_minutes,
                "result_lines": 1,
                "hit": bool(mvo),
            },
        )
    elif "last see" in q or "last seen" in q:
        target = _extract_last_seen_target(question)
        if target:
            if target.lower() in known_people:
                intent = "person_last_seen"
                row = memory.find_person_last_seen(target)
                if row:
                    answer = (
                        f"Last seen '{target}' at {row.get('timestamp_local')} | "
                        f"people={row.get('people', [])} | snapshot={row.get('snapshot')}"
                    )
                else:
                    hit = False
                    answer = f"I could not find recent sightings for '{target}'."
            else:
                intent = "object_last_seen"
                row = memory.find_object_last_seen(target)
                if row:
                    answer = (
                        f"Last seen '{target}' at {row.get('timestamp_local')} | "
                        f"objects={row.get('objects')} | snapshot={row.get('snapshot')}"
                    )
                else:
                    hit = False
                    answer = f"I could not find object '{target}' in memory."
        else:
            hit = False
            answer = "Please specify who or what you want to look up."
    else:
        context = _build_llm_context(question)
        llm_answer = _query_groq(question, context)
        if llm_answer:
            used_llm = True
            answer = llm_answer
        else:
            hit = False
            answer = (
                "I could not resolve that from deterministic tools and no Groq response was available. "
                "Try asking for summary, recent snapshots, memory stats, or last-seen queries."
            )

    duration_ms = (time.perf_counter() - started) * 1000.0
    _append_metric(
        "chat_query",
        {
            "question": question,
            "intent": intent,
            "action": action,
            "hit": bool(hit),
            "used_llm": bool(used_llm),
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {
        "answer": answer,
        "intent": intent,
        "action": action,
        "hit": bool(hit),
        "used_llm": bool(used_llm),
        "summary": summary_payload,
        "snapshot": snapshot_payload,
    }


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


# ── Session aggregate ─────────────────────────────────────────────────────────
# One definition of the `recognize_session.aggregate` contract, shared by the CLI
# loop and the API worker. They used to each carry a ~70-line copy of this and had
# already drifted apart (the API hardcoded its gaze metrics to zero).
SESSION_AGGREGATE_KEYS: tuple[str, ...] = (
    "session_id",
    "frames_total",
    "frames_with_faces",
    "frames_empty",
    "frames_dropped",
    "average_faces_per_frame",
    "detections_total",
    "known_detections",
    "unknown_detections",
    "peak_simultaneous_faces",
    "face_recognition_enabled",
    "avg_fps",
    "moving_avg_fps",
    "min_fps",
    "max_fps",
    "faces_per_sec",
    "avg_detection_latency_ms",
    "min_detection_latency_ms",
    "max_detection_latency_ms",
    "detection_calls",
    "avg_confidence",
    "recognition_rate",
    "unknown_rate",
    "unknown_alert_events",
    "unknown_alert_density_per_min",
    "unique_individuals_seen",
    "current_people_visible",
    "active_subjects",
    "active_objects",
    "detection_timeline",
    "object_detections_total",
    "object_general_detections",
    "object_custom_detections",
    "object_avg_confidence",
    "object_avg_confidence_general",
    "object_avg_confidence_custom",
    "gaze_enabled",
    "gaze_model_loaded",
    "gaze_base_interval_frames",
    "gaze_interval_frames_final",
    "gaze_target_fps_drop",
    "gaze_inference_calls",
    "gaze_inference_avg_ms",
    "gaze_inference_min_ms",
    "gaze_inference_max_ms",
    "object_class_counts_total",
    "object_class_counts_general",
    "object_class_counts_custom",
    "object_detection_timeline",
    "yolo_state_final",
    "yolo_model_paths",
    "memory_snapshots_auto",
    "memory_snapshots_manual",
    "memory_snapshots_total_session",
    "memory_snapshot_total_store",
    "memory_query_counts",
    "memory_query_hits",
    "memory_query_misses",
    "chat_queries_total",
    "chat_queries_hit",
    "chat_queries_llm",
    "behavior_interactions_total",
    "behavior_attention_total_sec",
    "behavior_top_objects",
    "behavior_attention_map",
    "behavior_events_count",
    "behavior_activity_patterns",
)


@dataclass
class SessionAggregateInput:
    """Counters a finished monitoring session hands to :meth:`build`.

    Field defaults describe "nothing happened": unset minimums stay at infinity so
    the builder can emit ``0.0`` for them, and gaze runs at full rate unless a
    caller says otherwise.
    """

    session_id: str = ""
    duration_sec: float = 0.0
    frames_total: int = 0
    frames_with_faces: int = 0
    frames_empty: int = 0
    frames_dropped: int = 0
    detections_total: int = 0
    known_detections: int = 0
    unknown_detections: int = 0
    peak_simultaneous_faces: int = 0
    face_recognition_enabled: bool = True
    confidence_sum: float = 0.0
    detection_calls: int = 0
    detection_latency_sum_ms: float = 0.0
    detection_latency_min_ms: float = float("inf")
    detection_latency_max_ms: float = 0.0
    fps_ema: float = 0.0
    fps_min: float = float("inf")
    fps_max: float = 0.0
    object_detections_total: int = 0
    object_general_detections: int = 0
    object_custom_detections: int = 0
    object_conf_sum: float = 0.0
    object_general_conf_sum: float = 0.0
    object_custom_conf_sum: float = 0.0
    unknown_alert_count: int = 0
    memory_auto_snapshots: int = 0
    memory_manual_snapshots: int = 0
    memory_total_store: int = 0
    memory_query_counts: dict[str, int] = field(default_factory=dict)
    memory_query_hits: dict[str, int] = field(default_factory=dict)
    memory_query_misses: dict[str, int] = field(default_factory=dict)
    chat_queries_total: int = 0
    chat_queries_hit: int = 0
    chat_queries_llm: int = 0
    unique_individuals_seen: int = 0
    current_people_visible: int = 0
    active_subjects: list[dict[str, Any]] = field(default_factory=list)
    active_objects: list[str] = field(default_factory=list)
    detection_timeline: dict[str, int] = field(default_factory=dict)
    object_detection_timeline: dict[str, int] = field(default_factory=dict)
    object_class_counts_total: dict[str, int] = field(default_factory=dict)
    object_class_counts_general: dict[str, int] = field(default_factory=dict)
    object_class_counts_custom: dict[str, int] = field(default_factory=dict)
    behavior_summary: dict[str, Any] = field(default_factory=dict)
    gaze_enabled: bool = False
    gaze_model_loaded: bool = False
    gaze_base_interval_frames: int = 1
    gaze_interval_frames_final: int = 1
    gaze_target_fps_drop: float = 0.0
    gaze_inference_calls: int = 0
    gaze_inference_sum_ms: float = 0.0
    gaze_inference_min_ms: float = float("inf")
    gaze_inference_max_ms: float = 0.0
    detector_state: dict[str, Any] = field(default_factory=dict)
    model_paths: dict[str, Any] = field(default_factory=dict)

    def build(self) -> dict[str, Any]:
        frames_total = _safe_int(self.frames_total)
        detections_total = _safe_int(self.detections_total)
        object_detections_total = _safe_int(self.object_detections_total)
        duration = max(_safe_float(self.duration_sec), 0.0)
        minutes = duration / 60.0

        behavior = self.behavior_summary if isinstance(self.behavior_summary, dict) else {}
        behavior_interactions = _safe_int(behavior.get("interactions_total"), 0)
        behavior_attention_sec = _safe_float(behavior.get("attention_total_sec"), 0.0)
        behavior_top_objects = behavior.get("top_objects", [])
        if not isinstance(behavior_top_objects, list):
            behavior_top_objects = []

        def ratio(numerator: float, denominator: float) -> float:
            return (numerator / denominator) if denominator else 0.0

        def real_min(value: float) -> float:
            return 0.0 if value == float("inf") else _safe_float(value)

        return {
            "session_id": str(self.session_id),
            "frames_total": frames_total,
            "frames_with_faces": _safe_int(self.frames_with_faces),
            "frames_empty": _safe_int(self.frames_empty),
            "frames_dropped": _safe_int(self.frames_dropped),
            "average_faces_per_frame": round(ratio(detections_total, frames_total), 4),
            "detections_total": detections_total,
            "known_detections": _safe_int(self.known_detections),
            "unknown_detections": _safe_int(self.unknown_detections),
            "peak_simultaneous_faces": _safe_int(self.peak_simultaneous_faces),
            "face_recognition_enabled": bool(self.face_recognition_enabled),
            "avg_fps": round(ratio(frames_total, duration), 3),
            "moving_avg_fps": round(_safe_float(self.fps_ema), 3),
            "min_fps": round(real_min(self.fps_min), 3),
            "max_fps": round(_safe_float(self.fps_max), 3),
            "faces_per_sec": round(ratio(detections_total, duration), 3),
            "avg_detection_latency_ms": round(
                ratio(_safe_float(self.detection_latency_sum_ms), _safe_int(self.detection_calls)), 2
            ),
            "min_detection_latency_ms": round(real_min(self.detection_latency_min_ms), 2),
            "max_detection_latency_ms": round(_safe_float(self.detection_latency_max_ms), 2),
            "detection_calls": _safe_int(self.detection_calls),
            "avg_confidence": round(ratio(_safe_float(self.confidence_sum), detections_total), 4),
            "recognition_rate": round(
                ratio(_safe_int(self.known_detections), detections_total), 6
            ),
            "unknown_rate": round(
                ratio(_safe_int(self.unknown_detections), detections_total), 6
            ),
            "unknown_alert_events": _safe_int(self.unknown_alert_count),
            "unknown_alert_density_per_min": round(
                ratio(_safe_int(self.unknown_alert_count), minutes), 3
            ),
            "unique_individuals_seen": _safe_int(self.unique_individuals_seen),
            "current_people_visible": _safe_int(self.current_people_visible),
            "active_subjects": list(self.active_subjects),
            "active_objects": list(self.active_objects),
            "detection_timeline": [
                {"time_local": t, "detections": int(c)}
                for t, c in list(self.detection_timeline.items())[-20:]
            ],
            "object_detections_total": object_detections_total,
            "object_general_detections": _safe_int(self.object_general_detections),
            "object_custom_detections": _safe_int(self.object_custom_detections),
            "object_avg_confidence": round(
                ratio(_safe_float(self.object_conf_sum), object_detections_total), 4
            ),
            "object_avg_confidence_general": round(
                ratio(
                    _safe_float(self.object_general_conf_sum),
                    _safe_int(self.object_general_detections),
                ),
                4,
            ),
            "object_avg_confidence_custom": round(
                ratio(
                    _safe_float(self.object_custom_conf_sum),
                    _safe_int(self.object_custom_detections),
                ),
                4,
            ),
            "gaze_enabled": bool(self.gaze_enabled),
            "gaze_model_loaded": bool(self.gaze_model_loaded),
            "gaze_base_interval_frames": _safe_int(self.gaze_base_interval_frames, 1),
            "gaze_interval_frames_final": _safe_int(self.gaze_interval_frames_final, 1),
            "gaze_target_fps_drop": _safe_float(self.gaze_target_fps_drop, 0.0),
            "gaze_inference_calls": _safe_int(self.gaze_inference_calls),
            "gaze_inference_avg_ms": round(
                ratio(_safe_float(self.gaze_inference_sum_ms), _safe_int(self.gaze_inference_calls)), 2
            ),
            "gaze_inference_min_ms": round(real_min(self.gaze_inference_min_ms), 2),
            "gaze_inference_max_ms": round(_safe_float(self.gaze_inference_max_ms), 2),
            "object_class_counts_total": dict(self.object_class_counts_total),
            "object_class_counts_general": dict(self.object_class_counts_general),
            "object_class_counts_custom": dict(self.object_class_counts_custom),
            "object_detection_timeline": [
                {"time_local": t, "detections": int(c)}
                for t, c in list(self.object_detection_timeline.items())[-20:]
            ],
            "yolo_state_final": dict(self.detector_state),
            "yolo_model_paths": dict(self.model_paths),
            "memory_snapshots_auto": _safe_int(self.memory_auto_snapshots),
            "memory_snapshots_manual": _safe_int(self.memory_manual_snapshots),
            "memory_snapshots_total_session": _safe_int(self.memory_auto_snapshots)
            + _safe_int(self.memory_manual_snapshots),
            "memory_snapshot_total_store": _safe_int(self.memory_total_store),
            "memory_query_counts": dict(self.memory_query_counts),
            "memory_query_hits": dict(self.memory_query_hits),
            "memory_query_misses": dict(self.memory_query_misses),
            "chat_queries_total": _safe_int(self.chat_queries_total),
            "chat_queries_hit": _safe_int(self.chat_queries_hit),
            "chat_queries_llm": _safe_int(self.chat_queries_llm),
            "behavior_interactions_total": behavior_interactions,
            "behavior_attention_total_sec": round(behavior_attention_sec, 3),
            "behavior_top_objects": behavior_top_objects,
            "behavior_attention_map": behavior.get("attention_map", {}),
            "behavior_events_count": _safe_int(behavior.get("events_count"), 0),
            "behavior_activity_patterns": {
                "transitions_per_min": round(ratio(behavior_interactions, minutes), 3),
                "unique_attended_objects": len(behavior_top_objects),
                "focus_ratio": round(ratio(behavior_attention_sec, duration), 4),
            },
        }


def build_session_aggregate(**kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper so callers can pass counters as keywords."""
    return SessionAggregateInput(**kwargs).build()


# ── Commands ──────────────────────────────────────────────────────────────────
def cmd_enroll(name: str, model: str) -> None:
    session_id = _session_id("enroll")
    start_dt = _now_utc()
    started = time.time()

    app, db, cap = _build_app(model), FaceDB.load(), _open_camera()
    reader = _AsyncCameraReader(cap)
    cv2.namedWindow(win := f"Enroll — {name}", cv2.WINDOW_NORMAL)

    samples: list[np.ndarray] = []
    frames_total = 0
    frames_dropped = 0
    last_capture_ts = -1e9

    print(f"Enrolling '{name}'. Keep one face visible. Press q to finish early.")

    try:
        while len(samples) < ENROLL_SAMPLES:
            ok, frame = reader.read(timeout_sec=1.0)
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
                bbox, emb, _, _ = picked
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
        reader.close()
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
            "schema_version": common.ENROLL_SCHEMA_VERSION,
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
    disable_gaze: bool,
    gaze_arch: str,
    gaze_weights: str,
    gaze_weights_source: str,
    disable_gaze_auto_download: bool,
    gaze_max_interval: int = GAZE_INTERVAL_DEFAULT,
    gaze_target_fps_drop: float = GAZE_TARGET_FPS_DROP_DEFAULT,
) -> None:
    app, face_reason = _try_build_face_app(model)
    if app is None:
        # Degrade loudly: a session without faces is still useful (objects, gaze,
        # memory), but the operator must be told why identities are not matched.
        print(f"Face recognition DISABLED: {face_reason}")
        print("Continuing with object detection only; no identities will be matched.")

    db = FaceDB.load()
    if not db.names:
        if app is None:
            # The "no identities" setup error only applies when matching is possible.
            # With face recognition off, every detection is an object anyway, and
            # `enroll` cannot even run, so refusing to start would be a dead end.
            print(
                "No enrolled identities and face recognition is unavailable; "
                "continuing with object detection only."
            )
        else:
            # Fail fast in the CLI: an interactive session with no identities would only
            # ever report Unknown, so this is a setup error, not a runtime condition.
            # The API deliberately degrades instead (a service should still serve frames).
            raise RuntimeError(
                "No enrolled identities. Run `pixi run python main.py enroll --name <name>` "
                "first, then start monitoring again."
            )

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

    cap = _open_camera()
    reader = _AsyncCameraReader(cap)
    gaze_enabled = not disable_gaze
    gaze_auto_download = not disable_gaze_auto_download
    gaze_scheduler = GazeScheduler(
        base_interval=GAZE_INTERVAL_DEFAULT,
        max_interval=gaze_max_interval,
        target_fps_drop=gaze_target_fps_drop,
    )
    gaze_inference_calls = 0
    gaze_inference_sum_ms = 0.0
    gaze_inference_min_ms = float("inf")
    gaze_inference_max_ms = 0.0
    last_gaze_rows: list[tuple[int, int, float, float] | None] | None = None
    gaze_runtime = (
        _load_gaze_runtime(
            gaze_arch=gaze_arch,
            gaze_weights=gaze_weights,
            gaze_weights_source=gaze_weights_source,
            gaze_auto_download=gaze_auto_download,
        )
        if gaze_enabled
        else None
    )
    gaze_model_loaded = gaze_runtime is not None
    gaze_active = gaze_enabled and gaze_model_loaded

    cv2.namedWindow(win := "Recognize", cv2.WINDOW_NORMAL)

    print(
        "Running unified stream: "
        + ("InsightFace + YOLO + memory." if app is not None else "YOLO + memory (faces disabled).")
    )
    _print_runtime_help()
    if custom_model_path:
        print(f"Custom YOLO model: {custom_model_path}")
    else:
        print("Custom YOLO model: not configured")
    if gaze_enabled:
        if gaze_model_loaded:
            loaded_path = str(gaze_runtime.get("weights_path", gaze_weights)) if gaze_runtime else gaze_weights
            loaded_arch = str(gaze_runtime.get("arch", gaze_arch)) if gaze_runtime else gaze_arch
            print(
                f"Gaze active: model=L2CS-Net {loaded_arch} weights={loaded_path} "
                f"mode={gaze_scheduler.mode_label()}"
            )
        else:
            print("Gaze requested but unavailable. Continuing with gaze OFF.")

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
    chat_queries_total = 0
    chat_queries_hit = 0
    chat_queries_llm = 0

    unknown_alert_count = 0
    last_unknown_alert_ts = -1e9

    label_counter: Counter[str] = Counter()
    latest_active_subjects: list[dict[str, Any]] = []
    latest_object_labels: list[str] = []

    people: dict[str, dict[str, Any]] = {}
    visible_prev: set[str] = set()
    behavior_tracker = _BehaviorTracker()

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
            ok, frame = reader.read(timeout_sec=1.0)
            if not ok:
                frames_dropped += 1
                cv2.waitKey(1)
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
            render_rows: list[tuple[np.ndarray, np.ndarray | None, str, float]] = []
            active_conf: dict[str, float] = {}

            for bbox, emb, _, landmarks in face_rows:
                label, score = _match(emb, db)
                render_rows.append((bbox, landmarks, label, score))
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

            gaze_rows: list[tuple[int, int, float, float] | None] = [None for _ in render_rows]
            ran_gaze_this_frame = False
            if gaze_active and render_rows:
                if gaze_scheduler.should_run():
                    ran_gaze_this_frame = True
                    gaze_t0 = time.perf_counter()
                    gaze_rows = _estimate_gaze_points(
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
                    ran_gaze_this_frame = True
                    gaze_rows = [
                        last_gaze_rows[i] if i < len(last_gaze_rows) else None
                        for i in range(len(render_rows))
                    ]

            gaze_observations: dict[str, str | None] = {}
            attention_rows: list[dict[str, Any]] = []
            face_snapshot_rows: list[dict[str, Any]] = []
            for i, (bbox, _landmarks, label, score) in enumerate(render_rows):
                color = GREEN if label != UNKNOWN_LABEL else AMBER
                _bracket_box(frame, bbox, color)
                _label_tag(frame, f"{label}  {score:.2f}", int(bbox[0]), int(bbox[1]) - 6, color)

                gaze_info = gaze_rows[i] if i < len(gaze_rows) else None
                target_info: dict[str, Any] | None = None
                gaze_payload: dict[str, Any] | None = None
                if ran_gaze_this_frame and gaze_info is not None:
                    gx, gy, pitch, yaw = gaze_info
                    cx = int((bbox[0] + bbox[2]) * 0.5)
                    cy = int((bbox[1] + bbox[3]) * 0.5)
                    cv2.line(frame, (cx, cy), (gx, gy), TEAL, 2, cv2.LINE_AA)
                    cv2.circle(frame, (gx, gy), 7, TEAL, -1)
                    _label_tag(
                        frame,
                        f"pitch:{pitch:+.2f} yaw:{yaw:+.2f}",
                        gx + 8,
                        max(20, gy - 8),
                        TEAL,
                    )
                    target_info = _infer_gaze_target(gaze_info, object_rows)
                    if target_info is not None:
                        _label_tag(
                            frame,
                            f"target:{target_info['label']}",
                            gx + 8,
                            min(frame.shape[0] - 10, gy + 22),
                            CYAN,
                        )
                    gaze_payload = {
                        "endpoint": [int(gx), int(gy)],
                        "pitch": round(float(pitch), 6),
                        "yaw": round(float(yaw), 6),
                    }

                if label != UNKNOWN_LABEL:
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
                        "bbox": _bbox_to_list(bbox),
                        "gaze": gaze_payload,
                        "target_object": str(target_info.get("label")) if isinstance(target_info, dict) else None,
                    }
                )

            behavior_events = behavior_tracker.update(gaze_observations, now_ts)
            for behavior_event in behavior_events:
                _append_metric("behavior_event", {"session_id": session_id, **behavior_event})
                evt = str(behavior_event.get("event"))
                person = str(behavior_event.get("person", "Unknown"))
                target = behavior_event.get("target_object")
                prev = behavior_event.get("previous_target")
                if evt == "switch":
                    message = f"{person} shifted attention from {prev} to {target}"
                elif evt == "start":
                    message = f"{person} started attending {target}"
                else:
                    message = f"{person} stopped attending {target}"
                add_event(
                    "behavior_event",
                    message,
                    extra={
                        "behavior_event": evt,
                        "person": person,
                        "target_object": target,
                        "duration_sec": behavior_event.get("duration_sec"),
                    },
                )

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

            state = detector.get_state()
            gaze_status = "ON" if gaze_active else ("OFF" if not gaze_enabled else "UNAVAILABLE")
            _hud(
                frame,
                [
                    (
                        f"Faces:{face_count} Objects:{object_count} FPS:{fps_ema:.1f}",
                        WHITE,
                    ),
                    (
                        (
                            f"Known:{known_detections} Unknown:{unknown_detections} "
                            f"G:{'ON' if state['general']['enabled'] else 'OFF'} "
                            f"C:{'ON' if state['custom']['enabled'] else 'OFF'}"
                        ),
                        (180, 180, 180),
                    ),
                    (
                        (
                            f"Gaze:{gaze_status} L:{'Y' if gaze_model_loaded else 'N'} "
                            f"Calls:{gaze_inference_calls}"
                        ),
                        (170, 170, 170),
                    ),
                    (
                        f"Snapshots(auto/manual): {memory_auto_snapshots}/{memory_manual_snapshots}",
                        (160, 160, 160),
                    ),
                    ("Q quit | G/O toggle YOLO | C/T/M/R/F/H", (140, 140, 140)),
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
            elif key == ord("c"):
                query = input("Chat query: ").strip()
                if query:
                    chat_queries_total += 1
                    runtime_context = {
                        "memory": memory,
                        "frame": frame.copy(),
                        "object_rows": _normalize_object_rows_for_json(object_rows),
                        "face_rows": _normalize_face_rows_for_json(face_snapshot_rows),
                        "people": sorted(visible_now),
                        "attention_rows": list(attention_rows),
                    }
                    result = _handle_chat_query(query, runtime_context=runtime_context)
                    if result.get("hit"):
                        chat_queries_hit += 1
                    if result.get("used_llm"):
                        chat_queries_llm += 1
                    print(result.get("answer", ""))
                    add_event(
                        "chat_query",
                        f"Chat intent={result.get('intent')} q='{query[:80]}'",
                        extra={
                            "intent": result.get("intent"),
                            "hit": result.get("hit"),
                            "used_llm": result.get("used_llm"),
                        },
                    )
                    if result.get("action") == "snapshot" and isinstance(result.get("snapshot"), dict):
                        memory_manual_snapshots += 1
                        snap = result["snapshot"]
                        add_event(
                            "memory_snapshot_manual",
                            "Manual snapshot saved (chat action)",
                            extra={
                                "image_path": snap.get("snapshot_path"),
                                "image_name": Path(str(snap.get("snapshot", ""))).name,
                            },
                        )
            elif key == ord("t"):
                snap = memory.save_snapshot(
                    frame,
                    object_rows,
                    current_time=now_ts,
                    manual=True,
                    faces=face_snapshot_rows,
                    object_detections=object_rows,
                    people=sorted(visible_now),
                    attention=attention_rows,
                )
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
        reader.close()
        cap.release()
        memory.save_all_memory()
        cv2.destroyAllWindows()

    end_dt = _now_utc()
    end_ts = time.time()
    duration_sec = max(end_ts - session_start, 0.0)

    for behavior_event in behavior_tracker.finalize(end_ts):
        _append_metric("behavior_event", {"session_id": session_id, **behavior_event})
        evt = str(behavior_event.get("event"))
        if evt == "end":
            person = str(behavior_event.get("person", "Unknown"))
            target = behavior_event.get("target_object")
            add_event(
                "behavior_event",
                f"{person} stopped attending {target}",
                extra={
                    "behavior_event": evt,
                    "person": person,
                    "target_object": target,
                    "duration_sec": behavior_event.get("duration_sec"),
                },
            )

    # Close out presence timing for those still visible at end.
    for name in visible_prev:
        info = people.get(name)
        if info and info["present"] and info["last_enter_ts"] is not None:
            info["presence_sec"] += max(end_ts - float(info["last_enter_ts"]), 0.0)
            info["last_enter_ts"] = None
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

    memory_stats = memory.get_memory_stats()
    aggregate = build_session_aggregate(
        session_id=session_id,
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
        fps_ema=fps_ema,
        fps_min=fps_min,
        fps_max=fps_max,
        object_detections_total=object_detections_total,
        object_general_detections=object_general_detections,
        object_custom_detections=object_custom_detections,
        object_conf_sum=object_conf_sum,
        object_general_conf_sum=object_general_conf_sum,
        object_custom_conf_sum=object_custom_conf_sum,
        unknown_alert_count=unknown_alert_count,
        memory_auto_snapshots=memory_auto_snapshots,
        memory_manual_snapshots=memory_manual_snapshots,
        memory_total_store=_safe_int(memory_stats.get("total_snapshots"), 0),
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
        gaze_model_loaded=gaze_model_loaded,
        **gaze_scheduler.metrics(),
        gaze_inference_calls=gaze_inference_calls,
        gaze_inference_sum_ms=gaze_inference_sum_ms,
        gaze_inference_min_ms=gaze_inference_min_ms,
        gaze_inference_max_ms=gaze_inference_max_ms,
        detector_state=detector.get_state(),
        model_paths={"general": general_model_path, "custom": custom_model_path},
        face_recognition_enabled=app is not None,
    )

    _append_metric(
        "recognize_session",
        {
            "schema_version": common.SESSION_SCHEMA_VERSION,
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
        f"Session summary | frames={frames_total} avg_fps={aggregate['avg_fps']:.2f} "
        f"faces={detections_total} objects={object_detections_total} "
        f"known={known_detections} unknown={unknown_detections}"
    )


def cmd_list() -> None:
    db = FaceDB.load()
    if not db.names:
        print("No enrolled identities found.")
        return
    print(f"Database: {DB_PATH}")
    # strict=False: a hand-edited or truncated face_db.npz should list what it has
    # rather than crash a diagnostic command.
    for name, count in zip(db.names, db.counts, strict=False):
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
        Path(train_result.save_dir)
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
            "schema_version": common.OBJECT_TRAIN_SCHEMA_VERSION,
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
    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
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
    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
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
    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
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


def cmd_memory_find_person(person_name: str) -> None:
    memory = SceneMemoryManager(base_dir=MEMORY_DIR, enable_vectors=False)
    row = memory.find_person_last_seen(person_name)
    _append_metric(
        "memory_query",
        {
            "query_type": "find_person",
            "person": person_name,
            "hit": bool(row),
            "result_count": 1 if row else 0,
        },
    )

    if not row:
        print(f"Person not found: {person_name}")
        return

    print(
        f"Last seen '{person_name}' at {row.get('timestamp_local')} | "
        f"people={row.get('people', [])} | snapshot={row.get('snapshot')}"
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


def cmd_session_summary(minutes: int, as_json: bool = False) -> None:
    summary = _build_situation_summary(minutes=minutes)
    rendered = _render_situation_summary(summary)
    _append_metric(
        "summary_query",
        {
            "source": "command",
            "minutes": int(minutes),
            "result_lines": rendered.count("\n") + 1,
            "hit": bool(summary.get("top_attention_pairs") or summary.get("snapshots_total")),
        },
    )
    if as_json:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return
    print(rendered)


def cmd_chat(question: str | None = None) -> None:
    if question:
        result = _handle_chat_query(question)
        print(result.get("answer", ""))
        return

    print("Interactive chat mode. Type 'quit' to exit.")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break
        result = _handle_chat_query(q)
        print(result.get("answer", ""))


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
        "gaze_enabled": bool(raw_aggregate.get("gaze_enabled", event.get("gaze_enabled", False))),
        "gaze_model_loaded": bool(
            raw_aggregate.get("gaze_model_loaded", event.get("gaze_model_loaded", False))
        ),
        "gaze_base_interval_frames": _safe_int(
            raw_aggregate.get("gaze_base_interval_frames", event.get("gaze_base_interval_frames", 0)),
            0,
        ),
        "gaze_interval_frames_final": _safe_int(
            raw_aggregate.get("gaze_interval_frames_final", event.get("gaze_interval_frames_final", 0)),
            0,
        ),
        "gaze_target_fps_drop": _safe_float(
            raw_aggregate.get("gaze_target_fps_drop", event.get("gaze_target_fps_drop", 0.0)),
            0.0,
        ),
        "gaze_inference_calls": _safe_int(
            raw_aggregate.get("gaze_inference_calls", event.get("gaze_inference_calls", 0)),
            0,
        ),
        "gaze_inference_avg_ms": _safe_float(
            raw_aggregate.get("gaze_inference_avg_ms", event.get("gaze_inference_avg_ms", 0.0)),
            0.0,
        ),
        "gaze_inference_min_ms": _safe_float(
            raw_aggregate.get("gaze_inference_min_ms", event.get("gaze_inference_min_ms", 0.0)),
            0.0,
        ),
        "gaze_inference_max_ms": _safe_float(
            raw_aggregate.get("gaze_inference_max_ms", event.get("gaze_inference_max_ms", 0.0)),
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
        "chat_queries_total": _safe_int(raw_aggregate.get("chat_queries_total"), 0),
        "chat_queries_hit": _safe_int(raw_aggregate.get("chat_queries_hit"), 0),
        "chat_queries_llm": _safe_int(raw_aggregate.get("chat_queries_llm"), 0),
        "behavior_interactions_total": _safe_int(raw_aggregate.get("behavior_interactions_total"), 0),
        "behavior_attention_total_sec": _safe_float(raw_aggregate.get("behavior_attention_total_sec"), 0.0),
        "behavior_top_objects": raw_aggregate.get("behavior_top_objects", []),
        "behavior_attention_map": raw_aggregate.get("behavior_attention_map", {}),
        "behavior_events_count": _safe_int(raw_aggregate.get("behavior_events_count"), 0),
        "behavior_activity_patterns": raw_aggregate.get("behavior_activity_patterns", {}),
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
    chat_query_events = [e for e in events if e.get("event_type") == "chat_query"]
    summary_query_events = [e for e in events if e.get("event_type") == "summary_query"]
    chat_query_event_hits = sum(1 for e in chat_query_events if bool(e.get("hit")))
    chat_query_event_llm = sum(1 for e in chat_query_events if bool(e.get("used_llm")))

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
    hist_chat_queries_total = 0
    hist_chat_queries_hit = 0
    hist_chat_queries_llm = 0
    hist_behavior_interactions = 0
    hist_behavior_attention_sec = 0.0
    hist_behavior_events = 0
    behavior_object_counter: Counter[str] = Counter()

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
        hist_chat_queries_total += _safe_int(a.get("chat_queries_total"), 0)
        hist_chat_queries_hit += _safe_int(a.get("chat_queries_hit"), 0)
        hist_chat_queries_llm += _safe_int(a.get("chat_queries_llm"), 0)
        hist_behavior_interactions += _safe_int(a.get("behavior_interactions_total"), 0)
        hist_behavior_attention_sec += _safe_float(a.get("behavior_attention_total_sec"), 0.0)
        hist_behavior_events += _safe_int(a.get("behavior_events_count"), 0)
        query_counts = a.get("memory_query_counts", {})
        if isinstance(query_counts, dict):
            hist_memory_queries_total += sum(_safe_int(v) for v in query_counts.values())
        class_counts = a.get("object_class_counts_total", {})
        if isinstance(class_counts, dict):
            object_label_counter.update({str(k): _safe_int(v) for k, v in class_counts.items()})
        behavior_top = a.get("behavior_top_objects", [])
        if isinstance(behavior_top, list):
            for row in behavior_top:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                weight = _safe_float(row[1], 0.0)
                if weight <= 0.0:
                    continue
                behavior_object_counter[str(row[0])] += max(1, round(weight))

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
                for name, count in zip(db.names, db.counts, strict=False)
            ],
        },
        "metrics": {
            "path": str(METRICS_LOG_PATH),
            "events_total": len(events),
            "parse_errors": _metrics_parse_errors(),
            "enroll_events": len(enroll_events),
            "recognize_sessions": len(recognize_events),
            "chat_queries": len(chat_query_events),
            "summary_queries": len(summary_query_events),
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
            "chat_queries_total": len(chat_query_events) or hist_chat_queries_total,
            "chat_queries_hit": chat_query_event_hits or hist_chat_queries_hit,
            "chat_queries_llm": chat_query_event_llm or hist_chat_queries_llm,
            "summary_queries_total": len(summary_query_events),
            "behavior_interactions_total": hist_behavior_interactions,
            "behavior_attention_total_sec": hist_behavior_attention_sec,
            "behavior_events_count": hist_behavior_events,
            "behavior_top_objects": behavior_object_counter.most_common(10),
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
    if isinstance(session_id, str) and session_id.startswith(("legacy-", "recognize-")):
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
    behavior_top = (
        latest_agg.get("behavior_top_objects", [])
        if isinstance(latest_agg.get("behavior_top_objects"), list)
        else []
    )
    behavior_interactions = _safe_int(latest_agg.get("behavior_interactions_total"), 0)

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
        f"Behavior Interactions   : {behavior_interactions}",
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

    if behavior_top:
        top_name = str(behavior_top[0][0]) if len(behavior_top[0]) >= 1 else "object"
        lines.append(f"Top attended object: {top_name}")

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
            (
                f"Object Gen/Custom       : {_safe_int(hist.get('object_general_detections'), 0)}/"
                f"{_safe_int(hist.get('object_custom_detections'), 0)}"
            ),
            f"Object Avg Confidence   : {_safe_float(hist.get('object_avg_confidence'), 0.0):.4f}",
            (
                f"Memory Snapshots A/M    : {_safe_int(hist.get('memory_auto_snapshots'), 0)}/"
                f"{_safe_int(hist.get('memory_manual_snapshots'), 0)}"
            ),
            f"Memory Queries Total    : {_safe_int(hist.get('memory_queries_total'), 0)}",
            f"Chat Queries Total      : {_safe_int(hist.get('chat_queries_total'), 0)}",
            (
                f"Chat Queries Hit/LLM    : {_safe_int(hist.get('chat_queries_hit'), 0)}/"
                f"{_safe_int(hist.get('chat_queries_llm'), 0)}"
            ),
            f"Summary Queries Total   : {_safe_int(hist.get('summary_queries_total'), 0)}",
            f"Behavior Interactions   : {_safe_int(hist.get('behavior_interactions_total'), 0)}",
            f"Behavior Attention (s)  : {_safe_float(hist.get('behavior_attention_total_sec'), 0.0):.1f}",
            f"Behavior Events         : {_safe_int(hist.get('behavior_events_count'), 0)}",
            f"Alert Density           : {_safe_float(hist.get('alert_density_per_min'), 0.0):.2f} alerts/min",
            f"Metrics Parse Errors    : {_metrics_parse_errors()}",
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


# ── Environment readiness ─────────────────────────────────────────────────────
# (status, name, detail) rows. "fail" items block a monitoring session; "warn"
# items disable one feature but let the pipeline run.
_REQUIRED_MODULES = ("numpy", "cv2", "PIL", "onnxruntime", "ultralytics", "torch")

# Modules whose absence disables exactly one capability rather than blocking a
# session. A missing or too-old entry here is reported as `warn`, never `fail`:
# the pipeline still runs, and `degraded_reason` names what is switched off.
_DEGRADABLE_MODULES = {"insightface": "face recognition"}

# Importable is not the same as usable: the pipeline calls APIs that only exist in
# recent releases, so a module can import cleanly and still break the first session.
# Only lower bounds are enforced here -- falling below a pinned minimum is a proven
# break, whereas exceeding an upper bound is a forward-looking risk.
_MIN_MODULE_VERSIONS: dict[str, tuple[tuple[int, ...], str]] = {
    "insightface": (INSIGHTFACE_MIN_VERSION, "FaceAnalysis(providers=...) requires 0.7+ (2.x supported)"),
    "numpy": ((1, 26), "pixi.toml pins numpy >=1.26,<3"),
    "torch": ((2, 5), "pixi.toml pins torch >=2.5,<3"),
    "ultralytics": ((8, 4), "pixi.toml pins ultralytics >=8.4,<9"),
    "onnxruntime": ((1, 17), "required for the InsightFace execution providers"),
    "PIL": ((11,), "pixi.toml pins pillow >=11,<12"),
}


def _parse_version(value: Any) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))
_OPTIONAL_MODULES = {
    "l2cs": "gaze estimation",
    "gdown": "gaze weight auto-download",
    "groq": "LLM chat fallback",
    "faiss": "vector scene search",
    "open_clip": "vector scene search",
    "dotenv": ".env loading",
    "fastapi": "API server",
    "uvicorn": "API server",
}


def _check_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"not importable ({type(exc).__name__})"
    version = str(getattr(module, "__version__", "") or "").strip()
    return True, version or "installed"


def _dir_summary(path: Path, pattern: str) -> str:
    if not path.exists():
        return f"{path} (absent)"
    try:
        count = sum(1 for _ in path.glob(pattern))
    except OSError:
        count = 0
    return f"{path} ({count} file(s))"


def collect_environment_report(check_camera: bool = False) -> list[tuple[str, str, str]]:
    """Describe what this machine can actually run. Pure apart from the optional
    camera probe, so it is safe to call from tests."""
    rows: list[tuple[str, str, str]] = []

    python_version = platform.python_version()
    pinned = python_version.startswith("3.11")
    rows.append(
        (
            "ok" if pinned else "warn",
            "Python",
            f"{python_version} (pixi workspace pins 3.11)"
            if not pinned
            else python_version,
        )
    )

    for name in (*_REQUIRED_MODULES, *_DEGRADABLE_MODULES):
        available, detail = _check_import(name)
        degradable = name in _DEGRADABLE_MODULES
        unavailable_status = "warn" if degradable else "fail"
        status = "ok" if available else unavailable_status
        minimum = _MIN_MODULE_VERSIONS.get(name)
        if available and minimum:
            parsed = _parse_version(detail)
            if parsed and parsed < minimum[0]:
                status = unavailable_status
                want = ".".join(str(part) for part in minimum[0])
                detail = f"{detail} is too old (need >= {want}: {minimum[1]})"
        if degradable:
            detail = f"{detail} - {_DEGRADABLE_MODULES[name]} disabled without it"
        rows.append((status, f"module:{name}", detail))

    for name, purpose in _OPTIONAL_MODULES.items():
        available, detail = _check_import(name)
        rows.append(
            ("ok" if available else "warn", f"module:{name}", f"{detail} - {purpose}")
        )

    general_path = _resolve_general_model_path(None)
    rows.append(
        (
            "ok" if Path(general_path).exists() else "warn",
            "general YOLO",
            f"{general_path} "
            + ("present" if Path(general_path).exists() else "missing (Ultralytics downloads it on first run)"),
        )
    )

    custom_path = _load_default_custom_model_path()
    rows.append(
        ("ok" if custom_path else "warn", "custom YOLO", custom_path or "not configured"),
    )

    gaze_weights = Path(GAZE_WEIGHTS_DEFAULT)
    rows.append(
        (
            "ok" if gaze_weights.exists() else "warn",
            "gaze weights",
            f"{gaze_weights} "
            + ("present" if gaze_weights.exists() else "missing (see `bootstrap`)"),
        )
    )

    db = FaceDB.load()
    rows.append(
        (
            "ok" if db.names else "warn",
            "enrolled identities",
            f"{len(db.names)} identity(ies) in {DB_PATH}"
            if db.names
            else f"none in {DB_PATH} - run `enroll --name <name>`",
        )
    )

    rows.append(("ok", "memory store", _dir_summary(MEMORY_DIR / "snapshots", "*.jpg")))
    rows.append(("ok", "incidents", _dir_summary(UNKNOWN_INCIDENTS_DIR, "*.jpg")))
    rows.append(
        (
            "ok",
            "retention caps",
            (
                f"auto snapshots {common.MEMORY_MAX_AUTO_SNAPSHOTS}, "
                f"incidents {common.UNKNOWN_INCIDENT_MAX_FILES}, "
                f"metrics {common.METRICS_MAX_BYTES} B x {common.METRICS_BACKUP_COUNT} backups"
            ),
        )
    )

    lock_backend = common.lock_backend_name()
    lock_timeouts = common.lock_timeouts()
    rows.append(
        (
            "ok" if lock_backend != "none" and lock_timeouts == 0 else "warn",
            "cross-process lock",
            f"backend {lock_backend}"
            + (
                ", no timeouts"
                if lock_timeouts == 0
                else f", {lock_timeouts} timeout(s): work proceeded without the lock"
            )
            + (
                ""
                if lock_backend != "none"
                else " - metrics and snapshot writes are only thread-safe here"
            ),
        )
    )

    rows.append(("ok", "camera source", f"{CAMERA_SOURCE} (from AI_STUDIO_CAM_CAMERA_INDEX)"))
    if check_camera:
        try:
            cap = _open_camera()
            cap.release()
            rows.append(("ok", "camera probe", "opened successfully"))
        except Exception as exc:
            rows.append(("fail", "camera probe", str(exc)))

    return rows


def cmd_doctor(check_camera: bool = False) -> None:
    rows = collect_environment_report(check_camera=check_camera)
    symbols = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}

    print("Environment readiness")
    print("-" * 72)
    for status, name, detail in rows:
        print(f"{symbols.get(status, '[ ?? ]')} {name:<20} {detail}")

    failures = [name for status, name, _ in rows if status == "fail"]
    warnings = [name for status, name, _ in rows if status == "warn"]
    print("-" * 72)
    print(f"{len(rows) - len(failures) - len(warnings)} ok, {len(warnings)} warning(s), {len(failures)} failure(s)")

    if failures:
        print("Blocking: " + ", ".join(failures))
        print("These must be installed before monitoring can start. See the README install steps.")
    elif warnings:
        print("The pipeline can start; each warning above disables one feature only.")
    else:
        print("Everything the pipeline needs is present.")

    if failures:
        sys.exit(1)


def cmd_bootstrap(download_gaze: bool = True) -> None:
    """Create runtime directories and fetch the model assets a session needs."""
    print("Bootstrapping runtime assets")
    print("-" * 72)

    for directory in (MEMORY_DIR / "snapshots", UNKNOWN_INCIDENTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[ ok ] directory  {directory}")

    problems: list[str] = []

    general_path = _resolve_general_model_path(None)
    if Path(general_path).exists():
        print(f"[ ok ] general YOLO already present: {general_path}")
    else:
        try:
            from ultralytics import YOLO

            YOLO("yolov8n.pt")
            print(f"[ ok ] general YOLO ready: {general_path}")
        except Exception as exc:
            problems.append(f"general YOLO weights unavailable ({exc})")
            print(f"[warn] general YOLO could not be prepared: {exc}")

    gaze_weights = Path(GAZE_WEIGHTS_DEFAULT)
    if gaze_weights.exists():
        print(f"[ ok ] gaze weights already present: {gaze_weights}")
    elif not download_gaze:
        print(f"[warn] gaze weights missing: {gaze_weights} (download skipped)")
    else:
        try:
            runtime = _load_gaze_runtime(
                gaze_arch=GAZE_ARCH_DEFAULT,
                gaze_weights=str(gaze_weights),
                gaze_weights_source=GAZE_WEIGHTS_SOURCE_DEFAULT,
                gaze_auto_download=True,
            )
        except Exception as exc:
            runtime = None
            print(f"[warn] gaze weights could not be downloaded: {exc}")
        if runtime is not None and Path(str(runtime.get("weights_path", gaze_weights))).exists():
            print(f"[ ok ] gaze weights ready: {runtime.get('weights_path')}")
        else:
            problems.append("gaze weights unavailable")
            print(
                "[warn] gaze weights still missing. Place L2CSNet_gaze360.pkl in models/ "
                "manually; gaze stays disabled until then."
            )

    db = FaceDB.load()
    if db.names:
        print(f"[ ok ] enrolled identities: {len(db.names)}")
    else:
        problems.append("no enrolled identities")
        print("[warn] no enrolled identities yet.")

    print("-" * 72)
    print("Next steps:")
    print("  1. pixi run python main.py doctor            # confirm the environment")
    print("  2. pixi run python main.py enroll --name <your name>")
    print("  3. pixi run python main.py recognize")

    if problems:
        print("Unresolved: " + "; ".join(problems))


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
    p_r.add_argument("--disable-gaze", action="store_true", help="Disable gaze prediction stream")
    p_r.add_argument(
        "--snapshot-interval",
        type=float,
        default=15.0,
        help="Automatic snapshot interval in seconds (default: 15)",
    )
    p_r.add_argument(
        "--gaze-arch",
        default=GAZE_ARCH_DEFAULT,
        choices=["ResNet50"],
        help=f"L2CS-Net backbone architecture (default: {GAZE_ARCH_DEFAULT})",
    )
    p_r.add_argument(
        "--gaze-weights",
        default=GAZE_WEIGHTS_DEFAULT,
        help=f"Path to L2CS-Net gaze weights (default: {GAZE_WEIGHTS_DEFAULT})",
    )
    p_r.add_argument(
        "--gaze-weights-source",
        default=GAZE_WEIGHTS_SOURCE_DEFAULT,
        help="Source URL used for auto-downloading gaze weights when local file is missing",
    )
    p_r.add_argument(
        "--gaze-max-interval",
        type=int,
        default=GAZE_INTERVAL_DEFAULT,
        help=(
            "Maximum frames between gaze inferences. 1 (default) keeps gaze at "
            "full rate so attention metrics stay per-frame; above 1 lets the "
            f"interval adapt to its own cost (recommended: {GAZE_MAX_INTERVAL_DEFAULT})"
        ),
    )
    p_r.add_argument(
        "--gaze-target-fps-drop",
        type=float,
        default=GAZE_TARGET_FPS_DROP_DEFAULT,
        help=(
            "Gaze cost as a fraction of the frame budget above which the adaptive "
            f"interval grows (default: {GAZE_TARGET_FPS_DROP_DEFAULT}; only used when "
            "--gaze-max-interval > 1)"
        ),
    )
    p_r.add_argument(
        "--disable-gaze-auto-download",
        action="store_true",
        help="Disable automatic gaze weight download fallback",
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
    p_mfp = sub.add_parser("memory-find-person", help="Find when a person was last seen")
    p_mfp.add_argument("--name", required=True, help="Person name")
    p_ms = sub.add_parser("memory-search", help="Search similar scenes")
    p_ms.add_argument("--text", required=True, help="Natural language scene query")
    p_ss = sub.add_parser("session-summary", help="Generate narrative summary of recent activity")
    p_ss.add_argument("--minutes", type=int, default=5, help="Lookback window in minutes (default: 5)")
    p_ss.add_argument("--json", action="store_true", help="Output raw summary JSON")
    p_chat = sub.add_parser("chat", help="Interactive chat over memory and logs")
    p_chat.add_argument("--question", default=None, help="Single-turn question (optional)")

    sub.add_parser("list", help="List enrolled identities")
    sub.add_parser("report", help="Generate metrics/report files")
    p_doc = sub.add_parser("doctor", help="Report which parts of the pipeline can run here")
    p_doc.add_argument(
        "--check-camera",
        action="store_true",
        help="Also probe the configured camera (opens and releases the device)",
    )
    p_boot = sub.add_parser("bootstrap", help="Create runtime directories and fetch model assets")
    p_boot.add_argument(
        "--no-gaze-download",
        action="store_true",
        help="Skip the L2CS gaze-weight download",
    )

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
            args.disable_gaze,
            args.gaze_arch,
            args.gaze_weights,
            args.gaze_weights_source,
            args.disable_gaze_auto_download,
            args.gaze_max_interval,
            args.gaze_target_fps_drop,
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
        "memory-find-person": lambda: cmd_memory_find_person(args.name),
        "memory-search": lambda: cmd_memory_search(args.text),
        "session-summary": lambda: cmd_session_summary(args.minutes, args.json),
        "chat": lambda: cmd_chat(args.question),
        "list": cmd_list,
        "report": cmd_report,
        "doctor": lambda: cmd_doctor(args.check_camera),
        "bootstrap": lambda: cmd_bootstrap(download_gaze=not args.no_gaze_download),
    }.get(args.cmd, parser.print_help)()


if __name__ == "__main__":
    main()
