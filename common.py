"""Helpers shared by the CLI core (``main.py``) and the API server (``server.py``).

This module deliberately imports nothing from the project and nothing from the
computer-vision stack, so ``server.py`` stays importable without insightface,
torch or ultralytics installed. It exists so the two entry points share:

* one JSONL lock and parse-error counter for the metrics log,
* one session-id format (the CLI used to drop the prefix entirely),
* one set of schema version numbers per record type.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bump a version when the shape of that record changes. Readers normalise older
# rows, so both entry points must agree on the number they write.
SESSION_SCHEMA_VERSION = 7
ENROLL_SCHEMA_VERSION = 4
OBJECT_TRAIN_SCHEMA_VERSION = 1

# Gaze inference scheduling, shared so the CLI and the API cannot disagree about
# which frames ran gaze. Adaptation is opt-in: at the default base/max of 1 every
# frame with faces runs gaze, so attention metrics stay per-frame rather than
# sampled. ``GAZE_MAX_INTERVAL_DEFAULT`` is only the recommended opt-in ceiling.
GAZE_INTERVAL_DEFAULT = 1
GAZE_MAX_INTERVAL_DEFAULT = 4
GAZE_TARGET_FPS_DROP_DEFAULT = 0.25
GAZE_RECOVERY_STREAK_MIN = 5

METRICS_FILENAME = "metrics_log.jsonl"


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


# ── Storage lifecycle ─────────────────────────────────────────────────────────
# The metrics log rotates by size so it cannot grow without bound, and rotated
# generations stay readable so reports keep their history. All three caps are
# overridable from the environment so a deployment can size them to its disk.
METRICS_MAX_BYTES = max(_env_int("AI_STUDIO_METRICS_MAX_BYTES", 4 * 1024 * 1024), 64 * 1024)
METRICS_BACKUP_COUNT = max(_env_int("AI_STUDIO_METRICS_BACKUPS", 2), 0)

# Auto snapshots are pruned to the newest N; manual snapshots are never pruned.
MEMORY_MAX_AUTO_SNAPSHOTS = max(_env_int("AI_STUDIO_MEMORY_MAX_AUTO_SNAPSHOTS", 5000), 0)

# Unknown-face incident captures are pruned to the newest N files.
UNKNOWN_INCIDENT_MAX_FILES = max(_env_int("AI_STUDIO_UNKNOWN_INCIDENT_MAX_FILES", 500), 0)

# ── Object-detection inference ────────────────────────────────────────────────
# The detector used to call `model(frame)` with *no* arguments, so every value
# below was unreachable and all tuning was guesswork. The defaults come from the
# offline benchmark (`main.py bench-detect`, see OBJECT_DETECTION_PLAN.md): on
# this project's own frames, imgsz=768 found 8 classes versus 5 at 640 with
# reference recall unchanged at 1.00, for ~150 ms versus ~90 ms per frame on a
# CPU-only box.
#
# Confidence and IoU stay at Ultralytics' defaults on purpose: lowering the
# global threshold adds low-confidence junk (surfboard/tie at 0.17), which is a
# per-class policy decision rather than a global one.
def normalize_imgsz(value: int | float) -> int:
    """Clamp an inference size to what YOLO can use: 320-1920, multiple of 32."""
    clamped = min(max(int(value), 320), 1920)
    return int(round(clamped / 32.0) * 32)


YOLO_CONF_DEFAULT = min(max(_env_float("AI_STUDIO_YOLO_CONF", 0.25), 0.01), 0.99)
YOLO_IOU_DEFAULT = min(max(_env_float("AI_STUDIO_YOLO_IOU", 0.7), 0.1), 0.95)
YOLO_IMGSZ_DEFAULT = normalize_imgsz(_env_int("AI_STUDIO_YOLO_IMGSZ", 768))
YOLO_MAX_DET_DEFAULT = min(max(_env_int("AI_STUDIO_YOLO_MAX_DET", 300), 1), 1000)

# ── Live pacing & instance capture ────────────────────────────────────────────
# The monitor loop targets at most FPS_CAP_DEFAULT frames per second; the cap is
# a ceiling, so a slow machine simply runs at whatever it can keep up with. The
# default is deliberately modest: every frame pays for two YOLO passes, face
# recognition and (sometimes) gaze, and the same machine also encodes the
# MJPEG stream for every connected dashboard.
#
# Auto snapshots — the "instances" that reports and chat ground on — run on
# their own wall-clock cadence (SNAPSHOT_INTERVAL_DEFAULT seconds), fully
# decoupled from FPS, so instance volume is tuned independently of processing
# speed. Enrollment keeps a faster cap because sample collection benefits.
FPS_CAP_DEFAULT = min(max(_env_int("AI_STUDIO_FPS_CAP", 12), 1), 60)
ENROLL_FPS_CAP_DEFAULT = min(max(_env_int("AI_STUDIO_ENROLL_FPS_CAP", 20), 1), 60)
SNAPSHOT_INTERVAL_DEFAULT = min(max(_env_float("AI_STUDIO_SNAPSHOT_INTERVAL", 8.0), 1.0), 3600.0)

# Single process-wide lock: the monitor worker thread and FastAPI request
# handlers both append to the metrics log, and rows can exceed the size where
# an unbuffered append stays atomic. This only serialises threads; see
# `process_lock` for the cross-process guarantee.
METRICS_LOCK = threading.Lock()
_METRICS_PARSE_ERRORS = 0

# ── Cross-process locking ─────────────────────────────────────────────────────
# `main.py` and `server.py` can run at the same time against the same working
# directory. Thread locks do not protect against that, so every read-merge-write
# cycle over a shared file also takes an OS-level advisory lock.
#
# The lock lives in a sidecar `<name>.lock` file rather than on the data file
# itself: writes replace the data file atomically via `os.replace`, so a handle
# to the old file would guard nothing.
_LOCK_TIMEOUT_SEC = 10.0
_LOCK_POLL_SEC = 0.05
_thread_state = threading.local()
# Counted rather than logged: a required lock that times out means work proceeded
# without the cross-process guarantee, which callers should be able to report on.
_LOCK_TIMEOUTS = 0


def lock_timeouts() -> int:
    return int(_LOCK_TIMEOUTS)


def _lock_backend() -> tuple[str, Any]:
    try:
        import fcntl

        return "posix", fcntl
    except ImportError:
        pass
    try:
        import msvcrt

        return "windows", msvcrt
    except ImportError:
        return "none", None


def lock_backend_name() -> str:
    """Which advisory-lock implementation is active on this platform."""
    return _lock_backend()[0]


def lock_path_for(path: Path | str) -> Path:
    return Path(f"{path}.lock")


def _try_lock(kind: str, module: Any, handle: Any) -> bool:
    try:
        if kind == "posix":
            module.flock(handle.fileno(), module.LOCK_EX | module.LOCK_NB)
        else:
            handle.seek(0)
            module.locking(handle.fileno(), module.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def _unlock(kind: str, module: Any, handle: Any) -> None:
    try:
        if kind == "posix":
            module.flock(handle.fileno(), module.LOCK_UN)
        else:
            handle.seek(0)
            module.locking(handle.fileno(), module.LK_UNLCK, 1)
    except OSError:
        pass


@contextmanager
def process_lock(
    path: Path | str,
    timeout: float = _LOCK_TIMEOUT_SEC,
    required: bool = True,
) -> Iterator[None]:
    """Serialise access to ``path`` across processes using a sidecar lock file.

    Re-entrant within the acquiring thread, and the depth is tracked so nesting
    inside one process never blocks on itself -- two separate ``flock`` calls in
    the same process would otherwise deadlock.

    With ``required=False`` the lock is best-effort: if it cannot be taken within
    ``timeout`` the work proceeds anyway. Readers use that so a slow writer can
    never stall a dashboard request. When no locking module exists (an exotic
    platform) this degrades to the caller's thread lock rather than failing.
    """
    lock_file = lock_path_for(path)
    try:
        key = str(Path(lock_file).absolute())
    except OSError:
        key = str(lock_file)

    depth: dict[str, int] = getattr(_thread_state, "depth", None) or {}
    _thread_state.depth = depth
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
            if depth[key] <= 0:
                depth.pop(key, None)
        return

    kind, module = _lock_backend()
    if kind == "none":
        yield
        return

    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        # The handle must stay open for the whole yield -- it *is* the lock -- so a
        # `with` block is not applicable here. It is closed in the finally below.
        handle = open(lock_file, "a+b")  # noqa: SIM115
    except OSError:
        # A read-only or missing directory must not turn a write into a crash.
        yield
        return

    acquired = False
    try:
        if kind == "windows":
            # `msvcrt.locking` locks a byte range, so the file needs a byte to lock.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()

        global _LOCK_TIMEOUTS
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while True:
            if _try_lock(kind, module, handle):
                acquired = True
                break
            if time.monotonic() >= deadline:
                if required:
                    # Degrade rather than hang: proceeding without the lock risks the
                    # race we are guarding against, but stalling monitoring forever
                    # is worse. The counter makes the degradation reportable.
                    _LOCK_TIMEOUTS += 1
                break
            time.sleep(_LOCK_POLL_SEC)

        depth[key] = 1
        try:
            yield
        finally:
            depth.pop(key, None)
    finally:
        if acquired:
            _unlock(kind, module, handle)
        handle.close()


def now_utc() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None = None) -> str:
    return (dt or now_utc()).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def session_id(prefix: str) -> str:
    """Correlation-friendly id shown in dashboards and logs."""
    stamp = now_utc().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def metrics_parse_errors() -> int:
    return int(_METRICS_PARSE_ERRORS)


def metric_generations(path: Path | str) -> list[Path]:
    """Existing generations of a rotating JSONL log, oldest first."""
    path = Path(path)
    generations = [
        path.with_name(f"{path.name}.{index}")
        for index in range(METRICS_BACKUP_COUNT, 0, -1)
    ]
    return [candidate for candidate in generations if candidate.exists()] + [path]


def _rotate_metrics(path: Path) -> None:
    """Shift generations once the active log exceeds its size cap.

    Called while holding ``METRICS_LOCK``. Rotation is best-effort: a reader can
    hold the file open (notably on Windows, where that blocks the rename), and
    losing a rotation is far better than failing the append. The log then simply
    keeps growing until a later append succeeds in rotating it.
    """
    try:
        if not path.exists() or path.stat().st_size < METRICS_MAX_BYTES:
            return
        if METRICS_BACKUP_COUNT == 0:
            path.unlink(missing_ok=True)
            return
        oldest = path.with_name(f"{path.name}.{METRICS_BACKUP_COUNT}")
        oldest.unlink(missing_ok=True)
        for index in range(METRICS_BACKUP_COUNT - 1, 0, -1):
            source = path.with_name(f"{path.name}.{index}")
            if source.exists():
                os.replace(source, path.with_name(f"{path.name}.{index + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        return


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line, serialised across threads."""
    path = Path(path)
    line = json.dumps(record, ensure_ascii=True) + "\n"
    with METRICS_LOCK:
        with process_lock(path):
            _rotate_metrics(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()


def read_jsonl(path: Path | str, include_backups: bool = True) -> list[dict[str, Any]]:
    """Read newline-delimited JSON, counting unreadable rows instead of hiding them.

    Rotated generations are read oldest first so reports keep their history, which
    also makes the read bounded by ``(METRICS_BACKUP_COUNT + 1) * METRICS_MAX_BYTES``.
    """
    global _METRICS_PARSE_ERRORS
    path = Path(path)

    rows: list[dict[str, Any]] = []
    # Take the lock for the listing and the read so rotation cannot move files out
    # from under us. Best-effort plus a short timeout: a reader must never hang a
    # request behind a slow writer.
    with process_lock(path, timeout=2.0, required=False):
        sources = metric_generations(path) if include_backups else [path]
        for source in sources:
            try:
                handle = source.open("r", encoding="utf-8")
            except OSError:
                # A concurrent rotation may have moved it between listing and opening.
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # Surface the count through `report` rather than discarding silently.
                        _METRICS_PARSE_ERRORS += 1
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
    return rows


def filter_events(
    rows: list[dict[str, Any]],
    *,
    event_type: str | None,
    limit: int,
    from_ts: str | None,
    to_ts: str | None,
) -> list[dict[str, Any]]:
    from_dt = parse_iso(from_ts)
    to_dt = parse_iso(to_ts)
    out: list[dict[str, Any]] = []
    for row in rows:
        if event_type and str(row.get("event_type")) != event_type:
            continue
        dt = parse_iso(row.get("timestamp_utc"))
        if from_dt and dt and dt < from_dt:
            continue
        if to_dt and dt and dt > to_dt:
            continue
        out.append(row)
    return out[-max(1, int(limit)) :]
