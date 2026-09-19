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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bump a version when the shape of that record changes. Readers normalise older
# rows, so both entry points must agree on the number they write.
SESSION_SCHEMA_VERSION = 6
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

# Single process-wide lock: the monitor worker thread and FastAPI request
# handlers both append to the metrics log, and rows can exceed the size where
# an unbuffered append stays atomic.
METRICS_LOCK = threading.Lock()
_METRICS_PARSE_ERRORS = 0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


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
        return dt.replace(tzinfo=timezone.utc)
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


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    """Append one JSON object as a single line, serialised across threads."""
    line = json.dumps(record, ensure_ascii=True) + "\n"
    with METRICS_LOCK:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read newline-delimited JSON, counting unreadable rows instead of hiding them."""
    global _METRICS_PARSE_ERRORS
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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
