"""Offline benchmark for the object detectors.

Why this exists: the live loop can only tell you *that* something was or was not
detected, in a scene nobody can replay. This module makes detection quality
measurable and repeatable offline — a fixed frame set, explicit parameters, and
per-class results — so every tuning change (thresholds, inference size, model)
can be justified with numbers instead of impressions.

Layering:

* Pure helpers (frame scoring, aggregation, reference matching, reporting) are
  importable without ``cv2`` or ``ultralytics``, so the unit tests need neither.
* :func:`load_frames` and :func:`run_benchmark` are the only heavy entry points
  and they import their dependencies lazily.

Reference images ship with Ultralytics and have labels verified on the machine
this was written on; they guard against the detector silently degrading, while
the project's own saved frames measure real-scene recall.
"""

from __future__ import annotations

import glob
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import common
from object_detection import DualYoloDetector

#: Labels present in the reference images shipped with Ultralytics. Counts are the
#: expected *minimum* detections (a stronger model may legitimately find more).
REFERENCE_LABELS: dict[str, dict[str, int]] = {
    "bus.jpg": {"bus": 1, "person": 4, "stop sign": 1},
    "zidane.jpg": {"person": 2, "tie": 1},
}

#: Frames below these thresholds cannot be detected in by any model, so they are
#: reported separately instead of dragging the recall numbers down.
DEFAULT_MIN_BRIGHTNESS = 90.0
DEFAULT_MIN_SHARPNESS = 50.0
DEFAULT_MIN_HEIGHT = 200


@dataclass(frozen=True)
class FrameQuality:
    """Why a frame was or was not included in the benchmark."""

    path: str
    height: int
    brightness: float
    sharpness: float
    usable: bool
    reason: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True)
class DetectorParams:
    """One point in the parameter space being compared."""

    conf: float = 0.25
    imgsz: int = 640
    iou: float = 0.7
    max_det: int = 300

    @property
    def label(self) -> str:
        return f"conf={self.conf:g} imgsz={self.imgsz}"

    def as_kwargs(self) -> dict[str, Any]:
        return {"conf": self.conf, "imgsz": self.imgsz, "iou": self.iou, "max_det": self.max_det}

    @classmethod
    def configured(cls) -> DetectorParams:
        """The parameters the application itself will use (env knobs included)."""
        return cls(
            conf=common.YOLO_CONF_DEFAULT,
            imgsz=common.YOLO_IMGSZ_DEFAULT,
            iou=common.YOLO_IOU_DEFAULT,
            max_det=common.YOLO_MAX_DET_DEFAULT,
        )


def score_image(gray: np.ndarray) -> tuple[float, float]:
    """Return ``(mean brightness, Laplacian variance)`` for a grayscale image.

    Variance of the Laplacian is the standard cheap focus measure: a black or
    uniform frame scores ~0, a well-lit textured scene scores in the hundreds.
    """
    import cv2

    if gray.size == 0:
        return 0.0, 0.0
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_frame(
    path: str,
    *,
    height: int,
    brightness: float,
    sharpness: float,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> FrameQuality:
    """Decide whether a frame is worth benchmarking, and say why when it is not."""
    if height < min_height:
        return FrameQuality(path, height, brightness, sharpness, False, "too_small")
    if brightness < min_brightness:
        return FrameQuality(path, height, brightness, sharpness, False, "too_dark")
    if sharpness < min_sharpness:
        return FrameQuality(path, height, brightness, sharpness, False, "blurred")
    return FrameQuality(path, height, brightness, sharpness, True, "")


def summarize_detections(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate detection rows into ``label -> count/max_conf/mean_conf``."""
    counts: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        counts.setdefault(label, []).append(float(row.get("confidence", 0.0)))

    summary: dict[str, dict[str, float]] = {}
    for label, scores in counts.items():
        summary[label] = {
            "count": float(len(scores)),
            "max_conf": round(max(scores), 3),
            "mean_conf": round(sum(scores) / len(scores), 3),
        }
    return dict(sorted(summary.items(), key=lambda kv: (-kv[1]["count"], kv[0])))


def match_reference(
    found: dict[str, float], expected: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Compare per-class counts against the reference labels.

    A label counts as recalled when at least the expected number of instances was
    found; extra instances are reported but never a failure (models legitimately
    discover more than the hand-written reference).
    """
    detail: dict[str, dict[str, float]] = {}
    for label, want in expected.items():
        got = float(found.get(label, 0.0))
        detail[label] = {
            "expected": float(want),
            "found": got,
            "recalled": 1.0 if got >= float(want) else 0.0,
        }
    return detail


def recall_of(matches: dict[str, dict[str, float]]) -> float:
    """Fraction of reference labels recalled across all reference images."""
    if not matches:
        return 0.0
    return round(sum(row["recalled"] for row in matches.values()) / len(matches), 3)


def default_param_grid(
    confs: Sequence[float] | None = None, sizes: Sequence[int] | None = None
) -> list[DetectorParams]:
    """The comparison grid used when the caller does not specify one.

    With no arguments this measures the *shipping* configuration against the
    historical 640/0.25 baseline, so the default output answers "did the change
    we just made actually help?" rather than an abstract sweep.
    """
    if not confs and not sizes:
        configured = DetectorParams.configured()
        baseline = DetectorParams(conf=0.25, imgsz=640)
        if configured.conf == baseline.conf and configured.imgsz == baseline.imgsz:
            return [configured]
        return [configured, baseline]

    conf_values = list(confs) if confs else [0.25, 0.15]
    size_values = list(sizes) if sizes else [640, 960]
    return [DetectorParams(conf=c, imgsz=s) for s in size_values for c in conf_values]


def load_frames(
    patterns: Sequence[str],
    *,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    min_height: int = DEFAULT_MIN_HEIGHT,
    limit: int | None = None,
) -> tuple[list[tuple[str, np.ndarray]], list[FrameQuality]]:
    """Load frames matching ``patterns``, split into usable and rejected ones."""
    import cv2

    # Preserve first-seen order while de-duplicating overlapping patterns.
    unique: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                seen.add(path)
                unique.append(path)

    usable: list[tuple[str, np.ndarray]] = []
    rejected: list[FrameQuality] = []
    for path in unique:
        frame = cv2.imread(path)
        if frame is None:
            rejected.append(FrameQuality(path, 0, 0.0, 0.0, False, "unreadable"))
            continue
        height = int(frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness, sharpness = score_image(gray)
        quality = classify_frame(
            path,
            height=height,
            brightness=brightness,
            sharpness=sharpness,
            min_brightness=min_brightness,
            min_sharpness=min_sharpness,
            min_height=min_height,
        )
        if quality.usable:
            usable.append((path, frame))
        else:
            rejected.append(quality)
        if limit is not None and len(usable) >= limit:
            break
    return usable, rejected


def reference_assets() -> list[tuple[str, str, dict[str, int]]]:
    """Locate the bundled reference images: ``(name, path, expected labels)``."""
    try:
        import ultralytics
    except Exception:
        return []

    assets = Path(ultralytics.__file__).parent / "assets"
    found: list[tuple[str, str, dict[str, int]]] = []
    for name, labels in REFERENCE_LABELS.items():
        path = assets / name
        if path.exists():
            found.append((name, str(path), labels))
    return found


def run_benchmark(
    model_path: str,
    params_list: Sequence[DetectorParams],
    frames: Sequence[tuple[str, np.ndarray]] = (),
    references: Sequence[tuple[str, str, dict[str, int]]] = (),
) -> list[dict[str, Any]]:
    """Run the detector over every parameter set and collect per-class results.

    This drives the real :class:`DualYoloDetector`, not raw Ultralytics, so the
    numbers describe the production path — including its parameter resolution.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    detector = DualYoloDetector(general_model_obj=model, enable_custom=False)

    results: list[dict[str, Any]] = []
    for params in params_list:
        detector.configure(**params.as_kwargs())
        started = time.perf_counter()
        projected: list[dict[str, Any]] = []
        for _path, frame in frames:
            projected.extend(detector.detect(frame))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        reference_detail: dict[str, dict[str, float]] = {}
        for name, path, labels in references:
            image = cv2.imread(path)
            if image is None:
                continue
            per_class: dict[str, float] = {}
            for row in detector.detect(image):
                label = str(row["label"])
                per_class[label] = per_class.get(label, 0.0) + 1.0
            for label, detail in match_reference(per_class, labels).items():
                reference_detail[f"{name}:{label}"] = detail

        results.append(
            {
                "params": detector.params(),
                "label": params.label,
                "ms_per_frame": round(elapsed_ms / max(len(frames), 1), 1),
                "classes": summarize_detections(projected),
                "frames": len(frames),
                "reference": reference_detail,
                "reference_recall": recall_of(reference_detail),
            }
        )
    return results


def format_report(results: Sequence[dict[str, Any]], rejected: Sequence[FrameQuality] = ()) -> str:
    """Render benchmark results as a plain-text table for the terminal."""
    lines: list[str] = []
    if rejected:
        reasons: dict[str, int] = {}
        for row in rejected:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
        summary = ", ".join(f"{name}={count}" for name, count in sorted(reasons.items()))
        lines.append(f"excluded frames: {len(rejected)} ({summary})")
        lines.append("")

    for entry in results:
        classes = entry.get("classes", {})
        lines.append(
            f"{entry.get('label', '?'):<28} {entry.get('ms_per_frame', 0):>7.0f} ms/frame   "
            f"{len(classes)} classes   reference recall={entry.get('reference_recall', 0):.2f}"
        )
        if classes:
            rendered = "  ".join(
                f"{label} x{int(row['count'])} (max {row['max_conf']:.2f})"
                for label, row in list(classes.items())[:8]
            )
            lines.append(f"    {rendered}")
        else:
            lines.append("    (no detections)")
    return "\n".join(lines)
