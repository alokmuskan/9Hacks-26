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

The parameter space is described by :class:`object_detection.DetectorConfig` — the
same type the live detector uses — rather than by a local copy of the fields, so a
benchmark result always describes a configuration the application could actually
run.

Reference images ship with Ultralytics and have labels verified on the machine
this was written on; they guard against the detector silently degrading, while
the project's own saved frames measure real-scene recall.

Nothing here measures **precision**, because almost nothing here is labelled. The
reference images have known labels, so recall against them is real. The project's
own frames have no ground truth whatsoever, so what :func:`summarize_per_frame`
reports for them is *detection statistics* — counts, rates and frame coverage. Those
can expose a false-positive flood, which a bare box count cannot; they cannot call
any individual detection wrong. The two must not be presented as if they were the
same kind of number.

Nor does anything here measure **end-to-end** latency. :func:`latency_stats` times
object detection alone, on saved frames, in a process that is doing nothing else. The
live loop additionally pays capture, face recognition, gaze, memory writes and
snapshot encoding, and those costs do not simply add because they contend for the same
CPU — see OBJECT_DETECTION_PLAN.md §2f. Detection is roughly a sixth of a frame.
"""

from __future__ import annotations

import glob
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from object_detection import DetectorConfig, DualYoloDetector

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


def summarize_per_frame(
    per_frame: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Detection statistics per class, counted in *frames* rather than boxes.

    :func:`summarize_detections` answers "how many boxes of each class?", which
    cannot separate a real object found once per frame from a false positive fired
    in every frame — both simply raise the count. Two numbers make that visible:
    how many frames contained the class at all, and the most boxes of it seen in
    any single frame.

    These are **detection statistics, not precision**. Precision needs to know
    which detections were wrong, and the project's own frames carry no labels: a
    class at 0.9 boxes per frame may be an object genuinely present in almost every
    frame, or a systematic false positive on some texture. Only hand-labelled
    frames can separate those, so this reports rates and coverage and never claims
    a precision figure.
    """
    frame_count = len(per_frame)
    totals: dict[str, int] = {}
    frame_hits: dict[str, int] = {}
    busiest_frame: dict[str, int] = {}
    confidences: dict[str, list[float]] = {}
    frames_with_any = 0

    for rows in per_frame:
        seen: dict[str, int] = {}
        for row in rows:
            label = str(row.get("label", "")).strip()
            if not label:
                continue
            seen[label] = seen.get(label, 0) + 1
            confidences.setdefault(label, []).append(float(row.get("confidence", 0.0)))
        if seen:
            frames_with_any += 1
        for label, count in seen.items():
            totals[label] = totals.get(label, 0) + count
            frame_hits[label] = frame_hits.get(label, 0) + 1
            busiest_frame[label] = max(busiest_frame.get(label, 0), count)

    per_class: dict[str, dict[str, float]] = {}
    for label, total in totals.items():
        scores = confidences[label]
        per_class[label] = {
            "detections": float(total),
            "frames_with_detection": float(frame_hits[label]),
            "frame_coverage": round(frame_hits[label] / frame_count, 3) if frame_count else 0.0,
            "detections_per_frame": round(total / frame_count, 3) if frame_count else 0.0,
            "max_in_one_frame": float(busiest_frame[label]),
            "max_conf": round(max(scores), 3),
            "mean_conf": round(sum(scores) / len(scores), 3),
        }

    total_detections = sum(totals.values())
    return {
        "frames": float(frame_count),
        "frames_with_any_detection": float(frames_with_any),
        "detections": float(total_detections),
        "detections_per_frame": round(total_detections / frame_count, 3) if frame_count else 0.0,
        # Busiest first: a flood is a high rate, so it should be the first line read.
        "per_class": dict(
            sorted(per_class.items(), key=lambda kv: (-kv[1]["detections_per_frame"], kv[0]))
        ),
    }


def latency_stats(frame_ms: Sequence[float]) -> dict[str, float]:
    """Summarise a per-frame timing series: mean, median, min, max, and warm-up.

    A single mean cannot answer the questions actually being asked of it. §2e
    measured the *same* configuration at 438 ms and then 503 ms — a 15% run-to-run
    move, larger than several of the differences being compared — and a mean cannot
    tell that kind of drift apart from a real difference. The median and the spread
    can.

    The first frame of a run pays one-off warm-up (lazy kernel selection, allocator
    growth, the first inference through a freshly loaded graph) and is far slower
    than the rest. Silently dropping it would flatter every number, so it is **kept
    in the mean** and reported separately as `first_frame_ms` alongside
    `mean_after_first_ms`. That way the size of the effect is visible instead of
    being assumed away.

    These are **detection-only** figures, measured on saved frames in one process.
    They are not end-to-end frame latency and must not be presented as such: see
    OBJECT_DETECTION_PLAN.md §2f for what the live loop adds (gaze alone is ~433 ms
    per call) and why the costs do not simply add under CPU contention.
    """
    values = [float(value) for value in frame_ms]
    if not values:
        return dict.fromkeys(
            (
                "frames",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "first_frame_ms",
                "mean_after_first_ms",
            ),
            0.0,
        )

    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    rest = values[1:]
    return {
        "frames": float(len(values)),
        "mean_ms": round(sum(values) / len(values), 1),
        "median_ms": round(median, 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "first_frame_ms": round(values[0], 1),
        "mean_after_first_ms": round(sum(rest) / len(rest), 1) if rest else round(values[0], 1),
    }


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
) -> list[DetectorConfig]:
    """The comparison grid used when the caller does not specify one.

    With no arguments this measures the *shipping* configuration against the
    historical 640/0.25 baseline, so the default output answers "did the change
    we just made actually help?" rather than an abstract sweep.
    """
    if not confs and not sizes:
        configured = DetectorConfig.configured()
        baseline = DetectorConfig(conf=0.25, imgsz=640)
        if configured.conf == baseline.conf and configured.imgsz == baseline.imgsz:
            return [configured]
        return [configured, baseline]

    conf_values = list(confs) if confs else [0.25, 0.15]
    size_values = list(sizes) if sizes else [640, 960]
    return [DetectorConfig(conf=c, imgsz=s) for s in size_values for c in conf_values]


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
    params_list: Sequence[DetectorConfig],
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
        # Kept per frame rather than flattened: a box count cannot distinguish one
        # object seen 45 times from 45 boxes in a single frame.
        per_frame: list[list[dict[str, Any]]] = []
        # Timed per frame as well as in total: one number around the whole loop
        # hides both the spread and the warm-up frame, and §2e watched a single
        # mean move 15% between identical runs with no way to see why.
        frame_ms: list[float] = []
        for _path, frame in frames:
            started = time.perf_counter()
            per_frame.append(detector.detect(frame))
            frame_ms.append((time.perf_counter() - started) * 1000.0)
        projected = [row for rows in per_frame for row in rows]
        latency = latency_stats(frame_ms)

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
                # The same figure as latency["mean_ms"], read from one place so the
                # two cannot drift apart.
                "ms_per_frame": latency["mean_ms"],
                "latency": latency,
                "classes": summarize_detections(projected),
                "detection_stats": summarize_per_frame(per_frame),
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

        # One mean cannot show drift: §2e measured the same configuration at 438 ms
        # and 503 ms on identical input. The spread is what says whether a difference
        # between two models is real, and the warm-up frame is named rather than
        # quietly dropped so the mean above can be read for what it is.
        latency = entry.get("latency") or {}
        if latency.get("frames"):
            lines.append(
                f"    {latency['median_ms']:.0f} median, {latency['min_ms']:.0f} min, "
                f"{latency['max_ms']:.0f} max over {int(latency['frames'])} frames"
                "   (detection only, not end-to-end)"
            )
            if latency["frames"] > 1:
                lines.append(
                    f"    first frame {latency['first_frame_ms']:.0f} ms vs "
                    f"{latency['mean_after_first_ms']:.0f} ms mean for the rest "
                    "(warm-up is included in the mean above, not dropped)"
                )

        if classes:
            rendered = "  ".join(
                f"{label} x{int(row['count'])} (max {row['max_conf']:.2f})"
                for label, row in list(classes.items())[:8]
            )
            lines.append(f"    {rendered}")
        else:
            lines.append("    (no detections)")

        # A box count cannot show a flood: "surfboard x45" reads the same whether
        # that is 45 real objects or one false positive firing in most frames.
        # Rates and frame coverage can, and they are statistics, not precision.
        stats = entry.get("detection_stats") or {}
        if stats.get("per_class"):
            frames = int(stats["frames"])
            lines.append(
                f"    {stats['detections_per_frame']:.2f} boxes/frame over {frames} frames, "
                f"{int(stats['frames_with_any_detection'])} with >=1 detection"
                "   (detection statistics, not precision - no labels)"
            )
            for label, row in stats["per_class"].items():
                lines.append(
                    f"      {label:<16} {row['detections_per_frame']:>5.2f}/frame  "
                    f"{int(row['frames_with_detection']):>3}/{frames} frames  "
                    f"max {int(row['max_in_one_frame'])}/frame"
                )

        # A bare "recall=0.80" hides which label actually failed, which is the one
        # thing the reader needs: four of five labels can be recalled by a model
        # that has plainly regressed on the fifth.
        missed = [
            name for name, row in entry.get("reference", {}).items() if not row.get("recalled")
        ]
        if missed:
            lines.append(f"    missed reference: {', '.join(missed)}")
    return "\n".join(lines)
