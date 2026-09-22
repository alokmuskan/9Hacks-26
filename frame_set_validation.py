"""Validate a labelled frame set before anything is measured on it.

`OBJECT_DETECTION_FRAME_SET_SPEC.md` states what the replacement frame set must
contain. A specification that can only be checked by eye is one that gets skimmed,
and every failure it guards against is a silent one:

* A label file that was never written reads to the benchmark as "no objects in this
  frame", so every detection in that frame scores as a false positive.
* A class spelled `phone` instead of `cell phone` matches nothing, reports precision
  0, and prints no error.
* Frames below the quality floor are dropped by `detection_bench.load_frames`, so the
  benchmark reports healthy numbers on whatever happened to survive — which is how a
  dark-state failure mode disappears from its own evaluation.

None of those announce themselves. This module turns the checkable half of the spec
into a command that does.

It validates the **dataset**, never the detector. Nothing here measures precision,
recall or accuracy, and none of it is Phase 3 policy: it asserts only that the frames
and labels are well-formed enough for a metric to mean something once one exists.

Layering follows `detection_bench`: the parsing and checking helpers are pure and
importable without ``cv2``, and :func:`probe_frame` is the one heavy function, imported
lazily. Image probing is injectable so tests need neither OpenCV nor real images.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from detection_bench import (
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MIN_HEIGHT,
    DEFAULT_MIN_SHARPNESS,
    score_image,
)

#: COCO's 80 class names in index order — the vocabulary `yolov8n` emits, and
#: therefore the only names a label can legitimately use. Anything outside this list
#: cannot be detected by any model benchmarked in OBJECT_DETECTION_PLAN.md §2e, so a
#: label naming it is either a typo or a sign the target needs fine-tuning instead.
COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Lighting states the capture spec defines. Enforced strictly rather than freely,
#: because spec §5 hangs on them: the dark frames are exactly the ones the quality
#: filter removes, so they have to be identifiable in order to be evaluated apart.
LIGHTING_STATES = ("lit", "dim", "dark")

#: ``<timestamp>_<venue>_<lighting>_<distance>_<angle>_<NNN>`` — spec §7.
NAME_FIELD_COUNT = 6
MIN_SEQUENCE_DIGITS = 3

#: A set with no target-free frames cannot measure precision at all: every detection
#: in it is correct by construction, so precision comes out at 1.00 and means nothing.
#: This is the single most important thing the validator exists to catch.
RECOMMENDED_NEGATIVE_FRACTION = 0.33

#: Boxes may sit a hair outside the frame from labelling noise; beyond this the label
#: is describing something the detector will never be shown (spec §6c: partial objects
#: are labelled to their *visible* extent).
EDGE_TOLERANCE = 0.01

#: ``cx cy w h``. The class name is everything *before* these, not the first token:
#: fifteen COCO names contain a space ("cell phone", "stop sign", "hair drier"), and
#: splitting on whitespace and taking token 0 would reject every one of them.
COORD_FIELD_COUNT = 4

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: ``.txt`` files that belong to the frame set but are not labels. In the flat
#: layout these sit beside the labels and would otherwise be reported as orphans,
#: which is a confusing way to be told that `targets.txt` is not a label file.
RESERVED_TXT_NAMES = frozenset({"targets.txt", "provenance.txt"})


@dataclass(frozen=True)
class Finding:
    """One problem, with enough context to act on without re-deriving it."""

    severity: str
    code: str
    subject: str
    detail: str

    @property
    def is_error(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def render(self) -> str:
        return f"[{self.severity:<7}] {self.code:<22} {self.subject}: {self.detail}"


@dataclass(frozen=True)
class FrameName:
    """A capture filename decomposed into the fields spec §7 requires."""

    stem: str
    timestamp: str
    venue: str
    lighting: str
    distance: str
    angle: str
    index: str


@dataclass(frozen=True)
class LabelRow:
    """One labelled instance, in normalised YOLO form."""

    line_no: int
    cls: str
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class FrameSetReport:
    """Everything the validator found, plus what it counted on the way."""

    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [row for row in self.findings if row.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [row for row in self.findings if row.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


def vocabulary() -> dict[int, str]:
    """The default class vocabulary: COCO-80, keyed by index."""
    return dict(enumerate(COCO_CLASSES))


def load_vocabulary(path: str | Path) -> dict[int, str]:
    """Read a vocabulary override: one class name per line, index = line order.

    A fine-tuned model's names will not be COCO's, so the vocabulary has to be
    overridable — but it must still be an explicit, checkable list rather than
    whatever the labels happen to contain, or the check becomes circular.
    """
    names = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return dict(enumerate(names))


def load_targets(path: str | Path) -> list[str]:
    """Read the in-scope class list: one exact model class name per line."""
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_frame_name(path: str | Path) -> tuple[FrameName | None, str]:
    """Split a capture filename into its documented fields.

    Returns ``(name, "")`` on success and ``(None, reason)`` on failure, so a frame
    that does not fit the convention is reported with *why* rather than merely
    counted — the metadata in the name is what makes the coverage matrix checkable
    without opening every image.
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) != NAME_FIELD_COUNT:
        return None, (
            f"expected {NAME_FIELD_COUNT} underscore-separated fields "
            "(<timestamp>_<venue>_<lighting>_<distance>_<angle>_<NNN>), "
            f"found {len(parts)}"
        )

    timestamp, venue, lighting, distance, angle, index = parts
    for field_name, value in (
        ("timestamp", timestamp),
        ("venue", venue),
        ("distance", distance),
        ("angle", angle),
    ):
        if not value:
            return None, f"{field_name} is empty"
    if lighting not in LIGHTING_STATES:
        return None, f"lighting {lighting!r} is not one of {', '.join(LIGHTING_STATES)}"
    if len(index) < MIN_SEQUENCE_DIGITS or not index.isdigit():
        return None, f"sequence {index!r} is not at least {MIN_SEQUENCE_DIGITS} digits"

    return FrameName(stem, timestamp, venue, lighting, distance, angle, index), ""


def parse_label_line(
    line: str, *, line_no: int, subject: str, vocab: Mapping[int, str]
) -> tuple[LabelRow | None, Finding | None]:
    """Parse one ``class cx cy w h`` row, or explain why it cannot be used."""
    tokens = line.split()
    if len(tokens) <= COORD_FIELD_COUNT:
        return None, Finding(
            SEVERITY_ERROR,
            "malformed-label",
            subject,
            f"line {line_no}: expected a class followed by {COORD_FIELD_COUNT} "
            f"coordinates, found {len(tokens)} field(s)",
        )

    # Class names may contain spaces, so the coordinates are taken from the end and
    # whatever remains in front of them is the name.
    raw_class = " ".join(tokens[:-COORD_FIELD_COUNT])
    coordinates = tokens[-COORD_FIELD_COUNT:]

    if raw_class.lstrip("-").isdigit():
        index = int(raw_class)
        if index not in vocab:
            return None, Finding(
                SEVERITY_ERROR,
                "unknown-class-index",
                subject,
                f"line {line_no}: class index {index} is outside the {len(vocab)}-class vocabulary",
            )
        cls = vocab[index]
    else:
        if raw_class not in set(vocab.values()):
            return None, Finding(
                SEVERITY_ERROR,
                "unknown-class",
                subject,
                f"line {line_no}: {raw_class!r} is not in the model vocabulary "
                "(exact spelling required - 'phone' does not match 'cell phone')",
            )
        cls = raw_class

    try:
        cx, cy, width, height = (float(value) for value in coordinates)
    except ValueError:
        return None, Finding(
            SEVERITY_ERROR,
            "malformed-label",
            subject,
            f"line {line_no}: coordinates are not numbers: {' '.join(coordinates)}",
        )

    if not all(value == value and abs(value) != float("inf") for value in (cx, cy, width, height)):
        return None, Finding(
            SEVERITY_ERROR, "malformed-label", subject, f"line {line_no}: non-finite coordinate"
        )

    out_of_range = [
        name
        for name, value in (("cx", cx), ("cy", cy), ("w", width), ("h", height))
        if not 0.0 <= value <= 1.0
    ]
    if out_of_range:
        return None, Finding(
            SEVERITY_ERROR,
            "unnormalised-box",
            subject,
            f"line {line_no}: {', '.join(out_of_range)} outside 0-1 "
            "(coordinates must be normalised, not pixels)",
        )
    if width <= 0.0 or height <= 0.0:
        return None, Finding(
            SEVERITY_ERROR,
            "degenerate-box",
            subject,
            f"line {line_no}: {cls} has zero-area box ({width} x {height})",
        )

    row = LabelRow(line_no, cls, cx, cy, width, height)
    if (
        cx - width / 2 < -EDGE_TOLERANCE
        or cx + width / 2 > 1 + EDGE_TOLERANCE
        or cy - height / 2 < -EDGE_TOLERANCE
        or cy + height / 2 > 1 + EDGE_TOLERANCE
    ):
        return row, Finding(
            SEVERITY_WARNING,
            "box-past-edge",
            subject,
            f"line {line_no}: {cls} extends past the frame edge "
            "(spec §6c labels partial objects to their visible extent only)",
        )
    return row, None


def parse_labels(
    text: str, *, subject: str, vocab: Mapping[int, str]
) -> tuple[list[LabelRow], list[Finding]]:
    """Parse a whole label file, collecting every problem rather than the first."""
    rows: list[LabelRow] = []
    findings: list[Finding] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row, finding = parse_label_line(line, line_no=line_no, subject=subject, vocab=vocab)
        if finding is not None:
            findings.append(finding)
        if row is not None:
            rows.append(row)
    return rows, findings


def probe_frame(path: Path) -> tuple[int, float, float] | None:
    """``(height, brightness, sharpness)`` for an image, or ``None`` if unreadable.

    Uses the benchmark's own scoring, so a frame the validator passes is a frame
    `detection_bench.load_frames` will actually keep.
    """
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness, sharpness = score_image(gray)
    return int(frame.shape[0]), brightness, sharpness


def _frame_dirs(root: Path) -> tuple[Path, Path]:
    """Locate the images and labels directories inside a frame-set root.

    Spec §7 lays the set out as ``images/`` + ``labels/``. A flat directory with the
    two side by side is also accepted, because that is what most labelling tools
    write and refusing it would only encourage people to copy files around.
    """
    images = root / "images"
    labels = root / "labels"
    if images.is_dir() and labels.is_dir():
        return images, labels
    return root, root


def validate_frame_set(
    root: str | Path,
    *,
    targets: Sequence[str] = (),
    vocab: Mapping[int, str] | None = None,
    min_height: int = DEFAULT_MIN_HEIGHT,
    min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    probe: Callable[[Path], tuple[int, float, float] | None] | None = None,
) -> FrameSetReport:
    """Check a frame set against the mechanical half of the capture spec.

    ``probe`` is injectable so the checks can be tested without OpenCV or real
    images; it defaults to :func:`probe_frame`.
    """
    root_path = Path(root)
    report = FrameSetReport()
    classes = dict(vocab) if vocab is not None else vocabulary()
    known_names = set(classes.values())
    target_set = set(targets)
    read_probe = probe or probe_frame

    if not target_set:
        report.findings.append(
            Finding(
                SEVERITY_WARNING,
                "no-targets-declared",
                str(root_path),
                "no in-scope class list supplied, so coverage and negative-frame "
                "checks cannot run (spec §2a; expected at <root>/targets.txt)",
            )
        )

    images_dir, labels_dir = _frame_dirs(root_path)
    # `images/` without `labels/` is a plausible mistake that `_frame_dirs` would
    # reinterpret as a flat layout, then find no images and say so obliquely.
    if (root_path / "images").is_dir() and not (root_path / "labels").is_dir():
        report.findings.append(
            Finding(
                SEVERITY_ERROR,
                "no-labels-directory",
                str(root_path),
                "images/ exists but labels/ does not, so no label file can be found (spec §7)",
            )
        )
        report.stats.update({"frames": 0})
        return report

    images = (
        sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images_dir.is_dir()
        else []
    )

    if not images:
        report.findings.append(
            Finding(SEVERITY_ERROR, "no-frames", str(root_path), "no image files found")
        )
        report.stats.update({"frames": 0})
        return report

    instances: dict[str, int] = {}
    class_frames: dict[str, int] = {}
    coverage: dict[str, dict[str, int]] = {
        name: dict.fromkeys(LIGHTING_STATES, 0) for name in targets
    }
    lighting_totals: dict[str, int] = dict.fromkeys(LIGHTING_STATES, 0)
    lighting_negatives: dict[str, int] = dict.fromkeys(LIGHTING_STATES, 0)
    empty_label_files: list[str] = []
    missing_label_files: list[str] = []
    distractor_only: list[str] = []
    labelled_stems: set[str] = set()
    too_small: list[str] = []
    too_dark_by_state: dict[str, list[str]] = {state: [] for state in LIGHTING_STATES}
    blurred: list[str] = []

    for image in images:
        subject = image.name
        # Recorded before the name is checked: a frame with a bad filename still has
        # an image, so its label file must not be reported as an orphan on top.
        labelled_stems.add(image.stem)
        name, problem = parse_frame_name(image)
        if name is None:
            report.findings.append(Finding(SEVERITY_ERROR, "bad-filename", subject, problem))
            lighting = ""
        else:
            lighting = name.lighting
            if lighting in lighting_totals:
                lighting_totals[lighting] += 1

        label_path = labels_dir / f"{image.stem}.txt"
        rows: list[LabelRow] = []
        has_label = label_path.exists()
        if not has_label:
            report.findings.append(
                Finding(
                    SEVERITY_ERROR,
                    "missing-label",
                    subject,
                    f"no label file at {label_path.name} - a missing file reads as "
                    "'nothing here', so every detection in this frame scores as a "
                    "false positive (spec §6d)",
                )
            )
        else:
            rows, findings = parse_labels(
                label_path.read_text(encoding="utf-8"), subject=subject, vocab=classes
            )
            report.findings.extend(findings)

        present = {row.cls for row in rows}
        for cls in present:
            class_frames[cls] = class_frames.get(cls, 0) + 1
        for row in rows:
            instances[row.cls] = instances.get(row.cls, 0) + 1

        hits = present & target_set
        if hits:
            if lighting in LIGHTING_STATES:
                for cls in hits:
                    coverage[cls][lighting] += 1
        elif rows:
            distractor_only.append(subject)
            if lighting in lighting_negatives:
                lighting_negatives[lighting] += 1
        else:
            # Both are target-free, and both are counted as negatives, because a missing
            # label file reads as "nothing here" too. They are kept apart anyway: "the
            # labeller called this frame empty" and "there is no label file" need
            # different fixes, and reporting the second as the first turns a set with no
            # labels at all into one that merely looks thoroughly annotated.
            if has_label:
                empty_label_files.append(subject)
            else:
                missing_label_files.append(subject)
            if lighting in lighting_negatives:
                lighting_negatives[lighting] += 1

        quality = read_probe(image)
        if quality is None:
            report.findings.append(
                Finding(SEVERITY_ERROR, "unreadable-frame", subject, "image could not be decoded")
            )
            continue
        height, brightness, sharpness = quality
        if height < min_height:
            too_small.append(subject)
            report.findings.append(
                Finding(
                    SEVERITY_ERROR,
                    "too-small",
                    subject,
                    f"{height}px tall, below the {min_height}px floor - excluded by "
                    "detection_bench.load_frames, so it is never evaluated",
                )
            )
        if brightness < min_brightness:
            too_dark_by_state.setdefault(lighting, []).append(subject)
            # A dark frame in the dark lighting state is the point of capturing one;
            # anywhere else it is a frame that will be silently dropped.
            expected = lighting == "dark"
            report.findings.append(
                Finding(
                    SEVERITY_INFO if expected else SEVERITY_WARNING,
                    "below-brightness-floor",
                    subject,
                    f"brightness {brightness:.1f} < {min_brightness:.1f} - "
                    + (
                        "expected for a dark-state frame; evaluate with a lowered "
                        "--min-brightness and report it as its own row (spec §5)"
                        if expected
                        else "will be excluded from every benchmark run (spec §5)"
                    ),
                )
            )
        if sharpness < min_sharpness:
            blurred.append(subject)
            report.findings.append(
                Finding(
                    SEVERITY_WARNING,
                    "blurred",
                    subject,
                    f"sharpness {sharpness:.1f} < {min_sharpness:.1f} - excluded by "
                    "detection_bench.load_frames (spec §5)",
                )
            )

    for label_path in sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []:
        if label_path.name in RESERVED_TXT_NAMES:
            continue
        if label_path.stem not in labelled_stems:
            report.findings.append(
                Finding(
                    SEVERITY_ERROR,
                    "orphan-label",
                    label_path.name,
                    "label file has no matching image",
                )
            )

    _check_in_scope(report, targets, target_set, known_names, instances, coverage)
    _check_negatives(
        report,
        targets,
        lighting_totals,
        lighting_negatives,
        empty_label_files,
        missing_label_files,
        distractor_only,
    )

    negatives = len(empty_label_files) + len(missing_label_files) + len(distractor_only)
    report.stats.update(
        {
            "frames": len(images),
            "target_classes": list(targets),
            "instances": dict(sorted(instances.items(), key=lambda kv: (-kv[1], kv[0]))),
            "frames_per_class": dict(sorted(class_frames.items(), key=lambda kv: (-kv[1], kv[0]))),
            "coverage": coverage,
            "frames_by_lighting": lighting_totals,
            "negative_frames": negatives,
            "negative_fraction": round(negatives / len(images), 3),
            "empty_label_files": len(empty_label_files),
            "missing_label_files": len(missing_label_files),
            "distractor_only_frames": len(distractor_only),
            "negatives_by_lighting": lighting_negatives,
            "too_small": too_small,
            "too_dark": {state: names for state, names in too_dark_by_state.items() if names},
            "blurred": blurred,
        }
    )
    return report


def _check_in_scope(
    report: FrameSetReport,
    targets: Sequence[str],
    target_set: set[str],
    known_names: set[str],
    instances: dict[str, int],
    coverage: dict[str, dict[str, int]],
) -> None:
    """Every declared target must be a real class and must actually appear."""
    for name in targets:
        if name not in known_names:
            report.findings.append(
                Finding(
                    SEVERITY_ERROR,
                    "target-not-in-vocabulary",
                    name,
                    "declared in-scope but not a class the model can emit, so no frame "
                    "can ever contain it (spec §2b - this needs fine-tuning, not capture)",
                )
            )
            continue
        if not instances.get(name):
            report.findings.append(
                Finding(
                    SEVERITY_ERROR,
                    "target-never-labelled",
                    name,
                    "declared in-scope but no frame labels it - the class is a hole in "
                    "the coverage matrix (spec §3)",
                )
            )
            continue
        empty_states = [state for state, count in coverage[name].items() if not count]
        if empty_states:
            report.findings.append(
                Finding(
                    SEVERITY_WARNING,
                    "coverage-hole",
                    name,
                    f"no frames in lighting state(s): {', '.join(empty_states)} (spec §3)",
                )
            )

    undeclared = sorted(set(instances) - target_set)
    if undeclared and target_set:
        report.findings.append(
            Finding(
                SEVERITY_INFO,
                "undeclared-classes",
                ", ".join(undeclared),
                "labelled but not declared in-scope - fine for negatives, but confirm "
                "they are deliberately out of scope (spec §2c)",
            )
        )


def _check_negatives(
    report: FrameSetReport,
    targets: Sequence[str],
    lighting_totals: dict[str, int],
    lighting_negatives: dict[str, int],
    empty_label_files: Sequence[str],
    missing_label_files: Sequence[str],
    distractor_only: Sequence[str],
) -> None:
    """Without target-free frames, precision is not computable — say so loudly."""
    frames = sum(lighting_totals.values())
    negatives = len(empty_label_files) + len(missing_label_files) + len(distractor_only)
    if frames == 0 or not targets:
        return

    if negatives == 0:
        report.findings.append(
            Finding(
                SEVERITY_ERROR,
                "no-negatives",
                "frame set",
                "every frame contains a target class, so precision is not computable: "
                "every detection is correct by construction and precision would read "
                "1.00 regardless of the model (spec §4)",
            )
        )
        return

    fraction = negatives / frames
    if fraction < RECOMMENDED_NEGATIVE_FRACTION:
        report.findings.append(
            Finding(
                SEVERITY_WARNING,
                "few-negatives",
                "frame set",
                f"{negatives}/{frames} frames ({fraction:.0%}) are target-free, below the "
                f"recommended {RECOMMENDED_NEGATIVE_FRACTION:.0%} (spec §4a) - precision "
                "will rest on a thin base",
            )
        )

    missing = [
        state
        for state, total in lighting_totals.items()
        if total and not lighting_negatives.get(state)
    ]
    if missing:
        report.findings.append(
            Finding(
                SEVERITY_WARNING,
                "negatives-missing-state",
                "frame set",
                f"lighting state(s) with frames but no target-free frames: "
                f"{', '.join(missing)} - precision there is unmeasurable (spec §4a)",
            )
        )


def format_validation_report(report: FrameSetReport, *, limit: int = 15) -> str:
    """Render the findings and the counts behind them for the terminal."""
    stats = report.stats
    lines: list[str] = []

    if not stats.get("frames"):
        lines.append("No frames found - nothing to validate.")
    else:
        negatives = int(stats["negative_frames"])
        lines.append(
            f"{stats['frames']} frames, {negatives} target-free "
            f"({float(stats['negative_fraction']):.0%}), "
            f"{len(stats['instances'])} classes labelled"
        )
        lighting = ", ".join(
            f"{state} {count}" for state, count in stats["frames_by_lighting"].items() if count
        )
        if lighting:
            lines.append(f"lighting: {lighting}")
        lines.append(
            f"target-free breakdown: {stats['missing_label_files']} missing label file(s), "
            f"{stats['empty_label_files']} empty label file(s), "
            f"{stats['distractor_only_frames']} distractor-only"
        )

    targets = stats.get("target_classes") or []
    if targets:
        lines.append("")
        lines.append("coverage (frames per target class x lighting):")
        header = "  " + f"{'class':<18}" + "".join(f"{state:>7}" for state in LIGHTING_STATES)
        lines.append(header)
        for name in targets:
            row = stats["coverage"][name]
            cells = "".join(f"{row.get(state, 0):>7}" for state in LIGHTING_STATES)
            lines.append(f"  {name:<18}{cells}")

    counts = stats.get("instances") or {}
    if counts:
        lines.append("")
        rendered = ", ".join(f"{name} x{count}" for name, count in list(counts.items())[:10])
        lines.append(f"labelled instances: {rendered}")

    grouped = [
        (SEVERITY_ERROR, report.errors),
        (SEVERITY_WARNING, report.warnings),
        (
            SEVERITY_INFO,
            [row for row in report.findings if row.severity == SEVERITY_INFO],
        ),
    ]
    for severity, rows in grouped:
        if not rows:
            continue
        lines.append("")
        lines.append(f"{severity}s ({len(rows)}):")
        for row in rows[:limit]:
            lines.append("  " + row.render())
        if len(rows) > limit:
            lines.append(f"  ... and {len(rows) - limit} more")

    lines.append("")
    if report.ok:
        lines.append("No blocking problems found. The set is well-formed; it is not a")
        lines.append("judgement about the detector, and no precision has been measured.")
    else:
        lines.append(f"{len(report.errors)} blocking problem(s). Fix these before using the set:")
        lines.append("a set that fails validation produces numbers that look fine and are not.")
    return "\n".join(lines)
