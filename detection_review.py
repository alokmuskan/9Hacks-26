"""Review the detections the system already made, and score them into a real number.

Why this exists
---------------

``detection_bench`` can measure *recall* against two bundled reference images and
*detection statistics* on this project's own frames. It cannot measure precision,
because nothing is labelled, and every measurement of "did we get better" was
therefore blocked on a labelled frame set that may never be captured.

It does not need to be. The detector's own output on frames that already exist is
a bounded, finite list — 99 boxes over the usable frames at the time of writing —
and a human can confirm or reject that list in a couple of minutes. That turns
precision from unmeasurable into a **census of the existing frames**, with no new
capture and no labelling of anything the detector did not already propose.

What it measures, and what it does not
--------------------------------------

* **Precision, exactly** — every reviewed box is either right or wrong, so the
  counts are exact for the frames reviewed. When a subset is reviewed instead
  (``--limit``), the result is a sample and is reported with a Wilson interval,
  because a subset chosen for convenience is not a random sample.
* **Not recall.** Recall needs every object in the frame enumerated, which is the
  "extensive labelling" this deliberately avoids. The optional per-row note asks
  only for objects the reviewer *happened to notice* were missed; those are
  reported as a diagnostic list of concrete misses, never as a recall figure,
  because the frames they came from were not chosen by a sampling rule.
* **Not generalisation.** These frames are one scene. A precision figure here
  describes these frames, and the report says so on every run rather than leaving
  the reader to assume otherwise.

The review page embeds real camera frames. It is written under ``reviews/``,
which is git-ignored, and carries ``noindex`` for the same reason.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Review artefacts: the detection list, the page and the pasted-back verdicts.
DEFAULT_REVIEW_DIR = "reviews"
DETECTIONS_FILENAME = "detections.json"
PAGE_FILENAME = "review.html"
VERDICTS_FILENAME = "verdicts.json"

#: Confidence bands used to spread a limited review across the whole range
#: instead of only the most confident (and therefore easiest) detections.
CONFIDENCE_BANDS = ((0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01))

#: Narrower than this and a partial review cannot support a claim at all.
MIN_COVERAGE_WARNING = 0.80

Z_95 = 1.96


@dataclass(frozen=True)
class Detection:
    """One box the detector produced, with the index a reviewer refers to."""

    index: int
    frame: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frame": self.frame,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Detection:
        bbox = row.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        return cls(
            index=int(row["index"]),
            frame=str(row.get("frame", "")),
            label=str(row.get("label", "")),
            confidence=float(row.get("confidence", 0.0)),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        )


@dataclass(frozen=True)
class ReviewRow:
    """A detection plus the annotated thumbnail shown for it."""

    detection: Detection
    thumb_b64: str


@dataclass(frozen=True)
class Verdict:
    """A reviewer's decision on one detection."""

    index: int
    correct: bool
    missed: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassScore:
    label: str
    reviewed: int
    correct: int

    @property
    def precision(self) -> float:
        return self.correct / self.reviewed if self.reviewed else 0.0


@dataclass(frozen=True)
class ReviewScore:
    reviewed: int
    total: int
    correct: int
    precision: float
    interval: tuple[float, float] | None
    per_class: tuple[ClassScore, ...]
    misses: tuple[tuple[str, str], ...]

    @property
    def coverage(self) -> float:
        return self.reviewed / self.total if self.total else 0.0

    @property
    def wrong(self) -> int:
        return self.reviewed - self.correct


def fingerprint(detections: Sequence[Detection]) -> str:
    """A stable id for a detection list.

    Box indices are only meaningful against the list they were assigned to, so a
    rebuild renumbers them. Scoring one build's verdicts against another build's
    list would produce a confident, wrong precision figure and say nothing — the
    exact failure this module exists to prevent — so the list carries an id and the
    score refuses when they disagree.
    """
    payload = json.dumps(
        [
            [d.frame, d.label, round(d.confidence, 6), [round(v, 3) for v in d.bbox]]
            for d in detections
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint_mismatch(stored: str | None, payload: Any) -> str | None:
    """Why a verdicts file does not belong to this detection list, or ``None`` if it does.

    An older verdicts file with no id is allowed through: it cannot be checked, and
    refusing it would be inventing a problem rather than finding one.
    """
    if not stored:
        return None
    if not isinstance(payload, dict):
        return "the verdicts file is not a JSON object"
    claimed = payload.get("fingerprint")
    if not isinstance(claimed, str) or not claimed:
        return None
    if claimed != stored:
        return (
            f"these verdicts were recorded against a different detection list "
            f"({claimed} vs {stored})"
        )
    return None


def wilson_interval(correct: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval: correct for small n and for proportions near 0 or 1.

    The normal approximation would give a zero-width interval at 100% correct,
    which is exactly the case a small review run is most likely to produce.
    """
    if total <= 0:
        return (0.0, 1.0)
    phat = correct / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5)
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def select_detections(detections: list[Detection], limit: int) -> list[Detection]:
    """Take an evenly spaced subset across the confidence range.

    Systematic rather than top-N: the most confident boxes are the easiest to
    confirm, so reviewing only those would report the best possible precision for
    the same effort. Spreading the picks keeps the low-confidence band — where a
    threshold change actually bites — represented.
    """
    if limit <= 0 or limit >= len(detections):
        return list(detections)

    ordered = sorted(detections, key=lambda d: (-d.confidence, d.index))
    step = len(ordered) / limit
    picked = [ordered[min(int(i * step), len(ordered) - 1)] for i in range(limit)]
    # Systematic sampling can repeat an index when the step rounds; keep first-seen order.
    seen: set[int] = set()
    unique: list[Detection] = []
    for detection in picked:
        if detection.index not in seen:
            seen.add(detection.index)
            unique.append(detection)
    return unique


def parse_verdicts(payload: Any) -> list[Verdict]:
    """Read the review page's output: ``{"verdicts": {"12": "y"|"n"}, "missed": {...}}``.

    Also accepts a bare ``{index: verdict}`` mapping, because that is what a human
    ends up typing when they hand-write one. Anything unreadable is skipped rather
    than guessed at: an unparsed answer is a missing answer, and treating it as
    "correct" would inflate the score.
    """
    if not isinstance(payload, dict):
        return []

    raw = payload.get("verdicts", payload)
    if not isinstance(raw, dict):
        return []

    missed_by_index: dict[int, list[str]] = {}
    missed = payload.get("missed")
    if isinstance(missed, dict):
        for key, value in missed.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                missed_by_index[index] = _split_notes(value)
            elif isinstance(value, list):
                missed_by_index[index] = [str(item).strip() for item in value if str(item).strip()]

    verdicts: list[Verdict] = []
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        decided = _as_bool(value)
        if decided is None:
            continue
        verdicts.append(Verdict(index=index, correct=decided, missed=tuple(missed_by_index.get(index, ()))))
    verdicts.sort(key=lambda v: v.index)
    return verdicts


def _split_notes(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"y", "yes", "true", "1", "correct", "c"}:
            return True
        if token in {"n", "no", "false", "0", "wrong", "w"}:
            return False
    return None


def score(detections: list[Detection], verdicts: list[Verdict]) -> ReviewScore:
    """Precision over the reviewed detections, per class and overall."""
    by_index = {d.index: d for d in detections}
    counted: list[tuple[Detection, bool]] = []
    for verdict in verdicts:
        detection = by_index.get(verdict.index)
        if detection is None:
            continue  # a verdict for a detection that is not in this list: ignore it
        counted.append((detection, verdict.correct))

    totals: dict[str, list[int]] = {}
    for detection, correct in counted:
        bucket = totals.setdefault(detection.label, [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if correct else 0

    per_class = tuple(
        ClassScore(label=label, reviewed=counts[0], correct=counts[1])
        for label, counts in sorted(totals.items(), key=lambda kv: (-kv[1][0], kv[0]))
    )

    correct_total = sum(1 for _, correct in counted if correct)
    reviewed = len(counted)
    interval = None if reviewed >= len(detections) else wilson_interval(correct_total, reviewed)

    # One entry per (frame, object): the same miss noted on several rows of one
    # frame is one miss, not three.
    misses: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for verdict in verdicts:
        detection = by_index.get(verdict.index)
        if detection is None:
            continue
        for note in verdict.missed:
            key = (detection.frame, note.lower())
            if key in seen:
                continue
            seen.add(key)
            misses.append((detection.frame, note))

    return ReviewScore(
        reviewed=reviewed,
        total=len(detections),
        correct=correct_total,
        precision=(correct_total / reviewed) if reviewed else 0.0,
        interval=interval,
        per_class=per_class,
        misses=tuple(misses),
    )


def annotate(image: Any, detection: Detection) -> Any:
    """Draw the box, its class, its confidence and the index a reviewer refers to."""
    import cv2

    x1, y1, x2, y2 = (round(v) for v in detection.bbox)
    height, width = image.shape[:2]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

    GREEN = (80, 220, 120)
    DARK = (30, 30, 30)
    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)

    caption = f"#{detection.index} {detection.label} {detection.confidence:.2f}"
    ty = y1 - 8 if y1 > 24 else min(height - 6, y2 + 18)
    cv2.putText(image, caption, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK, 3, cv2.LINE_AA)
    cv2.putText(image, caption, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    return image


def encode_thumbnail(image: Any, max_side: int = 360, quality: int = 75) -> str:
    """Base64 JPEG of the annotated frame, scaled so the page stays openable."""
    import cv2

    height, width = image.shape[:2]
    scale = max_side / max(height, width, 1)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def build_rows(
    frames: Sequence[tuple[str, Any]],
    detector: Any,
    *,
    limit: int = 0,
    progress: Any = None,
) -> list[ReviewRow]:
    """Detect on every usable frame, then render the rows to be reviewed."""
    detections: list[Detection] = []
    images: dict[str, Any] = {}
    for path, image in frames:
        images[path] = image
        for row in detector.detect(image):
            box = row["bbox"]
            detections.append(
                Detection(
                    index=len(detections) + 1,
                    frame=path,
                    label=str(row.get("label", "")),
                    confidence=float(row.get("confidence", 0.0)),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        if progress is not None:
            progress(path)

    rows: list[ReviewRow] = []
    for detection in select_detections(detections, limit):
        canvas = images[detection.frame].copy()
        annotate(canvas, detection)
        rows.append(ReviewRow(detection=detection, thumb_b64=encode_thumbnail(canvas)))
    return rows


def render_page(rows: list[ReviewRow], *, source: str, list_id: str = "") -> str:
    """The self-contained review page: every row with a yes/no pair and an optional note."""
    cards: list[str] = []
    for row in rows:
        detection = row.detection
        frame_name = Path(detection.frame).name
        cards.append(
            f"""
    <li class="row" data-index="{detection.index}">
      <div class="meta">
        <span class="idx">#{detection.index}</span>
        <span class="cls">{html.escape(detection.label)}</span>
        <span class="conf">{detection.confidence:.2f}</span>
        <span class="frame">{html.escape(frame_name)}</span>
      </div>
      <img alt="frame {html.escape(frame_name)} with box #{detection.index}"
           src="data:image/jpeg;base64,{row.thumb_b64}">
      <div class="controls">
        <button class="yes" type="button" onclick="mark({detection.index}, true, event)">Correct</button>
        <button class="no" type="button" onclick="mark({detection.index}, false, event)">Wrong</button>
        <span class="state" id="state-{detection.index}">unreviewed</span>
      </div>
      <input class="missed" id="missed-{detection.index}" type="text"
             placeholder="objects you can see here that were NOT boxed (optional, comma-separated)">
    </li>"""
        )

    body = "\n".join(cards)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex, nofollow">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detection review</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #14161a; color: #e7e9ee; }}
  header {{ position: sticky; top: 0; background: #1b1e24; padding: 14px 18px; border-bottom: 1px solid #2b3038; }}
  h1 {{ margin: 0 0 6px; font-size: 16px; }}
  .honest {{ color: #b6bcc8; font-size: 12px; max-width: 900px; }}
  .bar {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-top: 10px; }}
  button {{ font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid #3a4049;
            background: #23272f; color: #e7e9ee; cursor: pointer; }}
  button.yes.on {{ background: #1f6f43; border-color: #2f9d62; }}
  button.no.on {{ background: #7a2530; border-color: #b8394a; }}
  ul {{ list-style: none; margin: 0; padding: 14px 18px; display: grid; gap: 14px;
        grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }}
  .row {{ background: #1b1e24; border: 1px solid #2b3038; border-radius: 8px; padding: 10px; }}
  .row.done {{ border-color: #3d4753; }}
  .row.active {{ border-color: #4c8bf5; box-shadow: 0 0 0 2px rgba(76, 139, 245, 0.35); }}
  .meta {{ display: flex; gap: 8px; align-items: center; font-size: 12px; margin-bottom: 8px; }}
  .idx {{ color: #8fa0b8; }}
  .cls {{ font-weight: 600; }}
  .conf {{ color: #8fa0b8; }}
  .frame {{ color: #6f7a8a; margin-left: auto; }}
  img {{ width: 100%; border-radius: 4px; display: block; }}
  .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
  .state {{ font-size: 12px; color: #8fa0b8; }}
  .missed {{ margin-top: 8px; width: 100%; box-sizing: border-box; padding: 6px 8px; border-radius: 6px;
             border: 1px solid #3a4049; background: #14161a; color: #e7e9ee; font: inherit; }}
</style>
</head>
<body>
<header>
  <h1>Detection review — {len(rows)} box(es) from {html.escape(source)}</h1>
  <p class="honest">
    Each box below is one detection. Mark it <strong>Correct</strong> if the box encloses the
    object the label names, <strong>Wrong</strong> otherwise. This measures
    <strong>precision only</strong>, and only for these frames — it cannot measure recall, and it
    says nothing about any other scene. Frames you leave <em>unreviewed</em> are reported as
    unreviewed, never as correct.
  </p>
  <div class="bar">
    <span id="progress">reviewed 0 / {len(rows)}</span>
    <span id="wrongcount">wrong 0</span>
    <button type="button" onclick="downloadVerdicts()">Download verdicts.json</button>
    <button type="button" onclick="copyVerdicts()">Copy JSON</button>
    <span id="copied" class="state"></span>
  </div>
  <p class="honest" style="margin: 8px 0 0">
    Keyboard: <strong>Y</strong> = correct, <strong>N</strong> = wrong (the highlighted box advances
    by itself), <strong>&uarr;</strong>/<strong>&darr;</strong> or <strong>J</strong>/<strong>K</strong>
    to move. Typing in a note field is never captured.
  </p>
</header>
<ul id="rows">
{body}
</ul>
<script>
const verdicts = {{}};
const missed = {{}};
function mark(index, correct, event) {{
  verdicts[index] = correct;
  const row = document.querySelector('.row[data-index="' + index + '"]');
  row.querySelector('button.yes').classList.toggle('on', correct);
  row.querySelector('button.no').classList.toggle('on', !correct);
  document.getElementById('state-' + index).textContent = correct ? 'correct' : 'wrong';
  row.classList.add('done');
  update();
}}
function collect() {{
  document.querySelectorAll('.missed').forEach(function (input) {{
    const index = input.id.replace('missed-', '');
    if (input.value.trim()) {{
      missed[index] = input.value.trim();
    }} else {{
      delete missed[index];
    }}
  }});
  return JSON.stringify({{fingerprint: "{list_id}", verdicts: verdicts, missed: missed}}, null, 2);
}}
function update() {{
  const n = Object.keys(verdicts).length;
  const w = Object.values(verdicts).filter(function (v) {{ return v === false; }}).length;
  document.getElementById('progress').textContent = 'reviewed ' + n + ' / {len(rows)}';
  document.getElementById('wrongcount').textContent = 'wrong ' + w;
}}
function downloadVerdicts() {{
  const blob = new Blob([collect()], {{type: 'application/json'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'verdicts.json';
  link.click();
  URL.revokeObjectURL(link.href);
}}
function copyVerdicts() {{
  const text = collect();
  const done = function () {{ document.getElementById('copied').textContent = 'copied'; }};
  if (navigator.clipboard) {{ navigator.clipboard.writeText(text).then(done, done); }}
  else {{ window.prompt('Copy this JSON', text); }}
}}

// Keyboard review: the whole list is 100-odd boxes, and a mouse round-trip per box is
// the difference between two minutes and ten. The cursor only marks the highlighted
// box, so a keystroke can never land on a box the reviewer cannot see.
let active = -1;
const cards = Array.from(document.querySelectorAll('.row'));
function setActive(i) {{
  if (active >= 0 && cards[active]) {{ cards[active].classList.remove('active'); }}
  active = Math.max(0, Math.min(cards.length - 1, i));
  const row = cards[active];
  if (!row) {{ return; }}
  row.classList.add('active');
  row.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
}}
function apply(correct) {{
  if (active < 0) {{ setActive(0); }}
  const row = cards[active];
  if (!row) {{ return; }}
  mark(Number(row.dataset.index), correct);
  let next = active + 1;
  while (next < cards.length && verdicts[cards[next].dataset.index] !== undefined) {{ next += 1; }}
  if (next < cards.length) {{ setActive(next); }}
}}
document.addEventListener('keydown', function (event) {{
  if (event.target && event.target.tagName === 'INPUT') {{ return; }}
  const key = event.key.toLowerCase();
  if (key === 'y') {{ apply(true); event.preventDefault(); }}
  else if (key === 'n') {{ apply(false); event.preventDefault(); }}
  else if (key === 'arrowdown' || key === 'j') {{ setActive(active + 1); event.preventDefault(); }}
  else if (key === 'arrowup' || key === 'k') {{ setActive(active - 1); event.preventDefault(); }}
}});

document.querySelectorAll('.missed').forEach(function (input) {{ input.addEventListener('change', update); }});
update();
setActive(0);
</script>
</body>
</html>
"""


def write_review(
    rows: list[ReviewRow],
    *,
    out_dir: str | Path = DEFAULT_REVIEW_DIR,
    source: str = "",
) -> tuple[Path, Path]:
    """Write the detection list and the review page. Returns ``(detections, page)``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    list_id = fingerprint([row.detection for row in rows])
    detections_path = directory / DETECTIONS_FILENAME
    detections_path.write_text(
        json.dumps(
            {
                "source": source,
                "fingerprint": list_id,
                "detections": [row.detection.to_json() for row in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    page_path = directory / PAGE_FILENAME
    page_path.write_text(render_page(rows, source=source, list_id=list_id), encoding="utf-8")
    return detections_path, page_path


def load_detections(path: str | Path) -> list[Detection]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("detections") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    return [Detection.from_json(row) for row in rows if isinstance(row, dict)]


def format_score(score: ReviewScore) -> str:
    """The scored result, with its scope and its limits stated on the output itself."""
    lines: list[str] = []
    if score.reviewed == 0:
        lines.append("Nothing reviewed yet: no verdicts recorded.")
        lines.append("  Open the review page, mark boxes Correct/Wrong, then Download verdicts.json.")
        return "\n".join(lines)

    lines.append(f"Reviewed {score.reviewed} of {score.total} detection(s)  (coverage {score.coverage * 100:.0f}%)")
    lines.append(f"Precision: {score.precision:.3f}  ({score.correct} correct, {score.wrong} wrong)")
    if score.interval is not None:
        low, high = score.interval
        lines.append(
            f"  95% Wilson interval {low:.3f}-{high:.3f}  (a subset was reviewed, so this is a sample)"
        )
    else:
        lines.append("  every detection was reviewed, so this is a census of these frames - no sampling error")

    lines.append("")
    lines.append(f"{'label':<18}{'reviewed':>9}{'correct':>9}{'wrong':>7}{'precision':>11}")
    for entry in score.per_class:
        lines.append(
            f"{entry.label:<18}{entry.reviewed:>9}{entry.correct:>9}"
            f"{entry.reviewed - entry.correct:>7}{entry.precision:>11.3f}"
        )

    if score.coverage < MIN_COVERAGE_WARNING:
        lines.append("")
        lines.append(
            f"[warn] only {score.coverage * 100:.0f}% reviewed: a subset chosen by hand is not a "
            "random sample,"
        )
        lines.append("       so treat the interval as optimistic rather than exact.")

    if score.misses:
        lines.append("")
        lines.append(f"Objects the reviewer saw but nothing was boxed around ({len(score.misses)}):")
        for frame, note in score.misses[:20]:
            lines.append(f"  {Path(frame).name}: {note}")
        if len(score.misses) > 20:
            lines.append(f"  ... and {len(score.misses) - 20} more")
        lines.append(
            "  These are concrete misses, not a recall figure: the frames they came from were not"
        )
        lines.append("  chosen by any sampling rule, so there is no denominator behind them.")

    lines.append("")
    lines.append("Scope of this number: the frames listed in the review directory, nothing else.")
    lines.append("  It measures precision, not recall, and describes these frames only.")
    return "\n".join(lines)
