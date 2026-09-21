"""Per-stage frame-budget accounting over recorded monitoring sessions.

:mod:`detection_bench` measures object detection alone, on saved frames, in one
process. That answers "how fast is the detector", not "what is a live frame made
of". This module answers the second question from the metrics the pipeline
*already* writes to ``metrics_log.jsonl``, so it needs no camera, no labels and no
new run.

The accounting is deliberately partial, because the record is partial:

* ``avg_detection_latency_ms`` and ``detection_calls`` time ``_detect(app, frame)``
  — **face detection** (``main.py``, ``server.py``) — not the YOLO object
  detector. Both loops call ``detector.detect(frame)`` untimed, so object
  detection appears in **no** session record. The field's name does not say this,
  which is why it is spelled out in the report.
* A stage whose latency was never written is ``untimed`` and renders as ``--``,
  never as ``0.0``. A stage that deliberately did not run renders as ``off``.
  Those are three different facts and collapsing them is how a missing
  measurement becomes a fast one.
* The remainder after the recorded stages is a **residual, not a measurement**.
  It holds object detection, capture, snapshot encoding, memory writes, HUD work,
  and idle time. Reporting it as a stage would invent a number no code produced.
* ``avg_fps`` is a session average, and a session average is not a per-frame cost.
  Four of this project's twelve recorded sessions have an exponential-moving
  average several times above their session mean (one reached 178 fps
  instantaneously against a 0.82 fps mean), so their mean period is dominated by
  stalls. Those sessions are flagged rather than averaged in silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

#: The event type a finished session is recorded under.
SESSION_EVENT_TYPE = "recognize_session"

#: A frame period within this factor of ``1 / fps_cap`` is treated as the pacer
#: being in control: the loop sleeps the remainder of each interval, so the
#: observed period settles just above the target.
CAP_TOLERANCE = 1.05

#: An EMA rate this many times the session average means most frames were far
#: faster than the average, i.e. a few stalls dominate it.
STALL_FACTOR = 2.0

#: An instantaneous frame rate above this cannot describe the pipeline, so it is
#: reported as an unusable measurement rather than quoted as a fast frame.
IMPLAUSIBLE_FPS = 1000.0

Pacing = Literal["throughput-bound", "cap-bound", "not-recorded"]

#: How a stage appears in the record. The three cases are not interchangeable:
#: a timed stage, one that did not run, and one that ran without a recorded
#: latency. Only the last can be silently mistaken for ``0 ms``.
StageState = Literal["measured", "disabled", "untimed"]

#: Rendered for a stage the session record ran but never timed.
NOT_RECORDED = "--"
#: Rendered for a stage that deliberately did not run this session.
DISABLED = "off"


@dataclass(frozen=True)
class Stage:
    """One recorded stage's cost per frame, and whether it was measured at all."""

    #: ``None`` unless the stage was actually timed.
    ms: float | None
    state: StageState

    @property
    def cell(self) -> str:
        if self.state == "measured":
            return f"{self.ms:.1f}"
        return DISABLED if self.state == "disabled" else NOT_RECORDED


@dataclass(frozen=True)
class SessionBudget:
    """One session's frame period, split into what is known and what is not."""

    session_id: str
    frames_total: int
    duration_sec: float
    #: Session throughput: ``frames_total / duration_sec``.
    avg_fps: float
    #: EMA of the per-frame rate, i.e. roughly what an ordinary frame achieved.
    ema_fps: float
    instant_low_fps: float
    instant_high_fps: float
    frame_period_ms: float
    face: Stage
    gaze: Stage
    objects: Stage
    residual_ms: float
    face_recognition: bool
    pacing: Pacing
    notes: tuple[str, ...] = ()

    @property
    def face_share(self) -> float | None:
        return None if self.face.ms is None else self.face.ms / self.frame_period_ms

    @property
    def gaze_share(self) -> float | None:
        return None if self.gaze.ms is None else self.gaze.ms / self.frame_period_ms

    @property
    def object_share(self) -> float | None:
        return None if self.objects.ms is None else self.objects.ms / self.frame_period_ms

    @property
    def residual_share(self) -> float:
        return self.residual_ms / self.frame_period_ms

    @property
    def stall_dominated(self) -> bool:
        """Most frames were much faster than the average, so the mean is not a per-frame cost."""
        return self.ema_fps > self.avg_fps * STALL_FACTOR

    @property
    def fully_recorded(self) -> bool:
        """Every recorded stage is known either way, so the residual means something."""
        return (
            self.face.state != "untimed"
            and self.gaze.state != "untimed"
            and self.objects.state != "untimed"
        )


@dataclass(frozen=True)
class Cohort:
    """Medians over sessions with a timed stage, counted per line.

    Each stage can be measured in a different number of sessions (face detection
    runs whenever face recognition is on, gaze only when its weights loaded), so
    the per-stage counts are carried rather than assumed equal.
    """

    #: Sessions where at least one recorded stage was timed, which is the only
    #: subset whose period can be split at all.
    sessions: int
    measured_face: int
    measured_gaze: int
    measured_objects: int
    median_period_ms: float
    median_face_ms: float | None
    median_gaze_ms: float | None
    median_objects_ms: float | None
    median_residual_ms: float
    median_face_share: float | None
    median_gaze_share: float | None
    median_objects_share: float | None
    median_residual_share: float


@dataclass(frozen=True)
class BudgetSummary:
    """Session-level throughput, plus per-stage medians over the accountable cohort."""

    sessions: int
    frames_total: int
    duration_sec: float
    median_avg_fps: float
    fps_range: tuple[float, float]
    median_ema_fps: float
    stall_dominated: int
    implausible_rate: int
    stages: Cohort
    partial: PhaseCounts


@dataclass(frozen=True)
class PhaseCounts:
    """How many sessions had a stage switched off, and how many lacked timings."""

    sessions: int
    face_off: int
    gaze_off: int
    untimed_stages: int


@dataclass(frozen=True)
class BudgetSet:
    """The usable budgets, plus every session row that could not become one."""

    budgets: tuple[SessionBudget, ...]
    skipped: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.budgets)


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_float(value, float(default)))


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _measure_stage(
    aggregate: Mapping[str, Any],
    calls_key: str,
    avg_key: str,
    frames: int,
    *,
    ran_this_session: bool,
) -> Stage:
    """Cost per frame of a stage, separating "did not run" from "ran but was not timed".

    ``ran_this_session`` comes from the session's own toggle rather than from the
    timings, because the timings cannot be trusted to report a stage that did not
    run: a disabled stage still increments its call count and still wraps a no-op
    in a timer, so it reports a number instead of being absent. Believing that
    number would record ~0.01 ms as a measured cost.
    """
    if not ran_this_session:
        return Stage(ms=None, state="disabled")
    calls = _as_int(aggregate.get(calls_key))
    if calls <= 0:
        return Stage(ms=None, state="disabled")
    average = _as_float(aggregate.get(avg_key))
    if average <= 0.0:
        return Stage(ms=None, state="untimed")
    return Stage(ms=calls * average / frames, state="measured")


def _measure_object_stage(aggregate: Mapping[str, Any], frames: int) -> Stage:
    """Cost per frame of object detection, which older records never timed.

    The field's *presence* is the signal here, not its value. Sessions written
    before schema 8 have no ``object_detection_calls`` at all: object detection ran
    on every frame, but nothing timed it, so the honest render is the same ``--`` as
    any untimed stage. Rendering ``off`` would claim the detector was switched off,
    which is a different and false statement.
    """
    if "object_detection_calls" not in aggregate:
        return Stage(ms=None, state="untimed")
    calls = _as_int(aggregate.get("object_detection_calls"))
    if calls <= 0:
        return Stage(ms=None, state="disabled")
    average = _as_float(aggregate.get("avg_object_detection_latency_ms"))
    if average <= 0.0:
        return Stage(ms=None, state="untimed")
    return Stage(ms=calls * average / frames, state="measured")


def _pacing(frame_period_ms: float, fps_cap: int) -> Pacing:
    """Whether the loop was pacing itself or working as fast as it could.

    ``fps_cap`` is absent from sessions recorded before schema 8, so this returns
    ``not-recorded`` rather than inferring from a default that may not have applied.
    """
    if fps_cap <= 0:
        return "not-recorded"
    interval_ms = 1000.0 / fps_cap
    return "cap-bound" if frame_period_ms <= interval_ms * CAP_TOLERANCE else "throughput-bound"


def build_budget(row: Mapping[str, Any]) -> tuple[SessionBudget | None, str]:
    """Turn one metrics row into a :class:`SessionBudget`.

    The second element explains why a row produced no budget, so a session is
    named rather than quietly dropped.
    """
    aggregate = row.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None, "no aggregate"

    session_id = _as_str(row.get("session_id")) or _as_str(aggregate.get("session_id")) or "?"
    frames = _as_int(aggregate.get("frames_total"))
    avg_fps = _as_float(aggregate.get("avg_fps"))
    if frames <= 0 or avg_fps <= 0:
        return (
            None,
            f"{session_id}: frames_total={frames}, avg_fps={avg_fps} (no measurable period)",
        )

    frame_period_ms = 1000.0 / avg_fps
    notes: list[str] = []
    face_recognition = bool(aggregate.get("face_recognition_enabled", True))

    calls = _as_int(aggregate.get("detection_calls"))
    face = _measure_stage(
        aggregate,
        "detection_calls",
        "avg_detection_latency_ms",
        frames,
        ran_this_session=face_recognition,
    )
    if face.state == "untimed":
        notes.append(
            f"face detection was called {calls} time(s) with no latency recorded, so the stage "
            "is unknown rather than free"
        )
    elif face.state == "disabled" and face_recognition:
        notes.append("face recognition was enabled but face detection was never called")

    gaze_enabled = bool(aggregate.get("gaze_enabled"))
    gaze = _measure_stage(
        aggregate,
        "gaze_inference_calls",
        "gaze_inference_avg_ms",
        frames,
        ran_this_session=gaze_enabled,
    )
    if gaze.state == "untimed":
        notes.append("gaze inference ran with no latency recorded")
    elif gaze.state == "disabled" and gaze_enabled:
        # Gaze is estimated per detected face, so a session with no faces has
        # nothing to estimate and legitimately incurs no gaze cost.
        if not aggregate.get("gaze_model_loaded"):
            reason = "weights never loaded"
        elif _as_int(aggregate.get("frames_with_faces")) == 0:
            reason = "no face was detected to estimate gaze for"
        else:
            reason = "no call recorded"
        notes.append(f"gaze was enabled but never ran ({reason})")

    objects = _measure_object_stage(aggregate, frames)
    if objects.state == "untimed":
        notes.append(
            "object detection is not timed in this session, so its cost sits in the remainder "
            "rather than being unknown"
        )

    residual_ms = frame_period_ms - ((face.ms or 0.0) + (gaze.ms or 0.0) + (objects.ms or 0.0))
    if residual_ms < 0:
        notes.append(
            f"recorded stages exceed the mean period by {abs(residual_ms):.1f} ms, so the split "
            "cannot be trusted for this session"
        )

    ema_fps = _as_float(aggregate.get("moving_avg_fps"))
    instant_high = _as_float(aggregate.get("max_fps"))
    if instant_high > IMPLAUSIBLE_FPS:
        notes.append(
            f"the instantaneous rate reaches {instant_high:.0f} fps, which no stage could produce: "
            "the interval was measured against the wall clock, which can step backwards "
            "mid-session, so this session's rate statistics are not usable (intervals are now "
            "measured with a monotonic clock)"
        )
    elif ema_fps > avg_fps * STALL_FACTOR:
        notes.append(
            f"instantaneous rate reached {instant_high:.1f} fps against a {avg_fps:.2f} fps session "
            "average, so that average is dominated by stalls, not by the cost of an ordinary frame"
        )

    pacing = _pacing(frame_period_ms, _as_int(aggregate.get("fps_cap")))
    if pacing == "cap-bound":
        notes.append(
            "the mean period is at the configured cap, so part of the remainder is idle pacing "
            "time rather than work"
        )

    return (
        SessionBudget(
            session_id=session_id,
            frames_total=frames,
            duration_sec=_as_float(row.get("duration_sec")),
            avg_fps=avg_fps,
            ema_fps=ema_fps,
            instant_low_fps=_as_float(aggregate.get("min_fps")),
            instant_high_fps=instant_high,
            frame_period_ms=frame_period_ms,
            face=face,
            gaze=gaze,
            objects=objects,
            residual_ms=residual_ms,
            face_recognition=face_recognition,
            pacing=pacing,
            notes=tuple(notes),
        ),
        "",
    )


def collect_budgets(rows: Sequence[Mapping[str, Any]], *, limit: int | None = None) -> BudgetSet:
    """Every session in ``rows`` that can be accounted for, and every one that cannot."""
    budgets: list[SessionBudget] = []
    skipped: list[str] = []
    for row in rows:
        if not isinstance(row.get("aggregate"), Mapping):
            continue
        budget, reason = build_budget(row)
        if budget is None:
            skipped.append(reason)
        else:
            budgets.append(budget)

    # Rows arrive oldest-first (rotated generations are read before the live file),
    # so the newest sessions are at the end.
    selected = budgets[-limit:] if limit and limit > 0 else budgets
    return BudgetSet(budgets=tuple(selected), skipped=tuple(skipped))


def _cohort(budgets: Sequence[SessionBudget]) -> Cohort:
    """Medians over sessions with a timed stage, with ``None`` where a stage has none.

    Sessions where every stage was switched off are excluded: their period is
    genuinely 100% unattributed, but including them would pull the stage medians
    toward a split that reflects the toggles rather than the pipeline.
    """
    timed = [
        b
        for b in budgets
        if b.face.ms is not None or b.gaze.ms is not None or b.objects.ms is not None
    ]
    if not timed:
        return Cohort(0, 0, 0, 0, 0.0, None, None, None, 0.0, None, None, None, 0.0)
    face = [b.face.ms for b in timed if b.face.ms is not None]
    gaze = [b.gaze.ms for b in timed if b.gaze.ms is not None]
    objects = [b.objects.ms for b in timed if b.objects.ms is not None]
    face_share = [s for s in (b.face_share for b in timed) if s is not None]
    gaze_share = [s for s in (b.gaze_share for b in timed) if s is not None]
    object_share = [s for s in (b.object_share for b in timed) if s is not None]
    return Cohort(
        sessions=len(timed),
        measured_face=len(face),
        measured_gaze=len(gaze),
        measured_objects=len(objects),
        median_period_ms=median([b.frame_period_ms for b in timed]),
        median_face_ms=median(face) if face else None,
        median_gaze_ms=median(gaze) if gaze else None,
        median_objects_ms=median(objects) if objects else None,
        median_residual_ms=median([b.residual_ms for b in timed]),
        median_face_share=median(face_share) if face_share else None,
        median_gaze_share=median(gaze_share) if gaze_share else None,
        median_objects_share=median(object_share) if object_share else None,
        median_residual_share=median([b.residual_share for b in timed]),
    )


def summarize(budgets: Sequence[SessionBudget]) -> BudgetSummary:
    """Medians, not means: the spread between sessions is larger than most differences."""
    if not budgets:
        return BudgetSummary(
            sessions=0,
            frames_total=0,
            duration_sec=0.0,
            median_avg_fps=0.0,
            fps_range=(0.0, 0.0),
            median_ema_fps=0.0,
            stall_dominated=0,
            implausible_rate=0,
            stages=Cohort(0, 0, 0, 0, 0.0, None, None, None, 0.0, None, None, None, 0.0),
            partial=PhaseCounts(0, 0, 0, 0),
        )

    fps_values = [b.avg_fps for b in budgets]
    partial = [b for b in budgets if not b.fully_recorded]

    return BudgetSummary(
        sessions=len(budgets),
        frames_total=sum(b.frames_total for b in budgets),
        duration_sec=sum(b.duration_sec for b in budgets),
        median_avg_fps=median(fps_values),
        fps_range=(min(fps_values), max(fps_values)),
        median_ema_fps=median([b.ema_fps for b in budgets]),
        stall_dominated=sum(1 for b in budgets if b.stall_dominated),
        implausible_rate=sum(1 for b in budgets if b.instant_high_fps > IMPLAUSIBLE_FPS),
        stages=_cohort(budgets),
        partial=PhaseCounts(
            sessions=len(partial),
            face_off=sum(1 for b in budgets if b.face.state == "disabled"),
            gaze_off=sum(1 for b in budgets if b.gaze.state == "disabled"),
            untimed_stages=sum(
                (1 if b.face.state == "untimed" else 0)
                + (1 if b.gaze.state == "untimed" else 0)
                + (1 if b.objects.state == "untimed" else 0)
                for b in budgets
            ),
        ),
    )


def _share(share: float | None, absolute: float | None) -> str:
    if share is None or absolute is None:
        return "not recorded in this cohort"
    return f"{share * 100:5.1f}%  ({absolute:.1f} ms/frame)"


def format_report(budget_set: BudgetSet) -> str:
    """Render the accounting for a terminal, gaps included rather than smoothed over."""
    lines: list[str] = []
    budgets = list(budget_set.budgets)

    if not budgets:
        lines.append("No session in the log carries a usable aggregate (frames_total and avg_fps).")
        for reason in budget_set.skipped:
            lines.append(f"  skipped: {reason}")
        return "\n".join(lines)

    summary = summarize(budgets)

    lines.append(
        f"{'session':<24}{'frames':>7}{'avg_fps':>8}{'ema_fps':>8}{'period_ms':>10}"
        f"{'face_ms':>9}{'gaze_ms':>9}{'obj_ms':>9}{'unattr_ms':>11}{'unattr':>8}  pacing"
    )
    for budget in budgets:
        lines.append(
            f"{budget.session_id:<24}{budget.frames_total:>7}{budget.avg_fps:>8.2f}"
            f"{budget.ema_fps:>8.2f}{budget.frame_period_ms:>10.1f}"
            f"{budget.face.cell:>9}{budget.gaze.cell:>9}{budget.objects.cell:>9}"
            f"{budget.residual_ms:>11.1f}{budget.residual_share * 100:>7.1f}%"
            f"  {budget.pacing}"
        )

    lines.append("")
    lines.append(
        f"{summary.sessions} session(s), {summary.frames_total} frames, {summary.duration_sec:.1f} s: "
        f"median {summary.median_avg_fps:.2f} fps (range {summary.fps_range[0]:.2f}-"
        f"{summary.fps_range[1]:.2f}), median EMA rate {summary.median_ema_fps:.2f} fps"
    )
    if summary.stall_dominated or summary.implausible_rate:
        lines.append(
            f"  {summary.stall_dominated} session(s) are stall-dominated (EMA rate above "
            f"{STALL_FACTOR:.0f}x the session average), so `period_ms` is a"
        )
        lines.append(
            "  session average spread over stalls, not the cost of an ordinary frame"
            + (
                f"; {summary.implausible_rate} has an unusable frame clock"
                if summary.implausible_rate
                else ""
            )
        )

    cohort = summary.stages
    lines.append("")
    if cohort.sessions:
        lines.append("Stage accounting, as a share of each session's mean period (median):")
        lines.append(
            f"  face detection   {_share(cohort.median_face_share, cohort.median_face_ms)}  "
            f"over {cohort.measured_face} session(s)"
        )
        lines.append(
            f"  gaze             {_share(cohort.median_gaze_share, cohort.median_gaze_ms)}  "
            f"over {cohort.measured_gaze} session(s)"
        )
        lines.append(
            f"  object detection {_share(cohort.median_objects_share, cohort.median_objects_ms)}  "
            f"over {cohort.measured_objects} session(s)"
        )
        lines.append(
            f"  unattributed     {cohort.median_residual_share * 100:5.1f}%  "
            f"({cohort.median_residual_ms:.1f} ms/frame)  - a remainder, not a measurement"
        )
    excluded = summary.sessions - cohort.sessions
    if excluded:
        lines.append(
            f"  {excluded} session(s) excluded from every line above: no recorded stage was timed "
            "there, so the"
        )
        lines.append(
            "  whole period is attributed to no one and there is no stage cost to compare against"
        )

    lines.append("")
    lines.append("What the record does and does not contain:")
    lines.append(
        "  recorded        face detection (`avg_detection_latency_ms`), gaze inference and, from"
    )
    lines.append(
        "                  schema 8 on, object detection (`avg_object_detection_latency_ms`)"
    )
    legacy = sum(1 for b in budgets if b.objects.state == "untimed")
    if legacy:
        lines.append(
            f"  NOT recorded    object detection in {legacy} session(s) written before schema 8: it ran"
        )
        lines.append(
            "                  on every frame but nothing timed it, so it sits in 'unattr_ms'"
        )
    lines.append(
        "  unattributed    capture, snapshot encoding, memory writes, HUD, idle, and object"
    )
    lines.append("                  detection in any session that carries no timing for it")
    if summary.partial.untimed_stages:
        lines.append(
            f"  untimed         {summary.partial.untimed_stages} stage(s) ran without a recorded latency "
            "and are shown as"
        )
        lines.append("                  '--', never as 0.0 ms: the timer measured a no-op")
    capped = sum(1 for b in budgets if b.pacing != "not-recorded")
    if capped:
        lines.append(
            f"  pacing          `fps_cap` is recorded for {capped} session(s), so idle time is split"
        )
        lines.append("                  from work where the loop was pacing itself")
    else:
        lines.append(
            "  pacing          `fps_cap` is absent from every record here, so idle time cannot be"
        )
        lines.append(
            "                  split from work even for a session that ran at its configured cap"
        )

    if budget_set.skipped:
        lines.append("")
        lines.append(
            f"Skipped rows with an aggregate but no measurable period ({len(budget_set.skipped)}):"
        )
        for reason in budget_set.skipped:
            lines.append(f"  - {reason}")

    noted = [b for b in budgets if b.notes]
    if noted:
        lines.append("")
        lines.append(f"Findings by session ({len(noted)}):")
        for budget in noted:
            lines.append(f"  {budget.session_id}")
            for note in budget.notes:
                lines.append(f"    - {note}")

    return "\n".join(lines)
