# Session handoff

**Date:** 2026-09-21
**Branch:** `dev` (last commit `392b5fc completed phase 2`)
**Written by:** Claude Code, at the end of a session that did **not** change any
production inference behaviour.

Read this top to bottom before starting tomorrow. §9 is the part to act on; §10 is the
commands to prove the state is what this file says it is.

---

## 1. What this session was scoped to do

The work continues `OBJECT_DETECTION_PLAN.md`. Object detection is being made more
robust on a CPU-only machine, and the whole plan is blocked on one thing: there is no
labelled, representative frame set to measure against.

You asked for two deliverables, in this order, and nothing else:

1. A **frame-set validator** — `python main.py validate-frames` — that mechanically
   checks a captured frame set against the spec, with unit tests.
2. A **latency distribution** — median and max per frame alongside the existing mean.

Then update `OBJECT_DETECTION_PLAN.md` where necessary, report actual results, and stop.

**Standing prohibitions that were in force all session and still apply:**

- Do not implement a surfboard deny-list.
- Do not implement per-class confidence thresholds.
- Do not make the Phase 3 label-policy decision.
- Do not fabricate or generate evaluation frames or labels.
- Do not claim the current dataset establishes representative expo-level precision or
  recall.

None of these were violated. No detector code was touched.

---

## 2. What was built

### 2.1 `frame_set_validation.py` — the validator (new module)

Turns the mechanical half of the frame-set spec into a command. Roughly 640 lines.

| Checked | Why it can't be left to inspection |
| --- | --- |
| Filename convention and parsing | The metadata that makes the coverage matrix checkable |
| Every image has a label file, every label file an image | A missing `.txt` reads as "no objects here", so every detection in that frame scores as a false positive |
| Class names and indices against the model vocabulary | `phone` does not match `cell phone`; it reports precision 0 and no error |
| Boxes normalised 0–1, non-degenerate, inside the frame | Pixel coordinates parse fine and match nothing |
| Height, brightness and sharpness floors | Exactly the frames `detection_bench.load_frames` silently drops |
| Coverage: per class × lighting state | A hole is invisible in an aggregate |
| Target-free frames exist, **in every lighting state** | Without them precision is not computable at all — it reads 1.00 by construction |
| Every frame contains a target class or is explicitly negative | An unlabelled frame is not the same as an empty one |

Exits non-zero when any error-level finding exists. `--json` writes the full finding
list for diffing between captures. The image probe is **injectable**, so the unit tests
need neither OpenCV nor real JPEGs.

**What it cannot check:** whether a label is *correct*, whether the collection is
biased, whether the right objects were chosen. Those stay `[human]` in the spec's
checklist and are the ones that let a set pass every automated check and still be
worthless. `[auto]` means "well-formed", never "right".

### 2.2 `detection_bench.py` — latency as a distribution

New `latency_stats()` returning `frames`, `mean_ms`, `median_ms`, `min_ms`, `max_ms`,
`first_frame_ms`, `mean_after_first_ms`. Reported beneath the existing mean by
`bench-detect`. `ms_per_frame` is now read from `latency["mean_ms"]` so the two cannot
drift apart.

The **cold first frame is named, not dropped** — it stays in the mean and is reported as
`first_frame_ms` against `mean_after_first_ms`. Silently discarding it would be the same
failure mode as `load_frames` discarding dark frames: the number improves and the reason
disappears.

Every latency figure remains **detection-only**: saved frames, one process, nothing else
running. Roughly a sixth of a live frame.

### 2.3 `OBJECT_DETECTION_FRAME_SET_SPEC.md` (new document)

What the labelled frame set must contain for Phase 3 to be answerable — target classes,
coverage matrix, negatives and controls, quality floors, label format, layout and naming,
provenance, fit/holdout split, a pre-handover checklist with `[auto]`/`[human]` markers,
and §12: what the set still will not establish.

The single most important point in it: **a set containing only true objects makes
precision uncomputable.** Every detection is correct by construction, so precision reads
1.00 regardless of the model. Negatives are as important as positives.

---

## 3. Current state — what is verified

All four of these were run, and these are the actual outputs:

| Command | Result |
| --- | --- |
| `python -m unittest discover -s tests -q` | **Ran 269 tests — OK** |
| `.\.venv-tools\Scripts\ruff.exe check .` | **All checks passed!** |
| `.\.venv-tools\Scripts\mypy.exe` | **Success: no issues found in 7 source files** |
| `python main.py validate-frames --root memory/snapshots` | **119 errors, 17 warnings, exit 1** |

mypy covers 7 source files rather than the previous 6, because `frame_set_validation.py`
was added to the enforced `files` list in `mypy.ini`. Note that `main.py` is listed there
but **not actually checked** — `[mypy-main] ignore_errors = True` — so the `main.py`
changes rest on tests, not on the type checker.

### 3.1 The validator run, in detail

Run against `memory/snapshots`, the project's own 59 saved monitoring frames:

```
59 frames, 59 target-free (100%), 0 classes labelled
target-free breakdown: 59 missing label file(s), 0 empty label file(s), 0 distractor-only

errors (119):
  59 × bad-filename    snap_<date>_<time>.jpg: expected 6 underscore-separated fields, found 3
  59 × missing-label   no label file at snap_<date>_<time>.txt
   1 × too-small       snap_2026-09-19_16-50-29.jpg: 64px tall, below the 200px floor

warnings (17):
   1 × no-targets-declared     no targets.txt, so coverage and negative checks were skipped
  16 × below-brightness-floor / blurred
```

**This is the correct answer for that directory.** Those are unlabelled monitoring
snapshots, not a frame set, and the validator refuses them. It confirms it reads real
JPEGs off disk and enforces against them. It says **nothing** about your labels — there
are none.

Two findings in there are worth carrying forward:

- **`too-small` is real.** `snap_2026-09-19_16-50-29.jpg` is 64 px tall, under the
  200 px floor, so `load_frames` drops it silently. That frame has been invisible to
  every benchmark run so far.
- **The dark-frame filter, by name.** Brightness 11.8, 0.0, 63.7 (×4), 84.1, 87.1 (×2)
  against a floor of 90; sharpness 1.2, 0.0, 44.6, 40.6 (×2) against a floor of 50. This
  is the §1 "dark rooms dominate the failure" risk, now visible as specific files
  instead of a count.

---

## 4. Files changed — and what is committed vs not

This matters: **some of the work is committed and some is not.** Verified against git.

### Already committed in `392b5fc completed phase 2`

| File | Contains |
| --- | --- |
| `detection_bench.py` | `latency_stats()`, per-frame timing, latency block in `format_report` |
| `main.py` | `cmd_validate_frames`, the `validate-frames` subparser and its dispatch |
| `mypy.ini` | `frame_set_validation.py` added to the enforced `files` list |
| `tests/test_detection_bench.py` | 5 `LatencyStatsTests` + 2 report tests |

### Uncommitted right now (4 files, `git status` shows ` M`)

| File | Uncommitted content |
| --- | --- |
| `frame_set_validation.py` | The multi-word-class-name parser fix, and the missing-vs-empty label file fix |
| `tests/test_frame_set_validation.py` | Both regression tests, plus fixture corrections |
| `OBJECT_DETECTION_FRAME_SET_SPEC.md` | §6b note on class names containing spaces |
| `OBJECT_DETECTION_PLAN.md` | §2g status lines and the verification section |

**The committed version of `frame_set_validation.py` still contains the parser bug.**
Confirmed: `git show HEAD:frame_set_validation.py` has neither `COORD_FIELD_COUNT` nor
`missing_label_files`. So if these four files were reverted, you would lose both fixes.

### Files created this session but not yet in git at all

- `SESSION_HANDOFF.md` (this file)

### Files from earlier in the session, already committed

`OBJECT_DETECTION_FRAME_SET_SPEC.md` exists in HEAD; its later edits are uncommitted.

---

## 5. Decisions made

| Decision | Rationale |
| --- | --- |
| **`imgsz=768` is the shipped default** (Phase 1) | 8 classes vs 5 on the project's own frames, at ~1.5× latency (59 → 86 ms). A deliberate, measured departure from "defaults equal today's behaviour" |
| **No model change** (Phase 2) | Every `yolo11` variant scored *worse* than `yolov8n` on classes and recall. The gate assumed a model would win; the honest outcome is that none did |
| **Labels are boxes, not counts** | The measured failure is up to six separate un-merged boxes in one frame. Count-only labels score that identically to a single box, collapsing the exact multiplicity the metric exists to detect |
| **Negatives are mandatory, ≈⅓ of frames** | Without them precision is uncomputable — it reads 1.00 by construction |
| **Split fit/holdout by scene, not by frame** | Frames from one 30-second burst are near-duplicates; a random split leaks the same person into both halves and the holdout stops being a holdout |
| **Cold first frame kept in the mean and reported separately** | Dropping it would improve the number and hide the reason |
| **Validator exits non-zero on failure** | So it can gate a pipeline rather than being advisory |
| **The image probe is injectable** | Keeps the module importable without OpenCV and the tests free of real images |
| **§11 items 1–5 must not start before the set exists** | Building the metric against no data is how the metric gets shaped by the code instead of by the problem |

**Decision left open, deliberately:** whether to build a *data-independent core* of the
metric consumer (IoU geometry, plus a loader for the format already frozen in §6b), or
build all of items 1–5, or hold to §11 exactly as written. I raised this conflict at the
end of the session — **I had recommended building the consumer, then found §11 forbids
it** — and you deferred the decision with the rest. It resolves itself once the frames
exist, because the constraint is satisfied by their existing. See §9 step 5.

---

## 6. Experiments and results

### 6.1 Measured latency (mean-only; predates the distribution work)

| Run | 768 | 640 | Frame set |
| --- | --- | --- | --- |
| Re-verification | **86 ms** | **59 ms** | 50 usable frames |
| Original | 92–158 ms | 60–98 ms | 45 usable + 2 reference |

Same 8-versus-5 class split, reference recall 1.00 in both.

**Why the distribution was needed:** §2e measured the *same configuration* at **438 ms**
and then **503 ms** — a 15% move on identical input, larger than several of the
differences being compared. A mean cannot separate that drift from a real difference
between two models.

### 6.2 The frame set is one scene, and it is curtains

This is the finding that blocks everything. The saved frames are a person seated in
front of floor-to-ceiling pale curtains, and the curtains read as `surfboard` — up to
**six separate un-merged boxes in a single frame**, peaking at **0.83 confidence**.

Consequences:

- Four models' worth of benchmarking was **inconclusive**, because over these frames a
  benchmark mostly ranks models by how they handle curtains.
- **Detection statistics are valid; precision and recall are not computable.** These
  frames carry no labels at all.
- The frame set **cannot discriminate models**. §6 Q1 is now answered the hard way: new
  frames showing the objects the system is expected to find are a precondition for any
  further detection work, not a nice-to-have.

### 6.3 Frame exclusions

Current: 59 frames in `memory/snapshots`, of which 9 fall below the brightness floor and
1 below the height floor. Earlier in the project, **38 of 83** saved frames were
effectively dark.

The interaction that matters: `DEFAULT_MIN_BRIGHTNESS = 90.0` silently discards exactly
the dark frames that test the documented "dark rooms dominate the failure" risk.

---

## 7. Bugs found and fixed

Nine test failures and two lint errors appeared on the first run of the new code. They
were **three root causes**, not nine problems — worth recording because fixing symptoms
one at a time would have taken nine attempts.

### 7.1 A genuine bug in the label parser (production code)

`parse_label_line` split the label line on whitespace and read the class from token 0.
**Fifteen COCO class names contain a space** — `cell phone`, `stop sign`, `hair drier`,
`potted plant`, `hot dog`, `traffic light`, `fire hydrant`, `parking meter`,
`sports ball`, `baseball bat`, `baseball glove`, `tennis racket`, `wine glass`,
`dining table`, `teddy bear`.

So `cell phone 0.744 0.612 0.058 0.091` parsed as six fields and was rejected as
malformed. **Every frame containing any of those fifteen classes would have failed
validation**, with a message about field counts rather than about anything being wrong.

This caused three of the nine failures, including a cascade: in
`test_a_well_formed_set_passes` the phone label failed to parse, so the frame looked
empty, which changed the negative-frame count, which wrongly fired `no-negatives` — an
error claiming precision was uncomputable, caused by a tokenizer.

Worth noting *how* it surfaced: **the file's own format example was rejected by its own
parser.** Fixed by reading coordinates from the end of the line (`COORD_FIELD_COUNT = 4`)
and taking the class name as whatever precedes them. A regression test covers five
multi-word names.

### 7.2 The validator's summary contradicted its own findings

Found by running it against `memory/snapshots`. It printed:

```
59 frames, 59 target-free (100%), 0 classes labelled
target-free breakdown: 59 empty label file(s), 0 distractor-only
```

while reporting `missing-label` 59 times. Both cannot be true. `frame_set_validation.py`
appended to `empty_label_files` whenever a frame had no parsed rows — and a **missing**
label file also leaves `rows` empty, so it landed in the same bucket as a deliberately
empty one.

Why it matters, and why it is the same failure class this module exists to catch: "59
empty label files" describes a set someone carefully labelled, marking every frame as
background. "There are no label files" describes a set nobody has labelled yet. Those
need entirely different actions, and the reassuring-sounding one was the wrong one.

Fixed by tracking `missing_label_files` and `empty_label_files` separately. Both still
count as target-free negatives, because a missing label genuinely does read as "nothing
here" — which is exactly why it is an error rather than a skip.

### 7.3 Test-side problems (mine, not the module's)

- Four CLI tests failed because `cmd_validate_frames` uses the real image probe and the
  suite's `cv2` stub returns `None` from `imread`, so every frame was reported
  unreadable. Fixed by patching the probe — what those tests exercise is exit codes and
  argument plumbing, not decoding.
- Three fixture errors: two fixtures had no target-free frame (so `no-negatives`
  correctly fired), and one truncation test miscounted its own expected remainder.
- Two ruff errors: an unused `# noqa: C901` (C901 is not in the selected rule set) and an
  unsorted import block.

### 7.4 The lesson

A check that has not been run is not evidence. Running the validator found a real
production bug on the first execution, and then found a second one when run against real
data. Both would have been invisible to inspection — the first was hidden by the tests
that were failing for their own reasons, the second only appeared on a set with no
labels at all.

---

## 8. What is still blocked

Everything downstream of the frame set.

| Blocked | Why |
| --- | --- |
| **Phase 3 — per-class thresholds and label policy** | Its gate is "measured precision/recall per class under the chosen policy". There are zero labels. Any threshold chosen now would be fitted to curtains |
| **The metric consumer** (spec §11 items 1–5) | `detection_bench.py` still has exactly one source of ground truth: `REFERENCE_LABELS`, a hardcoded dict of *counts* for two bundled images. There is **no labels loader, no IoU matching, and no precision code of any kind**. `match_reference` compares per-class counts against expected minimums — structurally incapable of expressing precision |
| **§6 Q2 — the end-to-end latency budget** | Needs the whole loop measured live under CPU contention, with gaze (~433 ms/call) dominating. The harness sees roughly a sixth of the frame. Different evidence entirely |
| **The surfboard deny-list** | Explicitly prohibited this session, and it would be a threshold fitted to curtains |
| **Any claim about expo-level precision or recall** | The current frames are one scene and carry no labels |

---

## 9. Tomorrow — exact next steps, in order

### Step 0 — confirm the state (commands in §10)

Do this first. If the four uncommitted files are missing, stop and tell me — the two
validator fixes are not in git.

### Step 1 — commit the uncommitted work

The parser fix and the label-file fix are real bug fixes to production code and are
currently sitting uncommitted. Commit them before doing anything else, so tomorrow's work
cannot lose them.

### Step 2 — capture and hand-label the frame set

Per `OBJECT_DETECTION_FRAME_SET_SPEC.md`. The shape:

```
frames/
  images/     2026-09-22T14-31-08_booth-a_lit_near_front_001.jpg
  labels/     2026-09-22T14-31-08_booth-a_lit_near_front_001.txt
  targets.txt
  provenance.md
```

Filename format is `<timestamp>_<venue>_<lighting>_<distance>_<angle>_<NNN>` — six
underscore-separated fields. Label format is YOLO: `class cx cy w h`, normalised 0–1.

**Shoot the negatives.** Empty scenes in every lighting state, the venue's real
distractor textures (the curtains, and whatever else), screens, glass and reflective
surfaces, crowds, near-misses. Target ≈⅓ of frames target-free. Without them precision
cannot be computed at all.

**Re-shoot the curtain scene on the deployment camera** as a regression control — that is
the measured failure, and the validator cannot tell a curtain from anything else.

### Step 3 — validate before measuring

```powershell
python main.py validate-frames --root frames
```

It exits non-zero on anything blocking, and names the file for each problem. Fix
everything it reports before any number is computed. A set that fails validation produces
numbers that look fine and are not.

Also work through the `[human]` items in spec §10. Those are the ones the validator
cannot see, and the ones where a set can pass every automated check and still be
worthless.

### Step 4 — record the fit/holdout split in `provenance.md`

Split by **scene or session**, keep it balanced across conditions, make sure negatives
appear in both halves, and freeze the holdout. If the set is too small to split, that is
the finding — report it and capture more rather than reporting a fitted number.

### Step 5 — resolve the §11 question, then build the metric consumer

Spec §11 says items 1–5 must not start before the set exists. Once it exists, that
constraint is satisfied and the question is what *scope* to build:

1. A labels loader — `.txt` files into per-frame box lists, with class-name mapping.
2. IoU matching — greedy pairing at a stated threshold (0.5 is conventional; **state
   whatever is chosen, because the number changes the result**), unmatched predictions as
   false positives, unmatched labels as false negatives.
3. Per-class precision, recall and F1, **plus the predicted-as breakdown for false
   positives** — the curtain case is real class none predicted `surfboard`, and a
   per-class false-positive count without the predicted-as breakdown cannot describe it.
4. A confidence sweep — precision/recall at several thresholds, so the operating point is
   chosen from a curve rather than asserted.
5. A fit/holdout switch — evaluate on one half, report on the other.

Tell me which scope you want and I will start there.

### Step 6 — only then, Phase 3

With measured precision/recall per class in hand, the label-policy decision becomes
answerable. Not before.

---

## 10. Commands to verify the current state

Run all six. The first four should reproduce §3 exactly.

```powershell
# 1. Test suite — expect: Ran 269 tests ... OK
python -m unittest discover -s tests -q

# 2. Lint — expect: All checks passed!
.\.venv-tools\Scripts\ruff.exe check .

# 3. Types — expect: Success: no issues found in 7 source files
#    (7, not 6: frame_set_validation.py is now in the enforced scope)
.\.venv-tools\Scripts\mypy.exe

# 4. Validator against the existing snapshots
#    expect: 119 errors, 17 warnings, exit 1, and the breakdown line reading
#    "59 missing label file(s), 0 empty label file(s), 0 distractor-only"
python main.py validate-frames --root memory/snapshots

# 5. Confirm which fixes are committed
#    The committed copy should NOT yet contain COORD_FIELD_COUNT (0 matches).
#    The working copy SHOULD (1 match). That difference is the parser fix.
git show HEAD:frame_set_validation.py | Select-String -Pattern COORD_FIELD_COUNT
Select-String -Path frame_set_validation.py -Pattern COORD_FIELD_COUNT

# 6. Confirm the four uncommitted files
git status --short
#    expect exactly:
#      M OBJECT_DETECTION_FRAME_SET_SPEC.md
#      M OBJECT_DETECTION_PLAN.md
#      M frame_set_validation.py
#      M tests/test_frame_set_validation.py
```

**If command 4 prints `59 empty label file(s)` instead of `59 missing label file(s)`**,
the missing-vs-empty fix is not in the working tree. Stop and tell me.

---

## 11. Files reference

| File | Role |
| --- | --- |
| `OBJECT_DETECTION_PLAN.md` | The phased plan. §2g covers this session's work |
| `OBJECT_DETECTION_FRAME_SET_SPEC.md` | What the labelled set must contain. §10 checklist, §11 what still has to be built |
| `frame_set_validation.py` | The validator module |
| `detection_bench.py` | The benchmark harness and `latency_stats()` |
| `tests/test_frame_set_validation.py` | ~50 tests, including the two regression tests |
| `tests/test_detection_bench.py` | `LatencyStatsTests` |
| `main.py` | `validate-frames` command |
| `mypy.ini` | The enforced file list |
| `SESSION_HANDOFF.md` | This file |

---

## 12. One-line summary

Two deliverables built and verified green — a frame-set validator and a latency
distribution — which found and fixed two real bugs; everything else is blocked on a
labelled, representative frame set that does not exist yet.
