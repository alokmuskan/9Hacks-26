# Object Detection Robustness Plan

Goal: make object detection recognise more real objects, reliably, without
exceeding the frame budget of a **CPU-only** machine.

Status: **Phases 0, 1 and 2 complete**, and the accuracy question has been answered as far
as the frames on disk allow. Phase 2 fixed the model-name fallback (§2d) and then benchmarked
`yolo11*` and **rejected** it (§2e). The harness gap that made that comparison inconclusive was
partly closed by a dataset validator and a latency distribution (§2g); the frame budget was
read out of the recorded sessions (§2h), which found that the session log records *face*
detection latency rather than object detection and could not close a per-frame budget — the
three fields whose absence was why are now recorded (§2h follow-up). The
labelled-frame-set block was worked around rather than accepted (§2i): reviewing the detector's
own output produced the project's first measured **precision — 0.911 over 124 boxes, 100%
reviewed** — and the one zero-precision class it found is now filtered by configuration.
Two further phases were **retired on measurement** rather than built (§2i).
**Recall is still unmeasured**, and the expo-level claim is still out of reach: it needs the
enumeration set in `OBJECT_DETECTION_FRAME_SET_SPEC.md`, which does not exist.
Every number below was measured on this machine (see *Evidence*), not assumed.

Re-verified after Phase 1 landed; that audit found several deliverables that were
claimed but not actually in place, and they are fixed (see *§2c Re-verification*).

| Phase | State |
| --- | --- |
| 0 — benchmark harness | ✅ done — `main.py bench-detect`, `detection_bench.py` |
| 1 — parameterise inference | ✅ done — env knobs, typed `DetectorConfig`, status + HUD reporting, defaults from the benchmark |
| 2 — model upgrade | ✅ done, gate half met — fallback fixed (§2d), `yolo11*` benchmarked and **rejected** (§2e) |
| 3 — per-class thresholds / label policy | 🟡 **answered for this scene** (§2i) — 0.911 precision measured, and the only zero-precision class (`surfboard`) is dropped by `AI_STUDIO_YOLO_EXCLUDED_CLASSES`. Expo-level policy still needs the labelled frame set |
| 4 — capture / low-light | ❌ **measured negative** (§2i) — CLAHE recovers nothing on the dark frames and loses two classes on the lit ones. Do not build |
| 5 — temporal smoothing | ⏸ **unverifiable with the frames on disk** (§2i) — snapshots are 15 s apart, so label flicker cannot be measured. Needs a short burst capture |
| 6 — cross-source dedupe / tiling | ⏸ dedupe has **nothing to fix** (§2i: 0 duplicate pairs at IoU ≥ 0.7, custom model off). Tiling untested |
| 7 — surfacing, docs, CI | 🟡 partly done — `frame-budget` (§2h) and `review-detections`/`score-detections` (§2i) are the surfacing; the CI benchmark job is still missing |

---

## 1. Evidence (measured on this machine)

**Hardware / runtime**

| Item | Measured |
| --- | --- |
| Torch build | `2.10.0+cpu` — **CUDA not available** |
| CPU | 12 cores; torch default 8 intra-op threads |
| Thread scaling, `yolov8n` @ imgsz 640 | 1 thread **75 ms** · 6 → 69 ms · 8 → 70 ms · 12 → 88 ms (flat) |
| Camera | **640×480 @ 30 fps**; requesting 1280×720 fails on the MSMF backend |
| Ultralytics / onnxruntime | `8.4.21` (supports v8, v9, v10, **v11, v12, v26**, RT-DETR) / `1.30.0` |
| Models on disk | `yolov8n.pt` only |

**Detector sanity check** (bundled reference images, current code path)

| Image | Result |
| --- | --- |
| `bus.jpg` 810×1080 | 6 boxes: bus 0.87, person ×4 0.87, stop sign 0.26 — 268 ms |
| `zidane.jpg` 1280×720 | 3 boxes: person ×2 0.84, tie 0.29 — 77 ms |

So the detector, the install and the wiring all work. The complaint is about
**recall on real scenes**, not a broken pipeline.

**On real saved frames from this project** (640×480; 45 of 83 frames met a
"usable" brightness filter, the rest are dark/blank)

| Configuration | Latency | Classes found |
| --- | --- | --- |
| `conf=0.25, imgsz=640` — **today's code path** | 106 ms | 3 — person ×7 (0.92), cell phone ×4 (0.46), remote ×1 (0.27) |
| `conf=0.15, imgsz=640` | 78 ms | 3 — cell phone ×7 (more hits, same classes) |
| `conf=0.25, imgsz=960` | 142 ms | 3 |
| `conf=0.15, imgsz=960` | 135 ms | **5** — adds cup (0.16), toothbrush (0.19) |

**Frame quality:** 38 of 83 saved frames are effectively dark
(mean brightness 11.8 → pure black; 63.7 → very dark), plus 64×64 placeholder
frames. Nothing is detectable in those, at any model or threshold.

**Frame budget context** (from this project's own `metrics_log.jsonl`):
gaze inference averaged **433 ms per call** while the whole loop ran at
**0.68 fps**. Object detection is ~70–106 ms. Gaze, not YOLO, is the dominant
cost — which is what makes a larger model affordable *if* gaze is throttled.

---

## 2. Root causes, ranked

1. **Inference parameters are never passed.** `DualYoloDetector._run_model`
   calls `model(frame, verbose=False)` — no `conf`, `imgsz`, `iou`, `max_det`,
   `agnostic_nms` or `classes`. Everything is Ultralytics' default
   (`conf=0.25`, `imgsz=640`), with **no per-class tuning possible**.
   Measured effect of exposing just two of these: **3 → 5 classes.**
2. ~~**Nano model.** `yolov8n.pt` is the weakest variant in the family; the
   installed Ultralytics supports v11/v12/v26, which are materially stronger
   at similar or modestly higher cost.~~ **Not supported by measurement — see
   §2e.** On this project's own frames every `yolo11` variant scored *worse* than
   `yolov8n` on both of the harness's metrics, and the largest was the worst value
   by a wide margin. The bottleneck was never model capacity.
3. **No measurement harness.** There is no repeatable way to say whether a
   change improved detection, so tuning is guesswork.
4. **Capture quality/size.** 640×480 capture, ~46 % of saved frames too dark to
   detect. `imgsz > 640` means upscaling — it still helps (cup/toothbrush
   appear) but cannot recover detail that was never captured.
5. **No temporal smoothing.** Every frame decides independently, so labels
   flicker — which reads as unreliability even when individual frames are fine.
6. **Cross-source duplicates.** General and custom models are concatenated and
   re-sorted with no NMS across sources, so the same object can appear twice.
7. **Silent model fallback bug.** ~~`_resolve_general_model_path` returns the
   configured value only *if the file already exists*, else `yolov8n.pt` — so a
   new model name set via `AI_STUDIO_GENERAL_YOLO_MODEL` is silently ignored,
   contradicting the README's "Ultralytics downloads it on first use".~~
   **Fixed in Phase 2 (§2d).** The audit found the same bug in a second place:
   `cmd_bootstrap` downloaded a hardcoded `yolov8n.pt` regardless of what was
   configured.

---

## 2b. Phase 0 + 1 results

**Harness:** `python main.py bench-detect` (45 usable frames + 2 labelled reference
images; 38 frames excluded as dark/blank/unreadable and reported by reason).
It drives the real `DualYoloDetector`, so results describe the production path,
and its default run compares the *shipping* configuration against the 640/0.25
baseline.

**Measured, shipping path vs baseline (same frame set, reference recall 1.00 both):**

| Configuration | Classes found | Latency per frame |
| --- | --- | --- |
| `imgsz=768` (new default) | **8** — person, remote, cell phone, toothbrush, surfboard, bottle, refrigerator, tie | 92–158 ms |
| `imgsz=640` (previous behaviour) | 5 — person, cell phone, remote, toothbrush, bottle | 60–98 ms |

Carried-item confidences also improve at 768 (remote 0.29 → 0.48). Latency varies
with machine state, so ranges are reported rather than single numbers; a first
run is always slower than steady state.

Re-verified later on a larger frame set (50 usable rather than 45, as snapshots
accumulated): **86 ms** at 768 versus **59 ms** at 640, same 8-versus-5 class
split, reference recall 1.00 both. The exclusion breakdown is now
`too_dark=9, too_small=1` — the `unreadable=28` entries were the stale 0-byte
files, since deleted (§2b).

**What changed in Phase 1**
- `common.py`: `AI_STUDIO_YOLO_IMGSZ` / `_CONF` / `_IOU` / `_MAX_DET` /
  `_AGNOSTIC_NMS`, with `normalize_imgsz()` clamping to 320–1920 in multiples of 32.
  The bounds (`YOLO_*_BOUNDS`) and `clamp()` live here once, and are applied by
  `DetectorConfig`, so a value cannot be valid on one path and invalid on another.
- `object_detection.py`: a typed, frozen `DetectorConfig`
  (`conf`, `iou`, `imgsz`, `max_det`, `agnostic_nms`) is the single definition of
  the inference parameters; its field defaults come from `common`, so the type, the
  env knobs and the benchmark cannot disagree. The detector now forwards
  `**config.as_kwargs()` on every call (previously it passed nothing at all),
  clamps through one `configure()`/`updated()` path, and reports the effective
  values in `get_state()`.
- `detection_bench.py`: the benchmark's own `DetectorParams` copy is gone — it
  reuses `DetectorConfig`, which removes the second set of hardcoded
  `iou`/`max_det` defaults that could silently drift from `common.py`.
- `server.py`: `/api/v1/monitor/status` exposes the effective values (`yolo.params`)
  and the HUD prints `Objects: <imgsz>px conf <conf>`; verified live.
- Docs: `README.md` (including the previously undocumented `bench-detect` command),
  `GETTING_STARTED.md`, `.env.example`.

**Measured trade, stated plainly:** the shipped default is `imgsz=768`, raised
from Ultralytics' 640 *on this evidence* — 8 classes versus 5 on the project's own
frames — and it costs roughly **1.5x** the per-frame object-detection latency
(59 → 86 ms in the re-verification run; 60–98 → 92–158 ms in the original). The
class gain is real, the latency cost is real, and the frame-budget context in §1
(gaze at ~433 ms dominates) is what makes it affordable. Setting
`AI_STUDIO_YOLO_IMGSZ=640` restores the old cost exactly.

**Incidental bug found by the harness:** 28 files in `unknown_incidents/` were
0 bytes. The current capture code writes valid 64 KB captures — those were stale
artifacts of an earlier version — but its `cv2.imwrite` return value was ignored,
so a failed write would leave exactly that kind of misleading empty file. The
capture path now verifies the write, deletes the reserved file on failure, and
reports it; three tests cover the failure modes. (The stale 0-byte files were
deleted during re-verification.)

## 2c. Re-verification of Phases 0 and 1

An audit of the delivered code against the specs above found the following. Each
line is either fixed, or recorded here as a known limitation rather than quietly
left to look like a pass.

**Deliverables that were claimed but not actually in place — now fixed**

| Finding | Fix |
| --- | --- |
| The Phase-0 wiring test never ran. All four `RealInferenceTests` (plus the sharpness test) were **skipped in the documented battery command and in CI**: the suite installs a `cv2` stub into `sys.modules` from an earlier module's `setUp`, and the availability probe rejected the stub. The test that exists to "guard refactors" guarded nothing. | The tests now *swap* the real `cv2` in for their duration and restore the stub afterwards, instead of probing what happens to be in `sys.modules`. A skip now means "no OpenCV/ultralytics here", nothing else. |
| `agnostic_nms` was listed in Phase 1's `DetectorConfig` and **did not exist** anywhere in the codebase. | Added as the fifth field, the `AI_STUDIO_YOLO_AGNOSTIC_NMS` knob (off by default = today's behaviour), forwarded on every call and reported in the status. |
| There was **no typed `DetectorConfig`**. The parameters were five loose attributes, and the benchmark defined a second near-copy with its own hardcoded `iou=0.7`/`max_det=300` — two definitions that could drift. | One frozen `DetectorConfig` in `object_detection.py`, used by both the detector and the benchmark. |
| The clamping bounds were duplicated between `common.py` and the detector. | `YOLO_*_BOUNDS` + `clamp()` in `common.py`, applied only by `DetectorConfig`. |
| `bench-detect` was undocumented in `README.md` and `GETTING_STARTED.md` — the harness existed but nothing told a reader it did. | Documented as *Detection Benchmark* in the README, with sample output and flags, plus a line in *Verify your install*. |
| `GETTING_STARTED.md` listed only 2 of the 4 knobs; the README's mypy paragraph omitted `detection_bench.py`; the test counts disagreed across three files (133 / 150 / 159 against an actual 198). | All corrected. Counts are no longer hardcoded where they will drift. |

**Verification (post-fix, on this machine)**

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests` | **Ran 198 tests, OK — 0 skipped** (was 190 with 5 skipped) |
| `ruff check .` | All checks passed. The first battery run had flagged a **real defect in the new test**, not a style preference: `tests/test_object_detection.py:164: B017 Do not assert blind exception`. It is fixed — the test now asserts `FrozenInstanceError`, which is what a frozen dataclass actually raises, instead of `Exception`, which every exception satisfies — and the re-run is clean. |
| `mypy` | Success: no issues found in 6 source files |
| `python main.py bench-detect` | 8 classes at 768 (86 ms) vs 5 at 640 (59 ms), reference recall 1.00 both; excluded frames now `too_dark=9, too_small=1` |

The skip count going 5 → 0 is the evidence that the Phase-0 wiring test is real
again. It had been reporting as skipped in the very command used to claim the gate.

The battery was run twice: once before the B017 fix (198 passed, ruff red) and
once after it (198 passed, ruff green), so the count above is not a stale result
from a different revision of the test file.

**Spec corrections — the specs, not the code, were wrong**

- Phase 0's gate said "zero production code touched". That was **not** met: the
  same commit changed `main.py` (the capture-write fix), `server.py` and
  `object_detection.py`, and additionally carried unrelated frontend work
  (`StreamImage`, `LiveMonitorPage`, `live.css`, `live-check.mjs`). The capture fix
  is disclosed above and the frontend work is harmless, but Phases 0–1 are not
  isolatable from it in history. The gate is reworded below to say what is
  actually required and what actually happened.
- Phase 1 said "defaults equal to today's behaviour so nothing changes until a
  value is chosen". The shipped default is `imgsz=768`, which **is** a behaviour
  change. It was made deliberately on the measured evidence in §2b and is recorded
  as such above, rather than pretending the spec was honoured.
- Phase 1's gate said "recall up at equal or better latency". Measured: classes
  5 → 8, reference recall 1.00 → 1.00 (the harness metric is blind to the gain,
  because both configurations already recall every reference label), latency
  **worse** by ~1.5x. The gate was unsatisfiable as written; it is restated below
  in terms of the trade that was actually measured.
- The harness is at `detection_bench.py` (repo root, alongside `common.py` and
  `object_detection.py`), not `tools/detection_bench.py` as written below.

**Known limitation, not fixed**

- CI (`.github/workflows/ci.yml`) installs only numpy/pillow/fastapi/httpx, so the
  detection-benchmark inference tests still report as **skipped** there. That is
  honest but it means the wiring guard runs locally only; wiring it into CI is
  Phase 7's job ("add a CI job running the benchmark smoke test").
- `_resolve_general_model_path` (root cause 7) was unfixed at this point, so the
  README's "Ultralytics downloads it on first use" was false for an env-set model
  name. **Fixed since — see §2d.**

## 2d. Phase 2 — the model-name fallback (done first)

Phase 2's second half is a benchmark of stronger checkpoints. That benchmark is
meaningless until the code stops ignoring which checkpoint it was asked to use, so
the fallback bug (root cause 7) was fixed first.

**The bug was in two places, not one.**

| Where | What it did |
| --- | --- |
| `_resolve_general_model_path` | Returned the configured value only if a file of that name already existed, else `yolov8n.pt`. `AI_STUDIO_GENERAL_YOLO_MODEL=yolo11s.pt` was therefore accepted, silently ignored, and reported as success. |
| `cmd_bootstrap` | Downloaded a **hardcoded** `YOLO("yolov8n.pt")` regardless of configuration, then printed `[ ok ] general YOLO ready: <whatever was configured>`. The one command whose job is to fetch the configured model was the one command guaranteed not to. |

The second one is what made the first hard to notice: `bootstrap` reported the
configured name as ready while placing a different checkpoint on disk.

**The fix** — the resolver is now honest, and fetching is a separate, reporting step:

- `_resolve_general_model_path()` returns what was configured, full stop. A bare
  name like `yolo11s.pt` is a *model name* (Ultralytics downloads it on first
  use), not a missing file. Substituting a different checkpoint is removed
  entirely, because a silent swap also corrupts benchmark output: the harness
  would print the name it was handed and measure something else.
- `_ensure_general_model()` performs the one-time download, or returns a reason it
  could not — offline, unknown name, unusable path. It never substitutes.
- `cmd_bootstrap` fetches the *configured* checkpoint and, on failure, prints the
  path **and** the reason and lists it under `Unresolved:` instead of claiming
  success.
- `doctor` now reports the configured checkpoint. Previously it could read
  `yolov8n.pt present` while the session was configured for something else, and it
  promised "Ultralytics downloads it on first run" even for a *path* that
  Ultralytics would never download. `_is_downloadable_model_name()` draws that
  distinction, so the row says "not downloaded yet" for a name and "not found" for
  a path.

**Tests** (`tests/test_environment_readiness.py`): a configured name is honoured;
an explicit model beats the configured default; only bare names count as
downloadable; an existing checkpoint is never re-downloaded; a failed fetch
returns its reason; `bootstrap` names the checkpoint it could not prepare and
lists it as unresolved; and `bootstrap` fetches the configured checkpoint rather
than a hardcoded one. The test that previously *asserted the buggy fallback*
(`test_general_model_resolves_to_a_usable_checkpoint`) is gone — it pinned
`_resolve_general_model_path(None) == "yolov8n.pt"` for a missing configured model,
i.e. it encoded root cause 7 as the contract.

**Status: implemented and verified.** `Ran 203 tests` / `OK` (was 198 — five net new)
and `ruff check .` clean. `mypy` is unaffected by construction: the only files
touched are `main.py`, which `mypy.ini` opts out via `[mypy-main] ignore_errors`,
and `tests/`, which is outside `files=`. The battery output shows the fix working
end to end — `[ ok ] general YOLO ready: ...\yolo11s.pt` is `bootstrap` fetching the
configured checkpoint rather than a hardcoded one.

The remaining Phase 2 work — benchmarking `yolo11n`/`s`/`m` on the 50-frame set — was
expected to be blocked on the open latency-budget question in §6. It was run anyway,
and the answer did not depend on the budget: **see §2e**.

## 2e. Phase 2 — model benchmark (`yolo11*` measured and rejected)

Same protocol as §2b: 50 usable saved frames plus the 2 labelled reference images, all
four models at `conf=0.25, imgsz=768`, driven through the production
`DualYoloDetector`. **Run twice.** Every class count and every box count below was
identical in both runs; only the latency moved, so it is given for both.

| Model | ms/frame (run 1 / run 2) | classes | reference recall | per-class detections |
| --- | --- | --- | --- | --- |
| `yolov8n.pt` | 85 / 79 | **8** | **1.00** | person 56, remote 8, cell phone 5, toothbrush 5, surfboard 2, tie 2, bottle 1, refrigerator 1 |
| `yolo11n.pt` | 77 / 77 | 5 | 0.80 | person 52, **surfboard 37**, cell phone 5, toothbrush 4, bottle 1 |
| `yolo11s.pt` | 177 / 166 | 5 | 0.80 | person 51, surfboard 8, cell phone 7, toothbrush 4, bottle 1 |
| `yolo11m.pt` | 438 / **503** | 6 | 0.80 | person 52, **surfboard 45**, toothbrush 10, cell phone 6, bottle 1, remote 1 |

**The premise did not survive contact.** Phase 2 was written on the assumption that
`yolov8n` was the weak link. Every `yolo11` variant scored worse on both headline
metrics, and the largest was the worst value by a wide margin.

**But the two headline metrics are too coarse to carry that conclusion either**, and
that is the more useful finding:

- *"classes found"* counts boxes above a threshold. `yolov8n`'s eighth class is
  `refrigerator x1 (max 0.28)`, plus `tie x2 (max 0.37)` — one or two marginal boxes
  sitting just above 0.25. "8 classes versus 5" reads as a large difference and is not
  one.
- *"reference recall"* is 4/5 against 5/5: **one label, across two images**. The
  harness now names it — `missed reference: bus.jpg:stop sign`, identically for all
  three `yolo11` variants — and §1 recorded `yolov8n` finding that same stop sign at
  **0.26** against a 0.25 threshold. So the whole "recall regression" is one box
  sitting essentially on the threshold line. Naming the label turned a number that
  read like a capability gap into a difference that is barely one.
- **Neither metric measures precision at all** — which is precisely where the real
  difference turned out to be.

### The frames are one scene, and it is curtains

The saved frames are a person seated in front of floor-to-ceiling pale curtains with
deep vertical folds. Three of them were opened to confirm this rather than assumed
from the metadata. It is not a varied scene.

That explains the one **robust, non-marginal** signal in the table, and the
per-frame statistics added during this phase sharpen it considerably. The surfboard
detections are not a steady one-per-frame drift; they arrive in **bursts**:

| Model | surfboard boxes | frames containing any | max in one frame | peak conf |
| --- | --- | --- | --- | --- |
| `yolov8n` | 2 | **1** / 50 | 2 | 0.31 |
| `yolo11n` | 37 | 18 / 50 | 4 | **0.83** |
| `yolo11s` | 8 | 3 / 50 | 5 | 0.78 |
| `yolo11m` | 45 | 16 / 50 | **6** | 0.79 |

Up to six *separate* `surfboard` boxes in a single frame, none merged by NMS, is what
several distinct curtain folds each being read as its own surfboard looks like. The
newer models do it far more confidently (0.83 against 0.31), and `yolov8n` does it in
one frame out of fifty. This is not threshold noise, and it is not visible in a box
count at all — 45 boxes reads identically whether it is 45 objects spread over 45
frames or six folds misread in sixteen.

**The confidence floor cannot be the fix, and the measurements say so.** For
`yolo11n`, the false `surfboard` peaks at **0.83** while real `cell phone` detections
peak at **0.64**. Any threshold that suppresses the curtains also suppresses the
phone. A per-class floor of "surfboard ≥ 0.85" would work, but it is a deny-list
wearing a threshold's clothes, and choosing it from these frames means fitting the
threshold to curtains. That is precisely the decision §2f says cannot be made yet.

### The one genuine regression

`remote` is named in §6 as an object that matters. `yolov8n` found **8** (max 0.48);
`yolo11n` found **0**, `yolo11s` found **0**, `yolo11m` found **1** (max 0.52). That is
a real miss at a threshold where the older model was finding them, not a marginal one.

### The one genuine gain

`cell phone` confidence rises monotonically with model size — 0.45 → 0.64 → 0.82 →
0.86 — on broadly the same boxes. The newer architectures are meaningfully more
certain about the phone, but it changed no class count and no recall number here.

### Latency

`yolo11n` and `yolov8n` are indistinguishable — 77 against 85 ms in the first run and
77 against 79 ms in the second, i.e. the two swap places, which is what noise looks
like. `yolo11s` is **~2.1x** in both runs (177/85 and 166/79). `yolo11m` is **5.2x** in
the first run and **6.4x** in the second (438/85 and 503/79), because it is the one
measurement that moved materially — 438 → 503 ms, +15%, on identical input.

Against the §1 frame budget (gaze ~433 ms per frame), `yolo11m` at 438–503 ms roughly
doubles total frame time in exchange for a net loss on every metric.

### Decision

**Keep `yolov8n.pt`.** No `yolo11` variant earned its cost on this evidence, and the
one with the strongest "bigger is better" claim is at least 5x the latency for fewer
classes and worse recall — a conclusion the run-to-run spread does not threaten, since
it is the *small* differences that the spread undermines, not this one.
`AI_STUDIO_GENERAL_YOLO_MODEL` remains the escape hatch, and it now actually works
(§2d).

What Phase 2 *did* establish is that **the harness cannot yet answer its own
question.** Root cause 3 ("no measurement harness") was only half fixed: the harness
exists and is reproducible, but it measures recall without precision, and it reported
an aggregate that hid which label regressed. Both were needed here and neither was
there.

### Harness changes made during this phase

Both were forced by findings the harness could not express, which is the pattern
worth noting: each time, a real result was invisible until the reporting changed.

| Change | What it fixed |
| --- | --- |
| `format_report` names the reference labels that were missed | `recall=0.80` said something regressed but not what; it turned out to be one marginal stop sign, not a capability gap. |
| `summarize_per_frame()` — detection statistics per class | A box count cannot distinguish one object seen 45 times from six false boxes in a single frame. Reports boxes/frame, frames-with-detection, frame coverage and max-in-one-frame, sorted busiest first, and labels itself **detection statistics, not precision**. |

`run_benchmark` now keeps each frame's detections separate instead of flattening them
immediately, and adds a `detection_stats` key. `summarize_detections` and every
pre-existing result key are unchanged, so the JSON output and the existing report
lines still read as before — the new block is additive.

### Still missing from the harness

- **No precision, and none is computable.** The reference images are labelled, so
  recall against them is a real measurement — over 5 labels in 2 images, which is
  enough to catch a catastrophic regression and nothing more. The project's own
  frames carry no labels at all, so every per-class number above is a *detection
  statistic*. A class at 0.9 boxes/frame may be a real object in almost every frame
  or a systematic false positive; only labelled frames separate those. No threshold
  should be set from these numbers.
- **The frame set cannot discriminate models.** §6's first question is now answered
  the hard way: these frames are one scene — a person, a phone and curtains — so a
  benchmark over them ranks models largely by how they handle curtains. Frames of the
  objects the system is actually expected to find are a precondition for any further
  work; Phase 3 onwards will hit the same wall.
- **Latency is a single mean, and it moves.** The same model and configuration
  measured **438 ms** in one run and **503 ms** in the next (~15%), which is larger
  than several of the differences being compared. `ms_per_frame` also folds the cold
  first frame into the average. Neither is fatal for the "is `yolo11m` affordable"
  question, where the gap is 5x, but neither supports a fine-grained budget claim.

## 2f. Phase 3 readiness

Phase 3 has **not been started** — there is no Phase 3 code in the repository, so
"run the Phase 3 pipeline" is not yet a thing that exists. What follows is what the
current evidence can and cannot support.

### The four kinds of number, kept apart

Conflating these is how a detection count turns into a "precision" claim by accident.

| Kind | Where it comes from | Status |
| --- | --- | --- |
| **Detection statistics** — boxes/frame, frame coverage, max-in-one-frame, confidence distribution | The project's 50 saved frames; no labels involved | Measured. Valid. Says nothing about correctness. |
| **Reference-label measurements** — `reference_recall`, and now *which* label missed | 5 labels over 2 bundled images with hand-verified contents | Measured and real, but the sample is two images. Catches catastrophic regressions; cannot rank models. |
| **Precision / per-class recall** | Requires labelled detections | **Not available. Not computed. Not estimated.** |
| **Conclusions about the expo** | Requires representative frames | **Cannot be made.** The frame set is one scene. |

### READY NOW

- Reproducible detection statistics per model and per class, including the burst
  structure that exposed the curtain false positives.
- Detection-only latency, to a resolution of "5x apart, yes; 15% apart, no".
- Recall against 5 verified labels, with the failing label named.
- The model decision, which is settled: **keep `yolov8n`** (§2e).
- **A diagnostic experiment**, worth recording because it is arithmetically free and
  it rules out the obvious Phase 3 design. Removing `surfboard` from the output takes
  the four models to **1.56 / 1.24 / 1.26 / 1.40** boxes per frame, from
  1.60 / 1.98 / 1.42 / 2.30 — one deny-list entry deletes 37–45 spurious boxes per 50
  frames from the newer models and 2 from `yolov8n`. **What this does not show** is
  that a deny-list is the right policy. It shows that on *this* scene the spurious
  boxes are concentrated in one class, which is a fact about curtains, not about the
  expo.

### BLOCKED

| Blocked item | What it needs |
| --- | --- |
| Any per-class confidence floor | Labelled frames. A floor fitted to curtains is a floor fitted to curtains. |
| Any allow/deny list justified by evidence | Same. The diagnostic above shows the shape of the problem, not the right policy. |
| Precision, per-class recall, false-positive rate | Frames with hand-labelled boxes. Nothing outside the 2 reference images has this. |
| Ranking `yolov8n` against `yolo11*` on scene recall | Frames containing the target objects, in expo-like lighting. |
| The §6 Q2 latency budget | See below. |

**On §6 Q2: the harness cannot measure it.** Q2 asks for an acceptable *end-to-end*
FPS. The harness measures object detection alone, on saved frames, in one process. It
does not measure, and cannot be tuned into measuring:

1. **The rest of the loop.** Capture, face recognition, gaze, memory writes and
   snapshot encoding are not exercised at all. §1 puts gaze at ~433 ms per call
   against detection's ~80 ms, so the harness sees roughly a sixth of the frame.
2. **CPU contention.** Detection alone may use all 12 cores; in the live loop it
   shares them with gaze and face recognition, so the costs do not simply add.
   Summing component times would be an estimate, and a wrong one — which is why none
   is given here.
3. **Gaze throttling.** `AI_STUDIO_GAZE_MAX_INTERVAL` moves the balance completely
   and is not exercised.
4. **A latency distribution.** The current figure is a single mean that includes the
   cold first frame; run-to-run spread on one configuration was ~15%.

Neither of the two cheap fixes has been made yet: reporting a median and a max per
frame instead of one mean, and reading the *existing* `metrics_log.jsonl` from a real
session, which already records `avg_fps` and gaze timings.

### NEXT ACTION

One step unblocks more than anything else: **capture a labelled frame set of the
objects the system is expected to find, in expo-like conditions** — the same objects
at a few distances and angles, in both lighting states, with the correct labels
recorded as they are captured.

That single artefact makes precision computable, makes per-class thresholds fittable
to something other than curtains, makes the model comparison meaningful, and gives
Phase 3 the evaluation set its gate ("measured precision/recall per class under the
chosen policy") assumes already exists. Nothing else on the blocked list becomes
reachable before it does.

**What that set must contain is specified separately**, in
[`OBJECT_DETECTION_FRAME_SET_SPEC.md`](OBJECT_DETECTION_FRAME_SET_SPEC.md) — scope and
target classes, the coverage matrix, the negatives and controls without which precision
is not computable at all, the label format, and a pre-handover checklist. It also
records a prerequisite that is easy to miss: **the harness cannot read a labelled frame
set today.** `detection_bench.py` has one source of ground truth, the hardcoded
`REFERENCE_LABELS` counts for the two bundled images, and no IoU matching or precision
code of any kind — so a loader and a box-matching metric have to be built before the
captured set can be used for anything.

## 2g. Dataset validator and latency distribution

Two pieces of groundwork done while the labelled set is being captured. Neither
touches the detector, and neither makes any Phase 3 decision.

### `validate-frames` — checking the spec mechanically

`frame_set_validation.py` plus `python main.py validate-frames --root frames`. The
spec's checklist is mostly checkable by machine, and the failures it guards against are
all silent ones, so leaving them to a human reading a list means they get skimmed.

| Checked automatically | Why it cannot be left to inspection |
| --- | --- |
| Every image has a label file, and every label file an image | A missing `.txt` reads as "no objects here", so every detection in that frame scores as a false positive |
| Class names and indices against the model vocabulary | `phone` does not match `cell phone`; it reports precision 0 and no error |
| Boxes normalised 0–1, non-degenerate, inside the frame | Pixel coordinates parse fine and match nothing |
| Height, brightness and sharpness floors | These are the frames `load_frames` silently drops, so the benchmark reports healthy numbers on whatever survived |
| Coverage: per class × lighting state | A hole is invisible in an aggregate |
| Target-free frames exist, in every lighting state | **Without them precision is not computable at all** — it reads 1.00 by construction |
| Filename convention, defined in spec §7 | The metadata that makes the coverage matrix checkable |

Validation failures exit non-zero. What it **cannot** check is anything requiring
judgement — whether a label is *correct*, whether the collection is biased, whether the
right objects were chosen. Those stay marked `[human]` in the spec's checklist, and
they are the ones that let a set pass every automated check and still be worthless.

**Status: written and unit-tested. Not yet run against a real frame set, because none
exists** — `frames/` is empty, and `memory/snapshots/` holds 83 unlabelled monitoring
snapshots whose `snap_<date>_<time>` names are not the spec §7 convention. That is not a
gap in the validator: those frames have no labels, so there is nothing for it to check
them against. It runs the day the captured set lands, which is the point of building it
first.

### Latency as a distribution

`latency_stats()` in `detection_bench.py`, reported by `bench-detect` beneath the
existing mean. One number was never enough: §2e measured the *same* configuration at
438 ms and then 503 ms — a 15% move on identical input, larger than several of the
differences being compared — and a mean cannot separate that drift from a real
difference between two models. The report now gives median, min and max alongside it.

The cold first frame is handled by **naming it, not dropping it**. It is left in the
mean and reported as `first_frame_ms` against `mean_after_first_ms`, so the size of the
warm-up effect is visible instead of being assumed away. `ms_per_frame` is unchanged and
is now read from `latency["mean_ms"]` so the two cannot drift apart.

Every latency figure here remains **detection-only**: saved frames, one process, nothing
else running. It is roughly a sixth of a live frame (§1 puts gaze at ~433 ms per call)
and the components contend for the same CPU rather than adding. **§6 Q2 is not answered
by this and is not closer to being answered** — it needs the whole loop.

**Status: written, unit-tested, green.** The distribution itself has not yet been read
off a live `bench-detect` run; the numbers below are from the existing mean-only runs.

### Verification — first run, and what it found

`python -m unittest discover -s tests -q` — **267 tests, 5 failures, 4 errors**.
`ruff check .` — **2 errors**. `mypy` — **`Success: no issues found in 7 source files`**.

Nine failures from three root causes, all fixed and re-run green (below):

1. **A genuine bug in the label parser.** `parse_label_line` split the line on
   whitespace and read the class from token 0, which breaks every class name containing
   a space — and COCO has fifteen (`cell phone`, `stop sign`, `hair drier`, …). So
   `cell phone 0.744 0.612 0.058 0.091` parsed as six fields and was rejected as
   malformed. This caused **three** of the nine: the parse test, and
   `test_a_well_formed_set_passes` via a cascade — a frame whose phone label failed to
   parse looked empty, which changed the negative-frame count, which wrongly fired
   `no-negatives`. The coordinates are now read from the end of the line and the class
   name is whatever precedes them. A regression test covers five multi-word names.
   Worth noting *how* it surfaced: the file's own format example was rejected by its
   own parser, which is the sort of contradiction that only a test finds.
2. **Four CLI tests failed** because `cmd_validate_frames` uses the real image probe and
   the suite's `cv2` stub returns `None` from `imread`, so every frame was reported
   unreadable. The probe is now patched in those tests: what they exercise is exit codes
   and argument plumbing, not decoding.
3. **Three test-fixture errors of mine**, not module bugs: two fixtures had no
   target-free frame (so `no-negatives` correctly fired), and one truncation test
   miscounted its own expected remainder (20 errors, not 21 — the 21st was
   `target-never-labelled`, which the fixture's own arity created).

Ruff's two: an unused `# noqa: C901` (C901 is not in the selected set) and an unsorted
import block in the new test file. Both fixed.

**Re-run after the fixes — green:**

```
python -m unittest discover -s tests -q   →  Ran 268 tests ... OK
.\.venv-tools\Scripts\ruff.exe check .    →  All checks passed!
.\.venv-tools\Scripts\mypy.exe            →  Success: no issues found in 7 source files
```

268 rather than the earlier 267 is the multi-word regression test added with the parser
fix.

The lesson is the one §2c already records: a check that has not been run is not
evidence, and running it found a real bug on the first try. The stronger version of that
lesson is here too — the nine failures were **not** nine problems. They were three, one
of them a genuine parser defect, and the rest were my own fixtures and my own test
harness. Fixing symptoms one at a time would have taken nine attempts; the second and
third root causes were only visible after the first was fixed, because a broken parser
was manufacturing failures in tests that had nothing wrong with them.

## 2h. The frame budget, read from the recorded sessions

§2f listed two cheap fixes that had not been made: a latency distribution, and "reading
the *existing* `metrics_log.jsonl` from a real session, which already records `avg_fps`
and gaze timings". The first landed in §2g. This is the second — `python main.py
frame-budget` (`frame_budget.py`) — which measures the live loop from data already on
disk: no camera, no labels, no re-run, and no change to production behaviour.

**Measured over the 12 recorded sessions** (1300 frames, 718.5 s of monitoring):

| Quantity | Measured |
| --- | --- |
| median session throughput | **0.83 fps** (range 0.68–9.39) |
| median EMA frame rate | **2.80 fps** |
| stall-dominated sessions (EMA above 2× the session average) | **5 of 12** — one reached 178 fps instantaneously against a 0.82 fps mean; another's frame clock is unusable (29,537 fps between frames) |
| face detection | **3.6%** of the mean period — 41.1 ms/frame, over the 8 sessions that ran it |
| gaze | **28.3%** — 334.5 ms/frame, over the 6 sessions that ran it |
| unattributed remainder | **69.1%** — 833.6 ms/frame |

Four findings, each verified in the code or in the record itself:

**1. `avg_detection_latency_ms` measures faces, not objects.** Both loops time
`core._detect(app, frame)` — face detection (`main.py:2799`, `server.py:1049`) — and call
`detector.detect(frame)` untimed on the very next lines. **Object detection appears in no
session record at all.** Every "detection latency" read out of the log so far has been
face-detection latency, and the field name does not distinguish them; §1's "object
detection is ~70–106 ms" is the *harness* measurement of the object detector, not this
field.

**2. A disabled stage reports a number instead of being absent.** With face recognition
off, `_detect` short-circuits on `app is None` — but the timer around it still runs and
`detection_calls` still increments. Two sessions record `avg_detection_latency_ms: 0.0`
with `max_detection_latency_ms: 0.01` and `detection_calls == frames_total` (324 and 362
frames). Read as `0 ms/frame` that says the stage became free; it is a timer measuring
nothing. `frame-budget` renders it `off` and never as `0.0`.

The same shape appears in gaze for the opposite reason: gaze is estimated per detected
face (`server.py:1155` gates it on non-empty face rows), so a session where no face is
found legitimately has no gaze cost. The report says which of the two it is rather than
leaving `0.0` to be interpreted either way.

**3. A session average is not a per-frame cost.** Five of the twelve sessions have an EMA
frame rate more than twice their session average, so their mean period is spread over
stalls rather than describing an ordinary frame. `bench-detect` already reports a median
and a max for this reason (§2g); the same distinction has to reach the session records
before a per-frame budget can be closed from them.

**4. Face recognition is associated with a 6–11× slower session, and the record does not
explain it.** Within a single revision (2026-09-19): `monitor-20260919-164022`, face
recognition off, ran **9.39 fps**; `monitor-20260919-185854` and `-213638`, face
recognition on with gaze never running, ran **0.82** and **0.76 fps**. Their recorded
face-detection cost is 11.3 and 10.8 ms/frame and their gaze cost is zero, so roughly
1200 ms/frame is attributed to nothing the log contains. This is stated as an association
in one revision, not a controlled comparison, and whether the missing time is work, a
stall, or an artifact of the mean is not answerable from the record — which is finding 3's
point as well.

**What this means for §6 Q2.** The question stays open, and §2f's claim that the harness
"sees roughly a sixth of the frame" does not survive the measurement: the detector's 86 ms
against the 1204 ms median period is about **7%**, nearer a fourteenth. The more useful
result is *why* neither source can answer it:

- the log records no object-detection timing, so the largest unmeasured share cannot be split;
- the log records no `fps_cap`, so idle pacing time cannot be told from work even for a
  session that ran at its configured cap;
- the log records a mean and an EMA but no per-frame distribution, and the two disagree by
  more than 2× in five of twelve sessions.

Closing it therefore needs instrumentation rather than analysis: time `detector.detect()`
into the aggregate, record `fps_cap`, and record a frame-duration distribution. *(All three
have since been recorded — see the **§2h follow-up** immediately after this section. The
diagnosis above stands as written: it is what the log looked like when this was measured, and
the sessions it describes still carry no object-detection timing.)* That is a
separate, small change, and it was **not** made here — this section changes no recorded
field and no runtime behaviour.

**Status: written, 30 unit tests, full battery green (299 tests, `ruff` clean, `mypy`
clean across 8 files). No Phase 3 decision is made or implied.**

### 2h follow-up: the three fields the log was missing

§2h concluded that the session log *cannot* close a frame budget, and named the three
reasons. All three are now recorded (session schema 7 -> 8):

| Field | Why it was needed |
| --- | --- |
| `object_detection_calls`, `avg_object_detection_latency_ms`, `max_object_detection_latency_ms` | Both loops called `detector.detect()` **untimed**, so object detection appeared in no session record at all. `avg_detection_latency_ms` was always face *recognition*, which the name concealed |
| `fps_cap` | Without the configured ceiling, `avg_fps` cannot be read as "held the cap" rather than "ran out of machine" -- and idle pacing time cannot be separated from work |
| `frame_period_p50_ms`, `frame_period_p95_ms` | A session averaging 0.82 fps contained a 178 fps instant. A mean cannot tell a slow pipeline apart from a stall-dominated one, and four of the twelve recorded sessions are stall-dominated |

`frame_budget.py` reads all three, and its `--` rendering is load-bearing: the presence of
`object_detection_calls` is the signal, because the twelve sessions already on disk ran the
detector on every frame without timing it. Those render `untimed` (`--`), never `off` ("the
stage did not run"), which would be a different and false statement. The report now prints how
many sessions predate the change instead of describing the old limitation as a current one.

**What this does and does not buy.** It makes the frame-budget question answerable from the
*next* session onwards. It cannot answer it for the sessions already recorded -- that data was
never captured and is not recoverable. No measurement in this document changes as a result.

## 2i. Closing the accuracy question without new data

The plan was blocked on a labelled frame set that cannot be captured or labelled. Before
building anything else, the three phases that were supposed to improve detection *without*
labels were measured on the frames already on disk. Two of them must not be built, and the
blocked one has a practical replacement.

### The three measurements

**Phase 6a (cross-source dedupe) has nothing to fix.** Over 79 usable frames and 101 boxes:
**0** same-class pairs at IoU ≥ 0.7, 0 degenerate boxes, 0 boxes outside the frame. The 6
pairs at IoU ≥ 0.5 are all `person` — two people standing close together — and the 2
cross-class pairs are a carried item overlapping its carrier, which is correct. Ultralytics'
NMS already merges same-class overlap inside each model call, and the custom model is
disabled, so cross-source duplication cannot occur at all today. Implementing the phase
would be fixing a non-problem.

**Phase 4 (low-light) is a measured negative.** CLAHE on the L channel of the LAB image,
applied to all 79 frames through the production detector:

| Subset | raw | CLAHE |
| --- | --- | --- |
| 12 dark frames (< 90 brightness) | 2 boxes on 2 frames | **2 boxes on 2 frames** — nothing recovered |
| 67 lit frames | 99 boxes, 8 classes | 100 boxes, **6 classes** — loses `refrigerator` and `surfboard` |

Latency is unchanged (81 → 84 ms lit; 96 → 94 ms dark). The dark frames are not dim, they
are unusable: brightness 11.8 and 13.0 are near-black and 63.7 is a covered lens, so there
is no detail to recover. On the frames that *are* usable it destroys two classes. The
phase's own gate anticipated exactly this — "if detection does not measurably recover, ship
it off by default and document that honestly" — so the recommendation is **do not build it**,
and retest only if the camera or the venue lighting changes.

**Phase 5 (temporal smoothing) cannot be verified with the frames on disk.** Snapshots are
**15 s apart** within a session (measured from the filenames: `18-59-02`, `-17`, `-33`,
`-49`) and hours apart between sessions. A short IoU tracker is meaningless at that
spacing, so "fewer label flips" has no denominator here. The phase is not wrong; it is
unverifiable, and its gate cannot be met honestly until someone captures a short burst.

### What replaces it: review the detections that already exist

`review-detections` → `score-detections` (`detection_review.py`). The detector's output on
frames that already exist is a *finite* list — **124 boxes over 75 usable frames** at the
time of writing — and a human can confirm or reject that list in a couple of minutes. This
is the practical alternative: it needs no new capture and no labelling of anything the
detector did not already propose, and it produces the project's first real **precision**
figure.

An accuracy number is exactly the kind of thing that gets laundered, so the rules are
explicit:

| Rule | Why |
| --- | --- |
| Unreviewed boxes stay unreviewed | Defaulting them to "correct" is the single change that would inflate the score |
| An unparseable verdict is skipped, not guessed | Same reason: a missing answer is not a positive one |
| A partial review prints a Wilson interval and a coverage warning below 80% | A subset picked by hand is not a random sample. `--limit` samples evenly across the confidence range rather than taking the easiest top-N |
| Noted misses are listed as misses, never as recall | They carry no denominator — the frames they came from were not chosen by any sampling rule |
| The scope is printed on every run | The number describes the frames under `reviews/` and nothing else |
| The page is git-ignored and `noindex` | It embeds real camera frames |

**What this does not fix.** Precision alone cannot tell whether detection got *better at
finding things* — that is recall, and recall needs every object in every frame enumerated.
So the honest position stays: detection can now be scored for **false boxes** on these
frames, and for nothing else. Phase 3's per-class policy becomes answerable **for this
scene** once a review is run; it remains unanswerable for the expo.

### The result: the first measured precision, and a rubric that mattered more than the detector

The review was run on the frames on disk: **124 boxes over 75 usable frames** (13 excluded —
`too_dark=12`, `too_small=1`), **100% reviewed → precision 0.911** (113 correct, 11 wrong).

| label | reviewed | correct | wrong | precision |
| --- | --- | --- | --- | --- |
| `person` | 96 | 89 | 7 | 0.927 |
| `remote` | 8 | 7 | 1 | 0.875 |
| `cell phone` | 7 | 6 | 1 | 0.857 |
| `toothbrush` | 5 | 5 | 0 | 1.000 |
| `surfboard` | 2 | 0 | 2 | **0.000** |
| `tie`, `book`, `bottle`, `refrigerator`, `snowboard` | 1–2 each | all | 0 | 1.000 |

**A first pass scored `person` at 0.645 (49/76) — same frames, same detector, same reviewer.**
The difference was not detection quality, it was the rubric: the first version of the page
never said what "Correct" meant, so duplicate boxes on one person and boxes clipped by the
frame edge were marked Wrong, which the rubric says to mark Correct. The rejected and
accepted boxes are statistically indistinguishable — median confidence 0.83 vs 0.87, median
area 28% vs 23% — and the person wrong-rate swung between **8.7% and 57.7%** across sessions
of the same scene, which is not what a confidence boundary looks like. The criteria are now
printed on the page itself, and the page's browser check asserts they are present.

This is recorded deliberately, because the other reading — "person detection is at 0.65, tune
the thresholds" — would have sent the project tuning a detector that was never the problem.

### What the result decided: the in-scope class knob

Only `surfboard` came back at zero precision, and both of its boxes were on curtain folds.
That is a policy fact rather than a threshold fact: no confidence threshold separates "curtain
fold" from "surfboard" without discarding real detections at the same time. So the decision
belongs in configuration, not in the model:

| Knob | Meaning |
| --- | --- |
| `AI_STUDIO_YOLO_IN_SCOPE_CLASSES` | Allowlist — keep only these labels |
| `AI_STUDIO_YOLO_EXCLUDED_CLASSES` | Denylist — drop exactly these labels |

Both are empty by default, which means *every label*, so nothing is filtered until one is set
deliberately. Exclusion wins over inclusion, so a label in both lists is dropped rather than
kept. Filtering happens once, in `DualYoloDetector.detect()`, so the live loop, the API, the
benchmark and the review all see the same set. `get_state()["in_scope"]` reports the effective
policy **and a per-label count of what was suppressed** — a filter that hides output silently
is indistinguishable from a detector that stopped working.

Measured on the same 75 frames with `AI_STUDIO_YOLO_EXCLUDED_CLASSES=surfboard`: **124 → 122
boxes**, `surfboard` gone, no other label changed by one box.

**This does not improve the detector, and must not be reported as if it did.** It removes boxes
the detector produced correctly for an object this project does not care about. The 0.911
figure above was measured *before* the filter, and the per-class policy is settled for **this
scene only**.

**Status: battery green — 361 tests (343 → 361), ruff clean, mypy clean across 9 files,
byte-compile clean. No Phase 3 decision is claimed for the expo: precision describes these
frames, and recall is still unmeasured.**

## 3. Phases

Every phase ends at a **gate**: new unit tests + the full backend battery
(`python -m unittest discover -s tests`, ruff, mypy) + the frontend checks where
UI is touched + a benchmark delta. No phase starts until the previous gate is
green.

A gate item only counts if it **runs**. A test that reports as skipped on the
machine that is claiming the gate is not evidence, and a gate whose wording no
measurement can satisfy is a bug in the plan (see §2c).

### Phase 0 — Measurement harness (no behaviour change)
- `detection_bench.py`: runs the detector over a fixed frame set — the
  project's own usable snapshots **plus** bundled reference images with known
  ground truth (`bus.jpg` → bus/person/stop sign; `zidane.jpg` → person/tie) —
  and reports per-class detections, mean confidence and ms/frame as a table.
- A fast test that asserts the wiring still finds a person in a reference image
  (guards refactors; it is not an accuracy test). It must **execute**, not skip,
  wherever the CV stack is installed.
- **Gate:** harness runs reproducibly; baseline recorded in this file; backend
  battery unchanged and green; **no change to the production inference path**.
  *Met, with the qualification in §2c:* the inference path was untouched, but the
  same commit also fixed the capture-write bug and carried unrelated frontend work,
  so Phase 0 is not isolatable from them in history.

### Phase 1 — Parameterise inference *(highest value per unit of risk)*
- A typed `DetectorConfig` (`conf`, `iou`, `imgsz`, `max_det`, `agnostic_nms`)
  forwarded into the Ultralytics call — one definition, shared with the benchmark.
- Env knobs (`AI_STUDIO_YOLO_CONF`, `AI_STUDIO_YOLO_IMGSZ`, …), clamped against
  one set of bounds, with the effective values exposed in
  `/api/v1/monitor/status` and the HUD.
- **Gate:** fake-model tests prove the kwargs are forwarded; env override/clamp
  tests; live checks and battery green; and a benchmark delta recorded in this
  file **stating both sides of the trade** (classes found *and* ms/frame), with the
  new default justified. *Met:* 5 → 8 classes at ~1.5x latency, recorded in §2b.
  Changing the default was a deliberate, measured departure from "defaults equal
  today's behaviour" — see §2c.

### Phase 2 — Model upgrade (measure, then decide)
- Benchmark candidates (`yolo11n`/`yolo11s`/`yolo11m`, `yolo26*` where
  applicable) on the same frame set: classes found, confidence distribution,
  ms/frame — and the end-to-end loop effect with gaze throttled.
- Fix the `_resolve_general_model_path` fallback so a model *name* is honoured
  and downloaded once (with a graceful, reported fallback when offline).
- **Gate:** chosen model fits the frame budget alongside the other detectors;
  measured recall gain; verified fallback with the network unavailable.
  *Half met.* The fallback is fixed and verified (§2d). The benchmark ran and
  **found no model worth switching to** — every `yolo11` variant lost on classes
  and recall, so the chosen model stays `yolov8n` (§2e). The gate assumed a model
  would win; the honest outcome is that the measurement says none did, and that the
  frame set could not have shown a real gain in any case.

### Phase 3 — Per-class thresholds and label policy
- Per-class confidence floors (a single global threshold either floods junk —
  cup/toothbrush at 0.15 — or misses real objects) plus optional allow/deny
  lists, with counts of what was filtered and why.
- **Gate:** measured precision/recall per class under the chosen policy;
  pure-function unit tests for the policy; no junk-class flood in the
  benchmark.

### Phase 4 — Capture and low-light robustness
- Attempt higher capture resolutions per backend (measured unsupported on
  MSMF here — retest with DSHOW/others), plus optional low-light
  preprocessing (CLAHE / auto-gamma) or camera brightness/exposure controls.
- **Gate:** measured on the **dark subset** of saved frames. If detection does
  not measurably recover, ship it **off by default** and document that
  honestly rather than claiming a win.

### Phase 5 — Temporal smoothing
- Short IoU tracker with label voting / confidence EMA so detections are
  stable frame-to-frame and the dashboard, HUD and memory metadata stop
  flickering. Post-processing only — no added inference cost.
- **Gate:** synthetic-sequence unit tests; live check shows fewer label flips;
  latency unchanged.

### Phase 6 — Cross-source dedupe and optional small-object tiling
- NMS across general + custom sources (removes duplicates).
- Optional tiled ("SAHI"-style) inference for small objects behind a flag, with
  its latency cost measured and stated; default off.
- **Gate:** no duplicate boxes on synthetic overlaps; tiling recall gain
  quantified with its cost.

### Phase 7 — Surfacing, docs, CI
- Show effective parameters, per-class confidences and a "why nothing was
  detected" diagnostic (frame brightness + threshold applied) in the UI.
- Document every knob in `README.md`, `GETTING_STARTED.md` and `.env.example`;
  add a CI job running the benchmark smoke test.
- **Gate:** docs match code; CI green.

---

## 4. Non-goals

- Training a custom model — already supported via `object_train` and the
  custom-YOLO path.
- Face recognition or gaze model changes.
- GPU-specific optimisations while this runtime is CPU-only.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| Frame budget: gaze already costs ~433 ms/frame | Pair bigger models with gaze throttling (`AI_STUDIO_GAZE_MAX_INTERVAL`); every phase measures ms/frame |
| Lower thresholds add false positives | Phase 3's per-class policy, measured in the benchmark |
| Benchmark overfits a small frame set | Reference images with known labels **plus** frames containing the objects that actually matter (to be supplied) |
| Dark rooms dominate the failure | Phase 4 gated on the measured dark subset |

## 6. Open questions

1. **Which objects matter most?** *Answered by §2e, the hard way.* The frames
   available here mostly contain a person, a phone and **curtains** — and a
   benchmark over them ranks models largely by how they handle curtains, which is
   how four models' worth of measurement ended up inconclusive. New frames showing
   the objects the system is expected to find are now a precondition for any
   further detection work, not a nice-to-have.
2. **Latency budget:** what end-to-end FPS is acceptable at the expo? This decides
   `imgsz` and which model variant is allowed. Still unanswered, and §2f records that
   the harness cannot answer it in its current form — it measures detection only,
   roughly a sixth of the frame (measured more precisely in §2h: about 7% of the median
   recorded period, and the recorded period is itself stall-dominated in five of twelve
   sessions). §2h also establishes that the session log could not answer it either, and names
   the three fields whose absence was why; those three are now recorded (§2h follow-up), so the
   question becomes answerable **from the next session onwards** — the twelve sessions already on
   disk still cannot be attributed, and no number in this document changes. §2e made the question
   less urgent for *model choice* (no variant earned its cost regardless of the budget), but it
   remains the gate for Phase 3's end-to-end verification and for any `imgsz` change.
