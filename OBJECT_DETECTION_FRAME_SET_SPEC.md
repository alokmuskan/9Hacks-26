# Labelled expo-like frame set — capture and labelling requirements

Companion to `OBJECT_DETECTION_PLAN.md`. That document records what could **not** be
decided from the frames available today (§2e, §2f); this one specifies exactly what
the replacement frame set must contain for those decisions to become possible, and how
to check it before handing it over.

Nothing in here is a measurement. Everything marked *heuristic* is a judgement call,
labelled as one, and should be overridden by better information.

---

## 0. The rule that governs the whole set

**Every frame must be a real capture from the deployment camera path.** No stock
photography, no downloaded images, no crops of the existing frames presented as new
ones, no synthetic or augmented frames. A set that is 20% stock images produces a
precision number that is 20% fiction, and there is no way to tell from the output.

The existing 50 saved frames are **not** to be deleted or replaced. They become the
negative-control baseline (§4) — they are the only frames that contain the one failure
mode that has actually been measured.

---

## 1. What the set is for

| Decision currently blocked | The specific measurement it needs |
| --- | --- |
| Any per-class confidence floor | Per-class precision and recall **at several candidate thresholds** |
| Any allow-list or deny-list | Per-class false-positive counts, with the firing class named |
| Ranking `yolov8n` against `yolo11*` on scene recall | Per-class recall on frames containing the target objects |
| Phase 3's gate ("measured precision/recall per class under the chosen policy") | All of the above, on a **held-out** split (§10) |

## 2. Scope — write the target list first

Do this **before** capturing. A capture session planned around a guessed object list
produces a set that cannot answer the question it was built for.

### 2a. Fill this in

| Target object | Exact model class name | Must it be detected? | Notes |
| --- | --- | --- | --- |
| e.g. attendee | `person` | yes | |
| e.g. demo handset | `cell phone` | yes | |
| | | | |
| | | | |

"Exact model class name" means the string the detector actually prints — `cell phone`,
not `phone`; `stop sign`, not `sign`; `tv`, not `monitor`. A label that is misspelled or
paraphrased matches nothing and silently reports precision 0 and recall 0, with no
error message. This is the single most common way a labelled set becomes worthless.

### 2b. Check every target against the model's vocabulary

`yolov8n` detects 80 fixed COCO classes. **Anything not in that list cannot be detected
at all**, at any threshold, by any of the models benchmarked in §2e. If a target object
is not in the vocabulary, it is out of scope for this entire line of work and needs
fine-tuning instead (`python main.py train-objects --data <dataset.yaml>` already exists
for that path) — record it as a separate workstream rather than capturing frames for it.

Confirm the vocabulary rather than trusting this document:

```powershell
python -c "from ultralytics import YOLO; print(sorted(YOLO('yolov8n.pt').names.values()))"
```

Classes confirmed present in this project's own observed output so far: `person`,
`cell phone`, `remote`, `cup`, `toothbrush`, `bottle`, `tie`, `refrigerator`,
`surfboard` (as a false positive on curtains), `bus`, `stop sign`.

### 2c. Record the decision

State plainly, in the provenance file (§9): **which classes were in scope and which
were explicitly excluded, and why.** An excluded class that later turns out to matter
invalidates the threshold work, and the reason is only recoverable if it was written
down at the time.

## 3. Coverage matrix

Populate every cell. **A cell with zero frames is a hole**, and a threshold fitted
across a hole is fitted to whatever the other cells happened to contain.

For each in-scope class, capture frames across every condition:

| Condition axis | Levels to cover | Why it is on the list |
| --- | --- | --- |
| Distance | near / mid / far, at least 3 distinct distances | A class detected only at 0.5 m is not detected at an expo booth |
| Angle | frontal, ~45°, ~profile | Rotation is a classic recall cliff |
| Lighting | **every distinct lighting state the venue has** — including the worst one | §5 of the plan names dark rooms as a top risk |
| Occlusion | partly occluded by another object, partly out of frame | Real scenes; the existing frames have none |
| Crowding | target alone, and target among other objects | Precision depends on this far more than recall does |
| Motion | stationary, and moving (handheld / walking) | Motion blur is a different failure from defocus |

Record the actual distances and angles used; "far" is not a measurement.

*Heuristic, not measured here:* at least 8 usable frames per (class × condition cell),
and at least 30 labelled instances of each class across the whole set before fitting any
threshold to it. Below that you are fitting to individual frames. This number is a
starting point, not a statistical result — the count actually needed depends on the
effect size being looked for, and none has been measured yet.

## 4. Negatives and controls — the half that is usually missed

Precision is computed from frames where the object is **absent** at least as much as
from frames where it is present. A set containing only true objects **cannot measure
precision at all** — every detection in it is correct by construction, so precision
comes out at 1.00 and means nothing.

The known failure in this project is a false positive on a *texture*, not on a similar
object (floor-to-ceiling pale curtains with deep vertical folds, read as `surfboard`,
up to six separate boxes in one frame, peaking at 0.83 confidence — §2e). A set of
object photographs would not have caught it and will not catch its successor.

### 4a. Required negative blocks

- **Empty scene** — the booth with no target objects present, at every lighting state.
- **Distractor textures** — shoot whatever the venue actually has: curtains, blinds,
  folding screens, drapes, banners, posters, slatted walls, grilles, foliage, fabric
  swags. These are the direct analogue of the curtain failure.
- **Screens and reflections** — monitors, TVs, glass, polished floors, mirrors,
  phone screens showing images. Screens can contain *pictures of the target objects*,
  which is a genuinely ambiguous case: decide explicitly whether a depicted object
  counts as a detection, and write the decision down.
- **Crowd** — many people and objects, where the model must separate instances.
- **The curtain scene, re-shot on the deployment camera.** This is the regression
  control for the one failure that has been measured. Without it, there is no way to
  tell whether a later policy change fixed the curtains or merely moved them.
- **Near-misses** — objects visually similar to the targets but not targets.

*Heuristic:* negatives should be roughly a third of the set, spread across the same
lighting and distance conditions as the positives. Frames that are negative for one
class are usually positive for another; label every class present in every frame, not
just the class the frame was captured for.

## 5. Frame quality — the filter that will silently eat your dark frames

`detection_bench.load_frames` rejects any frame failing **all three** of
`detection_bench.py:56-58`:

| Threshold | Default | Effect |
| --- | --- | --- |
| `DEFAULT_MIN_BRIGHTNESS` | 90.0 | Frame is excluded as `too_dark` |
| `DEFAULT_MIN_SHARPNESS` | 50.0 | Frame is excluded as `blurred` |
| `DEFAULT_MIN_HEIGHT` | 200 px | Frame is excluded as `too_small` |

This matters more than it looks. §1 records that **38 of 83 existing saved frames were
effectively dark and were filtered out**, and §5 lists "dark rooms dominate the failure"
as a top risk. Those two facts interact badly: the default filter will silently discard
exactly the frames that test the documented failure mode, and the benchmark will then
report healthy numbers on the well-lit remainder.

So:

- Capture the dark state **deliberately and knowingly**, and say in the provenance file
  how many frames it contains.
- Evaluate the dark subset **separately** with a lowered floor
  (`--min-brightness 40`), and report it as its own row. Never merge it into a headline
  number.
- If the venue's dark state genuinely falls below what any model can detect in, that is
  itself the finding — record it as such rather than dropping the frames.
- Check the harness's own accounting: `bench-detect` prints
  `Frames : N usable, M excluded` and a per-reason breakdown. If `excluded` is large,
  the headline numbers describe a different set from the one you captured.

The camera is 640×480 (§1), which is 480 px tall and clears the 200 px floor — but no
frame in this set should be a placeholder or a thumbnail. The existing set contains
64×64 placeholders; the new one must not.

## 6. Labels — format, and why counts are not enough

### 6a. What each level of labelling buys

| Level | Format | Unlocks | Misses |
| --- | --- | --- | --- |
| **Counts only** | `frame.jpg, cell phone, 2` | Per-class recall; frame-level precision ("did the class occur here at all") | Cannot see *how many* boxes were spurious in a frame |
| **Boxes** | YOLO `class cx cy w h`, normalised 0–1 | True per-box precision/recall at an IoU threshold; false positives counted per class; a precision/recall curve across confidence | Nothing that matters here |

**Use boxes.** The reason is specific to this project, not a general preference. The
measured failure is *six separate un-merged `surfboard` boxes in a single frame* (§2e).
Count-only labels would record "this frame contains no surfboard" and score that frame
as one error regardless of whether the model fired one box or six — collapsing exactly
the multiplicity that distinguishes a real object from curtain folds. The metric would
be blind to the one thing it exists to catch.

### 6b. Format

YOLO format, one `.txt` per image, same basename:

```
# 2026-09-22T14-31-08_booth-a_lit_near_001.txt
# class cx cy w h   — all normalised 0..1, cx/cy = box centre
person      0.512 0.480 0.211 0.640
cell phone  0.744 0.612 0.058 0.091
```

- One row per instance. No row for an absent class — absence is expressed by omission.
- `cx cy w h` normalised to image width/height, not pixels.
- Class **index** (0–79) is also acceptable and is what YOLO tooling writes natively;
  if you use indices, ship the index→name mapping alongside so it can be checked.
- Partial objects at the frame edge **are** labelled, with the visible extent only, and
  flagged in the notes. They are a real and common case and the model should be
  measured on them.

### 6c. Labelling rules to fix in advance

Write these down before labelling starts, because retrofitting them means re-labelling:

- **Minimum visible size** — below what fraction of the frame is an object too small to
  require detection? (Decide; state it; apply it consistently.)
- **Occlusion** — how much of an object must be visible to be labelled?
- **Reflections and depictions** — does a person on a poster, or a phone on a screen,
  count? (§4a)
- **Crowds** — how are overlapping people labelled individually?
- **Borders** — objects cut off by the frame edge.

### 6d. Quality control on the labels

- **Re-label a random 10% twice and compare.** Report the disagreement rate. If labels
  disagree with themselves, no metric computed from them means anything. This is the
  cheapest possible check on whether the numbers are trustworthy and it is routinely
  skipped.
- **A second person labels an overlapping subset** if one is available. Same purpose.
- **Open every image and confirm the label file is non-empty-or-correctly-empty.** A
  truncated or missing `.txt` reads to the harness as "nothing here", which scores as a
  false positive on every detection in that frame. Silent, and it looks like a model
  failure.

## 7. Layout and naming

```
frames/
  images/   2026-09-22T14-31-08_booth-a_lit_near_001.jpg
  labels/   2026-09-22T14-31-08_booth-a_lit_near_001.txt
  targets.txt
  provenance.md
```

`targets.txt` is the machine-readable form of the §2a table: one exact model class
name per line, `#` for comments. `python main.py validate-frames` reads it to check
coverage and the negative-frame requirement, and cannot run those two checks without
it. A flat directory with images and labels side by side is also accepted, because
that is what most labelling tools write.

The filename encodes the metadata so the coverage matrix (§3) is checkable without
opening every image. Use a fixed, documented field order:

```
<timestamp>_<venue>_<lighting>_<distance>_<angle>_<NNN>
```

- `lighting` — `lit` / `dim` / `dark` (matching whatever states §3 identified)
- `distance` — `near` / `mid` / `far`, or the actual metres
- `angle` — `front` / `q45` / `profile`
- `NNN` — zero-padded sequence, unique within that combination

Keep the raw frames. Do not pre-resize, re-compress, re-encode or "clean up" images
before labelling — the captured frame is what the detector will see, and any processing
between capture and evaluation invalidates the comparison with the live loop.

## 8. Provenance record

`provenance.md` must state, and the numbers here are what make the set above
reproducible:

| Field | Value |
| --- | --- |
| Capture date(s) and local time range | |
| Venue and booth identifier | |
| Camera and resolution actually used | |
| Lighting states present, and which frames represent each | |
| Target classes in scope, and the exact model class names (§2) | |
| Classes explicitly excluded, and why (§2c) | |
| Distances and angles used | |
| Total frames captured / usable / excluded, with exclusion reasons | |
| Negatives as a fraction of the set | |
| Who labelled, when, and against which version of the rules in §6c | |
| Label disagreement rate from the §6d re-label check | |
| Known problems with the set | |

That last row is not a formality. A set with a known weakness that is written down can
still be used carefully; the same weakness undocumented produces a confident wrong
answer later.

## 9. Split: fitting set and held-out set

**A threshold fitted on a frame set cannot be validated on that same frame set.**
Reporting "precision 0.94 at conf 0.4" when 0.4 was chosen by looking at those frames
measures how well the threshold was fitted, not how well it will work at the expo. This
is the most likely way for this work to produce a number that is confidently wrong.

So:

- Split the set into a **fit** part and a **holdout** part **before** any threshold work.
- Split by **capture session or scene**, not by random frame. Frames from the same
  30-second burst are near-duplicates; splitting them at random puts the same person in
  both halves and leaks.
- Keep the split roughly balanced across every condition axis in §3, and make sure the
  negatives appear in both halves.
- Freeze the holdout. Do not look at it, tune on it, or re-split it because the fit
  half gave an awkward answer.
- Record which files are in which half, in `provenance.md`.

If the set is too small to split, that is the finding — report it and capture more,
rather than reporting a fitted number.

## 10. Pre-handover checklist

Most of this is now checked mechanically. Run:

```powershell
python main.py validate-frames --root frames
```

It exits non-zero when anything blocking is wrong, and reports each problem with the
file it is in. It checks the structural half of this list — filename convention,
image/label pairing, class names against the model's vocabulary, normalised
coordinates, the height and brightness floors, coverage holes, and whether the
negatives exist. It **cannot** check the half that needs a human: whether the labels
are *right*, whether the collection is biased, or whether the right objects were
chosen. Those remain manual, and are marked **[human]** below.

Run through this before the set is used. Each item is checkable by inspection.

**Scope**
- [ ] **[human]** Every in-scope target has its exact model class name recorded (§2a)
- [x] *auto* — Every in-scope target checked against the model's 80-class vocabulary (§2b)
- [ ] **[human]** Out-of-scope classes and the reason are recorded (§2c)

**Coverage**
- [x] *auto* — Every declared target appears in at least one frame; per-class × lighting
      counts are tabulated
- [ ] **[human]** Every (class × condition) cell in §3 has at least one frame — the
      validator checks lighting and presence, not distance, angle, occlusion or crowding
- [ ] **[human]** All distinct venue lighting states are represented, including the worst
- [ ] **[human]** Distances and angles are recorded as actual values, not `near`/`far`

**Negatives**
- [x] *auto* — Target-free frames exist at all, and exist in each lighting state that
      has frames
- [x] *auto* — Their proportion is reported, and flagged below the §4a recommendation
- [ ] **[human]** Empty-scene frames exist, in every lighting state
- [ ] **[human]** The venue's real distractor textures were shot (curtains and whatever else)
- [ ] **[human]** Screens, glass and reflective surfaces are present
- [ ] **[human]** The curtain scene was re-shot on the deployment camera as a regression
      control — the validator cannot tell a curtain from anything else

**Frames**
- [ ] **[human]** No stock, downloaded, synthetic or augmented images (§0)
- [ ] **[human]** No placeholders, thumbnails, or re-encoded/resized images
- [x] *auto* — Frames below the 200 px height floor are listed by name
- [x] *auto* — Frames below the brightness and sharpness floors are listed by name, and
      dark-state frames are marked as expected rather than as mistakes (§5)
- [x] *auto* — Undecodable images are reported

**Labels**
- [x] *auto* — Class names and class indices checked against the model's vocabulary
- [x] *auto* — Boxes are normalised 0–1, non-degenerate, and inside the frame
- [x] *auto* — Every image has a corresponding label file, **including frames with no
      objects** — a present-but-empty file, not a missing one (§6d)
- [x] *auto* — Orphan label files with no image are reported
- [ ] **[human]** Labelling rules from §6c were fixed in writing before labelling started
- [ ] **[human]** The 10% re-label check was run and the disagreement rate recorded — the
      validator reads labels but cannot tell a right label from a wrong one
- [ ] **[human]** Objects at the frame edge and partially occluded objects are labelled,
      per §6c — the validator flags boxes past the edge but cannot know if they are right

**Records**
- [ ] **[human]** `provenance.md` is complete, including known problems (§8)
- [ ] **[human]** Fit/holdout split is recorded, split by session or scene, holdout frozen (§9)

The **[human]** items are the ones where the set can pass every automated check and
still be worthless: labels that are confidently wrong, a collection that only contains
scenes where the model already works, or two halves of a split that share the same
person. No validator can see those.

## 11. What still has to be built to consume this

**The harness cannot read a labelled frame set today.** This is a real gap, not a
formality, and it should be expected to take as long as the capture.

`detection_bench.py` currently has exactly one source of ground truth:
`REFERENCE_LABELS` (lines 49–52), a hardcoded dict of *counts* keyed to the two images
bundled with Ultralytics. There is no loader for an external labels file, no IoU
matching, and no precision computation anywhere in the module. `match_reference`
compares per-class **counts** against expected minimums — enough for the coarse recall
figure in §2e, structurally incapable of expressing precision.

Needed, roughly in order:

1. **A labels loader** — read the `.txt` files in §6b into per-frame box lists, with the
   class-name mapping.
2. **IoU matching** — greedily pair predictions to labels at a stated IoU threshold
   (0.5 is conventional; state whatever is chosen, because the number changes the
   result), counting unmatched predictions as false positives and unmatched labels as
   false negatives.
3. **Per-class precision, recall and F1**, plus the confusion of which class the false
   positives were predicted *as* — the curtain case is exactly this: real class none,
   predicted `surfboard`. A per-class false-positive count without the predicted-as
   breakdown cannot describe it.
4. **A confidence sweep** — precision/recall at several thresholds, so the operating
   point is chosen from a curve rather than asserted.
5. **A fit/holdout switch** — evaluate on one half, report on the other (§9).
6. **Per-frame latency as a distribution, not a mean** — median and max, since the
   current single mean includes the cold first frame and moved ~15% between identical
   runs (§2e).

Items 1–5 are Phase 3 work and must not be started before the set exists; building them
against no data is how the metric gets shaped by the code instead of by the problem.
Item 6 is independent of the frame set and can be done at any time.

## 12. What this set will still not establish

Stated up front so it is not discovered later as a disappointment:

- **§6 Q2, the end-to-end latency budget.** Unchanged by any of this. A labelled frame
  set measures detection quality; Q2 asks for whole-loop FPS under CPU contention, with
  gaze (~433 ms/call against detection's ~80 ms) dominating. The harness sees roughly a
  sixth of the frame. Different evidence entirely — see §2f.
- **Generalisation beyond one venue.** One booth, one camera, one week is still one
  venue. A threshold that is right for this booth may be wrong for the next one.
- **Anything about the excluded classes** (§2c), or about the ~80 COCO classes that are
  not targets.
- **Whether the collection itself is biased.** The person choosing the frames decides
  what the model is measured on. Shooting only scenes where the model is visibly working
  is the failure mode, and no metric computed afterwards can detect it.
