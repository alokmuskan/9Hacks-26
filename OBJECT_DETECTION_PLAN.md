# Object Detection Robustness Plan

Goal: make object detection recognise more real objects, reliably, without
exceeding the frame budget of a **CPU-only** machine.

Status: **Phases 0 and 1 complete and verified.** Phase 2 (model upgrade) is next.
Every number below was measured on this machine (see *Evidence*), not assumed.

| Phase | State |
| --- | --- |
| 0 — benchmark harness | ✅ done — `main.py bench-detect`, `detection_bench.py`, 21 tests |
| 1 — parameterise inference | ✅ done — env knobs, status + HUD reporting, defaults from the benchmark |
| 2 — model upgrade | ⏭ next |
| 3–7 | not started |

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
2. **Nano model.** `yolov8n.pt` is the weakest variant in the family; the
   installed Ultralytics supports v11/v12/v26, which are materially stronger
   at similar or modestly higher cost.
3. **No measurement harness.** There is no repeatable way to say whether a
   change improved detection, so tuning is guesswork.
4. **Capture quality/size.** 640×480 capture, ~46 % of saved frames too dark to
   detect. `imgsz > 640` means upscaling — it still helps (cup/toothbrush
   appear) but cannot recover detail that was never captured.
5. **No temporal smoothing.** Every frame decides independently, so labels
   flicker — which reads as unreliability even when individual frames are fine.
6. **Cross-source duplicates.** General and custom models are concatenated and
   re-sorted with no NMS across sources, so the same object can appear twice.
7. **Silent model fallback bug.** `_resolve_general_model_path` returns the
   configured value only *if the file already exists*, else `yolov8n.pt` — so a
   new model name set via `AI_STUDIO_GENERAL_YOLO_MODEL` is silently ignored,
   contradicting the README's "Ultralytics downloads it on first use".

---

## 2b. Phase 0 + 1 results

**Harness:** `python main.py bench-detect` (45 usable frames + 2 labelled reference
images; 38 frames excluded as dark/blank/unreadable and reported by reason).
It drives the real `DualYoloDetector`, so results describe the production path,
and its default run compares the *shipping* configuration against the 640/0.25
baseline.

**Measured, shipping path vs baseline (same 45 frames, reference recall 1.00 both):**

| Configuration | Classes found | Latency per frame |
| --- | --- | --- |
| `imgsz=768` (new default) | **8** — person, remote, cell phone, toothbrush, surfboard, bottle, refrigerator, tie | 92–158 ms |
| `imgsz=640` (previous behaviour) | 5 — person, cell phone, remote, toothbrush, bottle | 60–98 ms |

Carried-item confidences also improve at 768 (remote 0.29 → 0.48). Latency varies
with machine state, so ranges are reported rather than single numbers; a first
run is always slower than steady state.

**What changed in Phase 1**
- `common.py`: `AI_STUDIO_YOLO_IMGSZ` / `_CONF` / `_IOU` / `_MAX_DET`, with
  `normalize_imgsz()` clamping to 320–1920 in multiples of 32.
- `object_detection.py`: the detector now forwards `conf`, `iou`, `imgsz` and
  `max_det` on every call (previously it passed nothing at all), resolves them
  through one `configure()` path, and reports them in `get_state()`.
- `server.py`: `/api/v1/monitor/status` exposes the effective values and the HUD
  prints `Objects: <imgsz>px conf <conf>`; verified live.
- Docs: `README.md`, `GETTING_STARTED.md`, `.env.example`.

**Incidental bug found by the harness:** 28 files in `unknown_incidents/` were
0 bytes. The current capture code writes valid 64 KB captures — those were stale
artifacts of an earlier version — but its `cv2.imwrite` return value was ignored,
so a failed write would leave exactly that kind of misleading empty file. The
capture path now verifies the write, deletes the reserved file on failure, and
reports it; three tests cover the failure modes.

## 3. Phases

Every phase ends at a **gate**: new unit tests + the full backend battery
(159 tests, ruff, mypy) + the frontend checks where UI is touched + a
benchmark delta. No phase starts until the previous gate is green.

### Phase 0 — Measurement harness (no behaviour change)
- `tools/detection_bench.py`: runs the detector over a fixed frame set — the
  project's own usable snapshots **plus** bundled reference images with known
  ground truth (`bus.jpg` → bus/person/stop sign; `zidane.jpg` → person/tie) —
  and reports per-class detections, mean confidence and ms/frame as a table.
- A fast test that asserts the wiring still finds a person in a reference image
  (guards refactors; it is not an accuracy test).
- **Gate:** harness runs reproducibly; baseline recorded in this file; backend
  battery unchanged and green; zero production code touched.

### Phase 1 — Parameterise inference *(highest value per unit of risk)*
- A typed `DetectorConfig` (`conf`, `iou`, `imgsz`, `max_det`, `agnostic_nms`)
  forwarded into the Ultralytics call, with **defaults equal to today's
  behaviour** so nothing changes until a value is chosen.
- Env knobs (`AI_STUDIO_YOLO_CONF`, `AI_STUDIO_YOLO_IMGSZ`, …), clamped, with
  the effective values exposed in `/api/v1/monitor/status` and the HUD.
- **Gate:** benchmark shows recall up at equal or better latency; fake-model
  tests prove the kwargs are forwarded; env override/clamp tests; live checks
  and battery green.

### Phase 2 — Model upgrade (measure, then decide)
- Benchmark candidates (`yolo11n`/`yolo11s`/`yolo11m`, `yolo26*` where
  applicable) on the same frame set: classes found, confidence distribution,
  ms/frame — and the end-to-end loop effect with gaze throttled.
- Fix the `_resolve_general_model_path` fallback so a model *name* is honoured
  and downloaded once (with a graceful, reported fallback when offline).
- **Gate:** chosen model fits the frame budget alongside the other detectors;
  measured recall gain; verified fallback with the network unavailable.

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

1. **Which objects matter most?** The frames available here mostly contain a
   person, a phone and a remote — a benchmark needs frames containing the
   objects the system is expected to find.
2. **Latency budget:** what end-to-end FPS is acceptable at the expo? That
   decides `imgsz` and which model variant is allowed.
