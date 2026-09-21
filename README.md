# Intelligent Monitoring Platform

A full-stack, real-time computer-vision monitoring system that combines **face recognition**, **object detection**, and **gaze estimation** into a single pipeline, then records **who looked at what, and for how long**.

The system ships two front ends over one shared core:

- a **CLI / desktop mode** (`main.py`) with an OpenCV window, HUD, and live keyboard controls;
- an **API mode** (`server.py`) exposing REST, WebSocket, and MJPEG endpoints for the bundled React dashboard.

**Status:** `Milestone: Frontend and Backend Working`

> **New here?** [GETTING_STARTED.md](GETTING_STARTED.md) is the short, verified path
> from clone to a live session (Windows pip and Linux pixi). This README is the
> full reference.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](GETTING_STARTED.md)
- [Deployment](DEPLOYMENT.md)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Model Assets and First-Run Downloads](#model-assets-and-first-run-downloads)
- [Usage](#usage)
  - [API Mode (Backend)](#api-mode-backend)
  - [Frontend Dashboard](#frontend-dashboard)
  - [First Run](#first-run)
  - [CLI Mode](#cli-mode)
  - [Gaze Scheduling](#gaze-scheduling)
  - [Runtime Keyboard Controls](#runtime-keyboard-controls)
- [API Reference](#api-reference)
- [WebSocket Events](#websocket-events)
- [Data and Storage Model](#data-and-storage-model)
  - [Storage Lifecycle](#storage-lifecycle)
- [Testing](#testing)
  - [Lint and Types](#lint-and-types)
  - [Continuous Integration](#continuous-integration)
  - [End-to-End](#end-to-end)
- [Troubleshooting](#troubleshooting)
- [Current Limitations](#current-limitations)
- [Security and Privacy Notice](#security-and-privacy-notice)
- [Related Documentation](#related-documentation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

The platform ingests a live camera stream and runs three models per processed frame:

| Stage | Model / Library | Output |
| --- | --- | --- |
| Face detection + recognition | InsightFace (`buffalo_sc` by default) | Identity label + similarity score, or `Unknown` |
| Object detection | Ultralytics YOLO (general + optional custom model) | Labelled bounding boxes with confidence |
| Gaze estimation | L2CS-Net (ResNet50 by default) | Pitch/yaw angles and a predicted gaze endpoint |

Gaze endpoints are fused with detected object boxes to infer an **attention target**. A temporal tracker stabilises those observations into discrete `start` / `switch` / `end` **behavior events**, which are persisted together with periodic JPEG **memory snapshots** and an append-only **metrics log**.

On top of that data, the platform provides a **query layer**: deterministic natural-language intents (summaries, last-seen lookups, presence, attention, counts) with an optional **Groq LLM fallback** grounded in retrieved evidence and returned with citations.

---

## Key Features

- Real-time pipeline: face recognition + dual YOLO object detection + gaze estimation
- Gaze-to-object attention inference (`inside` box, otherwise `nearest` within a distance threshold)
- Temporal behavior tracking with rolling-window stabilisation (`start`, `switch`, `end`)
- Searchable memory: automatic and manual snapshots with JSON metadata
- Situation summaries over a configurable lookback window
- Chat assistant with deterministic intents, citations, and a grounded LLM fallback
- Two-step confirmation for mutating chat actions (snapshot, start/stop monitor, toggles, summary)
- FastAPI backend with REST, WebSocket fan-out, and MJPEG streaming
- Observable startup state machine (`starting` → `camera_opening` → `warming_up` → `ready` / `failed`)
- Camera failure recovery with bounded backoff and degraded-state reporting
- React (Vite) dashboard with live monitor, memory, reports, chatbot, and enrollment pages
- ASCII security report and structured metrics for offline analysis

---

## Architecture

### High-level flow

```
                       ┌────────────────────────┐
   Camera / video ───► │  Async frame reader     │  (background thread, latest-frame slot)
                       └───────────┬────────────┘
                                   ▼
                       ┌────────────────────────┐
                       │ Face detect + recognize │  InsightFace
                       └───────────┬────────────┘
                                   ▼
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌────────────────────┐                    ┌────────────────────┐
   │ Object detection    │                   │ Gaze estimation     │
   │ YOLO general/custom │                   │ L2CS-Net            │
   └──────────┬──────────┘                   └──────────┬──────────┘
              └───────────────┬──────────────────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Gaze → object fusion      │  inside box (+8 px) / nearest (≤120 px)
                 └───────────┬──────────────┘
                             ▼
                 ┌──────────────────────────┐
                 │ Behavior tracker          │  window 5 frames, confirm 3, end timeout 1.0 s
                 └───────────┬──────────────┘
                             ▼
     ┌───────────────────────┼────────────────────────┬─────────────────────┐
     ▼                       ▼                        ▼                     ▼
 metrics_log.jsonl   memory/snapshots + metadata   JPEG frame        unknown_incidents
     │                       │                        │                     │
     └───────────────┬───────┴────────────────────────┘                     │
                     ▼                                                      │
        Summaries / Chat / Reports ◄── REST API ◄── MJPEG + WebSocket       │
                     │                                                      │
                     └──────────────► React dashboard ◄─────────────────────┘
```

### Execution modes

Both modes share the same persistence layer and core library (`main.py`):

- **CLI mode** (`main.py`) — direct terminal control, local OpenCV window, keyboard shortcuts.
- **API mode** (`server.py`) — imports `main.py` as its core, owns one global `PipelineManager` that runs the monitor or enrollment worker in a background thread and fans events out to websocket clients.

The `PipelineManager` owns:

- a latest-frame store (`frame_bytes`, `timestamp_utc`, `sequence`)
- the monitor or enroll worker thread
- a control queue for runtime toggles
- the websocket pub/sub hub
- a recent-event cache

---

## Project Structure

```
.
├── main.py                  # CLI entry point + core inference/utility library
├── server.py                # FastAPI backend (REST, WebSocket, MJPEG, PipelineManager)
├── common.py                # Plumbing shared by both entry points (schema versions,
│                            #   metrics log + rotation, session ids, gaze scheduling)
├── object_detection.py      # DualYoloDetector (general + custom) and box normalisation
├── scene_memory.py          # Snapshot store, metadata index, retention, last-seen and search
├── frontend/                # React + Vite dashboard
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api/client.js
│       ├── context/         # Global state, polling, websocket wiring
│       ├── realtime/        # WebSocket client + stream stall detection
│       ├── components/      # Sidebar, StreamImage, LoadingScreen
│       └── pages/           # Dashboard, LiveMonitor, Memory, Reports, Chat, Register
├── ai-surveillance/
│   ├── server.py            # Thin launcher that re-exports the FastAPI app
│   └── inspect_model.py     # Utility to inspect an InsightFace model pack
├── tests/                   # Backend unit tests (unittest)
├── docs/
│   ├── ARCHITECTURE.md      # Pipeline and runtime internals
│   └── CHAT_AND_SUMMARY.md  # Chat intents, summaries, and API chat behavior
├── pixi.toml                # Environment definition (Python 3.11, linux-64)
├── pixi.lock                # Locked dependency set
├── requirements.txt         # pip fallback for non-linux-64 platforms (verified on Windows/Python 3.13)
├── ruff.toml                # Lint rule set (explicit, not tool defaults)
├── mypy.ini                 # Static type configuration (staged scope)
├── .github/workflows/ci.yml # CI: backend tests, lint/type checks, frontend
└── .env.example             # Environment variable template
```

### Which files matter most

| File | Role |
| --- | --- |
| `main.py` | Core logic: models, camera handling, gaze math, behavior tracker, metrics, chat, reports |
| `server.py` | The whole HTTP/WS surface; worker loops that mirror the CLI pipeline |
| `object_detection.py` | YOLO wrapper and detection schema shared by both modes |
| `scene_memory.py` | All persisted snapshot state, retention and retrieval helpers |
| `common.py` | The shared contract between the CLI and the API; keep schema versions and defaults here, not duplicated |
| `frontend/src/context/SurveillanceContext.jsx` | Dashboard state, polling, operation/error handling |
| `pixi.toml` / `pixi.lock` | Reproducible backend environment |

> Note: the runtime pipelines deliberately call `SceneMemoryManager(..., enable_vectors=False)`. The optional CLIP + FAISS vector index in `scene_memory.py` is present but not fed by the monitor/enroll pipelines.

---

## Requirements

### Backend

- **Operating system:** Linux is the supported target. `pixi.toml` declares `platforms = ["linux-64"]` and `pixi.lock` resolves Linux packages only.
- **Python:** 3.11 (pinned as `3.11.*`).
- **Environment manager:** [pixi](https://pixi.sh) (recommended), or a manual virtual environment using the same dependencies.
- **Hardware:**
  - A readable camera source (`/dev/video*` device path, an index, or a stream URL).
  - **GPU:** optional. InsightFace and torch select `CUDAExecutionProvider` / `cuda:0` when available and fall back to CPU otherwise. Gaze inference and YOLO run meaningfully faster with a CUDA-capable GPU.
- **Optional:** a Groq API key to enable the LLM chat fallback (deterministic intents work without it).

### Frontend

- **Node.js** 20.19+ or 22.12+ (required by Vite 7); Node 22 recommended
- npm

### Python dependencies (`pixi.toml`)

Conda-forge:

| Package | Constraint |
| --- | --- |
| `python` | `3.11.*` |
| `insightface` | `>=0.7.3,<0.8` (face recognition; degradable, see below) |
| `opencv` | `>=4.10,<5` |
| `ultralytics` | `>=8.4.0,<9` |
| `pytorch` | `>=2.5,<3` |
| `torchvision` | `>=0.20,<1` |
| `pillow` | `>=11,<12` |
| `numpy` | `>=1.26,<3` |
| `faiss-cpu` | `>=1.9,<2` |
| `open-clip-torch` | `>=3.0,<4` |
| `huggingface_hub` | `>=1.7.1,<2` |
| `pytest` | `>=9.0.2,<10` |

PyPI:

| Package | Source |
| --- | --- |
| `gdown` | PyPI (Google Drive downloads for gaze weights) |
| `l2cs` | git (`Ahmednull/L2CS-Net`, pinned revision) |
| `face_detection` | git (`elliottzheng/face-detection`) |
| `groq`, `python-dotenv`, `fastapi`, `uvicorn`, `httpx`, `websockets` | PyPI |

### Frontend dependencies (`frontend/package.json`)

Runtime: `react` 19, `react-dom` 19, `react-router-dom` 7, `recharts` 3, `lucide-react`
Dev: `vite` 7, `@vitejs/plugin-react` 5, `vitest` 3

---

## Installation

### 1. Backend environment

```bash
pixi install
```

Optionally run an interactive shell inside the environment:

```bash
pixi shell
```

**Not on Linux, or prefer not to use pixi?** The pixi workspace is pinned to `linux-64`. A pip fallback is provided for other platforms — **verified on Windows 11 + Python 3.13 with every feature working** (face recognition via insightface 2.0, YOLO, gaze, memory, chat):

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
python main.py doctor
```

Notes for the pip path: `l2cs` / `face_detection` are Git dependencies (a working `git` is required, no compiler needed); insightface installs from the pure-Python 2.x wheel on Windows, where the conda-forge 0.7.x build does not exist. `doctor` tells you precisely which parts came up rather than leaving you to guess.

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

### 3. Configuration

Copy the template and edit it:

```bash
cp .env.example .env
```

See [Configuration](#configuration) for the full list of variables.

---

## Configuration

All variables are optional; the defaults below reflect the code.

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `AI_STUDIO_CAM_CAMERA_INDEX` | `/dev/video42` on Linux **when that device exists**, otherwise `0` | backend | Camera device path, numeric index, or stream URL |
| `AI_STUDIO_CAM_INCLUDE_INDEX_FALLBACK` | `0` | backend | Set to `1` to also try the numeric index when a `/dev/video*` path is configured |
| `AI_STUDIO_CAM_SCAN_ALL_DEVICES` | `0` | backend (Linux) | Set to `1` to scan every `/dev/video*` device before giving up |
| `AI_STUDIO_GENERAL_YOLO_MODEL` | `yolov8n.pt` | backend | General YOLO checkpoint. Ultralytics downloads it on first use; `bootstrap` fetches it up front |
| `AI_STUDIO_METRICS_MAX_BYTES` | `4194304` (4 MiB) | backend | Size at which `metrics_log.jsonl` rotates; floored at 64 KiB |
| `AI_STUDIO_METRICS_BACKUPS` | `2` | backend | Rotated generations to keep (`0` truncates instead of rotating) |
| `AI_STUDIO_MEMORY_MAX_AUTO_SNAPSHOTS` | `5000` | backend | Automatic snapshots to keep; `0` disables pruning. Manual snapshots are never pruned |
| `AI_STUDIO_UNKNOWN_INCIDENT_MAX_FILES` | `500` | backend | Unknown-face captures to keep, oldest pruned first; `0` disables pruning |
| `AI_STUDIO_YOLO_IMGSZ` | `768` | backend | Object-detection inference size (320–1920, snapped to a multiple of 32). Benchmarked: 768 found 8 classes vs 5 at 640 on this project's own frames at ~150 ms vs ~90 ms per frame |
| `AI_STUDIO_YOLO_CONF` | `0.25` | backend | Object-detection confidence threshold (0.01–0.99). Lower finds more at the cost of low-confidence noise |
| `AI_STUDIO_YOLO_IOU` | `0.7` | backend | NMS IoU threshold (0.1–0.95) |
| `AI_STUDIO_YOLO_MAX_DET` | `300` | backend | Maximum boxes per frame (1–1000) |
| `AI_STUDIO_YOLO_AGNOSTIC_NMS` | `false` | backend | Class-agnostic NMS: one box pool across classes, so one object cannot be reported under two labels. Can also drop a genuinely distinct overlapping label |
| `AI_STUDIO_FPS_CAP` | `12` | backend | Monitor-loop ceiling in frames/s. The cap is a ceiling — slow machines run at whatever they keep up with. Lower = less CPU, higher = snappier detection |
| `AI_STUDIO_ENROLL_FPS_CAP` | `20` | backend | Same ceiling for enrollment sessions (kept faster on purpose; sample collection wants rate) |
| `AI_STUDIO_SNAPSHOT_INTERVAL` | `8.0` | backend | Seconds between auto snapshots — the instances reports and chat ground on. Runs on wall-clock time, independent of FPS; clamped to 1–3600 |
| `GROQ_API_KEY` *(or `groq_api_key`)* | unset | backend | Enables the LLM chat fallback |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | backend | Groq model name |
| `PORT` | `8000` | backend | Port for `python server.py` |
| `VITE_API_BASE` | `http://localhost:8000` | frontend | Backend base URL (also used to derive the WebSocket URL) |

Camera selection tries multiple candidate sources and backends (FFmpeg / V4L2 / platform default) before failing, and the failure message lists what was attempted.

---

## Model Assets and First-Run Downloads

| Asset | Default location | How it is obtained |
| --- | --- | --- |
| InsightFace face pack | `~/.insightface/models/<pack>` | Downloaded automatically on first use. If the pack is extracted with a nested directory layout, the app repairs it in place. |
| General YOLO weights | `yolov8n.pt` (or `AI_STUDIO_GENERAL_YOLO_MODEL`) | Ultralytics downloads the checkpoint if it is not present; `bootstrap` fetches it explicitly |
| Custom YOLO weights | `custom_model_path.txt` pointer, else newest of `runs/detect/*/weights/best.pt`, `models/custom_yolo*.pt` | Produced by `train-objects`; optional |
| L2CS-Net gaze weights | `models/L2CSNet_gaze360.pkl` | Auto-downloaded via `gdown` from a Google Drive folder, or placed manually. Cannot be embedded in the repository due to size. |

If gaze weights cannot be resolved, gaze estimation is disabled and the rest of the pipeline continues normally.

---

## Usage

### API Mode (Backend)

```bash
pixi run python server.py
# or, to pick up .env values as-is:
AI_STUDIO_CAM_CAMERA_INDEX=/dev/video42 pixi run python server.py
```

The backend listens on `http://0.0.0.0:8000` (`PORT` overrides the port). A thin launcher is also available:

```bash
pixi run python ai-surveillance/server.py
```

Then start a session:

```bash
curl -X POST http://localhost:8000/api/v1/monitor/start -H 'Content-Type: application/json' -d '{}'
curl http://localhost:8000/api/v1/monitor/status
```

Watch the MJPEG feed in a browser at `http://localhost:8000/api/v1/stream/video`.

### Frontend Dashboard

```bash
cd frontend
npm run dev
```

The dashboard runs at `http://localhost:5173` and talks to `VITE_API_BASE` (default `http://localhost:8000`). Pages:

| Route | Page | Purpose |
| --- | --- | --- |
| `/` | Dashboard | Detection totals, known/unknown faces, recent event table |
| `/live` | Live Monitor | MJPEG feed, start/stop, runtime toggles, live counts, stalled/startup overlays |
| `/memory` | Memory | Snapshot stats, recent snapshots, last-seen lookups, history search |
| `/reports` | Reports | Situation summaries, behavior snapshot, recent log rows |
| `/chat` | Chatbot | Chat with citations and confirmation buttons for mutating actions |
| `/register` | Enroll Face | Start/stop an enrollment session and watch capture progress |

For a production build:

```bash
cd frontend
npm run build
```

### First Run

Two commands stand between a fresh checkout and a working session:

```bash
# Create runtime directories and fetch the model assets (safe to re-run)
pixi run python main.py bootstrap

# Report exactly what this machine can run
pixi run python main.py doctor
```

`bootstrap` creates `memory/snapshots/` and `unknown_incidents/`, ensures the general YOLO checkpoint is present (downloading `yolov8n.pt` via Ultralytics if needed), and attempts the L2CS gaze-weight download. It never aborts on a missing asset: it reports what is unresolved and prints the next steps.

`doctor` prints one row per requirement with `ok` / `warn` / `fail`:

- **`fail`** items block monitoring (they are the required inference modules: `numpy`, `cv2`, `PIL`, `onnxruntime`, `ultralytics`, `torch`). A failure exits with status 1.
- **`warn`** items each disable exactly one feature, and the pipeline still runs. Missing `l2cs` disables gaze, missing `groq` disables the LLM chat fallback, missing `faiss`/`open_clip` disables vector search, missing gaze weights disables gaze, and no enrolled identities means every face reads as `Unknown`.
- **Degradable modules** are the `warn` case for a *core* capability: a missing or too-old `insightface` disables face recognition only. The session starts, object detection/gaze/memory keep working, `degraded_reason` names the exact cause, and the session metric `face_recognition_enabled` records that identity matching was off. It is deliberately **not** a `fail`, because a machine without a usable InsightFace can still run everything else. Run `bootstrap`/`doctor` and install the pinned version to restore it.

Add `--check-camera` to actually open and release the configured camera. A successful run on this machine looks like:

```
[ ok ] module:torch          2.10.0+cpu
[warn] module:l2cs          not importable (ModuleNotFoundError) - gaze estimation
[warn] module:insightface   0.2.1 is too old (need >= 0.7.3: FaceAnalysis(providers=...) requires 0.7.x) - face recognition disabled without it
[warn] gaze weights         models/L2CSNet_gaze360.pkl missing (see `bootstrap`)
[ ok ] camera source        0 (from AI_STUDIO_CAM_CAMERA_INDEX)
```

### CLI Mode

All commands run from the repository root so that relative artifact paths resolve correctly.

```bash
# Enroll an identity (press q to finish early)
pixi run python main.py enroll --name Alok

# Run the unified pipeline in a local OpenCV window
pixi run python main.py recognize

# Run it with specific models / streams disabled
pixi run python main.py recognize --disable-gaze --disable-custom --snapshot-interval 30

# Enrolled identities
pixi run python main.py list

# Fine-tune a custom YOLO model on your own dataset YAML
pixi run python main.py train-objects --data path/to/data.yaml --epochs 30 --set-default

# Memory queries
pixi run python main.py memory-stats
pixi run python main.py memory-recent --minutes 10
pixi run python main.py memory-find --object laptop
pixi run python main.py memory-find-person --name Alok
pixi run python main.py memory-search --text "person using laptop"

# Summaries
pixi run python main.py session-summary --minutes 5
pixi run python main.py session-summary --minutes 10 --json

# Chat
pixi run python main.py chat
pixi run python main.py chat --question "What happened in the last 10 minutes?"

# ASCII security report -> report.txt
pixi run python main.py report

# Offline detection benchmark (see "Detection Benchmark" below)
pixi run python main.py bench-detect

# Frame budget from recorded sessions (see "Frame Budget" below)
pixi run python main.py frame-budget

# Review the detections the system already made (measures precision - see "Detection Review")
pixi run python main.py review-detections
pixi run python main.py score-detections --review-dir reviews
```

`review-detections` options: `--frames`, `--out-dir`, `--limit`, `--min-brightness`, `--verdicts`.

Global flag: `--model {buffalo_l,buffalo_m,buffalo_s,buffalo_sc,antelopev2}` (default `buffalo_sc`).

`recognize` options: `--general-model`, `--custom-model`, `--disable-general`, `--disable-custom`, `--disable-gaze`, `--snapshot-interval`, `--gaze-arch`, `--gaze-weights`, `--gaze-weights-source`, `--disable-gaze-auto-download`, `--gaze-max-interval`, `--gaze-target-fps-drop`.

### Detection Benchmark

Detection quality is measurably different from detection *working*, and the live loop can only tell you the latter — in a scene nobody can replay. `bench-detect` runs the **real detector** over a fixed frame set (your own saved snapshots, plus the reference images bundled with Ultralytics that have known labels) and reports per-class counts, mean confidence and ms/frame, so a tuning change can be justified with numbers:

```bash
pixi run python main.py bench-detect
pixi run python main.py bench-detect --imgsz 640 768 960 --conf 0.25 0.15
pixi run python main.py bench-detect --json bench.json
```

```
excluded frames: 10 (too_dark=9, too_small=1)

conf=0.25 imgsz=768    86 ms/frame   8 classes   reference recall=1.00
    person x56 (max 0.92)  remote x8 (max 0.48)  cell phone x5 (max 0.45) ...
conf=0.25 imgsz=640    58 ms/frame   5 classes   reference recall=1.00
    person x56 (max 0.94)  cell phone x7 (max 0.46)  remote x2 (max 0.29) ...
```

Frames too dark, blurred or small to detect anything in are **excluded and reported by reason** instead of dragging the recall numbers down — on a real capture set a large fraction of frames are unusable, and averaging those in measures the room's lighting rather than the detector. With no arguments the benchmark compares the *shipping* configuration against the historical 640/0.25 baseline, so the default run answers "did the change I just made help?".

Run it after changing any `AI_STUDIO_YOLO_*` knob. `--frames` accepts globs, `--min-brightness` sets the usable-frame floor, and `--json` writes the raw per-class results.

### Frame Budget

`bench-detect` measures the detector alone, on saved frames. `frame-budget` reads the opposite half of the evidence — the sessions the pipeline has **already recorded** into `metrics_log.jsonl` — and reports what a live frame period was actually made of, with no camera and no new run:

```bash
pixi run python main.py frame-budget
pixi run python main.py frame-budget --limit 5 --json budget.json
```

```
session                  frames avg_fps ema_fps period_ms  face_ms  gaze_ms  unattr_ms  unattr  pacing
monitor-20260920-104542      38    0.68    1.73    1470.6     65.6    433.2      971.8   66.1%  not-recorded
monitor-20260921-191819     362    4.75    4.73     210.5      off      off      210.5  100.0%  not-recorded

12 session(s), 1300 frames, 718.5 s: median 0.83 fps (range 0.68-9.39), median EMA rate 2.80 fps
  5 session(s) are stall-dominated (EMA rate above 2x the session average)

Stage accounting, as a share of each session's mean period (median):
  face detection     3.6%  (41.1 ms/frame)  over 8 session(s)
  gaze              28.3%  (334.5 ms/frame)  over 6 session(s)
  unattributed      69.1%  (833.6 ms/frame)  - a remainder, not a measurement
```

It refuses three things, each because the record does not support them:

- **It never reports a missing measurement as zero.** `avg_detection_latency_ms` in the log times **face detection**, not the YOLO object detector — both loops call `detector.detect()` untimed — so object detection appears in *no* session record. The report says so, and the remainder is labelled `unattr_ms` rather than given a stage name it never had.
- **It does not treat a session average as a per-frame cost.** Five of the twelve recorded sessions have an EMA frame rate more than 2× their session average (one reached 178 fps instantaneously against a 0.82 fps mean), so their period is dominated by stalls and is flagged rather than smoothed in.
- **It renders three different states differently**: a timed stage, `off` (the stage did not run), and `--` (it ran with no recorded latency). A disabled stage still counts its calls and still wraps a no-op in a timer, so it reports ~0.01 ms; believing that would record a missing measurement as a fast one.

Sessions that cannot produce a period at all are listed by name under `Skipped rows` instead of being dropped.

### Detection Review

`bench-detect` can measure recall against two bundled reference images and detection statistics on your own frames — but **not precision**, because your frames carry no labels. `review-detections` gets a precision figure without new capture and without labelling anything: the detector proposes a finite list of boxes, and you confirm or reject that list.

```bash
pixi run python main.py review-detections          # writes reviews/review.html + reviews/detections.json
# open reviews/review.html, mark each box Correct / Wrong, click "Download verdicts.json"
pixi run python main.py score-detections --review-dir reviews
```

To revisit a previous run — for example to correct answers after reading the rubric —
pass the recorded verdicts back in and the page opens with them already marked, so a
correction pass is only the boxes you want to change:

```bash
pixi run python main.py review-detections --verdicts reviews/verdicts.json
```

```
Reviewed 102 of 102 detection(s)  (coverage 100%)
Precision: 0.686  (70 correct, 32 wrong)
  every detection was reviewed, so this is a census of these frames - no sampling error

label              reviewed  correct  wrong  precision
person                   76       49     27      0.645
remote                    8        7      1      0.875
cell phone                7        5      2      0.714
toothbrush                5        5      0      1.000
surfboard                 2        0      2      0.000
tie                       2        2      0      1.000
bottle                    1        1      0      1.000
refrigerator              1        1      0      1.000
```

That table is a real run over this repository's saved snapshots, not an illustration.
`surfboard` at 0.000 and `person` at 0.645 are the findings that matter, and both point
the same way: the model proposes classes the scene does not contain, and it splits one
person into more than one box. Re-running `review-detections` into the same directory is
safe — the detection list is fingerprinted, and `score-detections` refuses a verdict file
recorded against a different list rather than scoring it against the wrong boxes.

At the time of writing that is **102 boxes over 68 usable frames** — a couple of minutes of clicking, not a labelling project. Three things it is careful about:

- **Unreviewed boxes are reported as unreviewed, never as correct.** A verdict it cannot parse is skipped rather than guessed, because assuming "correct" is the one thing that would inflate the number it exists to produce.
- **A partial review is reported as a sample.** `--limit N` picks evenly across the confidence range (not the easiest top-N boxes) and the score prints a Wilson interval; below 80% coverage it says the interval is optimistic, because a subset picked by hand is not a random sample.
- **It measures precision, not recall.** Recall needs every object in every frame enumerated — the "extensive labelling" this exists to avoid. The optional per-row note records objects you *noticed* were missed; those are printed as concrete misses with no denominator, never as a recall figure.
- **A correction pass cannot drift onto the wrong boxes.** `--verdicts` only preloads a file carrying the same list id; anything else prints a warning and starts blank, and `score-detections` refuses a mismatch outright.

The number describes **the frames in `reviews/`** and nothing else; every run says so. The page embeds real camera frames, so it lives in the git-ignored `reviews/` directory and carries `noindex`.

### Gaze Scheduling

By default gaze runs on **every** processed frame that contains faces, and the session metrics report an interval of 1. This keeps attention and behavior data per-frame rather than sampled.

Gaze can instead throttle itself when inference is expensive. Pass `--gaze-max-interval N` (`N > 1`) to enable it:

```bash
pixi run python main.py recognize --gaze-max-interval 4
```

The interval grows by one whenever gaze inference costs more than `--gaze-target-fps-drop` (default `0.25`) of the frame budget, and steps back down after a run of cheap frames. Frames skipped this way reuse the previous gaze estimate, so attention tracking stays continuous; only genuine inferences are counted in `gaze_inference_calls`. The effective values are written to the session aggregate (`gaze_base_interval_frames`, `gaze_interval_frames_final`, `gaze_target_fps_drop`) and the CLI prints the active mode at startup.

The API exposes the same two controls as `gaze_max_interval` and `gaze_target_fps_drop` on `POST /api/v1/monitor/start`.

### Runtime Keyboard Controls

Available in the `recognize` window:

| Key | Action |
| --- | --- |
| `q` | Quit |
| `g` | Toggle general YOLO |
| `o` | Toggle custom YOLO |
| `c` | Chat query (live-aware; can capture a snapshot) |
| `t` | Manual memory snapshot |
| `m` | Memory statistics |
| `r` | Recent snapshots (last 5 minutes) |
| `f` | Find when an object was last seen |
| `h` | Print the control help |

---

## API Reference

Base URL: `http://localhost:8000`

### Monitoring

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/monitor/start` | Start the monitor worker; returns `status: "starting"` and startup deadlines |
| `POST` | `/api/v1/monitor/stop` | Stop the active worker |
| `GET` | `/api/v1/monitor/status` | Running state, mode, session id, startup phase, frame sequence, YOLO state, config |
| `PATCH` | `/api/v1/monitor/toggles` | Queue runtime toggles (`general_yolo`, `custom_yolo`, `gaze`) |
| `POST` | `/api/v1/monitor/snapshot` | Capture a manual snapshot from the latest frame |

### Streaming and live data

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/stream/video` | MJPEG stream (`multipart/x-mixed-replace`) |
| `WS` | `/api/v1/stream/events/ws` | Realtime event stream (see below) |
| `GET` | `/api/v1/detections/latest` | Latest normalised faces, objects, attention, counts |
| `GET` | `/api/v1/behavior/latest` | Behavior summary and recent behavior events |
| `GET` | `/api/v1/events/recent` | Recently published events |

### Memory

| Method | Path | Query parameters |
| --- | --- | --- |
| `GET` | `/api/v1/memory/stats` | - |
| `GET` | `/api/v1/memory/recent` | `minutes` (1-1440), `limit` (1-200) |
| `GET` | `/api/v1/memory/find/object` | `name` |
| `GET` | `/api/v1/memory/find/person` | `name` |
| `GET` | `/api/v1/memory/search` | `text`, `top_k` (1-50) |

### Summaries, chat, logs, enrollment

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/v1/summaries/session` | Situation summary; `minutes`, `json` |
| `POST` | `/api/v1/chat/query` | Chat; accepts `message`/`question`, `session_id`, `confirm_action_id` |
| `GET` | `/api/v1/logs` | Filter metric rows by `event_type`, `limit`, `from`, `to` |
| `POST` | `/api/v1/enroll/start` | Start an enrollment session (`name`, `model`, `fps_cap`) |
| `GET` | `/api/v1/enroll/status` | Enrollment progress |
| `POST` | `/api/v1/enroll/stop` | Stop enrollment |

### Static media

Captured media is mounted directly and requires no authentication:

- `/snapshots` → `memory/snapshots/`
- `/incidents` → `unknown_incidents/`

### Legacy aliases

`POST /start`, `POST /stop`, `GET /status`, `GET /video_feed`, `GET /logs`, `POST /enroll`, `GET /enroll/status` are retained for older clients.

### Chat behavior

1. A mutating request (snapshot, start/stop, toggles, summary) returns a **proposal** with `confirm_action_id`; a second call with that id executes it. Confirmations expire after 2 minutes.
2. Otherwise, deterministic intents are attempted first: session summary, person count, memory stats, recent snapshots, last-seen (person or object), presence, attention.
3. If no deterministic intent matches, the API builds grounding from metrics, memory hits, snapshots, and runtime context, and only then calls Groq. Responses include `citations` and `grounded` flags.

---

## WebSocket Events

Endpoint: `WS /api/v1/stream/events/ws`

Every message uses the same envelope:

```json
{
  "type": "detections",
  "timestamp": "2026-03-14T06:00:00+00:00",
  "session_id": "monitor-20260314-113000",
  "payload": {}
}
```

Published types: `pipeline_state`, `detections`, `behavior_event`, `memory_event`, `chat_result`, `summary_result`.

Clients should tolerate dropped events; the frontend additionally polls `status` and `detections/latest` every 2 seconds and derives a "stalled" state when the frame sequence or timestamps stop advancing.

---

## Data and Storage Model

All runtime artifacts are written relative to the working directory and are git-ignored.

| Path | Format | Contents |
| --- | --- | --- |
| `metrics_log.jsonl` | JSONL, append-only | Event stream: `enroll`, `recognize_session`, `behavior_event`, `chat_query`, `summary_query`, `chat_action_proposed`, `chat_action_executed`, `memory_query`, `object_train`, plus legacy event types. Rotates at `AI_STUDIO_METRICS_MAX_BYTES` into `.1`, `.2`, … |
| `memory/metadata.json` | JSON array | Snapshot index: timestamps, path, object labels, manual flag, optional faces/objects/people/attention |
| `memory/snapshots/*.jpg` | JPEG | Automatic (default every 15 s) and manual snapshots. Auto captures are pruned to `AI_STUDIO_MEMORY_MAX_AUTO_SNAPSHOTS`; manual ones are retained |
| `unknown_incidents/*.jpg` | JPEG | Frames containing unrecognised faces (throttled to one per 3 s), pruned to the newest `AI_STUDIO_UNKNOWN_INCIDENT_MAX_FILES` |
| `face_db.npz` | NumPy archive | Enrolled identities: names, L2-normalised centroids, sample counts |
| `custom_model_path.txt` | Text | Pointer to the active custom YOLO checkpoint |
| `report.txt` | Text | ASCII security dashboard generated by `report` |

`recognize_session` records aggregate performance and behavior rollups (FPS, detection latency, per-person presence, object class counts, attention map, top attended objects, chat/memory query counters, the gaze interval actually used, and `face_recognition_enabled`, so a session without face recognition is distinguishable from one where nobody was in frame). Both entry points write the same schema versions — `7` for sessions and `4` for enrollment — and readers normalise older rows.

### Storage Lifecycle

All three growth paths are bounded, and every cap is configurable through the environment variables above:

- **Metrics log** rotates by size into numbered generations (`.1`, `.2`, …). `read_jsonl` reads generations oldest-first, so reports keep their history while the read stays bounded by `(backups + 1) x max bytes`. Rotation is best-effort: if a reader holds the file open (notably on Windows), the append still succeeds and rotation is retried on the next write rather than failing the write.
- **Auto snapshots** are pruned oldest-first once the store exceeds the cap, with the JPEG removed from disk and the entry dropped from `metadata.json`. The cap is enforced against the index on disk, so a long-lived worker and per-request API handlers cannot each believe the store is under the limit.
- **Manual snapshots are never pruned.** They are explicit user actions; deleting them silently is precisely the data-loss bug this codebase already had.

Pruning is announced on stdout when it happens. `memory-stats` reports the active cap alongside the current count.

### Cross-Process Safety

`main.py` and `server.py` can be launched at the same time against the same working directory, and they share `metrics_log.jsonl` and `memory/metadata.json`. Thread locks do not protect against that, so every read-merge-write cycle over a shared file additionally takes an **OS-level advisory lock**:

| Platform | Mechanism |
| --- | --- |
| Linux / macOS | `fcntl.flock` (`LOCK_EX`) |
| Windows | `msvcrt.locking` (`LK_NBLCK`) |

The lock lives in a sidecar `<name>.lock` file, not on the data file itself, because writes replace the data file atomically with `os.replace` — a handle to the old file would guard nothing. Locking is re-entrant per thread, so nested acquisitions inside one process cannot deadlock on themselves.

Without this, concurrent processes silently destroyed data. The same workload run both ways, four processes writing twelve snapshots each:

```
UNSYNCHRONISED : expected 48 entries -> got 13
WITH OS LOCK   : expected 48 entries -> got 48  (jpegs on disk: 48)
```

On Windows the unlocked version does not merely lose entries — concurrent `os.replace` raises a sharing violation, because the destination is open in another process.

Waiting is bounded (10 s for writers, 2 s for readers). If a required lock cannot be taken, the work **proceeds anyway rather than hanging** and the event is counted; `doctor` reports the backend and the timeout count, so a degraded run is visible instead of silent.

---

## Testing

### Backend

```bash
pixi run python -m unittest discover -s tests -q
```

The suite prints its own count when it runs, which is the authoritative number — it is deliberately not restated here, because a hardcoded count goes stale the next time a test is added. Coverage: the detection schema and toggles, gaze L2CS helpers and interval scheduling, the shared session aggregate contract, environment readiness and first-run bootstrap, storage lifecycle (metrics rotation, snapshot and incident retention), cross-process locking (real subprocess contention, including control cases proving those tests detect unsynchronised writers), persistence integrity (including concurrent snapshot and metrics writers), behavior tracking and summaries, chat and snapshot actions, metrics schema normalisation, scene memory, the CLI `recognize` loop end-to-end, the offline detection benchmark, and the FastAPI surface (status schema, monitor lifecycle, camera recovery, stream generator, WebSocket envelope, chat sessions, and two-step action confirmation).

> The detection benchmark's inference tests need the real computer-vision stack. The rest of the suite stubs `cv2`, so those tests swap the real module in for the duration and put the stub back afterwards; on a machine without OpenCV or `ultralytics` they report as skipped rather than silently passing.

> The server tests import `server.py` at module scope, which creates `memory/snapshots/` and `unknown_incidents/`. Chat tests append to `metrics_log.jsonl`. These paths are git-ignored.

### Frontend

```bash
cd frontend
npm test          # vitest
npm run build     # production build
```

Frontend tests cover the WebSocket client (event forwarding, backoff reconnects) and stream stall detection.

### Lint and Types

```bash
ruff check .     # lint (config: ruff.toml)
mypy             # static types (config: mypy.ini)
```

Install the tools with `pip install ruff mypy` (they are listed in `requirements.txt`). The rule set in `ruff.toml` is listed explicitly rather than inherited from the tool's defaults, so results do not drift when ruff is upgraded. The file also records **which families are deliberately not selected and why** — notably `BLE`/`S` (broad `except Exception` is how the pipeline degrades around optional dependencies, and `try/except/pass` is how best-effort cleanup is written) and `D`/`ANN` (behaviour is documented in the README and the test suite rather than in docstrings). Individual rules within the selected families that would fight the existing style (`SIM117`, `TRY003`, `TRY300`) are ignored in the same file, each with a reason.

`mypy` is **staged, not all-or-nothing**: `common.py`, `object_detection.py`, `detection_bench.py`, `scene_memory.py` and `server.py` are enforced and currently clean, while `main.py` is opted out *explicitly* in `mypy.ini` with its remaining finding count recorded there. It reports ~108 issues today, 104 of them possible-`None` dereferences in the long CLI/report/chat helpers; guarding those is a refactor, not a config change. Deleting the one override line is all that is needed to start enforcing it.

Formatting is configured (`line-length = 100`, double quotes) but **not applied to the existing code** — running `ruff format .` would rewrite every file at once, so it is left as an opt-in. `ruff format --check` is therefore not yet enforced in CI.

### Continuous Integration

`.github/workflows/ci.yml` runs three jobs on every push to `main` and every pull request:

| Job | What it runs |
| --- | --- |
| Backend tests | `python -m unittest discover -s tests -q`, then a byte-compile of every module |
| Lint and types | `ruff check .` and `mypy` |
| Frontend | `npm ci`, `npm test`, `npm run build` |

The backend job installs only `numpy`, `pillow`, `fastapi` and `httpx`, because the suite stubs cv2 and insightface — the full computer-vision stack is not needed to run the tests. It also installs a CPU-only `torch` so the one L2CS decoding test executes rather than skipping; remove that step to make the job lighter and that test will report as skipped instead of failing.

### End-to-End

Run the backend and the dashboard together, then load the dashboard in a browser. On a machine with the pinned InsightFace installed:

```bash
python server.py                # terminal 1
cd frontend && npm run dev      # terminal 2  -> http://localhost:5173
```

The following were verified against a live server and a real headless browser:

| Check | Result |
| --- | --- |
| Every endpoint the dashboard calls | 200 (status, detections, behavior, memory stats/recent/find/search, logs, session summary, chat) |
| Monitor start / snapshot / toggles / stop | Lifecycle responds; snapshot and toggles return 409 with a clear reason when no session is running |
| CORS from the dashboard origin | Preflight 200, origin allowed |
| WebSocket `/api/v1/stream/events/ws` | Connects; first frame is a valid `{type, timestamp, session_id, payload}` envelope |
| Dashboard render | React app mounts, renders nav and counters, and lists real events fetched from the backend |
| Browser console | No errors |

Face recognition needs the pinned InsightFace. Without a usable one the session still starts and runs: object detection, gaze, memory and the dashboard are unaffected, `degraded_reason` reports `face_recognition_unavailable:<cause>`, and the session aggregate is written with `face_recognition_enabled: false`.

---

## Troubleshooting

**The camera cannot be opened**

`_open_camera()` tries each candidate source across several backends and throws with the list of attempts. Check that `AI_STUDIO_CAM_CAMERA_INDEX` points at a real device, and run `python main.py doctor --check-camera` to see the probe result directly.

**About `/dev/video42`:** that path is the signature of a `v4l2loopback` virtual device, so it only works while something (OBS, ffmpeg) is feeding it. It is used as the Linux default **only when the device actually exists**; otherwise the default is `0`. If you want the virtual-camera workflow, start the producer first and set the variable explicitly:

```bash
AI_STUDIO_CAM_CAMERA_INDEX=/dev/video42 pixi run python main.py recognize
```

Set `AI_STUDIO_CAM_SCAN_ALL_DEVICES=1` to scan every `/dev/video*` device before giving up, or `AI_STUDIO_CAM_INCLUDE_INDEX_FALLBACK=1` to also try the numeric index when a device path is configured.

**Monitoring starts but the frontend shows "Stalled"**

The stream is driven by a monotonic frame `sequence` plus UTC timestamps. Stalls are only reported after a 15-second warmup grace period, so the message indicates the worker is not producing frames. Check the pipeline status endpoint for `degraded`, `degraded_reason`, and `last_error`.

**`Face recognition DISABLED: insightface ... cannot be used: FaceAnalysis(providers=...) requires insightface >= 0.7.3`**

The installed InsightFace is an old 0.2.x release whose `FaceAnalysis` has no `providers` argument (or it is not installed at all). Face recognition is a degradable capability, so the session keeps running with objects, gaze and memory; identities simply are not matched, and `degraded_reason` says so. To restore it:

```bash
pixi install                        # Linux: pixi.toml pins insightface>=0.7.3,<0.8
# or, outside pixi (works on Windows too):
pip install -U "insightface>=0.7.3,<3"
```

The 0.7.x line is **source-only on PyPI** (needs a C++ toolchain and Cython), but the **2.x line ships a pure-Python wheel** (`py3-none-any`) that installs with no compiler and was verified against this project's API — `FaceAnalysis(providers=..., allowed_modules=...)`, `prepare`, `get` — including on Windows/Python 3.13. `python main.py doctor` reports an unusable insightface as a `warn`, not a `fail`, because an importable module is not necessarily a usable one — and an unusable one no longer blocks the session.

**Gaze is unavailable**

The L2CS checkpoint could not be resolved. Place the weights at `models/L2CSNet_gaze360.pkl`. The pipeline continues without gaze.

> **Heads-up:** the Google Drive folder advertised by upstream L2CS-Net returns **404** (as of September 2026), so `gdown` auto-download cannot work. A verified mirror of the same Gaze360 ResNet50 checkpoint exists as `py-feat/l2cs` on Hugging Face (`l2cs_gaze360_resnet50.safetensors`, key names identical to upstream: `fc_yaw_gaze.weight [90, 2048]`, etc.). Convert it once and drop it in `models/`:
>
> ```python
> # pip install safetensors  (or parse the header with stdlib; see git history)
> from safetensors.torch import load_file
> import torch
> state = load_file("l2cs_gaze360_resnet50.safetensors")
> torch.save(state, "models/L2CSNet_gaze360.pkl")
> ```

**Chat replies "Insufficient evidence"**

No deterministic intent matched and either no grounding citations were found or no `GROQ_API_KEY` is configured. Ask about summaries, last-seen, memory stats, presence, attention, or counts to use the deterministic paths.

**`pixi install` fails on macOS or Windows**

Expected: `pixi.toml` restricts the workspace to `linux-64` and `pixi.lock` contains Linux packages only. Use Linux, WSL, or a manually built equivalent environment.

**Enrollment reports no samples**

Enrollment requires exactly one detectable face at a time; multiple faces or zero faces are rejected. The default target is 25 samples captured every 0.25 s.

---

## Current Limitations

Verified against the current code:

- **Two verified environments.** Linux via the pixi workspace (`linux-64`, `insightface 0.7.x` from conda-forge) and Windows via the `requirements.txt` pip path (verified on Python 3.13 with `insightface 2.0`, whose pure-Python wheel needs no compiler). Run `doctor` on either to see what came up.
- **Face recognition degrades rather than blocks.** Without a usable `insightface` (not installed, or an old 0.2.x whose `FaceAnalysis` has no `providers` argument) a session still runs with object detection, gaze and memory; identity matching is simply off, `degraded_reason` explains why, and `face_recognition_enabled` is `false` in the session aggregate.
- **One active worker.** A single global `PipelineManager` runs either monitor or enroll, never both.
- **Vector search is not wired in.** `scene_memory.py` supports CLIP + FAISS semantic search, but the monitor and enroll pipelines construct the memory manager with `enable_vectors=False`, so the index is never populated during normal operation. `memory-search` falls back to lexical matching.
- **Gaze runs every processed frame by default.** Adaptive throttling is implemented and test-covered but opt-in (`--gaze-max-interval N`, or `gaze_max_interval` on the API). Enabling it trades attention-data fidelity for speed, since skipped frames reuse the previous gaze estimate.
- **Some pipeline logic is still duplicated.** The session aggregate, metrics-log helpers and gaze scheduler are shared, but the CLI loop (`cmd_recognize`) and the API worker (`_run_monitor_worker`) still reimplement the surrounding per-frame flow.
- **Custom object detection needs your own data.** `train-objects` requires a YOLO dataset YAML; no dataset ships with the repository.
- **Retention deletes data.** `memory/snapshots/` and `unknown_incidents/` are pruned by default, so long-running monitoring no longer grows without bound. The limits are generous (5000 auto snapshots, 500 incident captures) and configurable, but if you need indefinite retention, raise the caps or set them to `0`.
- **No authentication and no rate limiting.** See the security note below.

---

## Security and Privacy Notice

This application performs **biometric processing** (face embeddings) and stores images of detected people.

- Face embeddings are persisted unencrypted in `face_db.npz`; snapshots and unknown-face incidents are written as plain JPEGs.
- The API has **no authentication or authorization**. Any reachable client can start/stop monitoring, enrol identities, read logs, and capture snapshots.
- CORS is configured with `allow_origins=["*"]` together with `allow_credentials=True`, which is permissive and not a safe production configuration.
- `memory/snapshots/` and `unknown_incidents/` are served as public static directories at `/snapshots` and `/incidents`.
- The default `server.py` entry point binds to `0.0.0.0`.

Treat this project as a research/development prototype. Before any real deployment, add authentication, restrict CORS and bind addresses, encrypt or otherwise protect biometric data, define a retention policy for snapshots and incident images, and ensure compliance with the privacy and biometric-data regulations that apply to you (for example GDPR, BIPA, or DPDP Act).

---

## Related Documentation

- [`DEPLOYMENT.md`](DEPLOYMENT.md) — expo/public demo runbook: tunnel setup, pre-flight checks, contingency plans
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline internals, startup/recovery model, behavior rules, storage schema
- [`docs/CHAT_AND_SUMMARY.md`](docs/CHAT_AND_SUMMARY.md) — chat intents, action confirmation flow, summaries, citations, telemetry

---

## Acknowledgements

This project builds on the following open-source work:

- [InsightFace](https://github.com/deepinsight/insightface) — face detection and recognition
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — object detection
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) — gaze estimation
- [OpenCLIP](https://github.com/mlfoundations/open_clip) and [FAISS](https://github.com/facebookresearch/faiss) — optional semantic image search
- [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn) — backend
- [React](https://react.dev), [Vite](https://vite.dev) — frontend
- [Groq](https://groq.com) — optional grounded LLM fallback

---

## License

No license file is included in this repository. Add a `LICENSE` file before distributing or reusing the code.
