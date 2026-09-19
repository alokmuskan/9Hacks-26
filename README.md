# Intelligent Monitoring Platform

A full-stack, real-time computer-vision monitoring system that combines **face recognition**, **object detection**, and **gaze estimation** into a single pipeline, then records **who looked at what, and for how long**.

The system ships two front ends over one shared core:

- a **CLI / desktop mode** (`main.py`) with an OpenCV window, HUD, and live keyboard controls;
- an **API mode** (`server.py`) exposing REST, WebSocket, and MJPEG endpoints for the bundled React dashboard.

**Status:** `Milestone: Frontend and Backend Working`

---

## Table of Contents

- [Overview](#overview)
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
  - [CLI Mode](#cli-mode)
  - [Runtime Keyboard Controls](#runtime-keyboard-controls)
- [API Reference](#api-reference)
- [WebSocket Events](#websocket-events)
- [Data and Storage Model](#data-and-storage-model)
- [Testing](#testing)
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
├── object_detection.py      # DualYoloDetector (general + custom) and box normalisation
├── scene_memory.py          # Snapshot store, metadata index, last-seen and search
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
└── .env.example             # Environment variable template
```

### Which files matter most

| File | Role |
| --- | --- |
| `main.py` | Core logic: models, camera handling, gaze math, behavior tracker, metrics, chat, reports |
| `server.py` | The whole HTTP/WS surface; worker loops that mirror the CLI pipeline |
| `object_detection.py` | YOLO wrapper and detection schema shared by both modes |
| `scene_memory.py` | All persisted snapshot state and retrieval helpers |
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
| `insightface` | `>=0.7.3,<0.8` |
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

If you are not using pixi, install the equivalent packages in a Python 3.11 virtual environment. The `l2cs` and `face_detection` packages must be installed from their Git repositories.

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
| `AI_STUDIO_CAM_CAMERA_INDEX` | `/dev/video42` on Linux, `0` elsewhere | backend | Camera device path, numeric index, or stream URL |
| `AI_STUDIO_CAM_INCLUDE_INDEX_FALLBACK` | `0` | backend | Set to `1` to also try the numeric index when a `/dev/video*` path is configured |
| `AI_STUDIO_CAM_SCAN_ALL_DEVICES` | `0` | backend (Linux) | Set to `1` to scan every `/dev/video*` device before giving up |
| `AI_STUDIO_GENERAL_YOLO_MODEL` | `.references/AI-Studio-Cam-(On-Hold)/models/yolov8n.pt` | backend | General YOLO checkpoint; falls back to `yolov8n.pt` if the path does not exist |
| `GROQ_API_KEY` *(or `groq_api_key`)* | unset | backend | Enables the LLM chat fallback |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | backend | Groq model name |
| `PORT` | `8000` | backend | Port for `python server.py` |
| `VITE_API_BASE` | `http://localhost:8000` | frontend | Backend base URL (also used to derive the WebSocket URL) |

Camera selection tries multiple candidate sources and backends (FFmpeg / V4L2 / platform default) before failing, and the failure message lists what was attempted.

---

## Model Assets and First-Run Downloads

| Asset | Default location | How it is obtained |
| --- | --- | --- |
| InsightFace face pack | `~/.insightface/models/<pack>` | Downloaded automatically on first use. If the pack is extracted with a nested directory layout, the app repairs it in place. |
| General YOLO weights | `yolov8n.pt` (or `AI_STUDIO_GENERAL_YOLO_MODEL`) | Ultralytics downloads the checkpoint if it is not present |
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

### CLI Mode

All commands run from the repository root so that relative artifact paths resolve correctly.

```bash
# Enroll an identity (press q to finish early)
pixi run python main.py enroll --name Hemanth

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
pixi run python main.py memory-find-person --name Hemanth
pixi run python main.py memory-search --text "person using laptop"

# Summaries
pixi run python main.py session-summary --minutes 5
pixi run python main.py session-summary --minutes 10 --json

# Chat
pixi run python main.py chat
pixi run python main.py chat --question "What happened in the last 10 minutes?"

# ASCII security report -> report.txt
pixi run python main.py report
```

Global flag: `--model {buffalo_l,buffalo_m,buffalo_s,buffalo_sc,antelopev2}` (default `buffalo_sc`).

`recognize` options: `--general-model`, `--custom-model`, `--disable-general`, `--disable-custom`, `--disable-gaze`, `--snapshot-interval`, `--gaze-arch`, `--gaze-weights`, `--gaze-weights-source`, `--disable-gaze-auto-download`.

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
| `metrics_log.jsonl` | JSONL, append-only | Event stream: `enroll`, `recognize_session`, `behavior_event`, `chat_query`, `summary_query`, `chat_action_proposed`, `chat_action_executed`, `memory_query`, `object_train`, plus legacy event types |
| `memory/metadata.json` | JSON array | Snapshot index: timestamps, path, object labels, manual flag, optional faces/objects/people/attention |
| `memory/snapshots/*.jpg` | JPEG | Automatic (default every 15 s) and manual snapshots |
| `unknown_incidents/*.jpg` | JPEG | Frames containing unrecognised faces (throttled to one per 3 s) |
| `face_db.npz` | NumPy archive | Enrolled identities: names, L2-normalised centroids, sample counts |
| `custom_model_path.txt` | Text | Pointer to the active custom YOLO checkpoint |
| `report.txt` | Text | ASCII security dashboard generated by `report` |

`recognize_session` records aggregate performance and behavior rollups (FPS, detection latency, per-person presence, object class counts, attention map, top attended objects, chat/memory query counters). The session schema version is `4` from the CLI and `5` from the API; enrollment is `2` (CLI) and `3` (API). Readers normalise older rows.

---

## Testing

### Backend

```bash
pixi run python -m unittest discover -s tests -q
```

44 tests across seven modules cover the detection schema and toggles, gaze L2CS helpers and interval adaptation, behavior tracking and summaries, chat and snapshot actions, metrics schema normalisation, scene memory, and the FastAPI surface (status schema, monitor lifecycle, camera recovery, stream generator, WebSocket envelope, chat sessions, and two-step action confirmation).

> The server tests import `server.py` at module scope, which creates `memory/snapshots/` and `unknown_incidents/`. Chat tests append to `metrics_log.jsonl`. These paths are git-ignored.

### Frontend

```bash
cd frontend
npm test          # vitest
npm run build     # production build
```

Frontend tests cover the WebSocket client (event forwarding, backoff reconnects) and stream stall detection.

---

## Troubleshooting

**The camera cannot be opened**

`_open_camera()` tries each candidate source across several backends and throws with the list of attempts. Check that `AI_STUDIO_CAM_CAMERA_INDEX` points at a real device. On Linux, `/dev/video42` typically indicates a `v4l2loopback` virtual device, so the producer feeding that device must be running. Set `AI_STUDIO_CAM_SCAN_ALL_DEVICES=1` to scan all devices.

**Monitoring starts but the frontend shows "Stalled"**

The stream is driven by a monotonic frame `sequence` plus UTC timestamps. Stalls are only reported after a 15-second warmup grace period, so the message indicates the worker is not producing frames. Check the pipeline status endpoint for `degraded`, `degraded_reason`, and `last_error`.

**Gaze is unavailable**

The L2CS checkpoint could not be resolved. Place the weights at `models/L2CSNet_gaze360.pkl`, or allow auto-download (`gdown` must be installed and the Google Drive source reachable). The pipeline continues without gaze.

**Chat replies "Insufficient evidence"**

No deterministic intent matched and either no grounding citations were found or no `GROQ_API_KEY` is configured. Ask about summaries, last-seen, memory stats, presence, attention, or counts to use the deterministic paths.

**`pixi install` fails on macOS or Windows**

Expected: `pixi.toml` restricts the workspace to `linux-64` and `pixi.lock` contains Linux packages only. Use Linux, WSL, or a manually built equivalent environment.

**Enrollment reports no samples**

Enrollment requires exactly one detectable face at a time; multiple faces or zero faces are rejected. The default target is 25 samples captured every 0.25 s.

---

## Current Limitations

Verified against the current code:

- **Linux-targeted environment.** The pixi workspace and lock file resolve `linux-64` only.
- **One active worker.** A single global `PipelineManager` runs either monitor or enroll, never both.
- **Vector search is not wired in.** `scene_memory.py` supports CLIP + FAISS semantic search, but the monitor and enroll pipelines construct the memory manager with `enable_vectors=False`, so the index is never populated during normal operation. `memory-search` falls back to lexical matching.
- **Gaze runs every processed frame.** Interval-adaptation helpers exist and are unit-tested, but the runtime pipelines do not call them; the reported gaze interval metrics are fixed at 1 frame.
- **Duplicated pipeline logic.** The CLI loop (`cmd_recognize`) and the API worker (`_run_monitor_worker`) reimplement the same flow and already differ in places (for example, gaze latency metrics are collected only in the CLI path).
- **Custom object detection needs your own data.** `train-objects` requires a YOLO dataset YAML; no dataset ships with the repository.
- **Metrics log has no rotation.** `metrics_log.jsonl` grows without bound.
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
