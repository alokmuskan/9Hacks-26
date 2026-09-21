# Getting Started

> Taking this to a public demo or expo? [`DEPLOYMENT.md`](DEPLOYMENT.md) is the runbook for that.

A short, verified path from clone to a live monitoring session. The full reference
(including architecture, storage internals, and every endpoint) lives in
[README.md](README.md).

---

## What you are starting

A real-time monitoring pipeline that runs **face recognition + object detection +
gaze estimation** over a camera, records who looked at what and for how long, and
serves it through either:

- a **dashboard** — `server.py` (API on `:8000`) + React frontend (on `:5173`), or
- a **desktop CLI** — `python main.py recognize` (OpenCV window + keyboard controls).

No database, no cloud services, no accounts. The only optional external service is
the Groq API key for LLM-backed chat answers; without it, chat answers
deterministically from the recorded data.

---

## Prerequisites

| Requirement | Needed for |
| --- | --- |
| **Python 3.11+** (3.13 verified) | backend |
| **Node.js** 20.19+/22.12+ and npm | dashboard |
| A camera (webcam index, device path, or video file/URL) | the pipeline |
| Git | the two git-pinned pip packages (`l2cs`, `face_detection`) |
| NVIDIA GPU + CUDA | *optional* — CPU works, just slower |

Everything heavy (YOLO weights, InsightFace `buffalo_sc` models) downloads
automatically on first use.

---

## Path A — Windows (pip) · verified

> Always run the project through the venv's own interpreter
> (`./.venv/Scripts/python.exe`), never a bare `python` — your PATH may point at
> an unrelated interpreter.

```bash
# 1. Environment
python -m venv .venv
. .venv/Scripts/activate            # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt     # insightface installs from the 2.x wheel (no compiler)

# 2. Configuration
cp .env.example .env                # then edit camera index / Groq key if needed

# 3. Health check — must end with "0 failure(s)"
python main.py doctor

# 4. First-run downloads (dirs + YOLO weights)
python main.py bootstrap

# 5. Run — two separate terminals
./.venv/Scripts/python.exe server.py        # terminal 1: API + WebSocket on :8000
cd frontend && npm install && npm run dev   # terminal 2: dashboard on :5173
```

Open **http://localhost:5173** and click **Start monitoring**.

## Path B — Linux (pixi)

```bash
pixi install                            # linux-64 workspace, insightface 0.7.x prebuilt
cd frontend && npm install && cd ..
cp .env.example .env
pixi run python main.py doctor          # must say 0 failure(s)
pixi run python main.py bootstrap
pixi run python server.py               # terminal 1
cd frontend && npm run dev              # terminal 2
```

---

## First real session

1. **Enroll** at least one identity (needs a real camera, one face at a time,
   25 samples taken automatically):
   ```bash
   python main.py enroll --name YourName
   ```
2. **Monitor** — either click *Start monitoring* in the dashboard, or run the CLI:
   ```bash
   python main.py recognize
   ```
3. **Ask** — in the dashboard's chat box, or press `c` in the CLI:
   "how many people were visible?", "when was the cup last seen?", "summarize this session".

CLI keys during `recognize`: `q` quit · `g`/`o` toggle YOLO models · `c` chat ·
`t` manual snapshot · `m` memory stats · `r` recent snapshots · `f` find object · `h` help.

---

## Feature matrix — what needs what

| Feature | Needs | If missing |
| --- | --- | --- |
| Object detection (YOLO) | `ultralytics` + weights | fails (required) |
| Face recognition | `insightface >= 0.7.3` | degrades: session runs, identities unmatched |
| Enrolled identities | `enroll --name ...` | faces detected, but all reported "Unknown" |
| Gaze estimation | `l2cs` + `models/L2CSNet_gaze360.pkl` | degrades: attention tracking off |
| Vector scene search | `faiss-cpu` + `open-clip-torch` | degrades: lexical fallback |
| LLM chat answers | `groq` + `GROQ_API_KEY` | degrades: deterministic data-backed answers |

Every degradation is *reported*, never silent: the session keeps running and
`degraded_reason` (API status) / the doctor output says exactly what is off.
The persisted session aggregate records `face_recognition_enabled` and
`gaze_enabled` so a face-less session is distinguishable from "nobody was in frame".

---

## Configuration (`.env`)

`main.py` loads `.env` from the project root at startup. Full annotated list in
[`.env.example`](.env.example); the essentials:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AI_STUDIO_CAM_CAMERA_INDEX` | `0` | Camera index, `/dev/video*` path, or video file/URL |
| `GROQ_API_KEY` | *(unset)* | Enables LLM chat fallback |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `PORT` | `8000` | API server port |
| `UVICORN_GRACEFUL_SHUTDOWN_SEC` | `5` | Max seconds to wait for open streams (MJPEG/WebSocket) when shutting down with Ctrl+C |
| `AI_STUDIO_GENERAL_YOLO_MODEL` | `yolov8n.pt` | General YOLO checkpoint |
| `AI_STUDIO_METRICS_MAX_BYTES` | `4194304` | Metrics log rotation size |
| `AI_STUDIO_METRICS_BACKUPS` | `2` | Rotated metric generations kept |
| `AI_STUDIO_MEMORY_MAX_AUTO_SNAPSHOTS` | `5000` | Auto-snapshot retention cap |
| `AI_STUDIO_UNKNOWN_INCIDENT_MAX_FILES` | `500` | Unknown-face capture cap |
| `AI_STUDIO_YOLO_IMGSZ` | `768` | Object-detection inference size (bigger finds more, costs CPU) |
| `AI_STUDIO_YOLO_CONF` | `0.25` | Object-detection confidence threshold |
| `AI_STUDIO_YOLO_IOU` | `0.7` | NMS IoU threshold |
| `AI_STUDIO_YOLO_MAX_DET` | `300` | Maximum boxes per frame |
| `AI_STUDIO_YOLO_AGNOSTIC_NMS` | `false` | One box pool across classes (stops double-labelling; can drop a distinct overlapping label) |
| `AI_STUDIO_FPS_CAP` | `12` | Monitor FPS ceiling (lower = less CPU) |
| `AI_STUDIO_ENROLL_FPS_CAP` | `20` | Enrollment FPS ceiling |
| `AI_STUDIO_SNAPSHOT_INTERVAL` | `8.0` | Seconds between auto snapshots (the instances used for analysis) |

Dashboard → backend URL: set `VITE_API_BASE` in `frontend/.env.local` (defaults to
`http://localhost:8000`).

---

## Troubleshooting quick hits

| Symptom | Fix |
| --- | --- |
| `doctor` reports a `fail` | A required module is missing/unusable — follow the printed message, re-run `doctor` |
| Monitor start returns `409` | Nothing to stop / already running — check `GET /api/v1/monitor/status` first |
| Camera won't open | Set `AI_STUDIO_CAM_CAMERA_INDEX`; `python main.py doctor --check-camera` probes it |
| "Face recognition DISABLED" | `pip install -U "insightface>=0.7.3,<3"` — 2.x is a no-compiler wheel on Windows |
| Gaze weights missing | Upstream Drive link is dead; see README *Gaze is unavailable* for the Hugging Face mirror + one-line conversion |
| Frontend shows "Stalled" | Worker not producing frames — check `degraded` / `degraded_reason` / `last_error` in the status endpoint |

---

## Verify your install

```bash
python -m compileall -q main.py server.py common.py scene_memory.py object_detection.py
python -m unittest discover -s tests          # the backend battery; the run prints its own count
python main.py bench-detect                   # detection quality on your own frames
python main.py frame-budget                   # where a recorded session's frame time went
python main.py review-detections              # box-by-box review -> a real precision figure
cd frontend && npm test                       # 7 tests
ruff check . && mypy                          # lint + types (see ruff.toml / mypy.ini)
```

`bench-detect`, `frame-budget` and `review-detections` answer three different questions and none replaces another: the first measures the detector on saved frames (recall against labelled reference images, detection statistics on yours), the second reads the sessions already in `metrics_log.jsonl` and reports what the frame period was made of — including which stages the log does *not* record — and the third turns the detector's own output into a measured **precision** figure with a two-minute review and no new capture. Only the last one measures whether individual detections are *correct*; follow it with `score-detections --review-dir reviews`.

The detection benchmark needs the real computer-vision stack (`ultralytics` + OpenCV). The rest of the battery stubs `cv2`, so its inference tests report as **skipped** on a machine without them — a skip there means "not checked here", not "passing".

CI runs the same battery on every push (`.github/workflows/ci.yml`).
