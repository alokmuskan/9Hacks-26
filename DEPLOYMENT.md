# Deployment Guide

This is the operational runbook for putting the monitoring system online for the **expo demo**.
Read the "Status" table first: two code changes are prerequisites and are marked **PENDING**
until they land. Everything else is verified working on the host machine.

Companion docs: `GETTING_STARTED.md` (local setup) · `README.md` (architecture + config reference).

---

## 1. Deployment model (why this architecture)

The app needs a **camera**, the full CV stack (torch, insightface, FAISS, YOLO), and stores
biometric face embeddings on local disk. The expo deployment keeps all of that on one machine —
the demo laptop — and exposes it through an outbound-only tunnel:

```
Visitors' phones / laptops
        │  https://<random>.trycloudflare.com   (free Cloudflare quick tunnel)
        ▼
Demo laptop: uvicorn :8000  ── API + WebSocket + MJPEG + built frontend (PENDING)
        ▼
Webcam → YOLO + face recognition + gaze  (all local; biometrics never leave the machine)
```

| Property | Value |
| --- | --- |
| Cost | **$0** — quick tunnels are free, no Cloudflare account needed |
| HTTPS | Automatic (terminated by Cloudflare) |
| Venue Wi-Fi compatible | Yes — tunnel is **outbound-only**, no port forwarding |
| URL stability | Quick-tunnel URL is **random per launch** — this is why same-origin serving (Phase 0) matters |
| Biometric data | Stays on the laptop (`face_db.npz`, `memory/`) — a privacy talking point for the demo |

Not this guide's scope: multi-node/cloud VM deploys, multi-camera scaling. The single-node
design (local-disk state) is intentional for this use case.

---

## 2. Status — what is ready vs. what must be done first

| # | Item | Status | Notes |
| --- | --- | --- | --- |
| 1 | Serve `frontend/dist` from FastAPI + SPA fallback | **PENDING — required** | Without it the SPA isn't served in production, and `VITE_API_BASE` would need to be baked at build time — impossible with a random tunnel URL. Same-origin also eliminates CORS in production entirely. |
| 2 | Frontend `VITE_API_BASE` defaults to same-origin | **PENDING — required** | Currently `http://localhost:8000` (frontend/src/api/client.js:1), which breaks for every visitor's phone. |
| 3 | Lock CORS to same-origin in production | **PENDING — required** | `allow_origins=["*"]` at server.py:1565. Keep localhost for dev; same-origin for deploy. |
| 4 | Access gate (shared token via env var) | **PENDING — strongly recommended** | The public URL streams your camera. Minimum viable: a token checked on API routes, entered once per browser. Decision for the owner: ship it, or rely on link secrecy + teardown. |
| 5 | Full local run verified (faces + YOLO + gaze, `doctor` 0 failures) | ✅ Ready | Verified on this machine; `.venv` has all six features. |
| 6 | Chat LLM (Groq) | ✅ Ready | Key in `.env` (never committed — verified), model `openai/gpt-oss-120b`. |
| 7 | Bounded storage / retention knobs | ✅ Ready | `AI_STUDIO_METRICS_MAX_BYTES`, `AI_STUDIO_MEMORY_MAX_AUTO_SNAPSHOTS`, `AI_STUDIO_UNKNOWN_INCIDENT_MAX_FILES`. |
| 8 | Commit + push pending work (CI fix, responsive UI, drawer) | **PENDING — required** | Deploy from a clean, green commit. `ci.yml` is not yet on `origin/main`. |

---

## 3. Host prerequisites (the demo laptop)

Current host: this Windows machine, project root `D:/User/Desktop/Expo`.

- [ ] Project venv present: `./.venv/Scripts/python.exe --version`
      (⚠️ `python` on PATH resolves to the lint-only `.venv-tools` — always use `./.venv/Scripts/python.exe`.)
- [ ] `./.venv/Scripts/python.exe main.py bootstrap` — creates dirs, fetches `yolov8n.pt`
- [ ] `./.venv/Scripts/python.exe main.py doctor` — must end with **0 failure(s)**
      (warnings are fine; each disables one feature. Gaze weights live at `models/L2CSNet_gaze360.pkl` —
      if missing on a fresh machine, see the README gaze-weights section.)
- [ ] Models pre-fetched **before the expo** (first-run downloads will fail on venue Wi-Fi):
      `yolov8n.pt`, InsightFace `buffalo_sc` (`~/.insightface/models/`), gaze weights.
- [ ] `cd frontend && npm ci && npm run build` → `frontend/dist` exists and is fresh
- [ ] Faces enrolled at home: `main.py enroll --name <name>` → `face_db.npz` travels with the laptop
- [ ] `cloudflared` installed (one-time, no account):
      ```powershell
      winget install --id Cloudflare.cloudflared
      ```
      or download the single exe from
      `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe`
- [ ] Windows power set to **never sleep** on AC; charger planned

---

## 4. Environment configuration for the demo

All values optional unless noted. `.env` is loaded by `main.py` at import time; `server.py`
inherits it (it imports `main`). Never commit `.env`.

| Variable | Demo value | Purpose |
| --- | --- | --- |
| `AI_STUDIO_CAM_CAMERA_INDEX` | `"0"` (webcam) | Plan C: set to a demo `.mp4` path instead |
| `GROQ_API_KEY` | existing key | Enables LLM chat; without it chat falls back to deterministic answers |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Current working model |
| `PORT` | `8000` (default) | uvicorn port |
| `UVICORN_GRACEFUL_SHUTDOWN_SEC` | `5` (default) | Bounded shutdown even with streams open |
| `VITE_API_BASE` | *(unset)* | **Leave unset** — Phase 0 makes same-origin the default; that's the whole point |
| *(pending)* `DEMO_ACCESS_TOKEN` | strong random string | Access gate, if shipped |

---

## 5. Deploy runbook (execute in order)

### Step 1 — Freeze the code
- [ ] All Phase 0 items merged; working tree clean: `git status --short` shows nothing
- [ ] CI green on the pushed commit (backend 151 tests, lint, frontend build)

### Step 2 — Build the frontend
```bash
cd frontend && npm ci && npm run build && cd ..
```

### Step 3 — Configure `.env`
- [ ] Camera source set (webcam `0`; demo video path as plan C)
- [ ] `GROQ_API_KEY` present, `GROQ_MODEL=openai/gpt-oss-120b`

### Step 4 — Pre-flight checks (no tunnel yet)
```bash
./.venv/Scripts/python.exe main.py doctor        # 0 failure(s)
./.venv/Scripts/python.exe server.py &           # or a second terminal
curl -s http://127.0.0.1:8000/api/v1/monitor/status | head -c 200
```
- [ ] Status JSON returns (this endpoint is the healthcheck)
- [ ] Browser at the served origin shows the dashboard (Phase 0: same origin :8000)
- [ ] Start a monitor session from the dashboard; verify MJPEG frames, counts, chat answers

### Step 5 — Open the tunnel
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```
- [ ] Copy the printed `https://<random>.trycloudflare.com` URL
- [ ] **Verify from a phone on mobile data** (not venue Wi-Fi): dashboard loads, live feed plays,
      WebSocket events tick, chat responds

### Step 6 — Demo smoke test (from the phone, before doors open)
- [ ] Live Monitor: video + HUD render, ~real FPS
- [ ] Walk in front of the camera → name appears as known face; counts update
- [ ] Chatbot: data question ("how many people in the last 10 minutes") and an open-ended question
      (LLM path — `used_llm: true`)
- [ ] Memory/Reports pages load historical data
- [ ] Ctrl+C on the server closes in ≈5 s (graceful-shutdown bound)

---

## 6. Contingency plans

| Failure | Plan |
| --- | --- |
| Venue Wi-Fi blocks the tunnel | Phone hotspot (LTE) — tunnels are outbound-only, works immediately |
| Webcam fails / bad lighting | `AI_STUDIO_CAM_CAMERA_INDEX` → path to a demo `.mp4`; identical pipeline, no camera needed (verified) |
| Tunnel URL got shared too widely | Kill tunnel, relaunch for a fresh random URL; enable the token gate if shipped |
| Groq unreachable at the venue | Nothing breaks — chat silently falls back to deterministic, data-backed answers |
| Laptop dies | Second laptop or phone-hotspot + resume; state files are all in the project dir, copy them over |

## 7. Teardown after the expo

1. `Ctrl+C` the tunnel, then the server (bounded shutdown, ≈5 s)
2. Decide on biometric data: delete `face_db.npz` and `memory/` if not needed post-event
3. Rotate `GROQ_API_KEY` if the machine or `.env` left your control
4. The tunnel disappears with the process — no resources to deprovision

## 8. Known limits (state them to judges before they find them)

- Single camera, single node; viewer count scales but camera count does not
- Quick-tunnel URLs are random per launch (stability requires a named tunnel + CF account)
- No auth by default — the gate (Phase 0 #4) is the mitigation for a public link
- Face recognition accuracy depends on lighting/angle — rehearse the booth lighting at home
