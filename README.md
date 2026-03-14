# Intelligent Monitoring Platform

Full-stack monitoring system with working backend API + frontend dashboard.

## Milestone

`Milestone: Frontend and Backend Working`

## What Works

- Real-time monitor pipeline (face recognition + object detection + gaze)
- Behavior tracking (who looked at what)
- Memory snapshots and searchable history
- Chat + summary queries over logs/memory
- FastAPI backend with REST + WebSocket streams
- React/Vite frontend dashboard connected to backend

## Project Structure

- `main.py` - CLI pipeline and local commands
- `server.py` - FastAPI backend
- `frontend/` - React client app
- `docs/ARCHITECTURE.md` - pipeline/runtime internals
- `docs/CHAT_AND_SUMMARY.md` - chat and summary behavior

## Backend Run

```bash
pixi install
AI_STUDIO_CAM_CAMERA_INDEX=/dev/video42 pixi run python server.py
```

Backend default: `http://localhost:8000`

## Frontend Run

```bash
cd frontend
npm install
npm run dev
```

Frontend default: `http://localhost:5173`

`VITE_API_BASE` controls backend URL (default `http://localhost:8000`).

## CLI Run (Optional)

```bash
pixi run python main.py enroll --name Hemanth
pixi run python main.py recognize
pixi run python main.py session-summary --minutes 5
pixi run python main.py chat --question "What happened in the last 5 minutes?"
```

## Key API Endpoints

- `POST /api/v1/monitor/start`
- `POST /api/v1/monitor/stop`
- `GET /api/v1/monitor/status`
- `GET /api/v1/stream/video`
- `WS /api/v1/stream/events/ws`
- `POST /api/v1/chat/query`
- `GET /api/v1/summaries/session?minutes=5`

## Environment

Use `.env` for secrets/config:

- `AI_STUDIO_CAM_CAMERA_INDEX`
- `AI_STUDIO_GENERAL_YOLO_MODEL`
- `GROQ_API_KEY` (or `groq_api_key`)
- `GROQ_MODEL`
- `PORT`

## Tests

Backend:

```bash
pixi run python -m unittest discover -s tests -q
```

Frontend:

```bash
cd frontend
npm test
npm run build
```
