# Architecture

## Runtime Overview

Two execution modes share the same persistence layer:

- CLI mode (`main.py`) for direct terminal control
- API mode (`server.py`) for browser/dashboard clients

`server.py` uses one global `PipelineManager` that owns:

- a latest-frame store (`frame_bytes`, `timestamp_utc`, `sequence`)
- a monitor or enroll worker thread
- a control queue (toggle commands)
- a websocket pub/sub hub
- recent event cache (`/api/v1/events/recent`)

## Monitor Worker Pipeline

Main monitor loop execution order:

1. Read frame from async camera reader
2. Drain control queue (`general_yolo`, `custom_yolo`, `gaze` toggles)
3. Run face detection/recognition (InsightFace)
4. Run object detection (YOLO general/custom)
5. Run gaze inference (when enabled and loaded)
6. Fuse gaze + objects into attention targets
7. Update behavior tracker (`start`/`switch`/`end`)
8. Persist snapshots + append metrics events
9. Draw HUD, JPEG-encode frame, publish detections event
10. Sleep to satisfy FPS cap (default `20`)

## Startup and Recovery Model

Monitor startup is explicit and observable:

- `POST /api/v1/monitor/start` returns `status: "starting"`
- pipeline status reports `startup_phase`, `startup_started_utc`, `startup_deadline_utc`
- startup phase transitions include:
  - `starting`
  - `camera_opening`
  - `warming_up`
  - `ready`
  - `failed`
  - `idle`

Camera failures use bounded backoff retry and may mark pipeline as degraded.  
`pipeline_state` events carry degraded/error metadata without crashing websocket fan-out.

## Behavior Inference Rules

Target mapping:

1. Gaze endpoint inside object box (`+8px` padding)
2. Otherwise nearest object by point-to-rect distance
3. Accept nearest only if distance `<= 120px`

Temporal stabilization:

- rolling window: `5` frames
- switch confirmation: `3` frames
- end timeout: `1.0s`

`behavior_event` payload fields:

- `event`: `start | switch | end`
- `person`
- `target_object`
- `previous_target` (switch only)
- `duration_sec`
- `event_time_utc`
- `session_id` (added by logger on write)

## API/Event Model

Websocket envelope (`/api/v1/stream/events/ws`):

```json
{
  "type": "detections",
  "timestamp": "2026-03-14T06:00:00+00:00",
  "session_id": "monitor-20260314-113000",
  "payload": {}
}
```

Published event types:

- `pipeline_state`
- `detections`
- `behavior_event`
- `memory_event`
- `chat_result`
- `summary_result`

`/api/v1/detections/latest` payload includes:

- normalized `faces`
- normalized `objects`
- `attention`
- `active_subjects`, `active_objects`
- `counts` (`face_count`, `object_count`, `known_detections`, `unknown_detections`)

## Chat and Action Control

Chat endpoint (`POST /api/v1/chat/query`) runs:

1. action-proposal parsing and deterministic intent matching
2. grounding build from metrics + memory + runtime context
3. optional Groq fallback when evidence exists

API chat supports action confirmation for mutating actions:

- proposal (`intent: action_proposal`) returns `confirm_action_id`
- execution call uses `confirm_action_id` (`intent: action_execute`)
- expired/invalid confirmations are rejected safely

Supported confirmed actions:

- `snapshot`
- `monitor_start`
- `monitor_stop`
- `run_summary`
- `toggle_general_yolo`
- `toggle_custom_yolo`
- `toggle_gaze`

## Storage Model

### `metrics_log.jsonl`

Append-only event stream (key types):

- `recognize_session`
- `behavior_event`
- `chat_query`
- `summary_query`
- `chat_action_proposed`
- `chat_action_executed`
- legacy event types remain supported

`recognize_session.aggregate` includes behavior and performance rollups:

- `behavior_interactions_total`
- `behavior_attention_total_sec`
- `behavior_top_objects`
- `behavior_attention_map`
- `behavior_events_count`
- `behavior_activity_patterns`

### `memory/metadata.json`

Backward-compatible snapshot entries:

- always: timestamps, snapshot path, object labels, manual flag
- optional enrichment:
  - `faces`
  - `object_detections`
  - `people`
  - `attention`
