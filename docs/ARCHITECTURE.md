# Architecture

## Pipeline Overview

`recognize` loop execution order:

1. Capture frame from camera
2. Run InsightFace detection/recognition
3. Run YOLO object detection (general/custom)
4. Run gaze estimation (if enabled and model loaded)
5. Fuse identity + gaze + objects into attention target inference
6. Update behavior tracker (start/switch/end events)
7. Render overlays and HUD
8. Persist periodic/manual snapshots and event logs

## Behavior Inference Rules

Gaze target mapping:

1. Endpoint-in-bbox match (`+8px` padding)
2. Else nearest bbox by point-to-rect distance
3. Accept nearest only if distance `<= 120px`

Temporal stabilization:

- Rolling window: `5` frames
- Switch confirmation: `3` frames
- Lost timeout for end event: `1.0s`

## Event Model

`behavior_event` payload fields:

- `event`: `start | switch | end`
- `person`
- `target_object`
- `previous_target` (switch only)
- `duration_sec`
- `event_time_utc`
- `session_id` (added at write time)

`recognize_session.aggregate` behavior metrics:

- `behavior_interactions_total`
- `behavior_attention_total_sec`
- `behavior_top_objects`
- `behavior_attention_map`
- `behavior_events_count`
- `behavior_activity_patterns`

## Storage Model

### `metrics_log.jsonl`

Append-only records, including:

- `recognize_session` (schema v4)
- `behavior_event`
- `chat_query`
- `summary_query`
- existing legacy records (`enroll`, `object_train`, `memory_query`, etc.)

### `memory/metadata.json`

Backward-compatible snapshot entries:

- always: timestamps, snapshot path, object labels, manual flag
- optional enrichment:
  - `faces`
  - `object_detections`
  - `people`
  - `attention`

## Chat / Summary Flow

Deterministic intents are evaluated first. If no intent matches, LLM fallback path is attempted:

- Build retrieval context from:
  - latest recognize session aggregate
  - recent behavior events
  - semantic memory search hits
- Query Groq model

Situation summaries (`session-summary` or chat request) use deterministic aggregation over time-windowed persisted logs.
