# Intelligent Monitoring System

A real-time AI monitoring pipeline that combines:

- Face recognition (InsightFace)
- Object detection (YOLO: general + custom)
- Gaze prediction (L2CS-Net)
- Behavior tracking (who looked at what, and for how long)
- Memory snapshots with searchable metadata
- Chat and situation summary queries over recent activity

## Milestone

This milestone adds working chatbot and summary capabilities on top of face/object/gaze + memory data.

## Setup

Install dependencies with Pixi:

```bash
pixi install
```

## Quick Start

Enroll a person:

```bash
pixi run python main.py enroll --name Hemanth
```

Start live monitoring:

```bash
pixi run python main.py recognize
```

Generate summary for the last 5 minutes:

```bash
pixi run python main.py session-summary --minutes 5
```

Ask a one-shot chatbot question:

```bash
pixi run python main.py chat --question "What happened in the last 5 minutes?"
```

## Commands

```bash
python main.py enroll --name <person>
python main.py recognize [--disable-general] [--disable-custom] [--disable-gaze]
python main.py train-objects --data <dataset.yaml> [--set-default]
python main.py memory-stats
python main.py memory-recent --minutes <n>
python main.py memory-find --object <label>
python main.py memory-find-person --name <person>
python main.py memory-search --text "<query>"
python main.py session-summary --minutes <n> [--json]
python main.py chat [--question "..."]
python main.py list
python main.py report
```

## Chatbot Behavior

Deterministic intents supported directly from logs/memory:

- session summary queries
- last-seen object queries
- last-seen person queries
- memory stats and recent snapshots
- attention-style questions (for example: who looked at laptop)

For open-ended prompts, the system can use Groq when configured.

## Runtime Controls (during recognize)

- `q`: quit
- `g`: toggle general YOLO
- `o`: toggle custom YOLO
- `c`: open chatbot prompt
- `t`: manual snapshot
- `m`: memory stats
- `r`: recent snapshots
- `f`: find last-seen object
- `h`: help

## Environment Variables

Use `.env` or shell exports:

- `AI_STUDIO_CAM_CAMERA_INDEX`
- `AI_STUDIO_GENERAL_YOLO_MODEL`
- `GROQ_API_KEY` (preferred)
- `groq_api_key` (compat fallback)
- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)

## Stored Data

- `metrics_log.jsonl`: append-only events/sessions
- `memory/metadata.json`: memory metadata
- `memory/snapshots/*`: captured frames
- `memory/embeddings.faiss`: vector index (if enabled)

## Test

```bash
pixi run python -m unittest discover -s tests -q
```

## Docs

- `docs/ARCHITECTURE.md`
- `docs/CHAT_AND_SUMMARY.md`
