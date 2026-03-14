# 9Hacks-26

Real-time vision console for:
- face enrollment and recognition (InsightFace)
- object detection with dual YOLO streams (general + custom)
- scene memory snapshots and retrieval
- unknown-incident capture and session reporting

## Features

- Live face recognition with persistent local face database (`face_db.npz`)
- Dual YOLO inference:
  - general model (default `yolov8n.pt` or configured path)
  - optional custom fine-tuned model
- Scene memory manager with:
  - periodic/manual snapshots
  - metadata storage
  - vector search via CLIP + FAISS (when available)
- Unknown face incident image capture
- Metrics logging (`metrics_log.jsonl`) and ASCII report generation (`report.txt`)

## Requirements

- Linux (project currently configured for `linux-64` in `pixi.toml`)
- Webcam/camera device
- [Pixi](https://pixi.sh/) recommended for environment + dependency management

## Setup

```bash
pixi install
```

## Quick Start

Enroll a person:

```bash
pixi run python main.py enroll --name "YourName"
```

Start live recognition + object detection:

```bash
pixi run python main.py recognize
```

Generate report:

```bash
pixi run python main.py report
```

## CLI Commands

```bash
python main.py enroll --name <name>
python main.py recognize [--general-model <path>] [--custom-model <path>] [--disable-general] [--disable-custom] [--snapshot-interval 15]
python main.py train-objects --data <dataset.yaml> [--base-model yolov8n.pt] [--epochs 30] [--imgsz 640] [--batch 16] [--project runs/detect] [--name custom-objects] [--set-default]
python main.py memory-stats
python main.py memory-recent --minutes 5
python main.py memory-find --object "<label>"
python main.py memory-search --text "<query>"
python main.py list
python main.py report
```

If no command is passed, `recognize` is used by default.

## Runtime Controls (recognize mode)

- `q`: quit
- `g`: toggle general YOLO
- `o`: toggle custom YOLO
- `t`: manual memory snapshot
- `m`: print memory stats
- `r`: list recent snapshots (last 5 minutes)
- `f`: find when an object was last seen
- `h`: print runtime help

## Environment Variables

- `AI_STUDIO_CAM_CAMERA_INDEX` (default: `42`)
- `AI_STUDIO_GENERAL_YOLO_MODEL` (default: `.references/AI-Studio-Cam-(On-Hold)/models/yolov8n.pt`)

## Generated Artifacts

- `face_db.npz`: enrolled face embeddings/centroids
- `unknown_incidents/`: unknown face snapshots
- `memory/`: scene snapshots + metadata + FAISS index
- `metrics_log.jsonl`: event/session metrics
- `report.txt`: generated ASCII summary report
- `custom_model_path.txt`: pointer to selected custom YOLO model (when set)
