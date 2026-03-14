# 9Hacks-26

Real-time vision pipeline for:
- face enrollment and recognition
- dual YOLO object detection (general + custom)
- gaze prediction (L2CS-Net)
- scene memory snapshots and search
- session metrics and report generation

## Stack

- Python 3.11
- InsightFace
- OpenCV
- Ultralytics YOLO
- PyTorch / TorchVision
- L2CS-Net (via pip git dependency)
- FAISS + OpenCLIP for memory search
- Pixi for environment and task management

## Setup

```bash
pixi install
```

## Core Commands

```bash
# enroll an identity
pixi run python main.py enroll --name "Hemanth"

# live face + object + gaze recognition
pixi run python main.py recognize

# generate report artifacts
pixi run python main.py report

# list enrolled identities
pixi run python main.py list
```

If no subcommand is provided, `recognize` is used by default.

## Recognize Options

```bash
python main.py recognize \
  --general-model <path-or-default> \
  --custom-model <path> \
  --snapshot-interval 15 \
  --gaze-arch ResNet18 \
  --gaze-weights models/L2CSNet_gaze360.pkl
```

Useful flags:
- `--disable-general`
- `--disable-custom`
- `--disable-gaze`
- `--gaze-weights-source <url>`
- `--disable-gaze-auto-download`

## Object Model Training

```bash
python main.py train-objects \
  --data <dataset.yaml> \
  --base-model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --set-default
```

## Memory Commands

```bash
python main.py memory-stats
python main.py memory-recent --minutes 5
python main.py memory-find --object "person"
python main.py memory-search --text "person near doorway"
```

## Runtime Hotkeys

- `q`: quit
- `g`: toggle general YOLO
- `o`: toggle custom YOLO
- `t`: take manual memory snapshot
- `m`: print memory statistics
- `r`: show recent snapshots
- `f`: find when object was last seen
- `h`: print help

## Local Artifacts

Generated runtime artifacts are intentionally gitignored:
- `face_db.npz`
- `memory/`
- `unknown_incidents/`
- `metrics_log.jsonl`
- `report.txt`
- `models/`
