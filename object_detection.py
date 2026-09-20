from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np

import common


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def find_latest_custom_model() -> str | None:
    patterns = [
        "runs/detect/*/weights/best.pt",
        "runs/detect/*/weights/last.pt",
        "models/custom_yolo*.pt",
        "models/*custom*.pt",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    if not candidates:
        return None

    return str(max((Path(p) for p in candidates if Path(p).exists()), key=lambda p: p.stat().st_mtime))


def normalize_yolo_boxes(
    xyxy: Any,
    conf: Any,
    cls_ids: Any,
    names: dict[int, str] | list[str] | None,
    source: str,
) -> list[dict[str, Any]]:
    boxes = _to_numpy(xyxy).reshape(-1, 4)
    scores = _to_numpy(conf).reshape(-1)
    classes = _to_numpy(cls_ids).reshape(-1)

    if boxes.shape[0] == 0:
        return []

    n = min(boxes.shape[0], scores.shape[0], classes.shape[0])
    output: list[dict[str, Any]] = []

    for i in range(n):
        cls_idx = int(classes[i])
        if isinstance(names, dict):
            label = str(names.get(cls_idx, cls_idx))
        elif isinstance(names, list) and 0 <= cls_idx < len(names):
            label = str(names[cls_idx])
        else:
            label = str(cls_idx)

        output.append(
            {
                "label": label,
                "confidence": float(scores[i]),
                "bbox": boxes[i].astype(np.float32),
                "source": source,
                "class_id": cls_idx,
            }
        )

    output.sort(key=lambda row: row["confidence"], reverse=True)
    return output


class DualYoloDetector:
    def __init__(
        self,
        general_model_path: str = "yolov8n.pt",
        custom_model_path: str | None = None,
        enable_general: bool = True,
        enable_custom: bool = True,
        general_model_obj: Any | None = None,
        custom_model_obj: Any | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        max_det: int | None = None,
    ) -> None:
        self.general_model_path = general_model_path
        self.custom_model_path = custom_model_path

        # Inference parameters are resolved once, here, so the live loop and the
        # offline benchmark cannot drift apart, and so `/monitor/status` can report
        # exactly what inference is using.
        self.conf: float = common.YOLO_CONF_DEFAULT
        self.iou: float = common.YOLO_IOU_DEFAULT
        self.imgsz: int = common.YOLO_IMGSZ_DEFAULT
        self.max_det: int = common.YOLO_MAX_DET_DEFAULT
        self.configure(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)

        self.general_model = general_model_obj
        self.custom_model = custom_model_obj

        self.general_enabled = bool(enable_general)
        self.custom_enabled = bool(enable_custom)

        if self.general_model is None and self.general_enabled:
            self.general_model = self._load_model(general_model_path)

        if self.custom_model is None and self.custom_enabled and custom_model_path:
            self.custom_model = self._load_model(custom_model_path)

        if self.general_model is None:
            self.general_enabled = False
        if self.custom_model is None:
            self.custom_enabled = False

    def configure(
        self,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        max_det: int | None = None,
    ) -> dict[str, Any]:
        """Apply inference parameters (clamped to usable ranges) and return them.

        One clamping path for the constructor, runtime tuning and the offline
        benchmark, so a value cannot be valid in one place and invalid in another.
        ``None`` leaves a parameter at its configured default.
        """
        if conf is not None:
            self.conf = min(max(float(conf), 0.01), 0.99)
        if iou is not None:
            self.iou = min(max(float(iou), 0.1), 0.95)
        if imgsz is not None:
            self.imgsz = common.normalize_imgsz(int(imgsz))
        if max_det is not None:
            self.max_det = min(max(int(max_det), 1), 1000)
        return self.params()

    def _load_model(self, model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is not available. Install dependency 'ultralytics'."
            ) from exc

        return YOLO(model_path)

    def _run_model(self, model: Any, frame: np.ndarray, source: str) -> list[dict[str, Any]]:
        if model is None:
            return []

        results = model(frame, verbose=False, **self.params())
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", None)
        if names is None:
            names = getattr(model, "names", None)
        if names is None and hasattr(model, "model"):
            names = getattr(model.model, "names", None)

        return normalize_yolo_boxes(
            getattr(boxes, "xyxy", None),
            getattr(boxes, "conf", None),
            getattr(boxes, "cls", None),
            names,
            source,
        )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        if self.general_enabled:
            rows.extend(self._run_model(self.general_model, frame, "general"))

        if self.custom_enabled:
            rows.extend(self._run_model(self.custom_model, frame, "custom"))

        rows.sort(key=lambda row: row["confidence"], reverse=True)
        return rows

    def toggle_general(self) -> bool:
        if self.general_model is None:
            self.general_enabled = False
            return self.general_enabled
        self.general_enabled = not self.general_enabled
        return self.general_enabled

    def toggle_custom(self) -> bool:
        if self.custom_model is None:
            self.custom_enabled = False
            return self.custom_enabled
        self.custom_enabled = not self.custom_enabled
        return self.custom_enabled

    def params(self) -> dict[str, Any]:
        """The inference kwargs handed to Ultralytics on every call."""
        return {"conf": self.conf, "iou": self.iou, "imgsz": self.imgsz, "max_det": self.max_det}

    def get_state(self) -> dict[str, Any]:
        return {
            "params": self.params(),
            "general": {
                "enabled": self.general_enabled,
                "loaded": self.general_model is not None,
                "model_path": self.general_model_path,
            },
            "custom": {
                "enabled": self.custom_enabled,
                "loaded": self.custom_model is not None,
                "model_path": self.custom_model_path,
            },
        }
