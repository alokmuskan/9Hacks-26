from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import common


@dataclass(frozen=True)
class DetectorConfig:
    """The parameters forwarded to Ultralytics on every inference call.

    One typed definition, shared by the live detector and the offline benchmark,
    so the two cannot disagree about what inference is using. The field defaults
    are read from `common` (and therefore from the environment), so there is one
    source of truth for both the values and the bounds they are clamped to.

    This used to be five loose attributes on the detector plus a second,
    near-identical dataclass in the benchmark with its own hardcoded numbers —
    which is exactly how the two paths could drift apart unnoticed.
    """

    conf: float = common.YOLO_CONF_DEFAULT
    iou: float = common.YOLO_IOU_DEFAULT
    imgsz: int = common.YOLO_IMGSZ_DEFAULT
    max_det: int = common.YOLO_MAX_DET_DEFAULT
    agnostic_nms: bool = common.YOLO_AGNOSTIC_NMS_DEFAULT

    def as_kwargs(self) -> dict[str, Any]:
        """The exact kwargs handed to the Ultralytics model call."""
        return {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "agnostic_nms": self.agnostic_nms,
        }

    @property
    def label(self) -> str:
        return f"conf={self.conf:g} imgsz={self.imgsz}"

    @classmethod
    def configured(cls) -> DetectorConfig:
        """The configuration the application itself will use (env knobs included)."""
        return cls()

    def updated(
        self,
        *,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        max_det: int | None = None,
        agnostic_nms: bool | None = None,
    ) -> DetectorConfig:
        """Return this configuration with overrides applied, clamped to usable ranges.

        ``None`` leaves a field alone. Clamping lives here rather than at the call
        sites so a value cannot be valid in the live loop and invalid in the
        benchmark.
        """
        return DetectorConfig(
            conf=self.conf if conf is None else common.clamp(float(conf), common.YOLO_CONF_BOUNDS),
            iou=self.iou if iou is None else common.clamp(float(iou), common.YOLO_IOU_BOUNDS),
            imgsz=self.imgsz if imgsz is None else common.normalize_imgsz(int(imgsz)),
            max_det=(
                self.max_det
                if max_det is None
                else int(common.clamp(int(max_det), common.YOLO_MAX_DET_BOUNDS))
            ),
            agnostic_nms=self.agnostic_nms if agnostic_nms is None else bool(agnostic_nms),
        )


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

    return str(
        max((Path(p) for p in candidates if Path(p).exists()), key=lambda p: p.stat().st_mtime)
    )


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
        agnostic_nms: bool | None = None,
        in_scope_classes: Any | None = None,
        excluded_classes: Any | None = None,
    ) -> None:
        self.general_model_path = general_model_path
        self.custom_model_path = custom_model_path

        # Which labels are worth reporting at all. `None` reads the configured
        # policy; an empty list means "every label", so the unset default keeps
        # today's behaviour exactly. Kept out of `DetectorConfig` on purpose:
        # that dataclass is the set of Ultralytics kwargs, and this is not one.
        self.in_scope_classes = (
            common.YOLO_IN_SCOPE_CLASSES_DEFAULT
            if in_scope_classes is None
            else common.normalize_in_scope_classes(in_scope_classes)
        )
        self.excluded_classes = (
            common.YOLO_EXCLUDED_CLASSES_DEFAULT
            if excluded_classes is None
            else common.normalize_in_scope_classes(excluded_classes)
        )
        # What the filter hid, per label, since start. A silent filter is
        # indistinguishable from a broken detector, so the counts are kept.
        self.suppressed_labels: dict[str, int] = {}

        # Inference parameters are resolved once, here, so the live loop and the
        # offline benchmark cannot drift apart, and so `/monitor/status` can report
        # exactly what inference is using.
        self.config = DetectorConfig.configured().updated(
            conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, agnostic_nms=agnostic_nms
        )

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
        agnostic_nms: bool | None = None,
    ) -> DetectorConfig:
        """Apply inference parameters (clamped to usable ranges) and return them.

        One clamping path for the constructor, runtime tuning (see
        `set_config`) and the offline benchmark, so a value cannot be valid in one
        place and invalid in another. ``None`` leaves a parameter at its current
        value.
        """
        self.config = self.config.updated(
            conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, agnostic_nms=agnostic_nms
        )
        return self.config

    def set_config(self, config: DetectorConfig) -> DetectorConfig:
        """Replace the whole configuration in one step."""
        self.config = config
        return self.config

    def set_in_scope_classes(self, value: Any) -> frozenset[str]:
        """Replace the in-scope label allowlist.

        ``None``, ``""`` or an empty iterable all mean "every label is in scope".
        Suppression counts are left alone: they are a record of what has been
        hidden so far, not a property of the current policy.
        """
        self.in_scope_classes = common.normalize_in_scope_classes(value)
        return self.in_scope_classes

    def set_excluded_classes(self, value: Any) -> frozenset[str]:
        """Replace the excluded-label denylist. ``None`` or empty excludes nothing."""
        self.excluded_classes = common.normalize_in_scope_classes(value)
        return self.excluded_classes

    def _keep_in_scope(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop labels the policy does not want, counting what was dropped.

        Exclusion wins over inclusion, so a label named in both lists is dropped
        rather than kept — the conservative reading of a contradictory config.
        """
        kept: list[dict[str, Any]] = []
        for row in rows:
            label = str(row.get("label", "")).strip().lower()
            in_scope = not self.in_scope_classes or label in self.in_scope_classes
            if in_scope and label not in self.excluded_classes:
                kept.append(row)
            else:
                self.suppressed_labels[label] = self.suppressed_labels.get(label, 0) + 1
        return kept

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

        if self.in_scope_classes or self.excluded_classes:
            rows = self._keep_in_scope(rows)

        rows.sort(key=lambda row: row["confidence"], reverse=True)
        return rows

    def set_general_enabled(self, enabled: bool) -> bool:
        """Ask for a state and return the state actually in effect.

        A model that was never loaded cannot be enabled, so this reports the refusal
        instead of pretending. `toggle_general` cannot express that: with no model it
        always returns ``False``, so a caller looping until the state matched the
        request would never finish. That is not hypothetical — the dashboard's
        "enable custom YOLO" request ran exactly such a loop against a detector with
        no checkpoint configured, which span forever inside the frame loop and left
        the user looking at a stalled stream.
        """
        if self.general_model is None:
            self.general_enabled = False
            return self.general_enabled
        self.general_enabled = bool(enabled)
        return self.general_enabled

    def set_custom_enabled(self, enabled: bool) -> bool:
        """Ask for a state and return the state actually in effect.

        See `set_general_enabled` for why this exists next to `toggle_custom`.
        """
        if self.custom_model is None:
            self.custom_enabled = False
            return self.custom_enabled
        self.custom_enabled = bool(enabled)
        return self.custom_enabled

    def toggle_general(self) -> bool:
        return self.set_general_enabled(not self.general_enabled)

    def toggle_custom(self) -> bool:
        return self.set_custom_enabled(not self.custom_enabled)

    def params(self) -> dict[str, Any]:
        """The inference kwargs handed to Ultralytics on every call."""
        return self.config.as_kwargs()

    def get_state(self) -> dict[str, Any]:
        return {
            "params": self.params(),
            "in_scope": {
                "classes": sorted(self.in_scope_classes),
                "excluded": sorted(self.excluded_classes),
                "filtering": bool(self.in_scope_classes or self.excluded_classes),
                "suppressed": dict(sorted(self.suppressed_labels.items())),
            },
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
