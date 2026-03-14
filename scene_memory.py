from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class SceneMemoryManager:
    def __init__(
        self,
        snapshot_interval_sec: float = 15.0,
        base_dir: str | Path = "memory",
        enable_vectors: bool = True,
    ) -> None:
        self.snapshot_interval_sec = float(snapshot_interval_sec)
        self.last_snapshot_time = 0.0

        self.base_dir = Path(base_dir)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.metadata_path = self.base_dir / "metadata.json"
        self.embedding_path = self.base_dir / "embeddings.faiss"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: list[dict[str, Any]] = self._load_metadata()

        self.enable_vectors = enable_vectors
        self.vectors_enabled = False
        self.vector_backend_error: str | None = None

        self._faiss = None
        self._faiss_index = None
        self._torch = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_device = None

        if enable_vectors:
            self._init_vector_backend()

    def _load_metadata(self) -> list[dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [entry for entry in raw if isinstance(entry, dict)]
        except Exception:
            pass
        return []

    def _save_metadata(self) -> None:
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def _init_vector_backend(self) -> None:
        try:
            import faiss
            import open_clip
            import torch

            self._faiss = faiss
            self._torch = torch

            clip_arch = "ViT-B-32-quickgelu"
            model, _, preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained="openai")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = model.to(device)
            self._clip_model.eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer(clip_arch)
            self._clip_device = device

            self._faiss_index = faiss.IndexFlatL2(512)
            if self.embedding_path.exists() and self.metadata:
                try:
                    self._faiss_index = faiss.read_index(str(self.embedding_path))
                except Exception:
                    self._faiss_index = faiss.IndexFlatL2(512)

            self.vectors_enabled = True
        except Exception as exc:
            self.vectors_enabled = False
            self.vector_backend_error = str(exc)
            self._faiss_index = None

    def should_take_snapshot(self, current_time: float) -> bool:
        return (float(current_time) - self.last_snapshot_time) >= self.snapshot_interval_sec

    def _frame_to_rgb_image(self, frame: np.ndarray) -> Image.Image:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3).")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # cv2 frames are BGR.
        rgb = frame[..., ::-1]
        return Image.fromarray(rgb)

    def _next_snapshot_path(self, timestamp_utc: datetime) -> Path:
        stamp = timestamp_utc.astimezone().strftime("snap_%Y-%m-%d_%H-%M-%S")
        candidate = self.snapshots_dir / f"{stamp}.jpg"
        suffix = 1
        while candidate.exists():
            candidate = self.snapshots_dir / f"{stamp}_{suffix:02d}.jpg"
            suffix += 1
        return candidate

    def _extract_labels(self, detections: list[Any]) -> list[str]:
        labels: list[str] = []
        for row in detections:
            label = None
            if isinstance(row, dict):
                label = row.get("label")
            elif isinstance(row, str):
                label = row
            if label:
                labels.append(str(label))

        return sorted(set(labels))

    def _embed_image(self, image: Image.Image) -> np.ndarray | None:
        if not self.vectors_enabled:
            return None

        assert self._torch is not None
        assert self._clip_model is not None
        assert self._clip_preprocess is not None
        assert self._clip_device is not None

        tensor = self._clip_preprocess(image).unsqueeze(0).to(self._clip_device)
        with self._torch.no_grad():
            emb = self._clip_model.encode_image(tensor).detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norm, 1e-8)
        return emb

    def save_snapshot(
        self,
        frame: np.ndarray,
        detections: list[Any],
        current_time: float | None = None,
        manual: bool = False,
    ) -> dict[str, Any]:
        ts_utc = datetime.now(timezone.utc)
        snapshot_path = self._next_snapshot_path(ts_utc)

        image = self._frame_to_rgb_image(frame)
        image.save(snapshot_path, format="JPEG", quality=95)

        labels = self._extract_labels(detections)

        if self.vectors_enabled and self._faiss_index is not None:
            emb = self._embed_image(image)
            if emb is not None:
                self._faiss_index.add(emb)

        entry = {
            "id": len(self.metadata) + 1,
            "timestamp_utc": ts_utc.isoformat(),
            "timestamp_local": ts_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "datetime": ts_utc.isoformat(),
            "snapshot": str(snapshot_path.relative_to(self.base_dir)),
            "snapshot_path": str(snapshot_path),
            "objects": labels,
            "manual": bool(manual),
            "object_count": len(labels),
        }

        self.metadata.append(entry)
        self._save_metadata()

        if self.vectors_enabled and self._faiss is not None and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.embedding_path))

        if current_time is not None:
            self.last_snapshot_time = float(current_time)
        else:
            self.last_snapshot_time = float(datetime.now().timestamp())

        return entry

    def get_memory_stats(self) -> dict[str, Any]:
        manual = sum(1 for row in self.metadata if row.get("manual"))
        auto = max(len(self.metadata) - manual, 0)
        return {
            "base_dir": str(self.base_dir),
            "total_snapshots": len(self.metadata),
            "manual_snapshots": manual,
            "auto_snapshots": auto,
            "last_snapshot": self.metadata[-1]["timestamp_local"] if self.metadata else None,
            "snapshot_interval_sec": self.snapshot_interval_sec,
            "vectors_enabled": self.vectors_enabled,
            "vector_backend_error": self.vector_backend_error,
        }

    def get_recent_snapshots(self, minutes: int = 5, limit: int = 20) -> list[dict[str, Any]]:
        if not self.metadata:
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=max(int(minutes), 0))

        rows: list[dict[str, Any]] = []
        for entry in reversed(self.metadata):
            dt = self._parse_dt(entry.get("datetime"))
            if dt is None:
                continue
            if dt >= cutoff:
                rows.append(entry)
                if len(rows) >= limit:
                    break
        rows.reverse()
        return rows

    def find_object_last_seen(self, object_name: str) -> dict[str, Any] | None:
        target = object_name.strip().lower()
        if not target:
            return None

        for entry in reversed(self.metadata):
            labels = [str(o).lower() for o in entry.get("objects", [])]
            if any(target in label for label in labels):
                return entry
        return None

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def search_similar_scene(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        text = text.strip()
        if not text:
            return []

        if self.vectors_enabled and self._faiss_index is not None and self.metadata:
            assert self._torch is not None
            assert self._clip_model is not None
            assert self._clip_tokenizer is not None
            assert self._clip_device is not None

            tokens = self._clip_tokenizer([text]).to(self._clip_device)
            with self._torch.no_grad():
                emb = self._clip_model.encode_text(tokens).detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norm, 1e-8)

            count = min(max(int(top_k), 1), len(self.metadata))
            distances, indices = self._faiss_index.search(emb, count)

            out: list[dict[str, Any]] = []
            for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                row = dict(self.metadata[idx])
                row["distance"] = float(dist)
                out.append(row)
            return out

        # Fallback lexical search when vectors are unavailable.
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for entry in self.metadata:
            labels = [str(o).lower() for o in entry.get("objects", [])]
            score = sum(1 for token in tokens if any(token in label for label in labels))
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda row: row[0], reverse=True)
        return [row[1] for row in scored[: max(int(top_k), 1)]]

    def save_all_memory(self) -> None:
        self._save_metadata()
        if self.vectors_enabled and self._faiss is not None and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.embedding_path))
