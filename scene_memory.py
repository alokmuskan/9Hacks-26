from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import common

# Guards metadata read/merge/write across every SceneMemoryManager instance in the
# process. The monitor worker and per-request API handlers all share one file.
# This is only the thread half of the guarantee: `_store_guard` below also takes
# the OS-level lock, so a separately-launched `main.py` cannot race `server.py`.
_STORE_LOCK = threading.RLock()


def _entry_key(entry: dict[str, Any]) -> str:
    for field in ("snapshot_path", "snapshot", "timestamp_utc", "datetime"):
        value = str(entry.get(field) or "").strip()
        if value:
            return value
    return ""


def _merge_entries(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union several in-memory views of the index into one canonical list.

    Entries are deduplicated by snapshot identity, ordered chronologically and
    re-indexed so `id` stays unique even when instances were loaded separately.
    Later groups win on conflicts, so pass the freshest view last.
    """
    merged: dict[str, dict[str, Any]] = {}
    for rows in groups:
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            key = _entry_key(entry)
            if not key:
                continue
            merged[key] = entry

    ordered = sorted(
        merged.values(),
        key=lambda row: str(row.get("timestamp_utc") or row.get("datetime") or ""),
    )
    for idx, row in enumerate(ordered, start=1):
        row["id"] = idx
    return ordered


def _read_metadata_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _write_metadata_file(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the index atomically so a crash mid-write cannot truncate it."""
    payload = json.dumps(rows, ensure_ascii=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


class SceneMemoryManager:
    def __init__(
        self,
        snapshot_interval_sec: float = 15.0,
        base_dir: str | Path = "memory",
        enable_vectors: bool = True,
        max_auto_snapshots: int | None = None,
    ) -> None:
        # Floor the interval so a zero/negative value cannot snapshot every frame.
        self.snapshot_interval_sec = max(float(snapshot_interval_sec), 1.0)
        self.last_snapshot_time = 0.0

        # Retention caps only automatic captures. Manual snapshots are user actions
        # and are never pruned: silently deleting them is the bug this module
        # already had once. 0 disables pruning entirely.
        if max_auto_snapshots is None:
            max_auto_snapshots = common.MEMORY_MAX_AUTO_SNAPSHOTS
        self.max_auto_snapshots = max(int(max_auto_snapshots), 0)
        self.pruned_snapshots = 0

        self.base_dir = Path(base_dir)
        self.snapshots_dir = self.base_dir / "snapshots"
        self.metadata_path = self.base_dir / "metadata.json"
        self.embedding_path = self.base_dir / "embeddings.faiss"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: list[dict[str, Any]] = self._load_metadata()

        self.enable_vectors = enable_vectors
        self.vectors_enabled = False
        self.vector_backend_error: str | None = None

        # Third-party handles for the optional vector backend. `Any` is honest here:
        # faiss/torch/open_clip objects have no stubs, and all of them stay None when
        # `enable_vectors` is false or the optional imports are missing.
        self._faiss: Any = None
        self._faiss_index: Any = None
        self._torch: Any = None
        self._clip_model: Any = None
        self._clip_preprocess: Any = None
        self._clip_tokenizer: Any = None
        self._clip_device: str | None = None

        if enable_vectors:
            self._init_vector_backend()

    @contextmanager
    def _store_guard(self) -> Iterator[None]:
        """Serialise a read-merge-write cycle against other processes and threads."""
        with _STORE_LOCK:
            with common.process_lock(self.metadata_path):
                yield

    def _load_metadata(self) -> list[dict[str, Any]]:
        # Guarded like the merge cycles: on Windows `os.replace` fails if another
        # process has the destination open, so a lock-free read here could make a
        # concurrent writer fail.
        with self._store_guard():
            return _read_metadata_file(self.metadata_path)

    def _save_metadata(self) -> None:
        """Merge the on-disk index with this instance's view, then replace atomically.

        Without the merge a stale in-memory list silently drops entries appended
        by another instance (for example a manual snapshot taken through the API
        while the monitor worker holds an older view).
        """
        with self._store_guard():
            disk_rows = _read_metadata_file(self.metadata_path)
            self.metadata = _merge_entries(disk_rows, self.metadata)
            _write_metadata_file(self.metadata_path, self.metadata)

    def _init_vector_backend(self) -> None:
        try:
            import faiss
            import open_clip
            import torch

            self._faiss = faiss
            self._torch = torch

            clip_arch = "ViT-B-32-quickgelu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_arch, pretrained="openai"
            )
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

    def _save_snapshot_image(self, image: Image.Image, timestamp_utc: datetime) -> Path:
        """Reserve a unique snapshot path and write the image, atomically.

        Path selection and file creation must happen together: `_next_snapshot_path`
        only checks `exists()`, so two concurrent writers would otherwise pick the
        same filename and one image would overwrite the other.
        """
        with self._store_guard():
            path = self._next_snapshot_path(timestamp_utc)
            image.save(path, format="JPEG", quality=95)
            return path

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

    def _normalize_bbox(self, value: Any) -> list[float]:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]

    def _normalize_object_rows(self, rows: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                class_id = int(row.get("class_id", -1))
            except Exception:
                class_id = -1
            try:
                confidence = float(row.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            out.append(
                {
                    "label": str(row.get("label", "object")),
                    "confidence": confidence,
                    "bbox": self._normalize_bbox(row.get("bbox", [0, 0, 0, 0])),
                    "source": str(row.get("source", "general")),
                    "class_id": class_id,
                }
            )
        return out

    def _normalize_face_rows(self, rows: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            out.append(
                {
                    "name": str(row.get("name", "Unknown")),
                    "confidence": confidence,
                    "bbox": self._normalize_bbox(row.get("bbox", [0, 0, 0, 0])),
                    "gaze": row.get("gaze") if isinstance(row.get("gaze"), dict) else None,
                    "target_object": row.get("target_object"),
                }
            )
        return out

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
        faces: list[dict[str, Any]] | None = None,
        object_detections: list[dict[str, Any]] | None = None,
        people: list[str] | None = None,
        attention: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ts_utc = datetime.now(UTC)
        image = self._frame_to_rgb_image(frame)
        snapshot_path = self._save_snapshot_image(image, ts_utc)

        labels = self._extract_labels(detections)
        object_rows = self._normalize_object_rows(object_detections or [])
        face_rows = self._normalize_face_rows(faces or [])

        if self.vectors_enabled and self._faiss_index is not None:
            emb = self._embed_image(image)
            if emb is not None:
                self._faiss_index.add(emb)

        entry = {
            "timestamp_utc": ts_utc.isoformat(),
            "timestamp_local": ts_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "datetime": ts_utc.isoformat(),
            "snapshot": str(snapshot_path.relative_to(self.base_dir)),
            "snapshot_path": str(snapshot_path),
            "objects": labels,
            "manual": bool(manual),
            "object_count": len(labels),
        }
        if object_rows:
            entry["object_detections"] = object_rows
        if face_rows:
            entry["faces"] = face_rows

        if people:
            entry["people"] = sorted({str(p) for p in people if str(p).strip()})
        elif face_rows:
            entry["people"] = sorted(
                {
                    str(row.get("name"))
                    for row in face_rows
                    if str(row.get("name", "")).strip() and str(row.get("name")) != "Unknown"
                }
            )

        if isinstance(attention, list) and attention:
            safe_attention: list[dict[str, Any]] = []
            for row in attention:
                if not isinstance(row, dict):
                    continue
                try:
                    distance_px = float(row.get("distance_px", 0.0))
                except Exception:
                    distance_px = 0.0
                safe_attention.append(
                    {
                        "name": str(row.get("name", "Unknown")),
                        "target_object": row.get("target_object"),
                        "method": row.get("method"),
                        "distance_px": distance_px,
                    }
                )
            if safe_attention:
                entry["attention"] = safe_attention

        # Merge under the store guard so neither another thread nor another process
        # can win the race between reading the index and replacing it.
        with self._store_guard():
            merged = _merge_entries(_read_metadata_file(self.metadata_path), self.metadata)
            entry["id"] = len(merged) + 1
            merged.append(entry)
            self.metadata = merged
            _write_metadata_file(self.metadata_path, self.metadata)

        if self.vectors_enabled and self._faiss is not None and self._faiss_index is not None:
            self._faiss.write_index(self._faiss_index, str(self.embedding_path))

        if current_time is not None:
            self.last_snapshot_time = float(current_time)
        else:
            self.last_snapshot_time = datetime.now(UTC).timestamp()

        if not manual:
            pruned = self._prune_auto_snapshots()
            if pruned:
                self.pruned_snapshots += pruned
                # Print rather than log: this module has no logger, and silently
                # deleting captures is exactly what a user needs to be told about.
                print(
                    f"Memory retention: pruned {pruned} auto snapshot(s); "
                    f"keeping the newest {self.max_auto_snapshots} "
                    f"(manual snapshots are kept)."
                )

        return entry

    def _prune_auto_snapshots(self) -> int:
        """Delete the oldest automatic snapshots once the cap is exceeded.

        The cap is enforced across every instance because the index on disk is the
        authority: a long-lived worker and per-request handlers would otherwise each
        believe the store was under the limit.
        """
        if self.max_auto_snapshots <= 0:
            return 0

        with self._store_guard():
            merged = _merge_entries(_read_metadata_file(self.metadata_path), self.metadata)
            auto_entries = [row for row in merged if not row.get("manual")]
            excess = len(auto_entries) - self.max_auto_snapshots
            if excess <= 0:
                self.metadata = merged
                return 0

            # `merged` is chronological, so the head of `auto_entries` is the oldest.
            doomed = auto_entries[:excess]
            doomed_keys = {_entry_key(row) for row in doomed if _entry_key(row)}

            for row in doomed:
                raw_path = str(row.get("snapshot_path") or "").strip()
                if not raw_path:
                    continue
                # A locked or already-deleted image must not stop the index update.
                with suppress(OSError):
                    Path(raw_path).unlink(missing_ok=True)

            kept = [row for row in merged if _entry_key(row) not in doomed_keys]
            self.metadata = kept
            _write_metadata_file(self.metadata_path, kept)
            return len(doomed)

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
            # Retention config, not a per-instance counter: a fresh manager reports 0
            # pruned events even when earlier instances pruned the store, so only
            # `total_snapshots` above describes the actual state.
            "max_auto_snapshots": self.max_auto_snapshots,
            "vectors_enabled": self.vectors_enabled,
            "vector_backend_error": self.vector_backend_error,
        }

    def get_recent_snapshots(self, minutes: int = 5, limit: int = 20) -> list[dict[str, Any]]:
        if not self.metadata:
            return []

        now = datetime.now(UTC)
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

    def find_person_last_seen(self, person_name: str) -> dict[str, Any] | None:
        target = person_name.strip().lower()
        if not target:
            return None

        for entry in reversed(self.metadata):
            people = [str(p).lower() for p in entry.get("people", [])]
            if any(target in person for person in people):
                return entry

            faces = entry.get("faces", [])
            if isinstance(faces, list):
                names = [str(row.get("name", "")).lower() for row in faces if isinstance(row, dict)]
                if any(target in name for name in names):
                    return entry
        return None

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
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
            # strict=False: faiss returns parallel arrays, but a mismatched pair should
            # not turn a scene search into an exception.
            for dist, idx in zip(distances[0].tolist(), indices[0].tolist(), strict=False):
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
