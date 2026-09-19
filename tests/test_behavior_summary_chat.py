import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _stubs import install as _install_stubs


class BehaviorSummaryChatTests(unittest.TestCase):
    def test_infer_gaze_target_prefers_inside_then_nearest(self):
        _install_stubs()
        main = importlib.import_module("main")

        rows = [
            {"label": "laptop", "confidence": 0.7, "bbox": np.array([40, 40, 80, 80]), "source": "general"},
            {"label": "phone", "confidence": 0.9, "bbox": np.array([100, 100, 120, 120]), "source": "custom"},
        ]
        inside = main._infer_gaze_target((50, 50, 0.1, 0.2), rows)
        self.assertIsNotNone(inside)
        self.assertEqual(inside["label"], "laptop")
        self.assertEqual(inside["method"], "inside")

        nearest = main._infer_gaze_target(
            (126, 110, 0.0, 0.0),
            rows,
            hit_padding_px=0.0,
            max_dist_px=20,
        )
        self.assertIsNotNone(nearest)
        self.assertEqual(nearest["label"], "phone")
        self.assertEqual(nearest["method"], "nearest")

    def test_behavior_tracker_emits_transitions_and_durations(self):
        _install_stubs()
        main = importlib.import_module("main")
        tracker = main._BehaviorTracker(smoothing_window=3, switch_confirmation=2, lost_timeout_sec=1.0)

        emitted = []
        emitted.extend(tracker.update({"Your Name": "laptop"}, 0.0))
        emitted.extend(tracker.update({"Your Name": "laptop"}, 0.2))
        emitted.extend(tracker.update({"Your Name": "laptop"}, 0.5))
        emitted.extend(tracker.update({"Your Name": "phone"}, 1.0))
        emitted.extend(tracker.update({"Your Name": "phone"}, 1.3))
        emitted.extend(tracker.update({"Your Name": "phone"}, 1.6))
        emitted.extend(tracker.update({}, 2.7))

        kinds = [row["event"] for row in emitted]
        self.assertIn("start", kinds)
        self.assertIn("switch", kinds)
        self.assertIn("end", kinds)

        summary = tracker.summary()
        self.assertGreater(summary["interactions_total"], 0)
        self.assertGreater(summary["attention_total_sec"], 0.0)

    def test_situation_summary_and_chat_query(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            metrics_path = base / "metrics.jsonl"
            memory_dir = base / "memory"

            old_metrics = main.METRICS_LOG_PATH
            old_memory = main.MEMORY_DIR
            old_db = main.DB_PATH
            main.METRICS_LOG_PATH = metrics_path
            main.MEMORY_DIR = memory_dir
            main.DB_PATH = base / "face_db.npz"
            np.savez_compressed(main.DB_PATH, names=np.array(["Your Name"]), centroids=np.zeros((1, 4), np.float32), counts=np.array([1], np.int32))

            try:
                now = main._now_utc().isoformat()
                rows = [
                    {
                        "timestamp_utc": now,
                        "event_type": "behavior_event",
                        "event": "end",
                        "person": "Your Name",
                        "target_object": "laptop",
                        "duration_sec": 120.0,
                    },
                    {
                        "timestamp_utc": now,
                        "event_type": "behavior_event",
                        "event": "end",
                        "person": "Arjun",
                        "target_object": "phone",
                        "duration_sec": 20.0,
                    },
                ]
                with metrics_path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

                memory = main.SceneMemoryManager(base_dir=memory_dir, enable_vectors=False)
                frame = np.zeros((24, 24, 3), dtype=np.uint8)
                memory.save_snapshot(frame, [{"label": "laptop"}], current_time=1.0, manual=False)
                memory.save_snapshot(frame, [{"label": "phone"}], current_time=2.0, manual=True)

                summary = main._build_situation_summary(minutes=5)
                text = main._render_situation_summary(summary)
                self.assertIn("In the last 5 minutes", text)
                self.assertIn("snapshot", text.lower())
                self.assertIn("most viewed object", text.lower())

                result = main._handle_chat_query("What happened in the last 5 minutes?")
                self.assertEqual(result["intent"], "session_summary")
                self.assertIn("In the last 5 minutes", result["answer"])
                self.assertFalse(result["used_llm"])

                logs = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                event_types = [row.get("event_type") for row in logs]
                self.assertIn("chat_query", event_types)
                self.assertIn("summary_query", event_types)
            finally:
                main.METRICS_LOG_PATH = old_metrics
                main.MEMORY_DIR = old_memory
                main.DB_PATH = old_db

    def test_chat_snapshot_action_requires_runtime_or_returns_snapshot(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            old_metrics = main.METRICS_LOG_PATH
            old_memory = main.MEMORY_DIR
            old_db = main.DB_PATH
            main.METRICS_LOG_PATH = base / "metrics.jsonl"
            main.MEMORY_DIR = base / "memory"
            main.DB_PATH = base / "face_db.npz"
            np.savez_compressed(main.DB_PATH, names=np.array([]), centroids=np.empty((0, 4), np.float32), counts=np.empty((0,), np.int32))
            try:
                blocked = main._handle_chat_query("take snapshot")
                self.assertEqual(blocked["intent"], "snapshot")
                self.assertFalse(blocked["hit"])

                memory = main.SceneMemoryManager(base_dir=main.MEMORY_DIR, enable_vectors=False)
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                result = main._handle_chat_query(
                    "take snapshot",
                    runtime_context={
                        "memory": memory,
                        "frame": frame,
                        "object_rows": [{"label": "laptop", "confidence": 0.9, "bbox": [0, 0, 8, 8], "source": "general"}],
                        "face_rows": [{"name": "Your Name", "confidence": 0.7, "bbox": [1, 1, 10, 10], "gaze": None, "target_object": None}],
                        "people": ["Your Name"],
                        "attention_rows": [{"name": "Your Name", "target_object": "laptop", "method": "inside", "distance_px": 0.0}],
                    },
                )
                self.assertEqual(result["action"], "snapshot")
                self.assertIsInstance(result["snapshot"], dict)
            finally:
                main.METRICS_LOG_PATH = old_metrics
                main.MEMORY_DIR = old_memory
                main.DB_PATH = old_db


if __name__ == "__main__":
    unittest.main()
