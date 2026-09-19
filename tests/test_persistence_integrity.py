import importlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import numpy as np

from scene_memory import SceneMemoryManager


def _install_stubs():
    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.CAP_V4L2 = 200
        cv2_stub.CAP_ANY = 0
        cv2_stub.CAP_PROP_BUFFERSIZE = 38
        cv2_stub.CAP_PROP_FRAME_WIDTH = 3
        cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
        cv2_stub.FONT_HERSHEY_SIMPLEX = 0
        cv2_stub.LINE_AA = 16
        cv2_stub.WINDOW_NORMAL = 0
        sys.modules["cv2"] = cv2_stub

    if "insightface" not in sys.modules:
        insightface_stub = types.ModuleType("insightface")
        app_stub = types.ModuleType("insightface.app")

        class _FaceAnalysis:
            def __init__(self, *args, **kwargs):
                pass

            def prepare(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                return []

        app_stub.FaceAnalysis = _FaceAnalysis
        insightface_stub.app = app_stub
        sys.modules["insightface"] = insightface_stub
        sys.modules["insightface.app"] = app_stub


class SceneMemoryWriteIntegrityTests(unittest.TestCase):
    """Regression coverage for the shared metadata index.

    The monitor worker keeps one long-lived SceneMemoryManager while every API
    request builds its own. Whichever instance wrote last used to replace the
    whole file from its own stale view, erasing the other instance's snapshots.
    """

    def test_entry_from_another_instance_survives_worker_write(self):
        with tempfile.TemporaryDirectory() as td:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            worker = SceneMemoryManager(base_dir=td, snapshot_interval_sec=15.0, enable_vectors=False)
            worker.save_snapshot(frame, [{"label": "laptop"}], current_time=1.0, manual=False)

            # A per-request handler instance, exactly as the API layer builds one.
            request_view = SceneMemoryManager(base_dir=td, enable_vectors=False)
            request_view.save_snapshot(frame, [{"label": "phone"}], current_time=2.0, manual=True)

            # The worker writes again from the view it loaded before that request.
            worker.save_snapshot(frame, [{"label": "bottle"}], current_time=3.0, manual=False)

            rows = json.loads(Path(td, "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3, "a concurrent writer's snapshot was lost")
            labels = {label for row in rows for label in row.get("objects", [])}
            self.assertEqual(labels, {"laptop", "phone", "bottle"})
            self.assertEqual([row["manual"] for row in rows], [False, True, False])

    def test_ids_stay_unique_and_chronological_across_instances(self):
        with tempfile.TemporaryDirectory() as td:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            first = SceneMemoryManager(base_dir=td, enable_vectors=False)
            second = SceneMemoryManager(base_dir=td, enable_vectors=False)

            first.save_snapshot(frame, [{"label": "one"}], current_time=1.0)
            second.save_snapshot(frame, [{"label": "two"}], current_time=2.0)
            first.save_snapshot(frame, [{"label": "three"}], current_time=3.0)
            second.save_snapshot(frame, [{"label": "four"}], current_time=4.0)

            rows = json.loads(Path(td, "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in rows], [1, 2, 3, 4])
            self.assertEqual([row["objects"][0] for row in rows], ["one", "two", "three", "four"])

    def test_long_lived_instance_picks_up_entries_written_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            worker = SceneMemoryManager(base_dir=td, snapshot_interval_sec=15.0, enable_vectors=False)
            worker.save_snapshot(frame, [{"label": "laptop"}], current_time=1.0)

            other = SceneMemoryManager(base_dir=td, enable_vectors=False)
            other.save_snapshot(frame, [{"label": "phone"}], current_time=2.0, manual=True)

            worker.save_snapshot(frame, [{"label": "bottle"}], current_time=3.0)

            stats = worker.get_memory_stats()
            self.assertEqual(stats["total_snapshots"], 3)
            self.assertEqual(stats["manual_snapshots"], 1)

    def test_distinct_snapshots_keep_distinct_files(self):
        with tempfile.TemporaryDirectory() as td:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            first = SceneMemoryManager(base_dir=td, enable_vectors=False)
            second = SceneMemoryManager(base_dir=td, enable_vectors=False)
            first.save_snapshot(frame, [{"label": "a"}], current_time=1.0)
            second.save_snapshot(frame, [{"label": "b"}], current_time=1.0)

            rows = json.loads(Path(td, "metadata.json").read_text(encoding="utf-8"))
            paths = [Path(row["snapshot_path"]) for row in rows]
            self.assertEqual(len({str(path) for path in paths}), 2)
            self.assertTrue(all(path.exists() for path in paths))

    def test_concurrent_writers_never_lose_or_collide_snapshots(self):
        """Snapshot path reservation and image write must be atomic.

        Without it, two writers pick the same filename, one image overwrites the
        other, and the merge deduplicates the two entries by path.
        """
        workers, per_worker = 6, 5
        with tempfile.TemporaryDirectory() as td:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            errors = []

            def burst(worker_id):
                try:
                    for index in range(per_worker):
                        manager = SceneMemoryManager(base_dir=td, enable_vectors=False)
                        manager.save_snapshot(
                            frame,
                            [{"label": f"obj{worker_id}"}],
                            current_time=float(index),
                            manual=True,
                        )
                except Exception as exc:  # surfaced through the assertion below
                    errors.append(repr(exc))

            threads = [threading.Thread(target=burst, args=(w,)) for w in range(workers)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            expected = workers * per_worker
            self.assertEqual(errors, [])
            rows = json.loads(Path(td, "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), expected, "snapshot entries were lost or merged away")
            self.assertEqual(len({row["snapshot_path"] for row in rows}), expected)
            self.assertEqual(len({row["id"] for row in rows}), expected)
            self.assertEqual(len(list(Path(td, "snapshots").glob("*.jpg"))), expected)


class MetricsLogWriteIntegrityTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def test_concurrent_appends_produce_one_valid_json_row_each(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.jsonl"
            original = self.main.METRICS_LOG_PATH
            self.main.METRICS_LOG_PATH = path
            try:
                def write_rows(worker_id):
                    for index in range(25):
                        self.main._append_metric(
                            "chat_query",
                            {"worker": worker_id, "index": index, "question": "x" * 400},
                        )

                threads = [threading.Thread(target=write_rows, args=(w,)) for w in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                lines = [
                    line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                self.assertEqual(len(lines), 200, "rows were lost or merged")

                seen = set()
                for line in lines:
                    row = json.loads(line)  # raises if two writers interleaved
                    self.assertEqual(row["event_type"], "chat_query")
                    seen.add((row["worker"], row["index"]))
                self.assertEqual(len(seen), 200)
            finally:
                self.main.METRICS_LOG_PATH = original

    def test_corrupt_rows_are_counted_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.jsonl"
            original = self.main.METRICS_LOG_PATH
            self.main.METRICS_LOG_PATH = path
            try:
                good = json.dumps({"timestamp_utc": self.main._iso(), "event_type": "ok"})
                path.write_text(f"{good}\n" '{"broken": \n' f"{good}\n", encoding="utf-8")

                before = self.main._metrics_parse_errors()
                rows = self.main._load_metric_events()
                self.assertEqual(len(rows), 2)
                self.assertEqual(self.main._metrics_parse_errors(), before + 1)
            finally:
                self.main.METRICS_LOG_PATH = original


if __name__ == "__main__":
    unittest.main()
