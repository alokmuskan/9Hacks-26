import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from _stubs import install as _install_stubs


class MetricsRotationTests(unittest.TestCase):
    def setUp(self):
        self.common = importlib.import_module("common")

    def test_log_rotates_once_it_exceeds_the_cap(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics_log.jsonl"
            with (
                mock.patch.object(self.common, "METRICS_MAX_BYTES", 400),
                mock.patch.object(self.common, "METRICS_BACKUP_COUNT", 2),
            ):
                for index in range(80):
                    self.common.append_jsonl(log, {"n": index, "pad": "x" * 20})

                active_rows = self.common.read_jsonl(log, include_backups=False)

            self.assertTrue(log.exists())
            self.assertTrue(Path(f"{log}.1").exists(), "no rotation happened")
            # Rotation runs before each append, so the active file may overshoot the
            # cap by exactly one row -- it is bounded, not strictly under the limit.
            self.assertLess(log.stat().st_size, 400 + 200)
            self.assertGreater(len(active_rows), 0)
            self.assertLess(len(active_rows), 20, "the active log is not bounded")
            self.assertEqual(len(self.common.metric_generations(log)), 3)

    def test_rotated_generations_stay_readable_and_ordered(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics_log.jsonl"
            with (
                mock.patch.object(self.common, "METRICS_MAX_BYTES", 300),
                mock.patch.object(self.common, "METRICS_BACKUP_COUNT", 5),
            ):
                for index in range(60):
                    self.common.append_jsonl(log, {"n": index})

                rows = self.common.read_jsonl(log)

            numbers = [row["n"] for row in rows]
            self.assertEqual(numbers, sorted(numbers), "history came back out of order")
            # Rotation drops the oldest generation once the backup count is reached,
            # so the tail is always intact even when the head is gone.
            self.assertEqual(numbers[-1], 59)
            self.assertTrue(numbers)

    def test_backup_count_zero_keeps_only_the_active_file(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics_log.jsonl"
            with (
                mock.patch.object(self.common, "METRICS_MAX_BYTES", 200),
                mock.patch.object(self.common, "METRICS_BACKUP_COUNT", 0),
            ):
                for index in range(40):
                    self.common.append_jsonl(log, {"n": index})

                self.assertEqual(self.common.metric_generations(log), [log])
                rows = self.common.read_jsonl(log)

            self.assertEqual([row["n"] for row in rows], [39])

    def test_read_can_skip_rotated_generations(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics_log.jsonl"
            with (
                mock.patch.object(self.common, "METRICS_MAX_BYTES", 200),
                mock.patch.object(self.common, "METRICS_BACKUP_COUNT", 3),
            ):
                for index in range(40):
                    self.common.append_jsonl(log, {"n": index})
                everything = self.common.read_jsonl(log)
                active_only = self.common.read_jsonl(log, include_backups=False)

            self.assertGreater(len(everything), len(active_only))

    def test_a_failing_rotation_never_loses_an_append(self):
        """A reader holding the file open (Windows) must not break writing."""
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics_log.jsonl"
            with (
                mock.patch.object(self.common, "METRICS_MAX_BYTES", 100),
                mock.patch.object(self.common, "METRICS_BACKUP_COUNT", 2),
                mock.patch.object(self.common.os, "replace", side_effect=OSError("in use")),
            ):
                # Enough rows to be far past the cap; every rotation attempt fails.
                for index in range(20):
                    self.common.append_jsonl(log, {"n": index})

            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 20, "an append was lost when rotation failed")


class SnapshotRetentionTests(unittest.TestCase):
    def _manager(self, base_dir, cap):
        from scene_memory import SceneMemoryManager

        return SceneMemoryManager(
            base_dir=base_dir,
            snapshot_interval_sec=1.0,
            enable_vectors=False,
            max_auto_snapshots=cap,
        )

    def _frame(self):
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def test_auto_snapshots_are_pruned_to_the_cap(self):
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(td, 3)
            for index in range(6):
                manager.save_snapshot(self._frame(), [], current_time=100.0 + index)

            rows = json.loads((Path(td) / "metadata.json").read_text(encoding="utf-8"))
            images = sorted((Path(td) / "snapshots").glob("*.jpg"))

            self.assertEqual(len(rows), 3, "the cap was not enforced on the index")
            self.assertEqual(len(images), 3, "pruned images were left on disk")
            self.assertEqual(manager.pruned_snapshots, 3)
            for row in rows:
                self.assertTrue(Path(row["snapshot_path"]).exists())

    def test_manual_snapshots_are_never_pruned(self):
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(td, 1)
            for index in range(4):
                manager.save_snapshot(
                    self._frame(), [], current_time=200.0 + index, manual=True
                )
            manager.save_snapshot(self._frame(), [], current_time=300.0)

            rows = json.loads((Path(td) / "metadata.json").read_text(encoding="utf-8"))
            manual = [row for row in rows if row.get("manual")]

            self.assertEqual(len(manual), 4, "a manual snapshot was pruned")
            self.assertEqual(len(rows), 5)
            self.assertEqual(manager.pruned_snapshots, 0)

    def test_retention_is_enforced_across_instances(self):
        """A long-lived worker and per-request managers must share one cap."""
        with tempfile.TemporaryDirectory() as td:
            writer = self._manager(td, 2)
            for index in range(3):
                self._manager(td, 2).save_snapshot(
                    self._frame(), [], current_time=400.0 + index
                )

            rows = json.loads((Path(td) / "metadata.json").read_text(encoding="utf-8"))
            images = sorted((Path(td) / "snapshots").glob("*.jpg"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(len(images), 2)
            self.assertTrue(writer.max_auto_snapshots == 2)

    def test_zero_cap_disables_pruning(self):
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(td, 0)
            for index in range(8):
                manager.save_snapshot(self._frame(), [], current_time=500.0 + index)

            rows = json.loads((Path(td) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 8)
            self.assertEqual(manager.pruned_snapshots, 0)

    def test_default_cap_comes_from_shared_config(self):
        """The pipelines construct managers without the parameter, so it must default."""
        from scene_memory import SceneMemoryManager

        common = importlib.import_module("common")
        with tempfile.TemporaryDirectory() as td:
            manager = SceneMemoryManager(base_dir=td, enable_vectors=False)

        self.assertEqual(manager.max_auto_snapshots, common.MEMORY_MAX_AUTO_SNAPSHOTS)
        self.assertGreater(manager.max_auto_snapshots, 0)

    def test_stats_report_the_retention_state(self):
        with tempfile.TemporaryDirectory() as td:
            manager = self._manager(td, 2)
            for index in range(5):
                manager.save_snapshot(self._frame(), [], current_time=600.0 + index)

            stats = manager.get_memory_stats()
            self.assertEqual(stats["max_auto_snapshots"], 2)
            self.assertEqual(stats["total_snapshots"], 2)
            # Deliberately not exposed: it is per-instance, so a per-request manager
            # would always report 0 and misrepresent the store.
            self.assertNotIn("pruned_snapshots", stats)


class IncidentRetentionTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def _populate(self, directory, count):
        for index in range(count):
            path = directory / f"unknown_2026-01-01_00-00-{index:02d}.jpg"
            path.write_bytes(b"jpeg")
            # Distinct, ascending modification times so ordering is unambiguous.
            stamp = 1_700_000_000 + index
            os.utime(path, (stamp, stamp))
        return sorted(directory.glob("*.jpg"))

    def test_captures_are_pruned_to_the_cap_keeping_the_newest(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._populate(directory, 10)

            with mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", directory):
                removed = self.main._prune_unknown_incidents(keep=4)

            remaining = sorted(directory.glob("*.jpg"))
            self.assertEqual(removed, 6)
            self.assertEqual(len(remaining), 4)
            self.assertTrue(
                all("05" in path.name or "06" in path.name or "07" in path.name or "08" in path.name or "09" in path.name for path in remaining)
            )

    def test_prune_is_a_noop_below_the_cap(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._populate(directory, 3)

            with mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", directory):
                removed = self.main._prune_unknown_incidents(keep=10)

            self.assertEqual(removed, 0)
            self.assertEqual(len(sorted(directory.glob("*.jpg"))), 3)

    def test_zero_cap_disables_pruning(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self._populate(directory, 5)

            with mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", directory):
                removed = self.main._prune_unknown_incidents(keep=0)

            self.assertEqual(removed, 0)
            self.assertEqual(len(sorted(directory.glob("*.jpg"))), 5)

    def test_missing_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", Path(td) / "nope"):
                self.assertEqual(self.main._prune_unknown_incidents(keep=1), 0)


if __name__ == "__main__":
    unittest.main()
