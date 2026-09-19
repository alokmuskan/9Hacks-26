import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import common
from scene_memory import _read_metadata_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_children(script: Path, args_per_child: list[list[str]], timeout: float = 120.0) -> None:
    """Start every child at once and wait for all of them."""
    processes = [
        subprocess.Popen(
            [sys.executable, str(script), *args],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for args in args_per_child
    ]
    for process in processes:
        try:
            _out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            # from None: the timeout detail is noise next to the real message.
            raise AssertionError("a child process deadlocked") from None
        if process.returncode != 0:
            raise AssertionError(
                f"child failed with {process.returncode}: {err.decode('utf-8', 'replace')[:2000]}"
            )


METRICS_CHILD = '''
import pathlib
import sys

sys.path.insert(0, sys.argv[4])
import common

log = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
tag = sys.argv[3]
for index in range(count):
    common.append_jsonl(log, {"worker": tag, "n": index})
'''

METADATA_CHILD = '''
import pathlib
import sys
import time

sys.path.insert(0, sys.argv[3])
import numpy as np

from scene_memory import SceneMemoryManager

base = sys.argv[1]
count = int(sys.argv[2])
manager = SceneMemoryManager(
    base_dir=base, snapshot_interval_sec=1.0, enable_vectors=False, max_auto_snapshots=0
)
frame = np.zeros((16, 16, 3), dtype=np.uint8)
for index in range(count):
    manager.save_snapshot(frame, [], current_time=1000.0 + index)
    time.sleep(0.005)
'''

# Emulates the pre-fix behaviour: read the index, append, write it back, with no
# cross-process coordination at all.
UNSYNCHRONISED_CHILD = '''
import pathlib
import sys
import time

sys.path.insert(0, sys.argv[4])
from scene_memory import _read_metadata_file, _write_metadata_file
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
tag = sys.argv[3]
for index in range(count):
    try:
        rows = _read_metadata_file(path)
        stamp = datetime.now(timezone.utc).isoformat()
        rows.append({"timestamp_utc": stamp, "snapshot_path": f"{tag}-{index}.jpg", "manual": False})
        _write_metadata_file(path, rows)
    except OSError:
        # The unguarded path really can fail here (Windows sharing violations on
        # os.replace). Losing the write is the point; aborting the child is not.
        pass
    time.sleep(0.005)
'''


class LockPrimitiveTests(unittest.TestCase):
    def test_platform_backend_is_detected(self):
        self.assertIn(common.lock_backend_name(), {"posix", "windows", "none"})

    def test_lock_uses_a_sidecar_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            with common.process_lock(target):
                self.assertTrue(common.lock_path_for(target).exists())
                self.assertEqual(common.lock_path_for(target).name, "data.jsonl.lock")

    def test_lock_is_reentrant_without_deadlocking(self):
        """Two nested acquisitions in one thread must not block on themselves."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            started = time.monotonic()
            with common.process_lock(target, timeout=5.0):
                with common.process_lock(target, timeout=5.0):
                    pass
            self.assertLess(time.monotonic() - started, 2.0)

    def test_lock_is_released_when_the_body_raises(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            with self.assertRaises(ValueError):
                with common.process_lock(target):
                    raise ValueError("boom")

            started = time.monotonic()
            with common.process_lock(target, timeout=5.0):
                pass
            self.assertLess(time.monotonic() - started, 2.0, "the lock leaked")

    def test_unavailable_lock_degrades_instead_of_hanging(self):
        """A held lock must time out rather than block forever."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            released = threading.Event()
            errors: list[str] = []

            def hold() -> None:
                try:
                    with common.process_lock(target, timeout=5.0):
                        time.sleep(1.5)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(str(exc))
                finally:
                    released.set()

            holder = threading.Thread(target=hold, daemon=True)
            holder.start()
            time.sleep(0.3)

            started = time.monotonic()
            with common.process_lock(target, timeout=0.3, required=False):
                elapsed = time.monotonic() - started

            holder.join(timeout=5.0)
            self.assertEqual(errors, [])
            self.assertTrue(released.is_set())
            self.assertLess(elapsed, 2.0, "a best-effort lock blocked past its timeout")

    def test_missing_lock_module_degrades_without_failing(self):
        """A platform with neither fcntl nor msvcrt must still be usable."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            with mock.patch.object(common, "_lock_backend", return_value=("none", None)):
                self.assertEqual(common.lock_backend_name(), "none")
                with common.process_lock(target):
                    pass
                common.append_jsonl(target, {"n": 1})

            self.assertFalse(common.lock_path_for(target).exists())
            self.assertEqual([row["n"] for row in common.read_jsonl(target)], [1])

    def test_required_timeout_is_counted(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "data.jsonl"
            before = common.lock_timeouts()
            with subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time;sys.path.insert(0, sys.argv[2]);"
                        "import common;"
                        "ctx = common.process_lock(sys.argv[1], timeout=5.0);"
                        "ctx.__enter__();time.sleep(1.5)"
                    ),
                    str(target),
                    str(PROJECT_ROOT),
                ],
                cwd=str(PROJECT_ROOT),
            ) as holder:
                time.sleep(0.6)
                try:
                    with common.process_lock(target, timeout=0.3):
                        pass
                finally:
                    holder.terminate()
                    holder.wait(timeout=10)

            self.assertGreater(common.lock_timeouts(), before)


class CrossProcessMetricsTests(unittest.TestCase):
    def test_all_appends_from_separate_processes_survive(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            log = Path(td) / "metrics_log.jsonl"
            script = Path(td) / "metrics_child.py"
            script.write_text(METRICS_CHILD, encoding="utf-8")

            workers = 3
            per_worker = 40
            _run_children(
                script,
                [[str(log), str(per_worker), f"w{i}", str(PROJECT_ROOT)] for i in range(workers)],
            )

            numbers = [
                json.loads(line)["worker"]
                for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(numbers), workers * per_worker)
            for index in range(workers):
                self.assertEqual(numbers.count(f"w{index}"), per_worker)


class CrossProcessMetadataTests(unittest.TestCase):
    def _child_payload(self, td, count):
        script = Path(td) / "metadata_child.py"
        script.write_text(METADATA_CHILD, encoding="utf-8")
        return script

    def test_snapshots_from_separate_processes_all_merge(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            script = self._child_payload(td, 0)
            workers = 3
            per_worker = 6
            _run_children(
                script, [[td, str(per_worker), str(PROJECT_ROOT)] for _ in range(workers)]
            )

            rows = _read_metadata_file(Path(td) / "metadata.json")
            images = sorted((Path(td) / "snapshots").glob("*.jpg"))

            self.assertEqual(len(rows), workers * per_worker, "entries were lost across processes")
            self.assertEqual(len(images), workers * per_worker, "images overwrote each other")
            self.assertEqual(len({row["id"] for row in rows}), workers * per_worker)
            for row in rows:
                self.assertTrue(Path(row["snapshot_path"]).exists())

    def test_the_scenario_actually_detects_unsynchronised_writers(self):
        """Proves the previous test has teeth: without the lock, entries are lost."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            script = Path(td) / "raw_child.py"
            script.write_text(UNSYNCHRONISED_CHILD, encoding="utf-8")
            index = Path(td) / "metadata.json"
            workers = 4
            per_worker = 12
            _run_children(
                script, [[str(index), str(per_worker), f"w{i}", str(PROJECT_ROOT)] for i in range(workers)]
            )

            rows = _read_metadata_file(index)
            self.assertLess(
                len(rows),
                workers * per_worker,
                "the unsynchronised version unexpectedly survived: the concurrency test is too weak",
            )

    def test_manual_snapshots_from_separate_processes_are_not_pruned(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            script = Path(td) / "metadata_child.py"
            script.write_text(METADATA_CHILD, encoding="utf-8")
            _run_children(script, [[td, "3", str(PROJECT_ROOT)] for _ in range(2)])

            rows = _read_metadata_file(Path(td) / "metadata.json")
            self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
