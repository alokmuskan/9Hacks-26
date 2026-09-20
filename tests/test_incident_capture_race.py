import importlib
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import numpy as np

from _stubs import CV2_CONSTANTS, CV2_FUNCTIONS
from _stubs import install as _install_stubs

# NOTE: stub installation and the `main` import happen in setUp, not at module
# import time. Discovery imports every test module before running any test, so a
# module-level `_install_stubs()` would claim `sys.modules["cv2"]` first, outside
# any test. See `_stubs` for the shared implementation.


class SharedStubGuardTests(unittest.TestCase):
    """The shared stub is a superset of what the pipeline calls.

    `cv2` is cached in `sys.modules` on first insert, so whichever module installs
    it first is what every later test sees. The stub must therefore be complete,
    not merely complete-enough for the module that happened to run first.
    """

    def test_shared_cv2_stub_provides_what_the_pipeline_calls(self):
        _install_stubs()
        cv2_stub = sys.modules["cv2"]
        for name in CV2_FUNCTIONS:
            self.assertTrue(
                hasattr(cv2_stub, name),
                f"the cv2 stub is missing {name}, which the pipeline calls",
            )
        for name in CV2_CONSTANTS:
            self.assertTrue(
                hasattr(cv2_stub, name),
                f"the cv2 stub is missing the constant {name}",
            )


class IncidentCaptureRaceTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.core = importlib.import_module("main")
        self.frame = np.zeros((16, 16, 3), dtype=np.uint8)
        self.bboxes = [np.array([1, 1, 8, 8], dtype=np.float32)]

        # Patch main's cv2 locally so this file's assertions do not depend on which
        # stub discovery happened to install first.
        fake_cv2 = mock.MagicMock()

        def imwrite(path, _image, *_args, **_kwargs):
            Path(path).write_bytes(b"jpeg")
            return True

        fake_cv2.imwrite.side_effect = imwrite
        patcher = mock.patch.object(self.core, "cv2", fake_cv2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _capture(self, count, timestamp, barrier=None):
        paths: list[str] = []
        errors: list[str] = []

        def worker():
            try:
                if barrier is not None:
                    barrier.wait(timeout=10)
                paths.append(self.core._save_unknown_snapshot(self.frame, self.bboxes, timestamp))
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        return paths, errors

    def test_concurrent_captures_produce_distinct_files(self):
        ts = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            barrier = threading.Barrier(6, timeout=10)
            with mock.patch.object(self.core, "UNKNOWN_INCIDENTS_DIR", directory):
                paths, errors = self._capture(6, ts, barrier=barrier)

            self.assertEqual(errors, [])
            self.assertEqual(len(paths), 6)
            self.assertEqual(len(set(paths)), 6, "two writers chose the same filename")
            self.assertEqual(len(sorted(directory.glob("*.jpg"))), 6, "a capture overwrote another")

    def test_successive_captures_in_the_same_second_do_not_collide(self):
        ts = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
        # Filenames use local time, so derive the expected prefix the same way.
        prefix = ts.astimezone().strftime("unknown_%Y-%m-%d_%H-%M-%S")

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            with mock.patch.object(self.core, "UNKNOWN_INCIDENTS_DIR", directory):
                for _ in range(4):
                    self.core._save_unknown_snapshot(self.frame, self.bboxes, ts)

            names = sorted(path.name for path in directory.glob("*.jpg"))
            self.assertEqual(len(names), 4)
            self.assertEqual(names[0], f"{prefix}.jpg")
            self.assertIn(f"{prefix}_03.jpg", names)

    def test_the_scenario_detects_the_old_check_then_write_pattern(self):
        """Control case: the previous exists()-then-write logic really does collide."""
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            barrier = threading.Barrier(6, timeout=10)
            created: list[str] = []
            lock = threading.Lock()

            def old_logic():
                base = "unknown_2026-09-19_12-00-00"
                with lock:
                    candidate = directory / f"{base}.jpg"
                    suffix = 1
                    while candidate.exists():
                        candidate = directory / f"{base}_{suffix:02d}.jpg"
                        suffix += 1
                # The old code chose the name at the top of the function and only
                # wrote the image at the end (after copying the frame and drawing
                # boxes). That wide window is what made the race likely, so model it
                # explicitly rather than relying on scheduler timing.
                time.sleep(0.01)
                try:
                    descriptor = os.open(candidate, os.O_CREAT | os.O_WRONLY)
                    os.close(descriptor)
                except OSError:
                    return
                with lock:
                    created.append(str(candidate))

            threads = [
                threading.Thread(target=lambda: (barrier.wait(timeout=10), old_logic()))
                for _ in range(6)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)

            self.assertLess(
                len(set(created)),
                len(created),
                "the old pattern unexpectedly avoided collisions: the race test is too weak",
            )

    def test_capture_returns_an_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            with mock.patch.object(self.core, "UNKNOWN_INCIDENTS_DIR", directory):
                path = self.core._save_unknown_snapshot(
                    self.frame, self.bboxes, datetime.now(UTC)
                )

            self.assertTrue(Path(path).exists())
            self.assertEqual(Path(path).parent, directory)


class IncidentCaptureWriteFailureTests(unittest.TestCase):
    """A capture that cannot be written must not leave an empty file behind.

    A reserved filename with no image in it reads as evidence while containing
    nothing — the detection benchmark found 28 such zero-byte files on this
    machine, so the capture path now verifies the write instead of ignoring
    `cv2.imwrite`'s return value.
    """

    def setUp(self):
        _install_stubs()
        self.core = importlib.import_module("main")
        self.frame = np.zeros((16, 16, 3), dtype=np.uint8)
        self.bboxes = [np.array([1, 1, 8, 8], dtype=np.float32)]
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        patcher = mock.patch.object(self.core, "UNKNOWN_INCIDENTS_DIR", self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch_imwrite(self, behaviour):
        fake_cv2 = mock.MagicMock()
        fake_cv2.imwrite.side_effect = behaviour
        patcher = mock.patch.object(self.core, "cv2", fake_cv2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failed_write_removes_the_reserved_file_and_returns_none(self):
        self._patch_imwrite(lambda *_args, **_kwargs: False)

        with mock.patch("builtins.print") as printed:
            result = self.core._save_unknown_snapshot(self.frame, self.bboxes, datetime.now(UTC))

        self.assertIsNone(result)
        self.assertEqual(list(self.directory.glob("*.jpg")), [], "an empty file was left behind")
        self.assertTrue(
            any("could not be written" in str(call) for call in printed.call_args_list),
            "the failure was silent",
        )

    def test_imwrite_claiming_success_but_writing_nothing_is_a_failure(self):
        self._patch_imwrite(lambda *_args, **_kwargs: True)

        result = self.core._save_unknown_snapshot(self.frame, self.bboxes, datetime.now(UTC))

        self.assertIsNone(result)
        self.assertEqual(list(self.directory.glob("*.jpg")), [])

    def test_successful_write_returns_a_non_empty_path(self):
        def imwrite(path, _image, *_args, **_kwargs):
            Path(path).write_bytes(b"jpeg-bytes")
            return True

        self._patch_imwrite(imwrite)

        result = self.core._save_unknown_snapshot(self.frame, self.bboxes, datetime.now(UTC))

        self.assertIsNotNone(result)
        written = Path(str(result))
        self.assertEqual(written.parent, self.directory)
        self.assertGreater(written.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
