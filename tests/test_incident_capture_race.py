import importlib
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import numpy as np

# NOTE: stub installation and the `main` import happen in setUp, not at module
# import time. Discovery imports every test module before running any test, so a
# module-level `_install_stubs()` would claim `sys.modules["cv2"]` first and the
# partial stub would then be used by every other test in the suite.


def _install_stubs():
    if "cv2" not in sys.modules:
        stub = types.ModuleType("cv2")
        stub.CAP_V4L2 = 200
        stub.CAP_ANY = 0
        stub.CAP_PROP_BUFFERSIZE = 38
        stub.CAP_PROP_FRAME_WIDTH = 3
        stub.CAP_PROP_FRAME_HEIGHT = 4
        stub.FONT_HERSHEY_SIMPLEX = 0
        stub.LINE_AA = 16
        stub.WINDOW_NORMAL = 0
        stub.INTER_LINEAR = 1
        stub.INTER_AREA = 3
        stub.COLOR_BGR2RGB = 4
        stub.IMWRITE_JPEG_QUALITY = 1
        stub.cvtColor = lambda frame, _mode: frame
        stub.resize = lambda frame, *_a, **_k: frame
        stub.rectangle = lambda *_a, **_k: None
        stub.addWeighted = lambda *_a, **_k: None
        stub.putText = lambda *_a, **_k: None
        stub.polylines = lambda *_a, **_k: None
        stub.line = lambda *_a, **_k: None
        stub.circle = lambda *_a, **_k: None
        stub.getTextSize = lambda text, *_a, **_k: ((len(text) * 8, 12), 2)
        # Any module's stub can end up being the one every other test sees, so this
        # must cover everything the pipeline calls. The real fix is one shared
        # helper module; this keeps the suite honest until then.
        stub.imwrite = lambda path, _image, *_a, **_k: True
        stub.imencode = lambda *_a, **_k: (True, np.zeros(4, dtype=np.uint8))
        stub.imshow = lambda *_a, **_k: None
        stub.waitKey = lambda *_a, **_k: -1
        stub.namedWindow = lambda *_a, **_k: None
        stub.destroyAllWindows = lambda: None
        stub.setLogLevel = lambda *_a, **_k: None
        stub.VideoCapture = lambda *_a, **_k: None
        stub.CAP_FFMPEG = 1900
        sys.modules["cv2"] = stub

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


class SharedStubGuardTests(unittest.TestCase):
    """Discovery order decides which cv2 stub wins; it must be complete."""

    def test_shared_cv2_stub_provides_what_the_pipeline_calls(self):
        _install_stubs()
        cv2_stub = sys.modules["cv2"]
        for name in (
            "putText",
            "rectangle",
            "polylines",
            "line",
            "circle",
            "getTextSize",
            "addWeighted",
            "resize",
            "cvtColor",
            "imwrite",
            "imencode",
        ):
            self.assertTrue(
                hasattr(cv2_stub, name),
                f"the first cv2 stub installed is missing {name}; a module-level "
                "_install_stubs() in another test file may have claimed sys.modules['cv2']",
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


if __name__ == "__main__":
    unittest.main()
