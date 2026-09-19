import importlib
import json
import re
import sys
import tempfile
import threading
import types
import unittest
from datetime import UTC
from pathlib import Path
from unittest import mock

import numpy as np

import common


def _make_cv2_stub():
    stub = types.ModuleType("cv2")
    stub.CAP_V4L2 = 200
    stub.CAP_ANY = 0
    stub.CAP_FFMPEG = 1900
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
    stub.resize = lambda frame, *_args, **_kwargs: frame
    stub.rectangle = lambda *_args, **_kwargs: None
    stub.addWeighted = lambda *_args, **_kwargs: None
    stub.putText = lambda *_args, **_kwargs: None
    stub.polylines = lambda *_args, **_kwargs: None
    stub.line = lambda *_args, **_kwargs: None
    stub.circle = lambda *_args, **_kwargs: None
    stub.getTextSize = lambda text, *_args, **_kwargs: ((len(text) * 8, 12), 2)
    stub.imencode = lambda _ext, _frame, _params=None: (True, np.zeros(4, dtype=np.uint8))
    stub.imshow = lambda *_args, **_kwargs: None
    stub.waitKey = lambda *_args, **_kwargs: -1
    stub.namedWindow = lambda *_args, **_kwargs: None
    stub.destroyAllWindows = lambda: None
    stub.setLogLevel = lambda *_args, **_kwargs: None
    stub.VideoCapture = lambda *_args, **_kwargs: None
    return stub


def _install_stubs():
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = _make_cv2_stub()

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


class SharedHelperTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")
        self.server = importlib.import_module("server")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def test_session_id_keeps_its_prefix_in_both_entry_points(self):
        pattern = re.compile(r"^[a-z]+-\d{8}-\d{6}$")
        for module, prefix in ((self.main, "recognize"), (self.server, "monitor")):
            value = module._session_id(prefix)
            self.assertRegex(value, pattern, "session ids must be prefixed and consistent")
            self.assertTrue(value.startswith(f"{prefix}-"))

    def test_both_entry_points_read_the_same_metrics_log(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.jsonl"
            with mock.patch.object(self.main, "METRICS_LOG_PATH", path), mock.patch.object(
                self.server, "METRICS_PATH", path
            ):
                self.main._append_metric("chat_query", {"question": "hello", "hit": True})

                rows = self.server._load_metric_events()
                self.assertEqual(len(rows), 1)

                filtered = self.server._filter_metric_events(
                    event_type="chat_query", limit=10, from_ts=None, to_ts=None
                )
                self.assertEqual(len(filtered), 1)
                self.assertEqual(filtered[0]["question"], "hello")

                self.assertEqual(
                    self.server._filter_metric_events(
                        event_type="enroll", limit=10, from_ts=None, to_ts=None
                    ),
                    [],
                )

    def test_metrics_lock_is_shared_between_entry_points(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.jsonl"
            rows_per_thread = 15

            def write_via_main():
                for index in range(rows_per_thread):
                    self.main._append_metric("chat_query", {"writer": "main", "index": index, "pad": "x" * 400})

            def write_via_common():
                for index in range(rows_per_thread):
                    common.append_jsonl(
                        path,
                        {
                            "timestamp_utc": common.iso(),
                            "event_type": "chat_query",
                            "writer": "common",
                            "index": index,
                            "pad": "x" * 400,
                        },
                    )

            with mock.patch.object(self.main, "METRICS_LOG_PATH", path):
                threads = [
                    threading.Thread(target=write_via_main if index % 2 == 0 else write_via_common)
                    for index in range(4)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(lines), 4 * rows_per_thread)
                for line in lines:
                    json.loads(line)  # raises if two writers interleaved

    def test_parse_error_counter_is_shared_across_readers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metrics.jsonl"
            good = json.dumps({"timestamp_utc": common.iso(), "event_type": "ok"})
            path.write_text(f"{good}\n" + '{"broken": \n', encoding="utf-8")

            before = common.metrics_parse_errors()
            with mock.patch.object(self.main, "METRICS_LOG_PATH", path), mock.patch.object(
                self.server, "METRICS_PATH", path
            ):
                self.assertEqual(len(self.main._load_metric_events()), 1)
                self.assertEqual(len(self.server._load_metric_events()), 1)
                # One counter for the process, not one per module.
                self.assertEqual(self.main._metrics_parse_errors(), common.metrics_parse_errors())

            self.assertEqual(common.metrics_parse_errors(), before + 2)

    def test_filter_events_applies_window_and_limit(self):
        rows = [
            {"event_type": "chat_query", "timestamp_utc": "2026-03-14T10:00:00+00:00"},
            {"event_type": "chat_query", "timestamp_utc": "2026-03-14T11:00:00+00:00"},
            {"event_type": "enroll", "timestamp_utc": "2026-03-14T11:30:00+00:00"},
        ]

        windowed = common.filter_events(
            rows,
            event_type=None,
            limit=10,
            from_ts="2026-03-14T10:30:00+00:00",
            to_ts="2026-03-14T11:15:00+00:00",
        )
        self.assertEqual([row["timestamp_utc"] for row in windowed], ["2026-03-14T11:00:00+00:00"])

        self.assertEqual(len(common.filter_events(rows, event_type="chat_query", limit=1, from_ts=None, to_ts=None)), 1)

    def test_value_helpers_handle_bad_input(self):
        self.assertEqual(common.safe_int("7"), 7)
        self.assertEqual(common.safe_int(None, 3), 3)
        self.assertEqual(common.safe_float("2.5"), 2.5)
        self.assertEqual(common.safe_float("nope", 1.5), 1.5)

        aware = common.parse_iso("2026-03-14T10:00:00+00:00")
        self.assertIsNotNone(aware)
        self.assertEqual(aware.tzinfo, UTC)

        naive = common.parse_iso("2026-03-14T10:00:00")
        self.assertIsNotNone(naive)
        self.assertEqual(naive.tzinfo, UTC, "naive timestamps are treated as UTC")

        self.assertIsNone(common.parse_iso(""))
        self.assertIsNone(common.parse_iso("not-a-date"))
        self.assertEqual(common.parse_iso(common.iso(aware)), aware)


if __name__ == "__main__":
    unittest.main()
