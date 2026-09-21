import importlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import common
from _stubs import install as _install_stubs


class _FakeReader:
    """Feeds a fixed number of frames, then asks the worker to stop."""

    def __init__(self, manager, frames=3):
        self._manager = manager
        self._remaining = frames

    def read(self, timeout_sec=1.0):
        if self._remaining <= 0:
            self._manager._stop_event.set()
            return False, None
        self._remaining -= 1
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        return None


class ChatTruthfulnessTests(unittest.TestCase):
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

    def test_attention_helper_reports_found_flag(self):
        with tempfile.TemporaryDirectory() as td:
            old_metrics = self.main.METRICS_LOG_PATH
            self.main.METRICS_LOG_PATH = Path(td) / "metrics.jsonl"
            try:
                reply, found = self.main._answer_attention_query(
                    "what is Alok looking at", ["Alok"]
                )
                self.assertFalse(found)
                self.assertIn("No recent attention events", reply)

                row = {
                    "timestamp_utc": self.main._iso(),
                    "event_type": "behavior_event",
                    "event": "start",
                    "person": "Alok",
                    "target_object": "laptop",
                }
                self.main.METRICS_LOG_PATH.write_text(json.dumps(row) + "\n", encoding="utf-8")

                reply, found = self.main._answer_attention_query(
                    "what is Alok looking at", ["Alok"]
                )
                self.assertTrue(found)
                self.assertIn("laptop", reply)
            finally:
                self.main.METRICS_LOG_PATH = old_metrics

    def test_attention_lookup_miss_is_not_reported_as_hit(self):
        with mock.patch.object(self.main, "_load_metric_events", return_value=[]):
            with TestClient(self.server.app) as client:
                row = client.post(
                    "/api/v1/chat/query",
                    json={"message": "what is Alok looking at"},
                ).json()

        self.assertEqual(row["intent"], "attention_lookup")
        self.assertFalse(row["hit"], "an unmatched attention query must not claim a hit")

    def test_attention_lookup_hit_reports_success(self):
        events = [
            {
                "timestamp_utc": self.main._iso(),
                "event_type": "behavior_event",
                "event": "start",
                "person": "Alok",
                "target_object": "laptop",
            }
        ]
        with mock.patch.object(self.main, "_load_metric_events", return_value=events):
            with TestClient(self.server.app) as client:
                row = client.post(
                    "/api/v1/chat/query",
                    json={"message": "what is Alok looking at"},
                ).json()

        self.assertEqual(row["intent"], "attention_lookup")
        self.assertTrue(row["hit"])
        self.assertIn("laptop", row["reply"])

    def test_greeting_is_not_marked_grounded(self):
        with (
            mock.patch.object(
                self.server,
                "_query_groq_grounded",
                side_effect=AssertionError("Groq should not be called"),
            ),
            mock.patch.object(
                self.server,
                "_build_chat_grounding",
                side_effect=AssertionError("no grounding for a greeting"),
            ),
        ):
            with TestClient(self.server.app) as client:
                row = client.post("/api/v1/chat/query", json={"message": "Hi"}).json()

        self.assertEqual(row["intent"], "greeting")
        self.assertEqual(row["citations"], [])
        self.assertFalse(row["grounded"], "a greeting is not grounded in evidence")

    def test_citation_backed_answer_is_grounded(self):
        fake_grounding = {
            "citations": [
                {
                    "id": "C1",
                    "source": "metrics",
                    "timestamp": "2026-03-14T00:00:00+00:00",
                    "detail": "known=2, unknown=0",
                }
            ]
        }
        with (
            mock.patch.object(self.server, "_build_chat_grounding", return_value=fake_grounding),
            mock.patch.object(
                self.server, "_query_groq_grounded", return_value=("Camera is offline [C1].", True)
            ),
        ):
            with TestClient(self.server.app) as client:
                row = client.post(
                    "/api/v1/chat/query",
                    json={"message": "what is the state of things right now"},
                ).json()

        self.assertTrue(row["grounded"])
        self.assertTrue(row["used_llm"])
        self.assertEqual(row["citations"][0]["id"], "C1")


class MonitorGuardTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.server = importlib.import_module("server")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def test_snapshot_interval_must_be_sane(self):
        with TestClient(self.server.app) as client:
            for bad in (0, -5, 0.01, 5000):
                row = client.post("/api/v1/monitor/start", json={"snapshot_interval": bad})
                self.assertEqual(
                    row.status_code, 422, f"snapshot_interval={bad} should be rejected"
                )

    def test_unsupported_gaze_arch_is_rejected(self):
        with TestClient(self.server.app) as client:
            row = client.post("/api/v1/monitor/start", json={"gaze_arch": "ResNet99"})
        self.assertEqual(row.status_code, 422)

    def test_requested_gaze_arch_is_honoured(self):
        def _fake_monitor_worker(_manager_self, **_kwargs):
            return None

        with mock.patch.object(
            self.server.PipelineManager, "_run_monitor_worker", _fake_monitor_worker
        ):
            with TestClient(self.server.app) as client:
                started = client.post("/api/v1/monitor/start", json={"gaze_arch": "ResNet18"})
                self.assertEqual(started.status_code, 200)
                config = client.get("/api/v1/monitor/status").json().get("config", {})

        self.assertEqual(
            config.get("gaze_arch"),
            "ResNet18",
            "the API used to silently rewrite gaze_arch to the default",
        )

    def test_manual_snapshot_increments_session_counter(self):
        self.server.MANAGER._manual_snapshots = 0

        class _FakeMemory:
            def __init__(self, **_kwargs):
                pass

            def save_snapshot(self, *_args, **_kwargs):
                return {"snapshot": "snap_test.jpg"}

        class _FakeCore:
            MEMORY_DIR = "memory"
            SceneMemoryManager = _FakeMemory

        context = {
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
            "object_rows": [],
            "face_rows": [],
            "people": [],
            "attention_rows": [],
        }

        with (
            mock.patch.object(self.server, "_core", return_value=_FakeCore()),
            mock.patch.object(
                self.server.MANAGER, "build_chat_runtime_context", return_value=context
            ),
        ):
            with TestClient(self.server.app) as client:
                row = client.post("/api/v1/monitor/snapshot").json()

        self.assertEqual(row["status"], "ok")
        self.assertEqual(self.server.MANAGER._manual_snapshots, 1)


class MonitorWorkerResilienceTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.server = importlib.import_module("server")
        self.main = importlib.import_module("main")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def _fake_core(self, appended, face_reason=None):
        main = self.main

        class _FakeDB:
            names: ClassVar[list[str]] = []

        class _FakeFaceDB:
            @staticmethod
            def load():
                return _FakeDB()

        class _FakeDetector:
            def __init__(self, **_kwargs):
                pass

            def detect(self, _frame):
                raise RuntimeError("simulated YOLO failure")

            def get_state(self):
                return {
                    "general": {"enabled": True, "loaded": True, "model_path": "general.pt"},
                    "custom": {"enabled": False, "loaded": False, "model_path": None},
                }

            def toggle_general(self):
                return True

            def toggle_custom(self):
                return False

        class _FakeMemory:
            def __init__(self, **_kwargs):
                pass

            def should_take_snapshot(self, _now):
                return False

            def save_all_memory(self):
                return None

            def get_memory_stats(self):
                return {"total_snapshots": 0}

        class _FakeCore:
            FaceDB = _FakeFaceDB
            DualYoloDetector = _FakeDetector
            SceneMemoryManager = _FakeMemory
            _BehaviorTracker = main._BehaviorTracker
            GazeScheduler = main.GazeScheduler
            build_session_aggregate = staticmethod(main.build_session_aggregate)
            MEMORY_DIR = "memory"
            CAMERA_SOURCE = "0"
            UNKNOWN_LABEL = "Unknown"
            EVENTS_TIMELINE_CAP = 500

            @staticmethod
            def _detect(_app, _frame):
                return []

            @staticmethod
            def _match(_emb, _db):
                return "Unknown", 0.0

            @staticmethod
            def _build_app(_model):
                return object()

            @staticmethod
            def _try_build_face_app(_model):
                return (object(), None) if face_reason is None else (None, face_reason)

            @staticmethod
            def _load_gaze_runtime(**_kwargs):
                return None

            @staticmethod
            def _resolve_general_model_path(_path):
                return "yolov8n.pt"

            @staticmethod
            def _load_default_custom_model_path():
                return None

            @staticmethod
            def _append_metric(event_type, payload):
                appended.append((event_type, payload))

            @staticmethod
            def _save_unknown_snapshot(_frame, _boxes, _ts):
                return "unknown.jpg"

            @staticmethod
            def _bbox_to_list(_bbox):
                return [0.0, 0.0, 0.0, 0.0]

            @staticmethod
            def _normalize_face_rows_for_json(rows):
                return list(rows)

            @staticmethod
            def _normalize_object_rows_for_json(rows):
                return list(rows)

            @staticmethod
            def _infer_gaze_target(*_args, **_kwargs):
                return None

            @staticmethod
            def _estimate_gaze_points(*_args, **_kwargs):
                return []

            @staticmethod
            def _bracket_box(*_args, **_kwargs):
                return None

            @staticmethod
            def _label_tag(*_args, **_kwargs):
                return None

            @staticmethod
            def _hud(*_args, **_kwargs):
                return None

            @staticmethod
            def _progress_bar(*_args, **_kwargs):
                return None

        return _FakeCore()

    def test_detector_failure_does_not_kill_the_session(self):
        """A raising object detector used to abort the whole API monitor worker."""
        manager = self.server.PipelineManager()
        appended = []
        fake_core = self._fake_core(appended)
        reader = _FakeReader(manager, frames=3)

        with (
            mock.patch.object(self.server, "_core", return_value=fake_core),
            mock.patch.object(
                self.server.PipelineManager,
                "_attempt_camera_recovery",
                lambda _self, **_kwargs: (object(), reader),
            ),
        ):
            manager._run_monitor_worker(
                model="buffalo_sc",
                general_model=None,
                custom_model=None,
                disable_general=False,
                disable_custom=True,
                snapshot_interval=15.0,
                disable_gaze=True,
                gaze_arch="ResNet50",
                gaze_weights="models/L2CSNet_gaze360.pkl",
                gaze_weights_source="https://example.com",
                disable_gaze_auto_download=True,
                fps_cap=20,
            )

        sessions = [
            payload for event_type, payload in appended if event_type == "recognize_session"
        ]
        self.assertEqual(len(sessions), 1, "the worker did not reach its session summary")

        payload = sessions[0]
        self.assertEqual(payload["schema_version"], common.SESSION_SCHEMA_VERSION)
        self.assertEqual(
            set(payload["aggregate"]),
            set(self.main.SESSION_AGGREGATE_KEYS),
            "the API worker aggregate drifted from the shared contract",
        )
        self.assertGreaterEqual(payload["aggregate"]["frames_total"], 3)
        self.assertEqual(payload["aggregate"]["object_detections_total"], 0)
        self.assertIn(
            "object_detect_error",
            [row.get("type") for row in payload["events"]],
            "the detector failure was not recorded as an event",
        )

    def test_face_recognition_unavailable_degrades_without_aborting(self):
        """An unusable insightface used to kill the monitor before the frame loop."""
        manager = self.server.PipelineManager()
        appended = []
        reason = "insightface 0.2.1 cannot be used: FaceAnalysis(providers=...) requires >= 0.7.3"
        fake_core = self._fake_core(appended, face_reason=reason)
        reader = _FakeReader(manager, frames=3)

        with (
            mock.patch.object(self.server, "_core", return_value=fake_core),
            mock.patch.object(
                self.server.PipelineManager,
                "_attempt_camera_recovery",
                lambda _self, **_kwargs: (object(), reader),
            ),
        ):
            manager._run_monitor_worker(
                model="buffalo_sc",
                general_model=None,
                custom_model=None,
                disable_general=False,
                disable_custom=True,
                snapshot_interval=15.0,
                disable_gaze=True,
                gaze_arch="ResNet50",
                gaze_weights="models/L2CSNet_gaze360.pkl",
                gaze_weights_source="https://example.com",
                disable_gaze_auto_download=True,
                fps_cap=20,
            )

        sessions = [
            payload for event_type, payload in appended if event_type == "recognize_session"
        ]
        self.assertEqual(len(sessions), 1, "the worker did not reach its session summary")
        aggregate = sessions[0]["aggregate"]
        self.assertFalse(aggregate["face_recognition_enabled"])

        self.assertIn(
            "face_recognition_unavailable",
            [row.get("type") for row in sessions[0]["events"]],
            "the reason was not recorded as a session event",
        )
        # `degraded` is a runtime flag the worker clears when the session ends, so the
        # durable proof is the published pipeline_state it emitted while running.
        degraded_states = [
            event["payload"]
            for event in manager.recent_events(limit=200)
            if event.get("type") == "pipeline_state" and event["payload"].get("degraded")
        ]
        self.assertTrue(
            any(
                "face_recognition_unavailable" in str(s.get("degraded_reason"))
                for s in degraded_states
            ),
            "no pipeline_state event reported the missing face recogniser",
        )


if __name__ == "__main__":
    unittest.main()
