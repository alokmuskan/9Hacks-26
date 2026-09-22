import asyncio
import importlib
import logging
import os
import subprocess
import sys
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from typing import ClassVar
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from object_detection import DualYoloDetector


class ServerBackendTests(unittest.TestCase):
    def setUp(self):
        self.server = importlib.import_module("server")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def test_frame_store_sequence_and_timestamp(self):
        store = self.server.LatestFrameStore()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        first = store.update(b"abc", raw_frame=frame)
        second = store.update(b"def", raw_frame=frame)

        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        self.assertIsNotNone(second["timestamp_utc"])
        self.assertEqual(store.get()["sequence"], 2)

    def test_event_envelope_contract(self):
        payload = {"ok": True}
        row = self.server.MANAGER._event("pipeline_state", payload, session_id="s1")

        self.assertEqual(set(row.keys()), {"type", "timestamp", "session_id", "payload"})
        self.assertEqual(row["type"], "pipeline_state")
        self.assertEqual(row["session_id"], "s1")
        self.assertEqual(row["payload"], payload)

    def test_publish_event_does_not_create_a_coroutine_for_a_closed_loop(self):
        manager = self.server.PipelineManager()
        loop = mock.Mock()
        loop.is_closed.return_value = True
        manager.set_event_loop(loop)
        publish = mock.AsyncMock()
        manager.event_hub.publish = publish

        manager._publish_event("pipeline_state", {"running": False})

        publish.assert_not_awaited()

    def test_fps_throttle_sleep_duration(self):
        dur = self.server.PipelineManager._compute_sleep_duration(0.05, 0.01)
        self.assertAlmostEqual(dur, 0.04, places=6)

        dur = self.server.PipelineManager._compute_sleep_duration(0.05, 0.10)
        self.assertEqual(dur, 0.0)

    def test_stream_generator_uses_frame_store_only(self):
        # If stream path accidentally touches core inference stack, this will fail.
        with mock.patch.object(
            self.server, "_core", side_effect=AssertionError("core should not be called")
        ):
            gen = self.server.build_video_stream_generator()
            data = asyncio.run(gen.__anext__())
            asyncio.run(gen.aclose())
            self.assertIn(b"Content-Type: image/jpeg", data)

    def test_camera_recovery_retries_and_recovers(self):
        manager = self.server.PipelineManager()

        class _FakeCore:
            def __init__(self):
                self.calls = 0

            def _open_camera(self):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("camera offline")
                return object()

            class _AsyncCameraReader:
                def __init__(self, _cap):
                    self.cap = _cap

        fake_core = _FakeCore()
        with mock.patch.object(self.server, "_core", return_value=fake_core):
            pair = manager._attempt_camera_recovery()

        self.assertIsNotNone(pair)
        self.assertEqual(fake_core.calls, 3)

    def test_event_hub_publish_keeps_latest_when_full(self):
        hub = self.server.EventHub(max_queue_size=1)

        async def _run():
            q = await hub.subscribe()
            await hub.publish({"id": 1})
            await hub.publish({"id": 2})
            row = await q.get()
            await hub.unsubscribe(q)
            return row

        row = asyncio.run(_run())
        self.assertEqual(row["id"], 2)

    def test_shutdown_noise_filter_suppresses_only_benign_cancellations(self):
        flt = self.server._ShutdownNoiseFilter()

        def rec(level, exc=None, msg="ok"):
            return logging.LogRecord(
                name="uvicorn.error",
                level=level,
                pathname=__file__,
                lineno=0,
                args=(),
                msg=msg,
                exc_info=exc,
            )

        cancelled = asyncio.CancelledError("queue.get cancelled during shutdown")
        benign = rec(
            logging.ERROR, exc=(asyncio.CancelledError, cancelled, cancelled.__traceback__)
        )
        self.assertFalse(flt.filter(benign), "benign CancelledError record must be dropped")

        real_fail = rec(logging.ERROR, exc=(RuntimeError, RuntimeError("camera exploded"), None))
        self.assertTrue(flt.filter(real_fail), "real RuntimeError record must pass")

        plain_msg = rec(logging.ERROR, msg="Exception in ASGI application\n...\nCancelledError")
        self.assertFalse(flt.filter(plain_msg), "formatted CancelledError text must be dropped")

        cancelled_msg = asyncio.CancelledError("queue.get cancelled during shutdown")
        with_message = rec(
            logging.ERROR, exc=(asyncio.CancelledError, cancelled_msg, cancelled_msg.__traceback__)
        )
        self.assertFalse(
            flt.filter(with_message), "CancelledError with a message must also be dropped"
        )

        kb = rec(logging.ERROR, exc=(KeyboardInterrupt, KeyboardInterrupt(), None))
        self.assertFalse(flt.filter(kb), "bare KeyboardInterrupt record must be dropped")

        mentions = rec(
            logging.ERROR, msg="upload failed after CancelledError occurred mid-transfer"
        )
        self.assertTrue(flt.filter(mentions), "text that merely mentions the name must pass")

        looks_cancelled = rec(
            logging.ERROR,
            msg="Exception in ASGI application\nTraceback ...\nasyncio.exceptions.CancelledError",
        )
        self.assertFalse(
            flt.filter(looks_cancelled),
            "message-embedded traceback ending in CancelledError must be dropped",
        )

        info_rec = rec(
            logging.INFO, exc=(asyncio.CancelledError, cancelled, cancelled.__traceback__)
        )
        self.assertTrue(flt.filter(info_rec), "non-ERROR records must always pass")

    def test_api_status_schema(self):
        with TestClient(self.server.app) as client:
            row = client.get("/api/v1/monitor/status")
            self.assertEqual(row.status_code, 200)
            payload = row.json()
            self.assertIn("running", payload)
            self.assertIn("mode", payload)
            self.assertIn("session_id", payload)
            self.assertIn("frame", payload)
            self.assertIn("sequence", payload["frame"])
            self.assertIn("startup_phase", payload)
            self.assertIn("startup_failure_reason", payload)
            self.assertIn("last_frame_sequence", payload)

    def test_monitor_lifecycle_start_stop(self):
        def _fake_monitor_worker(manager_self, **_kwargs):
            while not manager_self._stop_event.is_set():
                time.sleep(0.01)
            with manager_self._lock:
                if manager_self._mode == "monitor":
                    manager_self._mode = "idle"
                    manager_self._thread = None
            manager_self._set_pipeline_state(mode="idle", running=False, degraded=False)

        with mock.patch.object(
            self.server.PipelineManager, "_run_monitor_worker", _fake_monitor_worker
        ):
            with TestClient(self.server.app) as client:
                started = client.post("/api/v1/monitor/start", json={})
                self.assertEqual(started.status_code, 200)
                self.assertEqual(started.json().get("mode"), "monitor")

                running = False
                for _ in range(20):
                    status = client.get("/api/v1/monitor/status").json()
                    if bool(status.get("running")):
                        running = True
                        break
                    time.sleep(0.01)
                self.assertTrue(running)

                stopped = client.post("/api/v1/monitor/stop")
                self.assertEqual(stopped.status_code, 200)
                status = client.get("/api/v1/monitor/status").json()
                self.assertFalse(bool(status.get("running")))
                self.assertEqual(status.get("mode"), "idle")

    def test_multiple_stream_clients_share_preencoded_frame(self):
        jpeg = b"frame-payload"
        self.server.MANAGER.frame_store.clear()
        self.server.MANAGER.frame_store.update(jpeg)

        with mock.patch.object(
            self.server, "_core", side_effect=AssertionError("core should not be called")
        ):
            g1 = self.server.build_video_stream_generator()
            g2 = self.server.build_video_stream_generator()
            c1 = asyncio.run(g1.__anext__())
            c2 = asyncio.run(g2.__anext__())
            asyncio.run(g1.aclose())
            asyncio.run(g2.aclose())

        self.assertIn(jpeg, c1)
        self.assertIn(jpeg, c2)

    def test_stream_generator_yields_preencoded_bytes_without_processing(self):
        raw = b"already-encoded"
        self.server.MANAGER.frame_store.clear()
        self.server.MANAGER.frame_store.update(raw)

        with mock.patch.object(
            self.server, "_encode_jpeg", side_effect=AssertionError("should not encode in stream")
        ):
            gen = self.server.build_video_stream_generator()
            chunk = asyncio.run(gen.__anext__())
            asyncio.run(gen.aclose())
        self.assertIn(raw, chunk)

    def test_stream_generator_keepalive_emits_when_sequence_static(self):
        raw = b"static-frame"
        self.server.MANAGER.frame_store.clear()
        self.server.MANAGER.frame_store.update(raw)

        with (
            mock.patch.object(self.server, "MJPEG_KEEPALIVE_SEC", 0.0),
            mock.patch.object(self.server, "FRAME_WAIT_IDLE_SEC", 0.0),
        ):
            gen = self.server.build_video_stream_generator()
            first = asyncio.run(gen.__anext__())
            second = asyncio.run(gen.__anext__())
            asyncio.run(gen.aclose())

        self.assertIn(raw, first)
        self.assertIn(raw, second)

    def test_websocket_initial_event_envelope_contract(self):
        with TestClient(self.server.app) as client:
            with client.websocket_connect("/api/v1/stream/events/ws") as ws:
                event = ws.receive_json()
                self.assertEqual(set(event.keys()), {"type", "timestamp", "session_id", "payload"})
                self.assertEqual(event.get("type"), "pipeline_state")

    def test_monitor_startup_timeout_marks_failed(self):
        def _fake_monitor_worker(manager_self, **_kwargs):
            time.sleep(0.02)
            manager_self._mark_startup_failed("startup_timeout_no_frames")

        with mock.patch.object(
            self.server.PipelineManager, "_run_monitor_worker", _fake_monitor_worker
        ):
            with TestClient(self.server.app) as client:
                started = client.post("/api/v1/monitor/start", json={})
                self.assertEqual(started.status_code, 200)
                failed = False
                for _ in range(30):
                    st = client.get("/api/v1/monitor/status").json()
                    if st.get("startup_phase") == "failed":
                        failed = True
                        self.assertEqual(st.get("running"), False)
                        self.assertTrue(st.get("startup_failure_reason"))
                        break
                    time.sleep(0.02)
                self.assertTrue(failed)

    def test_monitor_startup_init_exception_marks_failed(self):
        manager = self.server.PipelineManager()

        class _FakeDB:
            names: ClassVar[list[str]] = []

        class _FakeFaceDB:
            @staticmethod
            def load():
                return _FakeDB()

        class _FakeCore:
            FaceDB = _FakeFaceDB
            MEMORY_DIR = "memory"

            @staticmethod
            def _resolve_general_model_path(_arg):
                return "models/general.pt"

            @staticmethod
            def _load_default_custom_model_path():
                return "models/custom.pt"

            class DualYoloDetector:
                def __init__(self, **_kwargs):
                    raise RuntimeError("detector init failed")

        with mock.patch.object(self.server, "_core", return_value=_FakeCore):
            manager._run_monitor_worker(
                model="buffalo_sc",
                general_model=None,
                custom_model=None,
                disable_general=False,
                disable_custom=False,
                snapshot_interval=15.0,
                disable_gaze=False,
                gaze_arch="ResNet18",
                gaze_weights="models/L2CSNet_gaze360.pkl",
                gaze_weights_source="https://example.com",
                disable_gaze_auto_download=False,
                fps_cap=20,
            )

        st = manager.status()
        self.assertEqual(st.get("startup_phase"), "failed")
        self.assertIn("startup_init_error", str(st.get("startup_failure_reason")))
        self.assertFalse(bool(st.get("running")))

    def test_chat_session_continuity_and_citations(self):
        fake_grounding = {
            "citations": [
                {
                    "id": "C1",
                    "source": "metrics",
                    "timestamp": "2026-03-14T00:00:00+00:00",
                    "detail": "test evidence",
                }
            ]
        }
        with (
            mock.patch.object(self.server, "_build_chat_grounding", return_value=fake_grounding),
            mock.patch.object(
                self.server, "_query_groq_grounded", return_value=("Grounded answer [C1].", True)
            ),
        ):
            with TestClient(self.server.app) as client:
                first = client.post("/api/v1/chat/query", json={"message": "hello"}).json()
                second = client.post(
                    "/api/v1/chat/query",
                    json={"message": "follow up", "session_id": first.get("session_id")},
                ).json()
                self.assertTrue(first.get("session_id"))
                self.assertEqual(second.get("session_id"), first.get("session_id"))
                self.assertTrue(isinstance(second.get("citations"), list))
                self.assertEqual(second["citations"][0]["id"], "C1")

    def test_chat_two_step_action_confirmation(self):
        with TestClient(self.server.app) as client:
            proposed = client.post("/api/v1/chat/query", json={"message": "stop monitoring"}).json()
            action = proposed.get("proposed_action") or {}
            self.assertEqual(action.get("type"), "monitor_stop")
            confirm_id = action.get("confirm_action_id")
            self.assertTrue(confirm_id)

            executed = client.post(
                "/api/v1/chat/query",
                json={"session_id": proposed.get("session_id"), "confirm_action_id": confirm_id},
            ).json()
            self.assertEqual((executed.get("executed_action") or {}).get("type"), "monitor_stop")
            self.assertTrue(executed.get("hit"))

            invalid = client.post(
                "/api/v1/chat/query",
                json={"session_id": proposed.get("session_id"), "confirm_action_id": "bad-id"},
            ).json()
            self.assertFalse(invalid.get("hit"))

    def test_chat_greeting_is_deterministic_without_groq(self):
        with (
            mock.patch.object(
                self.server,
                "_query_groq_grounded",
                side_effect=AssertionError("Groq should not be called"),
            ),
            mock.patch.object(
                self.server,
                "_build_chat_grounding",
                side_effect=AssertionError("Grounding should not be called for greeting"),
            ),
        ):
            with TestClient(self.server.app) as client:
                row = client.post("/api/v1/chat/query", json={"message": "Hi"}).json()
                self.assertEqual(row.get("intent"), "greeting")
                self.assertTrue(row.get("hit"))
                self.assertEqual(row.get("citations"), [])
                self.assertIn("Pipeline mode", str(row.get("reply")))

    def test_chat_person_count_handles_minutes_typo(self):
        now_iso = self.server._iso()
        fake_events = [
            {
                "event_type": "recognize_session",
                "timestamp_utc": now_iso,
                "aggregate": {
                    "unique_individuals_seen": 2,
                    "active_subjects": [{"name": "Alok"}, {"name": "Hemanth"}],
                },
                "people": {
                    "Alok": {"detections": 10},
                    "Hemanth": {"detections": 8},
                },
            }
        ]

        class _FakeMemory:
            def get_recent_snapshots(self, minutes=5, limit=20):
                return []

        fake_grounding = {
            "citations": [
                {
                    "id": "C1",
                    "source": "recognize_session",
                    "timestamp": now_iso,
                    "detail": "known=18, unknown=0, active_people=2",
                }
            ]
        }
        fake_context = {
            "memory": _FakeMemory(),
            "people": [],
            "attention_rows": [],
            "object_rows": [],
            "face_rows": [],
        }

        with (
            mock.patch.object(self.server, "_load_metric_events", return_value=fake_events),
            mock.patch.object(
                self.server.MANAGER, "build_chat_runtime_context", return_value=fake_context
            ),
            mock.patch.object(self.server, "_build_chat_grounding", return_value=fake_grounding),
            mock.patch.object(
                self.server,
                "_query_groq_grounded",
                side_effect=AssertionError("Groq should not be called"),
            ),
        ):
            with TestClient(self.server.app) as client:
                row = client.post(
                    "/api/v1/chat/query",
                    json={"message": "How many persons did you detect in last 5 mminutes"},
                ).json()
                self.assertEqual(row.get("intent"), "person_count")
                self.assertTrue(row.get("hit"))
                self.assertIn("2 distinct persons", str(row.get("reply")))
                self.assertTrue(isinstance(row.get("citations"), list))


class ToggleRefusalTests(unittest.TestCase):
    """A toggle the detector cannot honour must not block the frame loop.

    Enabling a model that is not loaded is the case that bit in practice:
    `toggle_custom()` always returns ``False`` when no checkpoint is configured, so
    the previous `while state != desired: state = detector.toggle_custom()` spun
    forever inside the worker thread. Frames stopped, the dashboard served its
    stalled placeholder, and the user saw "Reconnecting" with no stated cause.
    The assertion is therefore a timeout: the handler has to return.
    """

    def setUp(self):
        self.server = importlib.import_module("server")

    def _manager(self):
        manager = self.server.PipelineManager()
        events: list[tuple[str, dict]] = []
        manager._publish_event = lambda event_type, payload: events.append((event_type, payload))
        return manager, events

    def _drain_with_timeout(self, manager, detector, state, command) -> bool:
        manager._control_q.put(command)
        finished = threading.Event()

        def drain() -> None:
            manager._drain_controls(detector, state)
            finished.set()

        threading.Thread(target=drain, daemon=True).start()
        return finished.wait(5.0)

    def test_asking_for_a_custom_model_that_is_not_loaded_returns(self):
        detector = DualYoloDetector(general_model_obj=object(), enable_custom=False)
        manager, events = self._manager()
        state = {"general": True, "custom": False, "gaze": False}

        command = {"type": "toggle", "general_yolo": None, "custom_yolo": True, "gaze": None}
        self.assertTrue(
            self._drain_with_timeout(manager, detector, state, command),
            "the toggle handler never returned, so it would have frozen the frame loop",
        )

        self.assertFalse(state["custom"], "custom must stay off with no checkpoint")
        payload = events[0][1]
        self.assertIn("custom", payload["toggle_refused"])
        self.assertIn("models/custom_yolo.pt", payload["toggle_refused"]["custom"])

    def test_asking_for_a_general_model_that_is_not_loaded_returns(self):
        detector = DualYoloDetector(general_model_obj=None, enable_general=False)
        manager, events = self._manager()
        state = {"general": False, "custom": False, "gaze": False}

        command = {"type": "toggle", "general_yolo": True, "custom_yolo": None, "gaze": None}
        self.assertTrue(self._drain_with_timeout(manager, detector, state, command))

        self.assertFalse(state["general"])
        self.assertIn("general", events[0][1]["toggle_refused"])

    def test_a_loaded_model_is_enabled_and_nothing_is_refused(self):
        detector = DualYoloDetector(
            general_model_obj=object(),
            custom_model_obj=object(),
            enable_general=True,
            enable_custom=False,
        )
        manager, events = self._manager()
        state = {"general": True, "custom": False, "gaze": False}

        command = {"type": "toggle", "general_yolo": None, "custom_yolo": True, "gaze": None}
        self.assertTrue(self._drain_with_timeout(manager, detector, state, command))

        self.assertTrue(state["custom"])
        self.assertEqual(events[0][1]["toggle_refused"], {})
        self.assertTrue(events[0][1]["toggle_update"]["custom"])


class EnrollLifecycleTests(unittest.TestCase):
    """Enrollment must refuse duplicates, and the stream must not pin a dead session."""

    def setUp(self):
        self.server = importlib.import_module("server")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def test_enrolling_a_name_the_db_already_knows_is_refused(self):
        """`upsert` merges samples by name, so a duplicate silently blurred two people."""
        db = self.server._core().FaceDB.load()
        self.assertTrue(db.names, "this test needs at least one enrolled identity")
        existing = str(db.names[0]).strip()

        # Case and whitespace must not be enough to sneak past the check.
        variant = "  " + existing.upper() + "  "
        with TestClient(self.server.app) as client:
            response = client.post("/api/v1/enroll/start", json={"name": variant})
        self.assertEqual(response.status_code, 409)
        self.assertIn(existing.casefold(), response.json()["detail"].casefold())

    def test_a_distinct_name_starts_normally(self):
        db = self.server._core().FaceDB.load()
        taken = {str(n).strip().casefold() for n in db.names}
        fresh = next(f"ci-test-{n}" for n in range(1000) if f"ci-test-{n}".casefold() not in taken)

        def _fake_worker(manager_self, **_kwargs):
            time.sleep(0.2)
            with manager_self._lock:
                if manager_self._mode == "enroll":
                    manager_self._mode = "idle"
                    manager_self._thread = None

        with mock.patch.object(self.server.PipelineManager, "_run_enroll_worker", _fake_worker):
            with TestClient(self.server.app) as client:
                response = client.post("/api/v1/enroll/start", json={"name": fresh})
                self.assertEqual(response.status_code, 200)
            self.server.MANAGER.stop()

    def test_stop_clears_the_last_frame_from_the_stream(self):
        """A stopped session must not leave its final frame frozen on screen."""
        manager = self.server.PipelineManager()
        manager.frame_store.update(b"stale-jpeg-bytes")

        manager.stop()

        frame = manager.get_stream_frame()
        self.assertNotEqual(frame["frame_bytes"], b"stale-jpeg-bytes")
        # Idle mode serves the idle placeholder, not the stale or stalled one.
        self.assertEqual(frame["frame_bytes"], manager._placeholder_idle)

    def test_a_worker_exit_clears_the_last_frame_even_without_stop(self):
        """Camera loss or a startup failure exits without `stop()` being called.

        The camera path itself is stubbed out: a real recovery attempt would reach
        for actual hardware and hang the suite.
        """
        manager = self.server.PipelineManager()
        manager.frame_store.update(b"stale-jpeg-bytes")

        with mock.patch.object(
            self.server.PipelineManager,
            "_attempt_camera_recovery",
            return_value=None,
        ):
            manager._run_enroll_worker(name="nobody", model="buffalo_sc", fps_cap=20)

        frame = manager.get_stream_frame()
        self.assertEqual(frame["frame_bytes"], manager._placeholder_idle)


class PacingAndCaptureDefaultsTests(unittest.TestCase):
    """Live-FPS pacing and instance-capture cadence are env-tunable defaults.

    The monitor loop treats fps_cap as a ceiling (a slow machine simply runs at
    whatever it can keep up with), while auto snapshots — the instances reports
    and chat ground on — run on their own wall-clock cadence. These tests pin
    the defaults, the status plumbing, and the env overrides with clamping.
    """

    def setUp(self):
        self.server = importlib.import_module("server")

    def test_request_defaults_are_tuned_for_capture(self):
        monitor = self.server.MonitorStartRequest()
        self.assertEqual(monitor.fps_cap, 12)
        self.assertEqual(monitor.snapshot_interval, 8.0)
        # Enrollment keeps the faster cap on purpose: sample collection wants rate.
        enroll = self.server.EnrollStartRequest(name="x")
        self.assertEqual(enroll.fps_cap, 20)

    def test_status_reports_active_pacing_and_cadence(self):
        manager = self.server.PipelineManager()
        st = manager.status()
        self.assertEqual(st["fps_cap"], self.server.FPS_CAP_DEFAULT)
        self.assertEqual(st["snapshot_interval"], self.server.SNAPSHOT_INTERVAL_DEFAULT)

        # What start_monitor stores must be exactly what status exposes.
        manager._fps_cap = 7
        manager._snapshot_interval = 3.5
        st = manager.status()
        self.assertEqual(st["fps_cap"], 7)
        self.assertEqual(st["snapshot_interval"], 3.5)

    def _probe(self, **env_overrides: str) -> list[str]:
        code = (
            "import common;"
            "print(common.FPS_CAP_DEFAULT, common.ENROLL_FPS_CAP_DEFAULT, "
            "common.SNAPSHOT_INTERVAL_DEFAULT)"
        )
        env = {**os.environ, **env_overrides}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or ".",
        )
        return proc.stdout.split()

    def test_env_overrides_apply_and_clamp(self):
        # Overrides apply.
        self.assertEqual(
            self._probe(
                AI_STUDIO_FPS_CAP="9",
                AI_STUDIO_ENROLL_FPS_CAP="33",
                AI_STUDIO_SNAPSHOT_INTERVAL="2.5",
            ),
            ["9", "33", "2.5"],
        )
        # Values outside the API's own validation ranges clamp instead of exploding.
        self.assertEqual(
            self._probe(AI_STUDIO_FPS_CAP="500", AI_STUDIO_SNAPSHOT_INTERVAL="0.1"),
            ["60", "20", "1.0"],
        )
        # Garbage falls back to the defaults.
        self.assertEqual(
            self._probe(AI_STUDIO_FPS_CAP="abc", AI_STUDIO_SNAPSHOT_INTERVAL="xyz"),
            ["12", "20", "8.0"],
        )


class RecentSnapshotsWindowTests(unittest.TestCase):
    """The /memory/recent window is capped at the last monitoring session."""

    def setUp(self):
        self.server = importlib.import_module("server")

    def tearDown(self):
        try:
            self.server.MANAGER._loop = None
            self.server.MANAGER.stop()
        except Exception:
            pass

    def test_recent_caps_to_session_window(self):
        memory = mock.MagicMock()
        memory.latest_session_window_minutes.return_value = 9
        captured: dict = {}

        def fake_recent(minutes, limit):
            captured["minutes"] = minutes
            return [{"id": 1}]

        memory.get_recent_snapshots.side_effect = fake_recent
        with mock.patch.object(self.server, "_core") as core:
            core.return_value.SceneMemoryManager.return_value = memory
            with TestClient(self.server.app) as client:
                row = client.get("/api/v1/memory/recent?minutes=60").json()

        self.assertEqual(captured["minutes"], 9)
        self.assertTrue(row["capped"])
        self.assertEqual(row["effective_minutes"], 9)
        self.assertEqual(row["requested_minutes"], 60)

    def test_recent_ignores_cap_without_session_history(self):
        memory = mock.MagicMock()
        memory.latest_session_window_minutes.return_value = None
        captured: dict = {}
        memory.get_recent_snapshots.side_effect = lambda minutes, limit: (
            captured.setdefault("minutes", minutes),
            [],
        )[1]
        with mock.patch.object(self.server, "_core") as core:
            core.return_value.SceneMemoryManager.return_value = memory
            with TestClient(self.server.app) as client:
                row = client.get("/api/v1/memory/recent?minutes=60").json()

        self.assertEqual(captured["minutes"], 60)
        self.assertFalse(row["capped"])
        self.assertIsNone(row["session_window_minutes"])

    def test_session_window_endpoint_reads_metrics(self):
        recent_end = datetime.now(UTC) - timedelta(seconds=30)
        events = [
            {"event_type": "recognize_session", "end_utc": recent_end.isoformat()},
            {"event_type": "chat_query"},
        ]
        with mock.patch.object(self.server, "_load_metric_events", return_value=events):
            with TestClient(self.server.app) as client:
                row = client.get("/api/v1/memory/session-window").json()

        self.assertEqual(row["session_window_minutes"], 1)
        self.assertIn("last_session", row)


if __name__ == "__main__":
    unittest.main()
