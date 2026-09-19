import asyncio
import importlib
import time
import unittest
from typing import ClassVar
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient


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

    def test_fps_throttle_sleep_duration(self):
        dur = self.server.PipelineManager._compute_sleep_duration(0.05, 0.01)
        self.assertAlmostEqual(dur, 0.04, places=6)

        dur = self.server.PipelineManager._compute_sleep_duration(0.05, 0.10)
        self.assertEqual(dur, 0.0)

    def test_stream_generator_uses_frame_store_only(self):
        # If stream path accidentally touches core inference stack, this will fail.
        with mock.patch.object(self.server, "_core", side_effect=AssertionError("core should not be called")):
            gen = self.server.build_video_stream_generator()
            data = next(gen)
            gen.close()
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

        with mock.patch.object(self.server.PipelineManager, "_run_monitor_worker", _fake_monitor_worker):
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

        with mock.patch.object(self.server, "_core", side_effect=AssertionError("core should not be called")):
            g1 = self.server.build_video_stream_generator()
            g2 = self.server.build_video_stream_generator()
            c1 = next(g1)
            c2 = next(g2)
            g1.close()
            g2.close()

        self.assertIn(jpeg, c1)
        self.assertIn(jpeg, c2)

    def test_stream_generator_yields_preencoded_bytes_without_processing(self):
        raw = b"already-encoded"
        self.server.MANAGER.frame_store.clear()
        self.server.MANAGER.frame_store.update(raw)

        with mock.patch.object(self.server, "_encode_jpeg", side_effect=AssertionError("should not encode in stream")):
            gen = self.server.build_video_stream_generator()
            chunk = next(gen)
            gen.close()
        self.assertIn(raw, chunk)

    def test_stream_generator_keepalive_emits_when_sequence_static(self):
        raw = b"static-frame"
        self.server.MANAGER.frame_store.clear()
        self.server.MANAGER.frame_store.update(raw)

        with mock.patch.object(self.server, "MJPEG_KEEPALIVE_SEC", 0.0), mock.patch.object(
            self.server, "FRAME_WAIT_IDLE_SEC", 0.0
        ):
            gen = self.server.build_video_stream_generator()
            first = next(gen)
            second = next(gen)
            gen.close()

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

        with mock.patch.object(self.server.PipelineManager, "_run_monitor_worker", _fake_monitor_worker):
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
        with mock.patch.object(self.server, "_build_chat_grounding", return_value=fake_grounding), mock.patch.object(
            self.server, "_query_groq_grounded", return_value=("Grounded answer [C1].", True)
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
        with mock.patch.object(self.server, "_query_groq_grounded", side_effect=AssertionError("Groq should not be called")), mock.patch.object(
            self.server, "_build_chat_grounding", side_effect=AssertionError("Grounding should not be called for greeting")
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

        with mock.patch.object(self.server, "_load_metric_events", return_value=fake_events), mock.patch.object(
            self.server.MANAGER, "build_chat_runtime_context", return_value=fake_context
        ), mock.patch.object(self.server, "_build_chat_grounding", return_value=fake_grounding), mock.patch.object(
            self.server, "_query_groq_grounded", side_effect=AssertionError("Groq should not be called")
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


if __name__ == "__main__":
    unittest.main()
