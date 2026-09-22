import importlib
import time
import unittest
from typing import ClassVar
from unittest import mock

import numpy as np

from _stubs import install as _install_stubs


class _FakeDB:
    names: ClassVar[list[str]] = ["Alok"]
    counts: ClassVar[list[int]] = [5]
    centroids: ClassVar[np.ndarray] = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


class _FakeFaceDB:
    @staticmethod
    def load():
        return _FakeDB()


class _FakeEmptyDB:
    names: ClassVar[list[str]] = []
    counts: ClassVar[list[int]] = []
    centroids: ClassVar[np.ndarray] = np.zeros((0, 3), dtype=np.float32)


class _FakeEmptyFaceDB:
    @staticmethod
    def load():
        return _FakeEmptyDB()


class _FakeDetector:
    def __init__(self, **_kwargs):
        self._raises = False

    def detect(self, _frame):
        if self._raises:
            raise RuntimeError("simulated YOLO failure")
        return []

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

    def get_recent_snapshots(self, **_kwargs):
        return []


class _FakeCap:
    def release(self):
        return None


class _FakeReader:
    """Yields the supplied frames, then keeps repeating the last one."""

    def __init__(self, frames):
        self._frames = frames
        self.index = 0
        self.reads = 0

    def read(self, timeout_sec=1.0):
        self.reads += 1
        if self.index < len(self._frames):
            frame = self._frames[self.index]
            self.index += 1
        else:
            frame = self._frames[-1]
        return True, frame.copy()

    def close(self):
        return None


class CliRecognizeLoopTests(unittest.TestCase):
    """`cmd_recognize` had no coverage at all before this file."""

    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def _fake_cv2(self, quit_after):
        """A full cv2 stand-in whose waitKey quits the loop after N frames."""
        fake = mock.MagicMock()
        counter = {"calls": 0}

        def wait_key(*_args, **_kwargs):
            counter["calls"] += 1
            return ord("q") if counter["calls"] >= quit_after else -1

        fake.waitKey.side_effect = wait_key
        fake.getTextSize.return_value = ((10, 12), 2)
        fake.imencode.return_value = (True, np.zeros(4, dtype=np.uint8))
        fake.LINE_AA = 16
        fake.COLOR_BGR2RGB = 4
        fake.INTER_LINEAR = 1
        fake.INTER_AREA = 3
        fake.IMWRITE_JPEG_QUALITY = 1
        fake.FONT_HERSHEY_SIMPLEX = 0
        fake.CAP_V4L2 = 200
        fake.CAP_ANY = 0
        fake.WINDOW_NORMAL = 0
        return fake

    def _run(
        self,
        frames=3,
        gaze_side_effect=None,
        detector_raises=False,
        face_unavailable_reason=None,
        empty_db=False,
        **overrides,
    ):
        """Run cmd_recognize against fakes; returns (appended, fake_gaze, detector)."""
        appended = []
        gaze_result = [(0, 0, 0.0, 0.0)]
        fake_gaze = mock.MagicMock(
            side_effect=gaze_side_effect or (lambda *_a, **_k: list(gaze_result))
        )
        fake_detector = _FakeDetector()
        fake_detector._raises = detector_raises
        reader = _FakeReader([np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(frames)])

        face = (
            np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            0.9,
            np.zeros((5, 2), dtype=np.float32),
        )

        kwargs = {
            "model": "buffalo_sc",
            "general_model": None,
            "custom_model": None,
            "disable_general": False,
            "disable_custom": True,
            "snapshot_interval": 15.0,
            "disable_gaze": True,
            "gaze_arch": "ResNet50",
            "gaze_weights": "models/L2CSNet_gaze360.pkl",
            "gaze_weights_source": "https://example.com",
            "disable_gaze_auto_download": True,
        }
        kwargs.update(overrides)

        gaze_runtime = {"arch": "ResNet50", "weights_path": "models/L2CSNet_gaze360.pkl"}
        gaze_return = None if kwargs["disable_gaze"] else gaze_runtime

        face_app = (
            (None, face_unavailable_reason)
            if face_unavailable_reason is not None
            else (object(), None)
        )
        patches = [
            mock.patch.object(self.main, "cv2", self._fake_cv2(frames)),
            mock.patch.object(self.main, "FaceDB", _FakeEmptyFaceDB if empty_db else _FakeFaceDB),
            mock.patch.object(self.main, "_try_build_face_app", return_value=face_app),
            mock.patch.object(self.main, "DualYoloDetector", return_value=fake_detector),
            mock.patch.object(self.main, "SceneMemoryManager", _FakeMemory),
            mock.patch.object(self.main, "_build_app", return_value=object()),
            mock.patch.object(self.main, "_open_camera", return_value=_FakeCap()),
            mock.patch.object(self.main, "_AsyncCameraReader", return_value=reader),
            mock.patch.object(self.main, "_detect", side_effect=lambda _app, _frame: [face]),
            mock.patch.object(self.main, "_estimate_gaze_points", new=fake_gaze),
            mock.patch.object(self.main, "_load_gaze_runtime", return_value=gaze_return),
            mock.patch.object(self.main, "_print_runtime_help", return_value=None),
            mock.patch.object(
                self.main,
                "_append_metric",
                side_effect=lambda event_type, payload: appended.append((event_type, payload)),
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.main.cmd_recognize(**kwargs)

        return appended, fake_gaze, fake_detector

    @staticmethod
    def _session(appended):
        sessions = [
            payload for event_type, payload in appended if event_type == "recognize_session"
        ]
        return sessions[0] if sessions else None

    def test_session_summary_matches_the_shared_aggregate_contract(self):
        appended, _gaze, _detector = self._run(frames=3)

        payload = self._session(appended)
        self.assertIsNotNone(payload, "the CLI never reached its session summary")
        self.assertEqual(
            payload["schema_version"], importlib.import_module("common").SESSION_SCHEMA_VERSION
        )

        aggregate = payload["aggregate"]
        self.assertEqual(set(aggregate), set(self.main.SESSION_AGGREGATE_KEYS))
        self.assertGreaterEqual(aggregate["frames_total"], 3)
        self.assertGreaterEqual(aggregate["known_detections"], 3)
        self.assertEqual(aggregate["unknown_detections"], 0)
        self.assertIn("Alok", payload["people"])
        self.assertGreaterEqual(payload["events_total_count"], 0)

    def test_a_backwards_wall_clock_step_cannot_invent_a_frame_rate(self):
        """One recorded session reports 29,537 fps against a 0.68 fps average.

        Its intervals were measured against the wall clock, which can be corrected
        backwards mid-session, turning one interval into microseconds. Pacing now
        reads a monotonic clock, so a backwards step must leave the rates sane.
        """
        monotonic_tick = [1000.0]
        wall_tick = [1_700_000_000.0]

        def monotonic() -> float:
            monotonic_tick[0] += 0.5
            return monotonic_tick[0]

        def wall_clock() -> float:
            wall_tick[0] -= 30.0  # 30 s earlier on every single call
            return wall_tick[0]

        with mock.patch.object(self.main.time, "monotonic", side_effect=monotonic):
            with mock.patch.object(self.main.time, "time", side_effect=wall_clock):
                appended, _gaze, _detector = self._run(frames=4)

        aggregate = self._session(appended)["aggregate"]
        self.assertGreaterEqual(aggregate["frames_total"], 3)
        # Half a second between frames is 2 fps. Measured with the stepping wall
        # clock instead, this reads as tens of thousands of fps.
        self.assertLess(aggregate["max_fps"], 5.0)
        self.assertLess(aggregate["moving_avg_fps"], 5.0)

    def test_default_gaze_runs_every_frame_and_reports_full_rate(self):
        appended, fake_gaze, _detector = self._run(frames=4, disable_gaze=False)

        aggregate = self._session(appended)["aggregate"]
        self.assertTrue(aggregate["gaze_enabled"])
        self.assertTrue(aggregate["gaze_model_loaded"])
        # One inference per processed frame, and the metrics say so.
        self.assertEqual(fake_gaze.call_count, aggregate["gaze_inference_calls"])
        self.assertGreaterEqual(aggregate["gaze_inference_calls"], 4)
        self.assertEqual(aggregate["gaze_base_interval_frames"], 1)
        self.assertEqual(aggregate["gaze_interval_frames_final"], 1)
        self.assertEqual(aggregate["gaze_target_fps_drop"], 0.0)

    def test_adaptive_gaze_skips_frames_and_reports_the_real_interval(self):
        appended, fake_gaze, _detector = self._run(
            frames=14,
            disable_gaze=False,
            gaze_side_effect=self._slow_gaze,
            gaze_max_interval=4,
            gaze_target_fps_drop=0.25,
        )

        aggregate = self._session(appended)["aggregate"]
        frames_total = aggregate["frames_total"]

        # Inference is expensive (20 ms against a ~20 ms frame), so the scheduler
        # must shed frames rather than run at full rate.
        self.assertGreater(aggregate["gaze_inference_calls"], 0)
        self.assertLess(
            aggregate["gaze_inference_calls"],
            frames_total,
            "adaptive gaze did not skip any frames",
        )
        # Only genuine inferences are counted, not the frames that reused a result.
        self.assertEqual(fake_gaze.call_count, aggregate["gaze_inference_calls"])
        self.assertGreater(aggregate["gaze_interval_frames_final"], 1)
        self.assertLessEqual(aggregate["gaze_interval_frames_final"], 4)
        self.assertEqual(aggregate["gaze_base_interval_frames"], 1)
        self.assertEqual(aggregate["gaze_target_fps_drop"], 0.25)

    @staticmethod
    def _slow_gaze(*_args, **_kwargs):
        time.sleep(0.02)
        return [(0, 0, 0.0, 0.0)]

    def test_detector_failure_records_an_event_without_aborting(self):
        appended, _gaze, _detector = self._run(frames=3, detector_raises=True)

        payload = self._session(appended)
        self.assertIsNotNone(payload, "a raising object detector aborted the CLI session")
        self.assertGreaterEqual(payload["aggregate"]["frames_total"], 3)
        self.assertIn(
            "object_detect_error",
            [row.get("type") for row in payload["events"]],
            "the detector failure was not recorded as an event",
        )

    def test_no_identities_is_not_fatal_when_face_recognition_is_unavailable(self):
        """`enroll` needs face recognition, so refusing to start would be a dead end.

        Previously an empty identity database raised unconditionally, which made the
        CLI unrunnable on a machine that cannot run `enroll` at all.
        """
        appended, _gaze, _detector = self._run(
            frames=3,
            empty_db=True,
            face_unavailable_reason="insightface not usable",
        )

        payload = self._session(appended)
        self.assertIsNotNone(payload, "the CLI refused to start without identities")
        self.assertFalse(payload["aggregate"]["face_recognition_enabled"])

    def test_face_recognition_unavailable_degrades_instead_of_aborting(self):
        """Was: an unusable insightface aborted the CLI session before the loop."""
        appended, _gaze, _detector = self._run(
            frames=3,
            face_unavailable_reason="insightface 0.2.1 cannot be used",
        )

        payload = self._session(appended)
        self.assertIsNotNone(payload, "a missing face recogniser aborted the CLI session")
        self.assertFalse(payload["aggregate"]["face_recognition_enabled"])
        self.assertGreaterEqual(payload["aggregate"]["frames_total"], 3)

    def test_gaze_metrics_are_reported_by_the_cli(self):
        appended, _gaze, _detector = self._run(frames=2)
        aggregate = self._session(appended)["aggregate"]
        for key in (
            "gaze_base_interval_frames",
            "gaze_interval_frames_final",
            "gaze_target_fps_drop",
        ):
            self.assertIn(key, aggregate)


if __name__ == "__main__":
    unittest.main()
