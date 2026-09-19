import importlib
import sys
import types
import unittest

import numpy as np


def _install_stubs():
    if "cv2" not in sys.modules:
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


class SessionAggregateContractTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def test_declared_key_set_is_unique_and_stable(self):
        keys = self.main.SESSION_AGGREGATE_KEYS
        self.assertEqual(len(keys), len(set(keys)), "duplicate keys in the declared contract")
        self.assertEqual(len(keys), 66, "the aggregate contract changed size without a schema bump")
        self.assertIn("gaze_interval_frames_final", keys)
        self.assertIn("behavior_activity_patterns", keys)

    def test_default_inputs_emit_exactly_the_contract_and_are_zero_safe(self):
        aggregate = self.main.SessionAggregateInput(session_id="recognize-20260314-113000").build()

        self.assertEqual(set(aggregate), set(self.main.SESSION_AGGREGATE_KEYS))

        # Nothing happened: no division by zero anywhere, and every sentinel collapses.
        self.assertEqual(aggregate["avg_fps"], 0.0)
        self.assertEqual(aggregate["min_fps"], 0.0)
        self.assertEqual(aggregate["faces_per_sec"], 0.0)
        self.assertEqual(aggregate["average_faces_per_frame"], 0.0)
        self.assertEqual(aggregate["min_detection_latency_ms"], 0.0)
        self.assertEqual(aggregate["gaze_inference_min_ms"], 0.0)
        self.assertEqual(aggregate["gaze_inference_avg_ms"], 0.0)
        self.assertEqual(aggregate["recognition_rate"], 0.0)
        self.assertEqual(aggregate["unknown_rate"], 0.0)
        self.assertEqual(aggregate["unknown_alert_density_per_min"], 0.0)
        self.assertEqual(aggregate["behavior_activity_patterns"]["focus_ratio"], 0.0)
        self.assertEqual(aggregate["behavior_activity_patterns"]["transitions_per_min"], 0.0)
        self.assertEqual(aggregate["memory_snapshots_total_session"], 0)

        # Gaze runs at full rate unless a caller says otherwise.
        self.assertEqual(aggregate["gaze_base_interval_frames"], 1)
        self.assertEqual(aggregate["gaze_interval_frames_final"], 1)
        self.assertEqual(aggregate["gaze_target_fps_drop"], 0.0)

    def test_derived_metrics_from_a_populated_session(self):
        aggregate = self.main.SessionAggregateInput(
            session_id="monitor-20260314-113000",
            duration_sec=60.0,
            frames_total=600,
            frames_with_faces=400,
            frames_empty=200,
            frames_dropped=5,
            detections_total=500,
            known_detections=400,
            unknown_detections=100,
            peak_simultaneous_faces=3,
            confidence_sum=400.0,
            detection_calls=600,
            detection_latency_sum_ms=6000.0,
            detection_latency_min_ms=5.0,
            detection_latency_max_ms=40.0,
            fps_ema=9.5,
            fps_min=8.0,
            fps_max=12.0,
            object_detections_total=300,
            object_general_detections=200,
            object_custom_detections=100,
            object_conf_sum=150.0,
            object_general_conf_sum=100.0,
            object_custom_conf_sum=60.0,
            unknown_alert_count=3,
            memory_auto_snapshots=4,
            memory_manual_snapshots=2,
            memory_total_store=11,
            chat_queries_total=5,
            chat_queries_hit=4,
            chat_queries_llm=1,
            unique_individuals_seen=2,
            current_people_visible=1,
            behavior_summary={
                "interactions_total": 4,
                "attention_total_sec": 30.0,
                "top_objects": [["laptop", 20.0]],
                "attention_map": {"Hemanth": {"laptop": 20.0}},
                "events_count": 7,
            },
            gaze_enabled=True,
            gaze_model_loaded=True,
            gaze_target_fps_drop=0.25,
            gaze_inference_calls=30,
            gaze_inference_sum_ms=300.0,
            gaze_inference_min_ms=5.0,
            gaze_inference_max_ms=20.0,
        ).build()

        self.assertEqual(set(aggregate), set(self.main.SESSION_AGGREGATE_KEYS))

        # Throughput and latency
        self.assertAlmostEqual(aggregate["avg_fps"], 10.0, places=6)
        self.assertAlmostEqual(aggregate["moving_avg_fps"], 9.5, places=6)
        self.assertAlmostEqual(aggregate["min_fps"], 8.0, places=6)
        self.assertAlmostEqual(aggregate["faces_per_sec"], 8.333, places=3)
        self.assertAlmostEqual(aggregate["average_faces_per_frame"], 0.8333, places=4)
        self.assertAlmostEqual(aggregate["avg_detection_latency_ms"], 10.0, places=6)
        self.assertEqual(aggregate["detection_calls"], 600)

        # Recognition quality
        self.assertAlmostEqual(aggregate["avg_confidence"], 0.8, places=6)
        self.assertAlmostEqual(aggregate["recognition_rate"], 0.8, places=6)
        self.assertAlmostEqual(aggregate["unknown_rate"], 0.2, places=6)
        self.assertAlmostEqual(aggregate["unknown_alert_density_per_min"], 3.0, places=6)

        # Objects
        self.assertAlmostEqual(aggregate["object_avg_confidence"], 0.5, places=6)
        self.assertAlmostEqual(aggregate["object_avg_confidence_general"], 0.5, places=6)
        self.assertAlmostEqual(aggregate["object_avg_confidence_custom"], 0.6, places=6)
        self.assertEqual(aggregate["object_detections_total"], 300)

        # Gaze
        self.assertTrue(aggregate["gaze_enabled"])
        self.assertTrue(aggregate["gaze_model_loaded"])
        self.assertAlmostEqual(aggregate["gaze_target_fps_drop"], 0.25, places=6)
        self.assertEqual(aggregate["gaze_inference_calls"], 30)
        self.assertAlmostEqual(aggregate["gaze_inference_avg_ms"], 10.0, places=6)
        self.assertAlmostEqual(aggregate["gaze_inference_min_ms"], 5.0, places=6)
        self.assertAlmostEqual(aggregate["gaze_inference_max_ms"], 20.0, places=6)

        # Memory and chat rollups
        self.assertEqual(aggregate["memory_snapshots_auto"], 4)
        self.assertEqual(aggregate["memory_snapshots_manual"], 2)
        self.assertEqual(aggregate["memory_snapshots_total_session"], 6)
        self.assertEqual(aggregate["memory_snapshot_total_store"], 11)
        self.assertEqual(aggregate["chat_queries_total"], 5)

        # Behavior
        self.assertEqual(aggregate["behavior_interactions_total"], 4)
        self.assertAlmostEqual(aggregate["behavior_attention_total_sec"], 30.0, places=6)
        self.assertEqual(aggregate["behavior_events_count"], 7)
        self.assertEqual(aggregate["behavior_top_objects"], [["laptop", 20.0]])
        self.assertAlmostEqual(aggregate["behavior_activity_patterns"]["transitions_per_min"], 4.0, places=6)
        self.assertEqual(aggregate["behavior_activity_patterns"]["unique_attended_objects"], 1)
        self.assertAlmostEqual(aggregate["behavior_activity_patterns"]["focus_ratio"], 0.5, places=6)

    def test_timelines_are_truncated_to_the_last_twenty_buckets(self):
        timeline = {f"10:00:{index:02d}": index for index in range(30)}
        aggregate = self.main.SessionAggregateInput(
            session_id="s", duration_sec=1.0, detection_timeline=timeline
        ).build()

        self.assertEqual(len(aggregate["detection_timeline"]), 20)
        self.assertEqual(aggregate["detection_timeline"][-1]["time_local"], "10:00:29")

    def test_bad_behavior_payload_does_not_break_the_builder(self):
        aggregate = self.main.SessionAggregateInput(
            session_id="s",
            duration_sec=10.0,
            behavior_summary={"interactions_total": "not-a-number", "top_objects": "nope"},
        ).build()

        self.assertEqual(aggregate["behavior_interactions_total"], 0)
        self.assertEqual(aggregate["behavior_top_objects"], [])
        self.assertEqual(aggregate["behavior_activity_patterns"]["unique_attended_objects"], 0)

    def test_written_aggregate_survives_the_legacy_normalizer(self):
        """The reader must accept what the writer now emits."""
        aggregate = self.main.SessionAggregateInput(
            session_id="monitor-x",
            duration_sec=30.0,
            frames_total=300,
            detections_total=100,
            known_detections=90,
            unknown_detections=10,
            gaze_enabled=True,
            gaze_model_loaded=True,
            gaze_inference_calls=4,
            gaze_inference_sum_ms=40.0,
            gaze_inference_min_ms=8.0,
            gaze_inference_max_ms=14.0,
        ).build()

        event = {
            "event_type": "recognize_session",
            "session_id": "monitor-x",
            "duration_sec": 30.0,
            "aggregate": aggregate,
            "people": {},
            "label_counts": {},
            "events": [],
            "timestamp_utc": self.main._iso(),
        }
        normalized = self.main._normalize_recognize_event(event, 0)
        self.assertIsNotNone(normalized)

        agg = normalized["aggregate"]
        self.assertEqual(agg["frames_total"], 300)
        self.assertEqual(agg["known_detections"], 90)
        self.assertEqual(agg["gaze_inference_calls"], 4)
        self.assertAlmostEqual(agg["gaze_inference_avg_ms"], 10.0, places=6)
        self.assertEqual(agg["gaze_base_interval_frames"], 1)
        self.assertEqual(agg["behavior_top_objects"], [])


if __name__ == "__main__":
    unittest.main()
