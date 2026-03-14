import importlib
import sys
import types
import unittest


def _install_stubs():
    if "cv2" not in sys.modules:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.CAP_V4L2 = 200
        cv2_stub.CAP_ANY = 0
        cv2_stub.CAP_PROP_BUFFERSIZE = 38
        cv2_stub.CAP_PROP_FRAME_WIDTH = 3
        cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
        cv2_stub.FONT_HERSHEY_SIMPLEX = 0
        cv2_stub.LINE_AA = 16
        cv2_stub.WINDOW_NORMAL = 0
        sys.modules["cv2"] = cv2_stub

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


class MetricsSchemaTests(unittest.TestCase):
    def test_normalize_recognize_event_supports_object_and_memory_metrics(self):
        _install_stubs()
        main = importlib.import_module("main")

        event = {
            "event_type": "recognize_session",
            "session_id": "s1",
            "duration_sec": 10,
            "aggregate": {
                "frames_total": 100,
                "frames_with_faces": 80,
                "detections_total": 120,
                "known_detections": 100,
                "unknown_detections": 20,
                "object_detections_total": 55,
                "object_general_detections": 40,
                "object_custom_detections": 15,
                "object_avg_confidence": 0.64,
                "memory_snapshots_auto": 4,
                "memory_snapshots_manual": 1,
                "memory_query_counts": {"find": 2, "recent": 1},
                "memory_query_hits": {"find": 1},
                "memory_query_misses": {"find": 1},
                "chat_queries_total": 3,
                "chat_queries_hit": 2,
                "chat_queries_llm": 1,
                "behavior_interactions_total": 7,
                "behavior_attention_total_sec": 123.4,
                "behavior_top_objects": [["laptop", 90.2]],
                "behavior_attention_map": {"Hemanth": {"laptop": 90.2}},
                "behavior_events_count": 11,
                "behavior_activity_patterns": {"transitions_per_min": 1.2},
                "gaze_enabled": True,
                "gaze_model_loaded": True,
                "gaze_base_interval_frames": 3,
                "gaze_interval_frames_final": 5,
                "gaze_target_fps_drop": 0.10,
                "gaze_inference_calls": 12,
                "gaze_inference_avg_ms": 8.4,
                "gaze_inference_min_ms": 7.1,
                "gaze_inference_max_ms": 12.3,
            },
            "events": [],
        }

        row = main._normalize_recognize_event(event, 0)
        self.assertIsNotNone(row)
        agg = row["aggregate"]

        self.assertEqual(agg["object_detections_total"], 55)
        self.assertEqual(agg["object_general_detections"], 40)
        self.assertEqual(agg["object_custom_detections"], 15)
        self.assertAlmostEqual(agg["object_avg_confidence"], 0.64, places=6)
        self.assertEqual(agg["memory_snapshots_auto"], 4)
        self.assertEqual(agg["memory_snapshots_manual"], 1)
        self.assertEqual(agg["memory_query_counts"], {"find": 2, "recent": 1})
        self.assertEqual(agg["chat_queries_total"], 3)
        self.assertEqual(agg["chat_queries_hit"], 2)
        self.assertEqual(agg["chat_queries_llm"], 1)
        self.assertEqual(agg["behavior_interactions_total"], 7)
        self.assertAlmostEqual(agg["behavior_attention_total_sec"], 123.4, places=6)
        self.assertEqual(agg["behavior_top_objects"], [["laptop", 90.2]])
        self.assertEqual(agg["behavior_attention_map"], {"Hemanth": {"laptop": 90.2}})
        self.assertEqual(agg["behavior_events_count"], 11)
        self.assertEqual(agg["behavior_activity_patterns"], {"transitions_per_min": 1.2})
        self.assertTrue(agg["gaze_enabled"])
        self.assertTrue(agg["gaze_model_loaded"])
        self.assertEqual(agg["gaze_base_interval_frames"], 3)
        self.assertEqual(agg["gaze_interval_frames_final"], 5)
        self.assertAlmostEqual(agg["gaze_target_fps_drop"], 0.10, places=6)
        self.assertEqual(agg["gaze_inference_calls"], 12)
        self.assertAlmostEqual(agg["gaze_inference_avg_ms"], 8.4, places=6)
        self.assertAlmostEqual(agg["gaze_inference_min_ms"], 7.1, places=6)
        self.assertAlmostEqual(agg["gaze_inference_max_ms"], 12.3, places=6)

    def test_normalize_recognize_event_backfills_missing_gaze_metrics(self):
        _install_stubs()
        main = importlib.import_module("main")

        event = {
            "event_type": "recognize_session",
            "session_id": "s2",
            "duration_sec": 5,
            "aggregate": {
                "frames_total": 20,
                "frames_with_faces": 10,
                "detections_total": 12,
                "known_detections": 8,
                "unknown_detections": 4,
            },
            "events": [],
        }

        row = main._normalize_recognize_event(event, 0)
        self.assertIsNotNone(row)
        agg = row["aggregate"]
        self.assertFalse(agg["gaze_enabled"])
        self.assertFalse(agg["gaze_model_loaded"])
        self.assertEqual(agg["gaze_base_interval_frames"], 0)
        self.assertEqual(agg["gaze_interval_frames_final"], 0)
        self.assertEqual(agg["gaze_target_fps_drop"], 0.0)
        self.assertEqual(agg["gaze_inference_calls"], 0)
        self.assertEqual(agg["gaze_inference_avg_ms"], 0.0)
        self.assertEqual(agg["gaze_inference_min_ms"], 0.0)
        self.assertEqual(agg["gaze_inference_max_ms"], 0.0)
        self.assertEqual(agg["chat_queries_total"], 0)
        self.assertEqual(agg["chat_queries_hit"], 0)
        self.assertEqual(agg["chat_queries_llm"], 0)
        self.assertEqual(agg["behavior_interactions_total"], 0)
        self.assertEqual(agg["behavior_attention_total_sec"], 0.0)
        self.assertEqual(agg["behavior_top_objects"], [])
        self.assertEqual(agg["behavior_attention_map"], {})
        self.assertEqual(agg["behavior_events_count"], 0)
        self.assertEqual(agg["behavior_activity_patterns"], {})


if __name__ == "__main__":
    unittest.main()
