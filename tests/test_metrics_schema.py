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


if __name__ == "__main__":
    unittest.main()
