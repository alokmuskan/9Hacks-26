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


class GazeAdaptiveTests(unittest.TestCase):
    def test_interval_increases_when_over_target(self):
        _install_stubs()
        main = importlib.import_module("main")

        interval, streak = main._adapt_gaze_interval(
            current_interval=3,
            base_interval=3,
            max_interval=12,
            overhead_ratio=0.2,
            target_drop=0.1,
            recovery_streak=2,
        )
        self.assertEqual(interval, 4)
        self.assertEqual(streak, 0)

    def test_interval_decreases_after_recovery_streak(self):
        _install_stubs()
        main = importlib.import_module("main")

        interval = 7
        streak = 0
        for _ in range(main.GAZE_RECOVERY_STREAK_MIN):
            interval, streak = main._adapt_gaze_interval(
                current_interval=interval,
                base_interval=3,
                max_interval=12,
                overhead_ratio=0.01,
                target_drop=0.1,
                recovery_streak=streak,
            )

        self.assertEqual(interval, 6)
        self.assertEqual(streak, 0)

    def test_interval_respects_bounds(self):
        _install_stubs()
        main = importlib.import_module("main")

        interval, streak = main._adapt_gaze_interval(
            current_interval=12,
            base_interval=3,
            max_interval=12,
            overhead_ratio=0.9,
            target_drop=0.1,
            recovery_streak=0,
        )
        self.assertEqual(interval, 12)
        self.assertEqual(streak, 0)

        interval, streak = main._adapt_gaze_interval(
            current_interval=3,
            base_interval=3,
            max_interval=12,
            overhead_ratio=0.0,
            target_drop=0.1,
            recovery_streak=main.GAZE_RECOVERY_STREAK_MIN - 1,
        )
        self.assertEqual(interval, 3)
        self.assertEqual(streak, main.GAZE_RECOVERY_STREAK_MIN)


if __name__ == "__main__":
    unittest.main()
