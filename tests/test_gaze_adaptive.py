import importlib
import unittest

from _stubs import install as _install_stubs


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


class GazeSchedulerTests(unittest.TestCase):
    """The scheduler is opt-in: at the default every frame runs gaze."""

    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def _scheduler(self, **kwargs):
        return self.main.GazeScheduler(**kwargs)

    def test_default_is_full_rate_and_never_adapts(self):
        scheduler = self._scheduler()

        self.assertFalse(scheduler.adaptive)
        self.assertEqual(scheduler.mode_label(), "full-rate")
        self.assertTrue(all(scheduler.should_run() for _ in range(50)))

        # Even absurd latency must not throttle the default configuration.
        for _ in range(20):
            scheduler.observe(latency_ms=5000.0, frame_ms=10.0)
        self.assertEqual(scheduler.interval, 1)
        self.assertEqual(
            scheduler.metrics(),
            {
                "gaze_base_interval_frames": 1,
                "gaze_interval_frames_final": 1,
                "gaze_target_fps_drop": 0.0,
            },
        )

    def test_max_interval_above_base_enables_adaptation(self):
        scheduler = self._scheduler(max_interval=4, target_fps_drop=0.25)

        self.assertTrue(scheduler.adaptive)
        self.assertEqual(scheduler.mode_label(), "adaptive(base=1, max=4, target-drop=0.25)")
        self.assertEqual(scheduler.metrics()["gaze_target_fps_drop"], 0.25)

    def test_expensive_inference_grows_the_interval_to_the_ceiling(self):
        scheduler = self._scheduler(max_interval=4, target_fps_drop=0.25)

        # Gaze costing a full frame budget each time: 1 -> 2 -> 3 -> 4, then clamp.
        seen_intervals = []
        for _ in range(12):
            if scheduler.should_run():
                scheduler.observe(latency_ms=100.0, frame_ms=100.0)
                seen_intervals.append(scheduler.interval)

        self.assertEqual(seen_intervals, [2, 3, 4, 4, 4])
        self.assertEqual(scheduler.metrics()["gaze_interval_frames_final"], 4)

    def test_scheduled_frames_follow_the_interval(self):
        scheduler = self._scheduler(max_interval=4, target_fps_drop=0.25)
        scheduler.observe(latency_ms=100.0, frame_ms=100.0)  # interval -> 2
        self.assertEqual(scheduler.interval, 2)

        runs = [index for index in range(1, 9) if scheduler.should_run()]
        # Runs on the 1st, 3rd, 5th and 7th processed frame of each window of 2.
        self.assertEqual(runs, [1, 3, 5, 7])

    def test_cheap_inference_recovers_the_interval_gradually(self):
        scheduler = self._scheduler(max_interval=4, target_fps_drop=0.25)
        scheduler.observe(latency_ms=100.0, frame_ms=100.0)  # -> 2
        scheduler.observe(latency_ms=100.0, frame_ms=100.0)  # -> 3
        self.assertEqual(scheduler.interval, 3)

        streak = self.main.GAZE_RECOVERY_STREAK_MIN
        for _ in range(streak):
            scheduler.observe(latency_ms=0.1, frame_ms=100.0)  # negligible overhead
        self.assertEqual(scheduler.interval, 2)

        for _ in range(streak):
            scheduler.observe(latency_ms=0.1, frame_ms=100.0)
        self.assertEqual(scheduler.interval, 1, "should never drop below the base interval")

    def test_interval_bounds_are_sanitised(self):
        scheduler = self._scheduler(base_interval=0, max_interval=0)
        self.assertEqual(scheduler.base_interval, 1)
        self.assertEqual(scheduler.max_interval, 1)
        self.assertFalse(scheduler.adaptive)

        # A max below the base cannot invert the window.
        scheduler = self._scheduler(base_interval=5, max_interval=2)
        self.assertEqual(scheduler.max_interval, 5)
        self.assertFalse(scheduler.adaptive)

    def test_zero_frame_time_does_not_divide_by_zero(self):
        scheduler = self._scheduler(max_interval=4)
        scheduler.observe(latency_ms=50.0, frame_ms=0.0)
        self.assertEqual(scheduler.interval, 1)


if __name__ == "__main__":
    unittest.main()
