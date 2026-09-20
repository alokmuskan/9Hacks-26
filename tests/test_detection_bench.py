import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

import common
import detection_bench

#: One point in the parameter space. This is the *detector's* type, re-exported
#: rather than redefined — see `object_detection.DetectorConfig`.
DetectorConfig = detection_bench.DetectorConfig


class _RealInferenceMixin:
    """Give a test the real OpenCV back, and put the stub back afterwards.

    The suite installs a `cv2` stub into `sys.modules` (see `_stubs`), which is
    what every other test wants — and precisely what used to make these tests
    skip. Availability cannot be answered by inspecting `sys.modules`, only by
    trying to import the real module, so this swaps rather than probes, and
    restores the stub via `addCleanup` so the rest of the battery is unaffected.
    """

    def use_real_inference(self) -> None:
        saved = sys.modules.pop("cv2", None)
        try:
            import cv2
        except Exception:
            if saved is not None:
                sys.modules["cv2"] = saved
            self.skipTest("real OpenCV is not installed")

        if not getattr(cv2, "__file__", None):
            # Only the stub is importable here, so this machine has no OpenCV.
            sys.modules["cv2"] = saved if saved is not None else cv2
            self.skipTest("real OpenCV is not installed")

        def restore() -> None:
            if saved is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = saved

        self.addCleanup(restore)

    def require_ultralytics_and_weights(self) -> None:
        if importlib.util.find_spec("ultralytics") is None:
            self.skipTest("ultralytics is not installed")
        if not Path("yolov8n.pt").exists():
            self.skipTest("yolov8n.pt weights are not present")


class FrameClassificationTests(_RealInferenceMixin, unittest.TestCase):
    """Frames that cannot be detected in must be excluded, not averaged in."""

    def test_dark_frame_is_rejected_with_a_reason(self):
        row = detection_bench.classify_frame("dark.jpg", height=480, brightness=11.8, sharpness=1.2)
        self.assertFalse(row.usable)
        self.assertEqual(row.reason, "too_dark")
        self.assertEqual(row.name, "dark.jpg")

    def test_small_frame_is_rejected(self):
        row = detection_bench.classify_frame("thumb.jpg", height=64, brightness=120.0, sharpness=90.0)
        self.assertFalse(row.usable)
        self.assertEqual(row.reason, "too_small")

    def test_blurred_frame_is_rejected(self):
        row = detection_bench.classify_frame("blur.jpg", height=480, brightness=120.0, sharpness=3.0)
        self.assertFalse(row.usable)
        self.assertEqual(row.reason, "blurred")

    def test_well_lit_textured_frame_is_usable(self):
        row = detection_bench.classify_frame("good.jpg", height=480, brightness=168.5, sharpness=254.4)
        self.assertTrue(row.usable)
        self.assertEqual(row.reason, "")

    def test_score_image_separates_black_from_textured(self):
        # `Laplacian` is the stub's blind spot: it returns zeros, so sharpness
        # only means anything with the real library.
        self.use_real_inference()
        black = np.zeros((64, 64), dtype=np.uint8)
        brightness, sharpness = detection_bench.score_image(black)
        self.assertLess(brightness, 1.0)
        self.assertLess(sharpness, 1.0)

        rng = np.random.default_rng(7)
        noise = (rng.random((64, 64)) * 255).astype(np.uint8)
        brightness, sharpness = detection_bench.score_image(noise)
        self.assertGreater(brightness, 100.0)
        self.assertGreater(sharpness, 50.0)


class SummarizeTests(unittest.TestCase):
    def test_aggregates_per_class_counts_and_confidence(self):
        rows = [
            {"label": "person", "confidence": 0.9},
            {"label": "person", "confidence": 0.8},
            {"label": "cell phone", "confidence": 0.46},
            {"label": "", "confidence": 0.5},
        ]
        summary = detection_bench.summarize_detections(rows)

        self.assertEqual([*summary], ["person", "cell phone"])
        self.assertEqual(summary["person"]["count"], 2.0)
        self.assertAlmostEqual(summary["person"]["max_conf"], 0.9)
        self.assertAlmostEqual(summary["person"]["mean_conf"], 0.85)
        self.assertEqual(summary["cell phone"]["count"], 1.0)

    def test_empty_input_is_harmless(self):
        self.assertEqual(detection_bench.summarize_detections([]), {})


class PerFrameDetectionStatsTests(unittest.TestCase):
    """Detection statistics, which are deliberately not precision.

    `summarize_detections` reports "surfboard x45" over 50 frames and nothing else.
    That is equally consistent with 45 real objects and with one false positive
    firing in almost every frame. Counting frames does not settle which it is —
    only labels can — but it replaces one ambiguous number with the rate and the
    coverage a reader can actually judge.
    """

    def test_counts_frames_rather_than_boxes(self):
        stats = detection_bench.summarize_per_frame(
            [
                [{"label": "person", "confidence": 0.9}],
                [{"label": "person", "confidence": 0.8}, {"label": "person", "confidence": 0.7}],
                [],
            ]
        )
        person = stats["per_class"]["person"]

        self.assertEqual(stats["frames"], 3)
        self.assertEqual(stats["frames_with_any_detection"], 2)
        self.assertEqual(person["detections"], 3)
        self.assertEqual(person["frames_with_detection"], 2)
        self.assertEqual(person["max_in_one_frame"], 2)
        self.assertAlmostEqual(person["detections_per_frame"], 1.0)
        self.assertAlmostEqual(person["frame_coverage"], 0.667, places=3)

    def test_a_spread_class_and_a_burst_are_distinguishable(self):
        """Same box count, different shape — the distinction this exists for."""
        spread = detection_bench.summarize_per_frame(
            [[{"label": "surfboard", "confidence": 0.8}] for _ in range(10)]
        )["per_class"]["surfboard"]
        burst = detection_bench.summarize_per_frame(
            [[{"label": "surfboard", "confidence": 0.8}] * 10] + [[] for _ in range(9)]
        )["per_class"]["surfboard"]

        self.assertEqual(spread["detections"], burst["detections"])
        self.assertEqual(spread["frames_with_detection"], 10)
        self.assertEqual(burst["frames_with_detection"], 1)
        self.assertEqual(spread["max_in_one_frame"], 1)
        self.assertEqual(burst["max_in_one_frame"], 10)

    def test_classes_are_ordered_busiest_first(self):
        stats = detection_bench.summarize_per_frame(
            [
                [{"label": "person", "confidence": 0.9}, {"label": "tie", "confidence": 0.3}],
                [{"label": "person", "confidence": 0.9}],
                [{"label": "person", "confidence": 0.9}],
            ]
        )
        self.assertEqual([*stats["per_class"]], ["person", "tie"])

    def test_unlabelled_rows_are_ignored_everywhere(self):
        stats = detection_bench.summarize_per_frame([[{"label": "", "confidence": 0.5}]])
        self.assertEqual(stats["per_class"], {})
        self.assertEqual(stats["frames_with_any_detection"], 0)

    def test_empty_input_is_harmless(self):
        stats = detection_bench.summarize_per_frame([])
        self.assertEqual(stats["frames"], 0)
        self.assertEqual(stats["detections_per_frame"], 0.0)
        self.assertEqual(stats["per_class"], {})


class LatencyStatsTests(unittest.TestCase):
    """A mean alone cannot show drift, and §2e watched one move 15% on fixed input."""

    def test_reports_mean_median_and_spread(self):
        stats = detection_bench.latency_stats([100.0, 200.0, 300.0, 400.0])

        self.assertEqual(stats["mean_ms"], 250.0)
        self.assertEqual(stats["median_ms"], 250.0)
        self.assertEqual(stats["min_ms"], 100.0)
        self.assertEqual(stats["max_ms"], 400.0)
        self.assertEqual(stats["frames"], 4.0)

    def test_the_median_is_the_middle_value_for_an_odd_count(self):
        stats = detection_bench.latency_stats([10.0, 20.0, 90.0])

        self.assertEqual(stats["median_ms"], 20.0)

    def test_the_cold_first_frame_is_kept_in_the_mean_not_dropped(self):
        """Silently discarding warm-up would flatter every latency the report prints."""
        stats = detection_bench.latency_stats([1000.0, 100.0, 100.0, 100.0])

        self.assertEqual(stats["first_frame_ms"], 1000.0)
        self.assertEqual(stats["mean_after_first_ms"], 100.0)
        self.assertEqual(stats["mean_ms"], 325.0)  # the warm-up frame is still in it
        self.assertEqual(stats["median_ms"], 100.0)

    def test_a_single_frame_run_is_harmless(self):
        stats = detection_bench.latency_stats([120.0])

        self.assertEqual(stats["frames"], 1.0)
        self.assertEqual(stats["mean_after_first_ms"], 120.0)
        self.assertEqual(stats["max_ms"], 120.0)

    def test_empty_input_is_harmless(self):
        stats = detection_bench.latency_stats([])

        self.assertEqual(stats["frames"], 0.0)
        self.assertEqual(stats["mean_ms"], 0.0)


class ReferenceMatchingTests(unittest.TestCase):
    def test_recall_uses_expected_counts_and_tolerates_extras(self):
        detail = detection_bench.match_reference(
            {"person": 6.0, "bus": 1.0}, {"person": 4, "bus": 1, "stop sign": 1}
        )

        self.assertEqual(detail["person"]["recalled"], 1.0)  # more than expected is fine
        self.assertEqual(detail["bus"]["recalled"], 1.0)
        self.assertEqual(detail["stop sign"]["recalled"], 0.0)
        self.assertAlmostEqual(detection_bench.recall_of(detail), 0.667, places=3)

    def test_recall_of_empty_is_zero(self):
        self.assertEqual(detection_bench.recall_of({}), 0.0)


class ParamGridTests(unittest.TestCase):
    def test_default_grid_measures_the_shipping_configuration(self):
        """The default run answers "did the change help?", not an abstract sweep."""
        grid = detection_bench.default_param_grid()
        labels = [row.label for row in grid]

        self.assertEqual(grid[0], DetectorConfig.configured())
        self.assertIn("conf=0.25 imgsz=640", labels, "the baseline to compare against is missing")

    def test_default_grid_matches_the_application_defaults(self):
        configured = DetectorConfig.configured()
        self.assertEqual(configured.imgsz, common.YOLO_IMGSZ_DEFAULT)
        self.assertEqual(configured.conf, common.YOLO_CONF_DEFAULT)

    def test_explicit_grid_covers_both_axes(self):
        grid = detection_bench.default_param_grid([0.25, 0.15], [640, 960])
        labels = [row.label for row in grid]
        self.assertEqual(len(grid), 4)
        self.assertIn("conf=0.25 imgsz=640", labels)
        self.assertIn("conf=0.15 imgsz=960", labels)

    def test_explicit_axes_are_respected(self):
        grid = detection_bench.default_param_grid([0.2], [768])
        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0].as_kwargs()["imgsz"], 768)
        self.assertAlmostEqual(grid[0].as_kwargs()["conf"], 0.2)

    def test_params_carry_the_full_kwarg_set(self):
        kwargs = DetectorConfig().as_kwargs()
        self.assertEqual(
            set(kwargs), {"conf", "imgsz", "iou", "max_det", "agnostic_nms"}
        )

    def test_a_grid_point_inherits_the_configured_nms_settings(self):
        """The benchmark and the live detector must not disagree about NMS.

        The benchmark used to carry its own hardcoded iou/max_det defaults, which
        could drift from `common.py` without anything noticing.
        """
        grid = detection_bench.default_param_grid([0.25], [640])
        self.assertEqual(grid[0].iou, common.YOLO_IOU_DEFAULT)
        self.assertEqual(grid[0].max_det, common.YOLO_MAX_DET_DEFAULT)
        self.assertEqual(grid[0].agnostic_nms, common.YOLO_AGNOSTIC_NMS_DEFAULT)


class ReportTests(unittest.TestCase):
    def test_report_shows_the_spread_next_to_the_mean(self):
        """§2e measured one configuration at 438 ms and then 503 ms on fixed input.

        A mean cannot tell that drift apart from a real difference between two models,
        so the spread has to be printed beside it — and the warm-up frame named rather
        than quietly dropped.
        """
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=768",
                    "ms_per_frame": 325.0,
                    "latency": detection_bench.latency_stats([1000.0, 100.0, 100.0, 100.0]),
                    "classes": {},
                    "reference_recall": 1.0,
                }
            ]
        )

        self.assertIn("100 median", text)
        self.assertIn("1000 max", text)
        self.assertIn("detection only, not end-to-end", text)
        self.assertIn("warm-up is included in the mean above, not dropped", text)

    def test_report_omits_the_spread_when_no_latency_was_collected(self):
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=640",
                    "ms_per_frame": 90.0,
                    "classes": {},
                    "reference_recall": 0.0,
                }
            ]
        )

        self.assertNotIn("median", text)
        self.assertNotIn("first frame", text)

    def test_report_shows_latency_classes_and_recall(self):
        results = [
            {
                "label": "conf=0.25 imgsz=640",
                "ms_per_frame": 106.4,
                "classes": {"person": {"count": 7.0, "max_conf": 0.92, "mean_conf": 0.88}},
                "reference_recall": 0.67,
            }
        ]
        text = detection_bench.format_report(results)
        self.assertIn("conf=0.25 imgsz=640", text)
        self.assertIn("106 ms/frame", text)
        self.assertIn("person x7 (max 0.92)", text)
        self.assertIn("reference recall=0.67", text)

    def test_report_states_when_nothing_was_detected(self):
        text = detection_bench.format_report(
            [{"label": "conf=0.25 imgsz=640", "ms_per_frame": 90.0, "classes": {}, "reference_recall": 0.0}]
        )
        self.assertIn("(no detections)", text)

    def test_report_names_the_reference_labels_that_were_missed(self):
        """`reference recall=0.80` is not actionable on its own.

        Four of five labels can be recalled by a model that has plainly regressed
        on the fifth, so the report has to say which one failed.
        """
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=768",
                    "ms_per_frame": 85.0,
                    "classes": {"person": {"count": 1.0, "max_conf": 0.9, "mean_conf": 0.9}},
                    "reference_recall": 0.8,
                    "reference": {
                        "bus.jpg:bus": {"recalled": 1.0},
                        "zidane.jpg:tie": {"recalled": 0.0},
                    },
                }
            ]
        )
        self.assertIn("missed reference: zidane.jpg:tie", text)

    def test_report_stays_quiet_when_every_reference_label_was_recalled(self):
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=768",
                    "ms_per_frame": 85.0,
                    "classes": {},
                    "reference_recall": 1.0,
                    "reference": {"bus.jpg:bus": {"recalled": 1.0}},
                }
            ]
        )
        self.assertNotIn("missed reference", text)

    def test_report_shows_detection_rates_next_to_the_box_counts(self):
        """A flooding class must not read as a success.

        "surfboard x45" is the same line whether those are 45 objects or one
        false positive in most frames; the rate and the coverage say which shape
        the flood has, and the header says outright that it is not precision.
        """
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=768",
                    "ms_per_frame": 85.0,
                    "classes": {"surfboard": {"count": 45.0, "max_conf": 0.79, "mean_conf": 0.6}},
                    "detection_stats": detection_bench.summarize_per_frame(
                        [[{"label": "surfboard", "confidence": 0.79}] for _ in range(45)]
                        + [[] for _ in range(5)]
                    ),
                    "reference_recall": 0.8,
                }
            ]
        )
        self.assertIn("0.90/frame", text)
        self.assertIn("45/50 frames", text)
        self.assertIn("not precision", text)

    def test_report_omits_rates_when_no_stats_were_collected(self):
        text = detection_bench.format_report(
            [
                {
                    "label": "conf=0.25 imgsz=640",
                    "ms_per_frame": 90.0,
                    "classes": {"person": {"count": 1.0, "max_conf": 0.9, "mean_conf": 0.9}},
                    "reference_recall": 1.0,
                }
            ]
        )
        self.assertNotIn("boxes/frame", text)

    def test_excluded_frames_are_summarised_by_reason(self):
        rejected = [
            detection_bench.FrameQuality("a.jpg", 480, 11.0, 1.0, False, "too_dark"),
            detection_bench.FrameQuality("b.jpg", 480, 12.0, 1.0, False, "too_dark"),
            detection_bench.FrameQuality("c.jpg", 64, 120.0, 90.0, False, "too_small"),
        ]
        text = detection_bench.format_report([], rejected)
        self.assertIn("excluded frames: 3", text)
        self.assertIn("too_dark=2", text)
        self.assertIn("too_small=1", text)


class RealInferenceTests(_RealInferenceMixin, unittest.TestCase):
    """Guards the wiring: the kwargs the benchmark passes must actually reach the model."""

    def setUp(self):
        self.use_real_inference()
        self.require_ultralytics_and_weights()
        self.references = detection_bench.reference_assets()
        if not self.references:
            self.skipTest("bundled ultralytics reference images are unavailable")

    def test_reference_images_are_detected(self):
        results = detection_bench.run_benchmark(
            "yolov8n.pt", [DetectorConfig(conf=0.25, imgsz=640)], [], self.references
        )
        detail = results[0]["reference"]
        self.assertGreaterEqual(results[0]["reference_recall"], 0.5)
        self.assertEqual(detail["bus.jpg:bus"]["recalled"], 1.0)
        self.assertEqual(detail["bus.jpg:person"]["recalled"], 1.0)

    def test_confidence_parameter_is_honoured(self):
        strict = detection_bench.run_benchmark(
            "yolov8n.pt", [DetectorConfig(conf=0.99, imgsz=640)], [], self.references
        )
        self.assertEqual(strict[0]["reference_recall"], 0.0)
        self.assertEqual(strict[0]["classes"], {})

    def test_benchmark_drives_the_production_detector(self):
        """The benchmark must measure the shipping wrapper, not raw Ultralytics.

        Params echoed back therefore come from the detector's own resolution
        (clamping + normalisation), which is what the live loop runs.
        """
        results = detection_bench.run_benchmark(
            "yolov8n.pt", [DetectorConfig(conf=0.25, imgsz=672)], [], self.references
        )
        self.assertEqual(results[0]["params"]["imgsz"], 672)

    def test_saved_project_frames_are_classified_before_use(self):
        frames, rejected = detection_bench.load_frames(["memory/snapshots/*.jpg"], limit=3)
        if not frames and not rejected:
            self.skipTest("no saved frames to check")
        for _path, frame in frames:
            self.assertGreaterEqual(frame.shape[0], detection_bench.DEFAULT_MIN_HEIGHT)
        for row in rejected:
            self.assertFalse(row.usable)
            self.assertTrue(row.reason)


if __name__ == "__main__":
    unittest.main()
