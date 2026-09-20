import importlib.util
import unittest
from pathlib import Path

import numpy as np

import common
import detection_bench


def _real_inference_available() -> bool:
    """True only when real inference can run here.

    Called from `setUp`, never at import time: importing `cv2` during discovery
    would claim `sys.modules["cv2"]` with the real library before the shared stub
    gets a chance to install, breaking every test that relies on the stub (see
    `_stubs` for the contract).
    """
    if importlib.util.find_spec("ultralytics") is None:
        return False
    if not Path("yolov8n.pt").exists():
        return False
    try:
        import cv2
    except Exception:
        return False
    # The stub is a hand-built `types.ModuleType`, so it has no `__file__`; the
    # real extension module does. (The stub used to be identifiable by missing
    # functions, but it now mirrors everything the pipeline calls.)
    return bool(getattr(cv2, "__file__", None)) and hasattr(cv2, "Laplacian")


class FrameClassificationTests(unittest.TestCase):
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
        if not _real_inference_available():
            self.skipTest("sharpness metrics need real OpenCV")
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

        self.assertEqual(grid[0], detection_bench.DetectorParams.configured())
        self.assertIn("conf=0.25 imgsz=640", labels, "the baseline to compare against is missing")

    def test_default_grid_matches_the_application_defaults(self):
        configured = detection_bench.DetectorParams.configured()
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
        kwargs = detection_bench.DetectorParams().as_kwargs()
        self.assertEqual(set(kwargs), {"conf", "imgsz", "iou", "max_det"})


class ReportTests(unittest.TestCase):
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


class RealInferenceTests(unittest.TestCase):
    """Guards the wiring: the kwargs the benchmark passes must actually reach the model."""

    def setUp(self):
        if not _real_inference_available():
            self.skipTest("needs real cv2, ultralytics and yolov8n.pt")
        self.references = detection_bench.reference_assets()
        if not self.references:
            self.skipTest("bundled ultralytics reference images are unavailable")

    def test_reference_images_are_detected(self):
        results = detection_bench.run_benchmark(
            "yolov8n.pt", [detection_bench.DetectorParams(conf=0.25, imgsz=640)], [], self.references
        )
        detail = results[0]["reference"]
        self.assertGreaterEqual(results[0]["reference_recall"], 0.5)
        self.assertEqual(detail["bus.jpg:bus"]["recalled"], 1.0)
        self.assertEqual(detail["bus.jpg:person"]["recalled"], 1.0)

    def test_confidence_parameter_is_honoured(self):
        strict = detection_bench.run_benchmark(
            "yolov8n.pt", [detection_bench.DetectorParams(conf=0.99, imgsz=640)], [], self.references
        )
        self.assertEqual(strict[0]["reference_recall"], 0.0)
        self.assertEqual(strict[0]["classes"], {})

    def test_benchmark_drives_the_production_detector(self):
        """The benchmark must measure the shipping wrapper, not raw Ultralytics.

        Params echoed back therefore come from the detector's own resolution
        (clamping + normalisation), which is what the live loop runs.
        """
        results = detection_bench.run_benchmark(
            "yolov8n.pt", [detection_bench.DetectorParams(conf=0.25, imgsz=672)], [], self.references
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
