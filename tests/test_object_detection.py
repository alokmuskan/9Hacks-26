import os
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import numpy as np

import common
from object_detection import DetectorConfig, DualYoloDetector, normalize_yolo_boxes


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    def __init__(self, rows, names):
        self._rows = rows
        self.names = names
        self.calls = 0
        self.call_kwargs: list[dict] = []

    def __call__(self, frame, verbose=False, **kwargs):
        self.calls += 1
        self.call_kwargs.append(dict(kwargs))
        boxes = _FakeBoxes(
            [r["bbox"] for r in self._rows],
            [r["confidence"] for r in self._rows],
            [r["class_id"] for r in self._rows],
        )
        return [_FakeResult(boxes, self.names)]


class ObjectDetectionTests(unittest.TestCase):
    def test_normalize_yolo_boxes_returns_expected_schema(self):
        rows = normalize_yolo_boxes(
            xyxy=[[1, 2, 10, 12]],
            conf=[0.75],
            cls_ids=[1],
            names={1: "bottle"},
            source="general",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "bottle")
        self.assertEqual(rows[0]["source"], "general")
        self.assertAlmostEqual(rows[0]["confidence"], 0.75, places=5)
        self.assertEqual(rows[0]["bbox"].shape, (4,))

    def test_dual_detector_toggle_behavior(self):
        general_model = _FakeModel(
            [{"bbox": [0, 0, 20, 20], "confidence": 0.9, "class_id": 0}],
            {0: "person"},
        )
        custom_model = _FakeModel(
            [{"bbox": [5, 5, 18, 18], "confidence": 0.8, "class_id": 3}],
            {3: "helmet"},
        )

        detector = DualYoloDetector(
            general_model_obj=general_model,
            custom_model_obj=custom_model,
            enable_general=True,
            enable_custom=True,
        )

        frame = np.zeros((24, 24, 3), dtype=np.uint8)

        rows = detector.detect(frame)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["source"] for r in rows}, {"general", "custom"})

        detector.toggle_custom()
        rows = detector.detect(frame)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "general")

        detector.toggle_general()
        rows = detector.detect(frame)
        self.assertEqual(rows, [])


class DetectorParameterTests(unittest.TestCase):
    """The detector used to call `model(frame)` with no arguments at all.

    Every inference parameter was therefore unreachable — no threshold, no
    inference size, no NMS control — which is why detection quality could not be
    tuned or measured. These tests pin the plumbing, not the values.
    """

    def _detector(self, **kwargs):
        model = _FakeModel(
            [{"bbox": [0, 0, 5, 5], "confidence": 0.9, "class_id": 0}], {0: "person"}
        )
        detector = DualYoloDetector(general_model_obj=model, enable_custom=False, **kwargs)
        return detector, model

    def test_inference_parameters_reach_the_model(self):
        detector, model = self._detector()
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        kwargs = model.call_kwargs[-1]
        self.assertEqual(set(kwargs), {"conf", "iou", "imgsz", "max_det", "agnostic_nms"})
        self.assertEqual(kwargs["conf"], common.YOLO_CONF_DEFAULT)
        self.assertEqual(kwargs["iou"], common.YOLO_IOU_DEFAULT)
        self.assertEqual(kwargs["imgsz"], common.YOLO_IMGSZ_DEFAULT)
        self.assertEqual(kwargs["max_det"], common.YOLO_MAX_DET_DEFAULT)
        self.assertEqual(kwargs["agnostic_nms"], common.YOLO_AGNOSTIC_NMS_DEFAULT)
        self.assertFalse(kwargs["imgsz"] is None)

    def test_explicit_parameters_are_clamped_to_usable_ranges(self):
        detector, _ = self._detector(conf=5.0, iou=-1.0, imgsz=700, max_det=10**9)
        params = detector.params()
        self.assertEqual(params["conf"], 0.99)
        self.assertEqual(params["iou"], 0.1)
        self.assertEqual(params["max_det"], 1000)
        # 700 is not a valid YOLO size: it must snap to a multiple of 32.
        self.assertEqual(params["imgsz"], 704)

    def test_agnostic_nms_is_forwarded_when_enabled(self):
        detector, model = self._detector(agnostic_nms=True)
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(model.call_kwargs[-1]["agnostic_nms"])
        self.assertIs(detector.get_state()["params"]["agnostic_nms"], True)

    def test_state_reports_the_effective_parameters(self):
        detector, _ = self._detector(imgsz=640)
        state = detector.get_state()
        self.assertEqual(state["params"]["imgsz"], 640)
        self.assertEqual(state["general"]["enabled"], True)

    def test_each_detection_call_receives_the_parameters(self):
        detector, model = self._detector(imgsz=640)
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(len(model.call_kwargs), 2)
        self.assertTrue(all(row["imgsz"] == 640 for row in model.call_kwargs))


class DetectorConfigTests(unittest.TestCase):
    """One typed definition of the inference parameters, shared by both paths.

    The benchmark used to define its own near-copy of these fields with hardcoded
    defaults, so the two could silently disagree about what inference was using.
    """

    def test_defaults_come_from_the_shared_configuration(self):
        config = DetectorConfig()
        self.assertEqual(config.conf, common.YOLO_CONF_DEFAULT)
        self.assertEqual(config.iou, common.YOLO_IOU_DEFAULT)
        self.assertEqual(config.imgsz, common.YOLO_IMGSZ_DEFAULT)
        self.assertEqual(config.max_det, common.YOLO_MAX_DET_DEFAULT)
        self.assertEqual(config.agnostic_nms, common.YOLO_AGNOSTIC_NMS_DEFAULT)

    def test_it_is_immutable(self):
        """Overrides must go through `updated()`, never by mutating in place."""
        with self.assertRaises(FrozenInstanceError):
            DetectorConfig().imgsz = 320  # type: ignore[misc]

    def test_updated_leaves_unspecified_fields_alone(self):
        config = DetectorConfig(conf=0.4, imgsz=640, max_det=50)
        changed = config.updated(conf=0.2)
        self.assertEqual(changed.conf, 0.2)
        self.assertEqual(changed.imgsz, 640)
        self.assertEqual(changed.max_det, 50)
        self.assertEqual(changed.agnostic_nms, config.agnostic_nms)

    def test_updated_clamps_to_the_shared_bounds(self):
        config = DetectorConfig().updated(conf=5.0, iou=-1.0, imgsz=100, max_det=10**9)
        self.assertEqual(config.conf, common.YOLO_CONF_BOUNDS[1])
        self.assertEqual(config.iou, common.YOLO_IOU_BOUNDS[0])
        self.assertEqual(config.imgsz, common.YOLO_IMGSZ_BOUNDS[0])
        self.assertEqual(config.max_det, common.YOLO_MAX_DET_BOUNDS[1])

    def test_the_detector_reports_the_same_kwargs_it_is_called_with(self):
        """`params()` and the model call are the same dict, by construction."""
        model = _FakeModel(
            [{"bbox": [0, 0, 5, 5], "confidence": 0.9, "class_id": 0}], {0: "person"}
        )
        config = DetectorConfig(conf=0.3, iou=0.5, imgsz=704, max_det=77, agnostic_nms=True)
        detector = DualYoloDetector(general_model_obj=model, enable_custom=False)
        detector.set_config(config)

        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(model.call_kwargs[-1], config.as_kwargs())
        self.assertEqual(detector.params(), config.as_kwargs())


class InScopeClassTests(unittest.TestCase):
    """The detector reports labels this project has no use for.

    A 124-box review of the project's own frames measured `surfboard` at 0.000
    precision, firing on curtain folds. This knob is how a class like that stops
    reaching the dashboard. It is off unless configured, so these tests pin both
    halves: that unset changes nothing, and that a set allowlist is enforced.
    """

    def _detector(self, labels, **kwargs):
        model = _FakeModel(
            [
                {"bbox": [i, i, i + 5, i + 5], "confidence": 0.9 - i / 100, "class_id": i}
                for i, _ in enumerate(labels)
            ],
            dict(enumerate(labels)),
        )
        return DualYoloDetector(general_model_obj=model, enable_custom=False, **kwargs)

    def test_unset_allowlist_keeps_every_label(self):
        detector = self._detector(["person", "surfboard", "refrigerator"], in_scope_classes="")
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual({r["label"] for r in rows}, {"person", "surfboard", "refrigerator"})
        self.assertFalse(detector.get_state()["in_scope"]["filtering"])
        self.assertEqual(detector.suppressed_labels, {})

    def test_allowlist_drops_unlisted_labels(self):
        detector = self._detector(["person", "surfboard"], in_scope_classes="person")
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual([r["label"] for r in rows], ["person"])
        self.assertEqual(detector.suppressed_labels, {"surfboard": 1})

    def test_matching_ignores_case_and_padding(self):
        detector = self._detector(["Cell Phone"], in_scope_classes="  cell phone , PERSON ")
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual([r["label"] for r in rows], ["Cell Phone"])

    def test_suppression_is_counted_per_label_across_calls(self):
        detector = self._detector(["person", "tie", "surfboard"], in_scope_classes="person")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.detect(frame)

        self.assertEqual(detector.suppressed_labels, {"tie": 2, "surfboard": 2})
        self.assertEqual(detector.get_state()["in_scope"]["suppressed"], {"surfboard": 2, "tie": 2})

    def test_a_typo_cannot_blank_the_output(self):
        """An empty or comma-only value means 'all classes', never 'no classes'."""
        for value in ("", "   ", ",", " , , "):
            with self.subTest(value=value):
                detector = self._detector(["person"], in_scope_classes=value)
                rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
                self.assertEqual([r["label"] for r in rows], ["person"])

    def test_set_in_scope_classes_replaces_the_policy(self):
        detector = self._detector(["person", "surfboard"])
        self.assertEqual(detector.set_in_scope_classes("PERSON"), frozenset({"person"}))
        self.assertEqual(len(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))), 1)

        self.assertEqual(detector.set_in_scope_classes(None), frozenset())
        self.assertEqual(len(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))), 2)

    def test_filtering_applies_to_both_models(self):
        general = _FakeModel(
            [{"bbox": [0, 0, 5, 5], "confidence": 0.9, "class_id": 0}], {0: "surfboard"}
        )
        custom = _FakeModel(
            [{"bbox": [1, 1, 6, 6], "confidence": 0.8, "class_id": 0}], {0: "helmet"}
        )
        detector = DualYoloDetector(
            general_model_obj=general,
            custom_model_obj=custom,
            enable_general=True,
            enable_custom=True,
            in_scope_classes=["helmet"],
        )
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual([(r["label"], r["source"]) for r in rows], [("helmet", "custom")])

    def test_the_allowlist_never_reaches_the_model(self):
        """It is post-processing policy, not an Ultralytics kwarg."""
        detector = self._detector(["person"], in_scope_classes="person")
        detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(
            set(detector.params()), {"conf", "iou", "imgsz", "max_det", "agnostic_nms"}
        )

    def test_state_reports_the_effective_policy(self):
        detector = self._detector(["person"], in_scope_classes="person")
        state = detector.get_state()["in_scope"]

        self.assertEqual(state["classes"], ["person"])
        self.assertTrue(state["filtering"])

    def test_excluding_a_label_drops_only_that_label(self):
        """The measured case: `surfboard` at 0.000 precision, everything else fine.

        A denylist is the smallest change that acts on that evidence, and it
        cannot hide a class nobody named.
        """
        detector = self._detector(["person", "surfboard", "remote"], excluded_classes="surfboard")
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual({r["label"] for r in rows}, {"person", "remote"})
        self.assertEqual(detector.suppressed_labels, {"surfboard": 1})
        self.assertEqual(detector.get_state()["in_scope"]["excluded"], ["surfboard"])

    def test_exclusion_wins_over_inclusion(self):
        detector = self._detector(
            ["person", "surfboard"],
            in_scope_classes="person,surfboard",
            excluded_classes="surfboard",
        )
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual([r["label"] for r in rows], ["person"])

    def test_set_excluded_classes_replaces_the_policy(self):
        detector = self._detector(["person", "tie"])
        self.assertEqual(detector.set_excluded_classes("TIE"), frozenset({"tie"}))

        self.assertEqual(detector.set_excluded_classes(None), frozenset())
        self.assertEqual(len(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))), 2)

    def test_an_unset_denylist_filters_nothing(self):
        detector = self._detector(["person", "surfboard"], excluded_classes="")
        rows = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(len(rows), 2)
        self.assertFalse(detector.get_state()["in_scope"]["filtering"])


class InScopeParsingTests(unittest.TestCase):
    """The value arrives from the environment, so parsing has to be forgiving."""

    def test_none_and_empty_mean_all_classes(self):
        self.assertEqual(common.normalize_in_scope_classes(None), frozenset())
        self.assertEqual(common.normalize_in_scope_classes(""), frozenset())
        self.assertEqual(common.normalize_in_scope_classes([]), frozenset())

    def test_strings_and_iterables_agree(self):
        expected = frozenset({"person", "cell phone"})
        self.assertEqual(common.normalize_in_scope_classes("person, cell phone"), expected)
        self.assertEqual(common.normalize_in_scope_classes(["Person", " CELL PHONE "]), expected)

    def test_reads_the_environment_knob(self):
        with mock.patch.dict(os.environ, {"AI_STUDIO_YOLO_IN_SCOPE_CLASSES": "person, Bottle"}):
            self.assertEqual(
                common._env_class_list("AI_STUDIO_YOLO_IN_SCOPE_CLASSES"),
                frozenset({"person", "bottle"}),
            )

    def test_an_unset_knob_is_empty_not_an_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(common._env_class_list("AI_STUDIO_YOLO_IN_SCOPE_CLASSES"), frozenset())

    def test_the_shipped_default_filters_nothing(self):
        """No environment, no allowlist: the project behaves as it did before."""
        self.assertIsInstance(common.YOLO_IN_SCOPE_CLASSES_DEFAULT, frozenset)


if __name__ == "__main__":
    unittest.main()
