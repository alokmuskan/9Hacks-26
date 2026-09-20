import unittest
from dataclasses import FrozenInstanceError

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
        self.assertEqual(
            set(kwargs), {"conf", "iou", "imgsz", "max_det", "agnostic_nms"}
        )
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


if __name__ == "__main__":
    unittest.main()
