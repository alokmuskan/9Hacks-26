import unittest

import numpy as np

from object_detection import DualYoloDetector, normalize_yolo_boxes


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

    def __call__(self, frame, verbose=False):
        self.calls += 1
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


if __name__ == "__main__":
    unittest.main()
