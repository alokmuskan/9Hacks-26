import tempfile
import unittest
from pathlib import Path

import numpy as np

from scene_memory import SceneMemoryManager


class SceneMemoryTests(unittest.TestCase):
    def test_save_and_query_without_vector_backend(self):
        with tempfile.TemporaryDirectory() as td:
            memory = SceneMemoryManager(base_dir=td, snapshot_interval_sec=1.0, enable_vectors=False)

            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            d1 = [{"label": "bottle", "confidence": 0.9, "bbox": np.array([0, 0, 10, 10]), "source": "general"}]
            d2 = [{"label": "phone", "confidence": 0.8, "bbox": np.array([1, 1, 12, 12]), "source": "custom"}]

            e1 = memory.save_snapshot(
                frame,
                d1,
                current_time=10.0,
                manual=False,
                faces=[
                    {
                        "name": "Hemanth",
                        "confidence": 0.91,
                        "bbox": [0, 0, 10, 10],
                        "gaze": {"endpoint": [5, 5], "pitch": 0.1, "yaw": -0.2},
                        "target_object": "bottle",
                    }
                ],
                object_detections=d1,
                people=["Hemanth"],
                attention=[{"name": "Hemanth", "target_object": "bottle", "method": "inside", "distance_px": 0.0}],
            )
            self.assertTrue(Path(e1["snapshot_path"]).exists())
            self.assertFalse(memory.should_take_snapshot(10.5))
            self.assertTrue(memory.should_take_snapshot(11.1))

            e2 = memory.save_snapshot(frame, d2, current_time=12.0, manual=True)
            self.assertTrue(Path(e2["snapshot_path"]).exists())

            stats = memory.get_memory_stats()
            self.assertEqual(stats["total_snapshots"], 2)
            self.assertEqual(stats["manual_snapshots"], 1)
            self.assertEqual(stats["auto_snapshots"], 1)
            self.assertFalse(stats["vectors_enabled"])

            recent = memory.get_recent_snapshots(minutes=60)
            self.assertGreaterEqual(len(recent), 2)

            found = memory.find_object_last_seen("phone")
            self.assertIsNotNone(found)
            self.assertIn("phone", [o.lower() for o in found.get("objects", [])])

            found_person = memory.find_person_last_seen("hemanth")
            self.assertIsNotNone(found_person)
            self.assertIn("Hemanth", found_person.get("people", []))

            hits = memory.search_similar_scene("phone", top_k=3)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("phone", [o.lower() for o in hits[0].get("objects", [])])

            self.assertIn("faces", e1)
            self.assertIn("object_detections", e1)
            self.assertIn("people", e1)
            self.assertIn("attention", e1)


if __name__ == "__main__":
    unittest.main()
