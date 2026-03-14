import importlib
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def _install_stubs():
    import sys

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
        cv2_stub.INTER_LINEAR = 1
        cv2_stub.INTER_AREA = 3
        cv2_stub.COLOR_BGR2RGB = 4
        cv2_stub.cvtColor = lambda frame, _: frame
        cv2_stub.resize = lambda frame, *_args, **_kwargs: frame
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


class _FakeGDown:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def download_folder(self, **_kwargs):
        self.calls += 1
        return self.rows


class GazeL2CSHelperTests(unittest.TestCase):
    def test_select_gaze360_weight_prefers_arch_token(self):
        _install_stubs()
        main = importlib.import_module("main")

        selected = main._select_gaze360_weight_path(
            [
                "/tmp/L2CSNet_gaze360.pkl",
                "/tmp/L2CSNet_gaze360_resnet18.pkl",
            ],
            preferred_arch="ResNet18",
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "L2CSNet_gaze360_resnet18.pkl")

    def test_infer_l2cs_arch_from_state_dict_resnet50(self):
        _install_stubs()
        main = importlib.import_module("main")

        state_dict = {
            "fc_yaw_gaze.weight": np.zeros((90, 2048), dtype=np.float32),
            "layer3.5.conv1.weight": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        self.assertEqual(main._infer_l2cs_arch_from_state_dict(state_dict), "ResNet50")

    def test_infer_l2cs_arch_from_state_dict_resnet18(self):
        _install_stubs()
        main = importlib.import_module("main")

        state_dict = {
            "fc_yaw_gaze.weight": np.zeros((90, 512), dtype=np.float32),
            "layer1.1.conv1.weight": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        self.assertEqual(main._infer_l2cs_arch_from_state_dict(state_dict), "ResNet18")

    def test_select_gaze360_weight_path_is_deterministic(self):
        _install_stubs()
        main = importlib.import_module("main")

        selected = main._select_gaze360_weight_path(
            [
                "/tmp/model.pkl",
                "/tmp/z_gaze360.pkl",
                "/tmp/a_gaze360.pkl",
                "/tmp/other_gaze360.onnx",
            ]
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "a_gaze360.pkl")

    def test_resolver_prefers_cached_path_without_download(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cached = base / "cached_gaze360.pkl"
            cached.write_bytes(b"x")
            gdown = _FakeGDown(rows=[])

            resolved = main._resolve_l2cs_weights_path(
                weights_path=base / "missing.pkl",
                weights_source="https://example.com",
                auto_download=True,
                gdown_module=gdown,
                cached_resolved_path=cached,
            )
            self.assertEqual(resolved, cached)
            self.assertEqual(gdown.calls, 0)

    def test_resolver_chooses_deterministic_gaze360_pkl_after_download(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            a = base / "B_gaze360.pkl"
            b = base / "A_gaze360.pkl"
            c = base / "A_other.pkl"
            for p in (a, b, c):
                p.write_bytes(b"x")

            gdown = _FakeGDown(rows=[str(a), str(b), str(c)])
            resolved = main._resolve_l2cs_weights_path(
                weights_path=base / "missing.pkl",
                weights_source="https://example.com",
                auto_download=True,
                gdown_module=gdown,
                cached_resolved_path=None,
                preferred_arch=None,
                force_download=False,
            )
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved).name, "A_gaze360.pkl")
            self.assertEqual(gdown.calls, 0)

    def test_resolver_finds_nested_local_gaze360_before_download(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "models"
            nested = base / "Gaze360"
            nested.mkdir(parents=True, exist_ok=True)
            nested_ckpt = nested / "L2CSNet_gaze360.pkl"
            nested_ckpt.write_bytes(b"x")
            gdown = _FakeGDown(rows=[])

            resolved = main._resolve_l2cs_weights_path(
                weights_path=base / "L2CSNet_gaze360.pkl",
                weights_source="https://example.com",
                auto_download=True,
                gdown_module=gdown,
                cached_resolved_path=None,
                preferred_arch=None,
                force_download=False,
            )
            self.assertEqual(resolved, nested_ckpt.resolve())
            self.assertEqual(gdown.calls, 0)

    def test_resolver_arch_preference_and_forced_download(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "models"
            base.mkdir(parents=True, exist_ok=True)

            local = base / "L2CSNet_gaze360.pkl"
            local.write_bytes(b"x")
            downloaded = base / "L2CSNet_gaze360_resnet18.pkl"
            downloaded.write_bytes(b"x")
            gdown = _FakeGDown(rows=[str(downloaded)])

            resolved_local = main._resolve_l2cs_weights_path(
                weights_path=base / "missing.pkl",
                weights_source="https://example.com",
                auto_download=True,
                gdown_module=gdown,
                cached_resolved_path=None,
                preferred_arch="ResNet18",
                force_download=False,
            )
            self.assertEqual(resolved_local, downloaded.resolve())
            self.assertEqual(gdown.calls, 0)

            resolved_forced = main._resolve_l2cs_weights_path(
                weights_path=base / "missing.pkl",
                weights_source="https://example.com",
                auto_download=True,
                gdown_module=gdown,
                cached_resolved_path=None,
                preferred_arch="ResNet18",
                force_download=True,
            )
            self.assertEqual(resolved_forced, downloaded.resolve())
            self.assertEqual(gdown.calls, 1)

    def test_expand_bbox_by_ratio_clamps_to_bounds(self):
        _install_stubs()
        main = importlib.import_module("main")

        ex1, ey1, ex2, ey2 = main._expand_bbox_by_ratio(
            np.array([5, 6, 20, 30], dtype=np.float32),
            frame_w=24,
            frame_h=32,
            ratio=0.10,
        )
        self.assertGreaterEqual(ex1, 0)
        self.assertGreaterEqual(ey1, 0)
        self.assertLessEqual(ex2, 23)
        self.assertLessEqual(ey2, 31)
        self.assertGreater(ex2, ex1)
        self.assertGreater(ey2, ey1)

    def test_decode_pitch_yaw_converts_to_radians_once(self):
        _install_stubs()
        main = importlib.import_module("main")
        import torch

        pitch_logits = torch.linspace(-1.5, 1.5, steps=90).unsqueeze(0)
        yaw_logits = torch.linspace(1.0, -1.0, steps=90).unsqueeze(0)
        softmax = torch.nn.Softmax(dim=1)
        idx_deg = (torch.arange(90, dtype=torch.float32) * 4.0) - 180.0

        pitch_rad, yaw_rad = main._decode_l2cs_pitch_yaw_rad(
            pitch_logits=pitch_logits,
            yaw_logits=yaw_logits,
            softmax=softmax,
            idx_tensor_deg=idx_deg,
            torch_module=torch,
        )

        expected_pitch = torch.sum(softmax(pitch_logits) * idx_deg, dim=1) * (np.pi / 180.0)
        expected_yaw = torch.sum(softmax(yaw_logits) * idx_deg, dim=1) * (np.pi / 180.0)
        self.assertAlmostEqual(float(pitch_rad[0].item()), float(expected_pitch[0].item()), places=6)
        self.assertAlmostEqual(float(yaw_rad[0].item()), float(expected_yaw[0].item()), places=6)

    def test_align_face_fallbacks_when_landmarks_missing_or_invalid(self):
        _install_stubs()
        main = importlib.import_module("main")

        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        aligned = main._align_face_from_landmarks(crop, None)
        self.assertTrue(np.array_equal(aligned, crop))

        # Eye distance < 20 should skip alignment.
        landmarks = np.array([[20.0, 20.0], [30.0, 22.0]], dtype=np.float32)
        aligned = main._align_face_from_landmarks(crop, landmarks)
        self.assertTrue(np.array_equal(aligned, crop))

    def test_align_face_fallbacks_for_small_crop(self):
        _install_stubs()
        main = importlib.import_module("main")

        crop = np.zeros((64, 24, 3), dtype=np.uint8)  # width < 32
        landmarks = np.array([[6.0, 10.0], [18.0, 10.0]], dtype=np.float32)
        aligned = main._align_face_from_landmarks(crop, landmarks)
        self.assertTrue(np.array_equal(aligned, crop))

    def test_gaze_endpoint_matches_l2cs_direction_math(self):
        _install_stubs()
        main = importlib.import_module("main")

        # Positive pitch moves arrow left in screen space.
        gx, gy = main._gaze_endpoint_from_pitch_yaw(
            cx=100.0,
            cy=100.0,
            length=50.0,
            pitch=float(np.deg2rad(30.0)),
            yaw=0.0,
            frame_w=640,
            frame_h=480,
        )
        self.assertEqual(gx, 75)
        self.assertEqual(gy, 100)

        # Positive yaw moves arrow up in screen space.
        gx, gy = main._gaze_endpoint_from_pitch_yaw(
            cx=100.0,
            cy=100.0,
            length=50.0,
            pitch=0.0,
            yaw=float(np.deg2rad(30.0)),
            frame_w=640,
            frame_h=480,
        )
        self.assertEqual(gx, 100)
        self.assertEqual(gy, 75)


if __name__ == "__main__":
    unittest.main()
