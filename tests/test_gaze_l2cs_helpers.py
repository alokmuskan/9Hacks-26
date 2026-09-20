import hashlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from _stubs import install as _install_stubs

# torch is a real (un-stubbed) dependency of the L2CS decoding path. Probe once so
# the one test that needs it can skip instead of failing on a machine without it
# (for example a minimal CI job).
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class _FakeGDown:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def download_folder(self, **_kwargs):
        self.calls += 1
        return self.rows


def _fake_urlopen_blob(blob):
    """Stand-in for urllib.request.urlopen serving one static bytes payload."""

    class _FakeResponse(io.BytesIO):
        def __init__(self, blob):
            super().__init__(blob)
            self.headers = {"Content-Length": str(len(blob))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    return lambda _request, timeout=None: _FakeResponse(blob)


class GazeL2CSHelperTests(unittest.TestCase):
    def test_normalize_l2cs_state_dict_unwraps_and_strips_module_prefix(self):
        _install_stubs()
        main = importlib.import_module("main")

        payload = {
            "state_dict": {
                "module.fc_yaw_gaze.weight": np.zeros((90, 512), dtype=np.float32),
                "module.layer1.1.conv1.weight": np.zeros((1, 1, 1, 1), dtype=np.float32),
            }
        }
        normalized = main._normalize_l2cs_state_dict(payload)
        self.assertIn("fc_yaw_gaze.weight", normalized)
        self.assertIn("layer1.1.conv1.weight", normalized)
        self.assertNotIn("module.fc_yaw_gaze.weight", normalized)

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

    @unittest.skipUnless(_TORCH_AVAILABLE, "torch is needed for this L2CS decoding test")
    def test_decode_pitch_yaw_converts_to_radians_once(self):
        _install_stubs()
        main = importlib.import_module("main")

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

class GazeMirrorFallbackTests(unittest.TestCase):
    """The Drive folder is dead upstream; bootstrap must recover via the mirror."""

    def _tiny_safetensors(self, state_dict):
        """Build a minimal but structurally valid safetensors file in memory."""
        import json as _json
        import struct as _struct

        header = {}
        blobs = []
        offset = 0
        for name, array in state_dict.items():
            raw = np.ascontiguousarray(array).tobytes()
            dtype = {"float32": "F32", "float16": "F16"}[str(array.dtype)]
            header[name] = {
                "dtype": dtype,
                "shape": list(array.shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            blobs.append(raw)
            offset += len(raw)
        header_json = _json.dumps(header).encode("utf-8")
        pad = (8 - (len(header_json) % 8)) % 8
        header_json += b" " * pad
        return _struct.pack("<Q", len(header_json)) + header_json + b"".join(blobs)

    def test_mirror_used_when_gdown_yields_nothing(self):
        _install_stubs()
        main = importlib.import_module("main")

        state = {
            "fc_yaw_gaze.weight": np.zeros((90, 2048), dtype=np.float32),
            "fc_yaw_gaze.bias": np.zeros(90, dtype=np.float32),
            "fc_pitch_gaze.weight": np.zeros((90, 2048), dtype=np.float32),
            "fc_pitch_gaze.bias": np.zeros(90, dtype=np.float32),
            "fc_finetune.weight": np.zeros((3, 2048), dtype=np.float32),
            "fc_finetune.bias": np.zeros(3, dtype=np.float32),
        }
        blob = self._tiny_safetensors(state)
        digest = hashlib.sha256(blob).hexdigest()

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "models" / "L2CSNet_gaze360.pkl"
            with (
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SIZE", len(blob)),
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SHA256", digest),
                mock.patch.object(
                    main,
                    "_load_safetensors_state_dict",
                    return_value={k: v.copy() for k, v in state.items()},
                ) as parse_spy,
            ):
                download_spy = mock.patch.object(
                    main,
                    "_download_gaze_weights_from_mirror",
                    wraps=main._download_gaze_weights_from_mirror,
                )
                with download_spy as dl:
                    with mock.patch.object(
                        main.urllib.request, "urlopen", side_effect=OSError("dead upstream")
                    ):
                        # gdown present but the Drive folder returns nothing usable.
                        result = main._resolve_l2cs_weights_path(
                            weights_path=target,
                            weights_source="https://drive.google.com/dead",
                            auto_download=True,
                            gdown_module=_FakeGDown([]),
                        )

            self.assertIsNone(result, "a failed download must not fabricate a path")
            self.assertFalse(target.exists())
            self.assertEqual(dl.call_count, 1, "mirror fallback must be attempted")
            self.assertEqual(parse_spy.call_count, 0)

    def test_mirror_download_writes_loader_compatible_pkl(self):
        _install_stubs()
        main = importlib.import_module("main")
        if not _TORCH_AVAILABLE:
            self.skipTest("torch is needed to verify the saved checkpoint loads")
        import torch

        state = {
            "fc_yaw_gaze.weight": np.arange(90 * 4, dtype=np.float32).reshape(90, 4),
            "fc_yaw_gaze.bias": np.arange(90, dtype=np.float32),
            "fc_pitch_gaze.weight": np.zeros((90, 4), dtype=np.float32),
            "fc_pitch_gaze.bias": np.zeros(90, dtype=np.float32),
            "fc_finetune.weight": np.zeros((3, 4), dtype=np.float32),
            "fc_finetune.bias": np.zeros(3, dtype=np.float32),
        }
        blob = self._tiny_safetensors(state)
        digest = hashlib.sha256(blob).hexdigest()

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "models" / "L2CSNet_gaze360.pkl"
            with (
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SIZE", len(blob)),
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SHA256", digest),
                mock.patch.object(
                    main, "_load_safetensors_state_dict", return_value={k: v.copy() for k, v in state.items()}
                ),
                mock.patch.object(
                    main.urllib.request,
                    "urlopen",
                    side_effect=_fake_urlopen_blob(blob),
                ),
            ):
                result = main._download_gaze_weights_from_mirror(target)

            self.assertIsNotNone(result)
            self.assertTrue(target.exists(), "converted pkl must be written")

            loaded = torch.load(str(target), map_location="cpu", weights_only=True)
            self.assertIsInstance(loaded, dict)
            for key, value in loaded.items():
                self.assertIsInstance(value, torch.Tensor, f"{key} must be a torch tensor")
                self.assertTrue(np.array_equal(value.numpy(), state[key]), f"{key} values must match")

    def test_checksum_mismatch_rejects_download(self):
        _install_stubs()
        main = importlib.import_module("main")

        state = {"fc_yaw_gaze.weight": np.zeros((90, 2048), dtype=np.float32)}
        blob = self._tiny_safetensors(state)
        wrong_digest = "0" * 64

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "models" / "L2CSNet_gaze360.pkl"
            with (
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SIZE", len(blob)),
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SHA256", wrong_digest),
                mock.patch.object(main.urllib.request, "urlopen", side_effect=_fake_urlopen_blob(blob)),
            ):
                result = main._download_gaze_weights_from_mirror(target)

            self.assertIsNone(result, "checksum mismatch must reject the file")
            self.assertFalse(target.exists(), "no file may be left behind on rejection")

    def test_existing_weights_are_never_touched(self):
        _install_stubs()
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "models" / "L2CSNet_gaze360.pkl"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing weights")

            result = main._download_gaze_weights_from_mirror(target)
            self.assertIsNone(result, "existing file means no download should run")
            self.assertEqual(target.read_bytes(), b"existing weights")

    def test_truncated_download_is_rejected(self):
        _install_stubs()
        main = importlib.import_module("main")

        blob = b"x" * 1024  # far smaller than the pinned size
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "models" / "L2CSNet_gaze360.pkl"
            with (
                mock.patch.object(main, "GAZE_WEIGHTS_MIRROR_SIZE", 10_000_000),
                mock.patch.object(main.urllib.request, "urlopen", side_effect=_fake_urlopen_blob(blob)),
            ):
                result = main._download_gaze_weights_from_mirror(target)

            self.assertIsNone(result)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
