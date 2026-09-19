import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from _stubs import install as _install_stubs

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class EnvironmentReportTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def _rows(self, **kwargs):
        return {name: (status, detail) for status, name, detail in self.main.collect_environment_report(**kwargs)}

    def test_every_required_module_is_reported(self):
        rows = self._rows()
        for name in self.main._REQUIRED_MODULES:
            self.assertIn(f"module:{name}", rows, f"{name} is not checked")

    def test_every_optional_module_names_the_feature_it_enables(self):
        rows = self._rows()
        for name, purpose in self.main._OPTIONAL_MODULES.items():
            self.assertIn(f"module:{name}", rows)
            self.assertIn(purpose, rows[f"module:{name}"][1])

    def test_missing_required_module_is_a_blocking_failure(self):
        # Deliberately does not assume the host has any particular module: the
        # subject is how a missing required import is *classified*.
        with mock.patch.object(
            self.main, "_check_import", side_effect=lambda name: (False, "not importable") if name == "torch" else (True, "1.0")
        ):
            degraded = self._rows()
            report = self.main.collect_environment_report()

        self.assertEqual(degraded["module:torch"][0], "fail")
        self.assertIn("module:torch", [name for status, name, _ in report if status == "fail"])

    def test_missing_optional_module_is_only_a_warning(self):
        with mock.patch.object(
            self.main, "_check_import", side_effect=lambda name: (False, "not importable") if name == "groq" else (True, "1.0")
        ):
            report = self.main.collect_environment_report()

        statuses = {name: status for status, name, _ in report}
        self.assertEqual(statuses["module:groq"], "warn")

    def test_no_enrolled_identities_is_advisory_not_blocking(self):
        with mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()):
            report = self.main.collect_environment_report()

        statuses = {name: status for status, name, _ in report}
        self.assertEqual(statuses["enrolled identities"], "warn")

    def test_camera_probe_failure_is_reported_not_raised(self):
        with mock.patch.object(self.main, "_open_camera", side_effect=RuntimeError("no camera")):
            report = self.main.collect_environment_report(check_camera=True)

        rows = {name: (status, detail) for status, name, detail in report}
        self.assertEqual(rows["camera probe"][0], "fail")
        self.assertIn("no camera", rows["camera probe"][1])

    def test_camera_probe_success_is_reported(self):
        with mock.patch.object(self.main, "_open_camera", return_value=mock.MagicMock()):
            report = self.main.collect_environment_report(check_camera=True)

        rows = {name: status for status, name, _ in report}
        self.assertEqual(rows["camera probe"], "ok")

    def test_doctor_exits_non_zero_on_a_blocking_failure(self):
        failing = [("fail", "module:torch", "not importable")]
        with mock.patch.object(self.main, "collect_environment_report", return_value=failing):
            with self.assertRaises(SystemExit) as caught:
                self.main.cmd_doctor()

        self.assertEqual(caught.exception.code, 1)

    def test_doctor_exits_zero_when_only_warnings_remain(self):
        advisory = [("warn", "module:groq", "not importable")]
        with mock.patch.object(self.main, "collect_environment_report", return_value=advisory):
            self.main.cmd_doctor()  # must not raise


class RunabilityConfigTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def test_general_model_default_is_downloadable(self):
        """Was a dead path into a different, git-ignored project."""
        self.assertEqual(self.main.DEFAULT_GENERAL_MODEL, "yolov8n.pt")
        self.assertNotIn(".references", self.main.DEFAULT_GENERAL_MODEL)

    def test_general_model_resolves_to_a_usable_checkpoint(self):
        with mock.patch.object(self.main, "DEFAULT_GENERAL_MODEL", "definitely-missing.pt"):
            self.assertEqual(self.main._resolve_general_model_path(None), "yolov8n.pt")

    def test_virtual_camera_is_only_the_default_when_it_exists(self):
        expected = (
            self.main._LINUX_VIRTUAL_CAMERA
            if sys.platform.startswith("linux") and Path(self.main._LINUX_VIRTUAL_CAMERA).exists()
            else "0"
        )
        self.assertEqual(self.main._DEFAULT_CAMERA_SOURCE, expected)

    def test_pixi_tasks_do_not_hardcode_a_camera_device(self):
        text = (PROJECT_ROOT / "pixi.toml").read_text(encoding="utf-8")
        self.assertNotIn("/dev/video42", text, "a hardcoded device makes the task unable to target a webcam")
        self.assertIn('doctor = "python main.py doctor"', text)
        self.assertIn('bootstrap = "python main.py bootstrap"', text)

    def test_enroll_first_error_is_actionable(self):
        with mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()):
            with self.assertRaises(RuntimeError) as caught:
                self.main.cmd_recognize(
                    model="buffalo_sc",
                    general_model=None,
                    custom_model=None,
                    disable_general=False,
                    disable_custom=True,
                    snapshot_interval=15.0,
                    disable_gaze=True,
                    gaze_arch="ResNet50",
                    gaze_weights="models/L2CSNet_gaze360.pkl",
                    gaze_weights_source="https://example.com",
                    disable_gaze_auto_download=True,
                )

        message = str(caught.exception)
        self.assertIn("enroll --name", message)
        self.assertIn("pixi run python main.py", message)

    def _fake_ultralytics(self, error: str | None = None):
        """A stand-in module so no real ultralytics import is attempted."""
        module = types.ModuleType("ultralytics")

        def yell(*_args, **_kwargs):
            if error:
                raise RuntimeError(error)
            return object()

        module.YOLO = yell
        return mock.patch.dict(sys.modules, {"ultralytics": module})

    def test_bootstrap_creates_directories_without_downloading(self):
        with tempfile.TemporaryDirectory() as td:
            memory_dir = Path(td) / "memory"
            incidents_dir = Path(td) / "incidents"
            weights = Path(td) / "yolov8n.pt"
            weights.write_bytes(b"weights")

            with (
                mock.patch.object(self.main, "MEMORY_DIR", memory_dir),
                mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", incidents_dir),
                mock.patch.object(self.main, "_resolve_general_model_path", return_value=str(weights)),
                mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()),
            ):
                self.main.cmd_bootstrap(download_gaze=False)

            self.assertTrue((memory_dir / "snapshots").is_dir())
            self.assertTrue(incidents_dir.is_dir())

    def test_bootstrap_never_raises_when_assets_are_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.object(self.main, "MEMORY_DIR", Path(td) / "memory"),
                mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", Path(td) / "incidents"),
                mock.patch.object(
                    self.main, "_resolve_general_model_path", return_value=str(Path(td) / "absent.pt")
                ),
                self._fake_ultralytics("no network"),
                mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()),
            ):
                self.main.cmd_bootstrap(download_gaze=False)

    def test_bootstrap_does_not_attempt_a_gaze_download_when_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.object(self.main, "MEMORY_DIR", Path(td) / "memory"),
                mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", Path(td) / "incidents"),
                mock.patch.object(
                    self.main, "_resolve_general_model_path", return_value=str(Path(td) / "absent.pt")
                ),
                self._fake_ultralytics(),
                mock.patch.object(self.main, "_load_gaze_runtime") as gaze_loader,
                mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()),
            ):
                self.main.cmd_bootstrap(download_gaze=False)

            gaze_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
