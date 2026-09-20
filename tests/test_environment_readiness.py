import importlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
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

    def test_a_configured_model_name_is_honoured(self):
        """The old resolver swapped in `yolov8n.pt` whenever the file was absent.

        So `AI_STUDIO_GENERAL_YOLO_MODEL=yolo11s.pt` was accepted, silently
        ignored, and reported as success — the checkpoint named in the README as
        configurable was the one thing that could not be configured.
        """
        with mock.patch.object(self.main, "DEFAULT_GENERAL_MODEL", "yolo11s.pt"):
            self.assertEqual(self.main._resolve_general_model_path(None), "yolo11s.pt")

    def test_an_explicit_model_wins_over_the_configured_default(self):
        with mock.patch.object(self.main, "DEFAULT_GENERAL_MODEL", "yolov8n.pt"):
            self.assertEqual(self.main._resolve_general_model_path("yolo11m.pt"), "yolo11m.pt")

    def test_only_bare_names_are_treated_as_downloadable(self):
        # A bare name is resolved against Ultralytics' release assets; a path is
        # only ever opened from disk, so `doctor` must not promise a download.
        self.assertTrue(self.main._is_downloadable_model_name("yolo11s.pt"))
        self.assertFalse(self.main._is_downloadable_model_name("runs/detect/train/weights/best.pt"))
        self.assertFalse(self.main._is_downloadable_model_name("models/custom.pt"))

    def test_an_existing_checkpoint_is_never_re_downloaded(self):
        with tempfile.TemporaryDirectory() as td:
            weights = Path(td) / "yolov8n.pt"
            weights.write_bytes(b"weights")
            # Any call at all would return the fake's error instead of "".
            with self._fake_ultralytics("should not have been called"):
                self.assertEqual(self.main._ensure_general_model(str(weights)), "")

    def test_ensure_general_model_reports_why_it_could_not_be_prepared(self):
        with tempfile.TemporaryDirectory() as td:
            with self._fake_ultralytics("no network"):
                reason = self.main._ensure_general_model(str(Path(td) / "yolo11s.pt"))

        self.assertEqual(reason, "no network")

    def test_virtual_camera_is_only_the_default_when_it_exists(self):
        expected = (
            self.main._LINUX_VIRTUAL_CAMERA
            if sys.platform.startswith("linux") and Path(self.main._LINUX_VIRTUAL_CAMERA).exists()
            else "0"
        )
        self.assertEqual(self.main._DEFAULT_CAMERA_SOURCE, expected)

    def test_yolo_inference_defaults_match_the_benchmark(self):
        """Pins the measured decision: see OBJECT_DETECTION_PLAN.md.

        On this project's own frames, imgsz=768 found 8 classes versus 5 at 640
        with reference recall unchanged, for ~150 ms versus ~90 ms per frame.
        """
        common = importlib.import_module("common")
        self.assertEqual(common.YOLO_IMGSZ_DEFAULT, 768)
        self.assertEqual(common.YOLO_CONF_DEFAULT, 0.25)
        self.assertEqual(common.YOLO_IOU_DEFAULT, 0.7)
        self.assertEqual(common.YOLO_MAX_DET_DEFAULT, 300)
        self.assertEqual(common.YOLO_AGNOSTIC_NMS_DEFAULT, False)

    def test_inference_size_is_normalised_and_clamped(self):
        common = importlib.import_module("common")
        self.assertEqual(common.normalize_imgsz(100), 320)
        self.assertEqual(common.normalize_imgsz(768), 768)
        self.assertEqual(common.normalize_imgsz(700), 704)
        self.assertEqual(common.normalize_imgsz(5000), 1920)

    def _probe_yolo_env(self, **env_overrides):
        code = (
            "import common;"
            "print(common.YOLO_CONF_DEFAULT, common.YOLO_IOU_DEFAULT, "
            "common.YOLO_IMGSZ_DEFAULT, common.YOLO_MAX_DET_DEFAULT, "
            "common.YOLO_AGNOSTIC_NMS_DEFAULT)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, **env_overrides},
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        return proc.stdout.split()

    def test_yolo_env_overrides_apply_and_clamp(self):
        self.assertEqual(
            self._probe_yolo_env(
                AI_STUDIO_YOLO_CONF="0.4", AI_STUDIO_YOLO_IMGSZ="640", AI_STUDIO_YOLO_MAX_DET="50"
            ),
            ["0.4", "0.7", "640", "50", "False"],
        )
        # Garbage and out-of-range values fall back or clamp instead of exploding.
        self.assertEqual(
            self._probe_yolo_env(AI_STUDIO_YOLO_CONF="5", AI_STUDIO_YOLO_IMGSZ="100"),
            ["0.99", "0.7", "320", "300", "False"],
        )
        self.assertEqual(
            self._probe_yolo_env(AI_STUDIO_YOLO_IMGSZ="nonsense"),
            ["0.25", "0.7", "768", "300", "False"],
        )

    def test_agnostic_nms_env_knob_accepts_the_usual_spellings(self):
        for enabled in ("1", "true", "TRUE", "yes", "on"):
            self.assertEqual(
                self._probe_yolo_env(AI_STUDIO_YOLO_AGNOSTIC_NMS=enabled)[-1],
                "True",
                f"{enabled!r} should enable agnostic NMS",
            )
        for disabled in ("0", "false", "no", "off"):
            self.assertEqual(
                self._probe_yolo_env(AI_STUDIO_YOLO_AGNOSTIC_NMS=disabled)[-1],
                "False",
                f"{disabled!r} should disable agnostic NMS",
            )
        # An unreadable value keeps the default rather than guessing.
        self.assertEqual(self._probe_yolo_env(AI_STUDIO_YOLO_AGNOSTIC_NMS="maybe")[-1], "False")

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

    def test_bootstrap_reports_a_checkpoint_it_could_not_prepare(self):
        """Offline, bootstrap must name what failed — never swap in another model."""
        with tempfile.TemporaryDirectory() as td:
            absent = str(Path(td) / "yolo11s.pt")
            output = io.StringIO()
            with (
                mock.patch.object(self.main, "MEMORY_DIR", Path(td) / "memory"),
                mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", Path(td) / "incidents"),
                mock.patch.object(self.main, "_resolve_general_model_path", return_value=absent),
                self._fake_ultralytics("no network"),
                mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()),
                redirect_stdout(output),
            ):
                self.main.cmd_bootstrap(download_gaze=False)  # must not raise

            text = output.getvalue()
            self.assertIn(absent, text, "the failed checkpoint must be named")
            self.assertIn("no network", text, "the reason must be reported")
            self.assertIn("Unresolved:", text)

    def test_bootstrap_fetches_the_configured_checkpoint(self):
        """Bootstrap used to call `YOLO("yolov8n.pt")` whatever was configured.

        So the one place that exists to make the model available up front was the
        one place guaranteed not to fetch the model the user had chosen.
        """
        with tempfile.TemporaryDirectory() as td:
            requested: list[str] = []
            module = types.ModuleType("ultralytics")

            def record(name, *_args, **_kwargs):
                requested.append(str(name))
                return object()

            module.YOLO = record
            configured = str(Path(td) / "yolo11s.pt")
            with (
                mock.patch.object(self.main, "MEMORY_DIR", Path(td) / "memory"),
                mock.patch.object(self.main, "UNKNOWN_INCIDENTS_DIR", Path(td) / "incidents"),
                mock.patch.object(self.main, "_resolve_general_model_path", return_value=configured),
                mock.patch.dict(sys.modules, {"ultralytics": module}),
                mock.patch.object(self.main.FaceDB, "load", return_value=self.main.FaceDB.empty()),
            ):
                self.main.cmd_bootstrap(download_gaze=False)

            self.assertEqual(requested, [configured])

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
