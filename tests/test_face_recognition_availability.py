import importlib
import unittest
from unittest import mock

import numpy as np

from _stubs import install as _install_stubs


class FaceAppBuilderTests(unittest.TestCase):
    """`_try_build_face_app` must turn any build failure into a reason, never raise."""

    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def test_success_returns_the_app_and_no_reason(self):
        sentinel = object()
        with mock.patch.object(self.main, "_build_app", return_value=sentinel):
            app, reason = self.main._try_build_face_app("buffalo_sc")

        self.assertIs(app, sentinel)
        self.assertIsNone(reason)

    def test_failure_returns_a_reason_instead_of_raising(self):
        with mock.patch.object(
            self.main, "_build_app", side_effect=RuntimeError("insightface 0.2.1 cannot be used")
        ):
            app, reason = self.main._try_build_face_app("buffalo_sc")

        self.assertIsNone(app)
        self.assertEqual(reason, "insightface 0.2.1 cannot be used")

    def test_an_empty_exception_message_falls_back_to_the_type_name(self):
        with mock.patch.object(self.main, "_build_app", side_effect=ValueError()):
            app, reason = self.main._try_build_face_app("buffalo_sc")

        self.assertIsNone(app)
        self.assertEqual(reason, "ValueError")

    def test_detect_without_a_face_app_returns_no_rows(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(self.main._detect(None, frame), [])

    def test_missing_face_analysis_raises_an_actionable_message(self):
        with mock.patch.object(self.main, "FaceAnalysis", None):
            with self.assertRaises(RuntimeError) as caught:
                self.main._make_face_analysis("buffalo_sc", ["CPUExecutionProvider"])

        message = str(caught.exception)
        self.assertIn("insightface", message)
        self.assertIn("0.7.3", message)
        self.assertIn("pixi install", message)


class SessionAggregateFaceFlagTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def test_the_flag_is_part_of_the_shared_contract(self):
        self.assertIn("face_recognition_enabled", self.main.SESSION_AGGREGATE_KEYS)

    def test_the_flag_defaults_to_enabled(self):
        aggregate = self.main.build_session_aggregate(session_id="s")
        self.assertTrue(aggregate["face_recognition_enabled"])
        self.assertEqual(set(aggregate), set(self.main.SESSION_AGGREGATE_KEYS))

    def test_the_flag_records_a_disabled_session(self):
        aggregate = self.main.build_session_aggregate(
            session_id="s", face_recognition_enabled=False
        )
        self.assertFalse(aggregate["face_recognition_enabled"])


class DoctorFaceRecognitionTests(unittest.TestCase):
    def setUp(self):
        _install_stubs()
        self.main = importlib.import_module("main")

    def _rows(self, checker):
        with mock.patch.object(self.main, "_check_import", side_effect=checker):
            report = self.main.collect_environment_report()
        return {name: (status, detail) for status, name, detail in report}

    def test_degradable_modules_are_reported(self):
        rows = self._rows(lambda _name: (True, "1.0"))
        for name, purpose in self.main._DEGRADABLE_MODULES.items():
            self.assertIn(f"module:{name}", rows)
            self.assertIn(purpose, rows[f"module:{name}"][1])

    def test_a_missing_degradable_module_is_only_a_warning(self):
        rows = self._rows(
            lambda name: (False, "not importable") if name == "insightface" else (True, "1.0")
        )
        self.assertEqual(rows["module:insightface"][0], "warn")

    def test_a_too_old_degradable_module_is_only_a_warning(self):
        rows = self._rows(
            lambda name: (True, "0.2.1") if name == "insightface" else (True, "9.9.9")
        )
        status, detail = rows["module:insightface"]
        self.assertEqual(status, "warn")
        self.assertIn("too old", detail)

    def test_a_missing_required_module_still_blocks(self):
        rows = self._rows(
            lambda name: (False, "not importable") if name == "torch" else (True, "9.9.9")
        )
        self.assertEqual(rows["module:torch"][0], "fail")


if __name__ == "__main__":
    unittest.main()
