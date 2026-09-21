import importlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import frame_set_validation as fsv
from _stubs import install as _install_stubs

#: A filename that satisfies the convention in OBJECT_DETECTION_FRAME_SET_SPEC.md §7:
#: <timestamp>_<venue>_<lighting>_<distance>_<angle>_<NNN>
GOOD_STEM = "2026-09-22T14-31-08_booth-a_lit_near_front_001"


def _stem(n: int = 1, lighting: str = "lit", distance: str = "near", angle: str = "front") -> str:
    return f"2026-09-22T14-31-08_booth-a_{lighting}_{distance}_{angle}_{n:03d}"


class _FrameSetBuilder:
    """A frame set on disk, with image decoding replaced by a lookup table.

    The validator's image probe is injectable precisely so this can exist: the
    checks under test are about *structure*, and requiring real JPEGs would make
    every test depend on OpenCV and on pixels nobody is asserting about.
    """

    DEFAULT_QUALITY = (480, 150.0, 200.0)

    def __init__(self, root: Path) -> None:
        self.root = root
        self.images = root / "images"
        self.labels = root / "labels"
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)
        self.quality: dict[str, tuple[int, float, float] | None] = {}

    def frame(
        self,
        stem: str,
        labels: str = "",
        *,
        quality: tuple[int, float, float] | None = None,
        label_file: bool = True,
    ) -> "_FrameSetBuilder":
        (self.images / f"{stem}.jpg").write_bytes(b"jpeg-bytes")
        if label_file:
            (self.labels / f"{stem}.txt").write_text(labels, encoding="utf-8")
        self.quality[stem] = quality or self.DEFAULT_QUALITY
        return self

    def probe(self, path: Path) -> tuple[int, float, float] | None:
        return self.quality.get(Path(path).stem, self.DEFAULT_QUALITY)

    def validate(self, **kwargs):
        return fsv.validate_frame_set(self.root, probe=self.probe, **kwargs)

    def codes(self, report) -> list[str]:
        return [row.code for row in report.findings]

    def errors(self, report) -> list[str]:
        return [row.code for row in report.errors]


class FrameSetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.builder = _FrameSetBuilder(Path(holder.name))

    def build(self) -> _FrameSetBuilder:
        return self.builder


class FrameNameTests(unittest.TestCase):
    def test_a_conventional_name_parses_into_its_fields(self):
        name, problem = fsv.parse_frame_name(f"{GOOD_STEM}.jpg")

        self.assertEqual(problem, "")
        assert name is not None
        self.assertEqual(name.timestamp, "2026-09-22T14-31-08")
        self.assertEqual(name.venue, "booth-a")
        self.assertEqual(name.lighting, "lit")
        self.assertEqual(name.distance, "near")
        self.assertEqual(name.angle, "front")
        self.assertEqual(name.index, "001")

    def test_the_wrong_number_of_fields_is_explained_not_just_rejected(self):
        name, problem = fsv.parse_frame_name("frame_001.jpg")

        self.assertIsNone(name)
        self.assertIn("6 underscore-separated fields", problem)
        self.assertIn("found 2", problem)

    def test_an_unknown_lighting_state_is_rejected(self):
        # Spec §5 depends on these being exactly identifiable: the dark frames are
        # the ones the quality filter drops, so they must be findable.
        name, problem = fsv.parse_frame_name(_stem(lighting="evening") + ".jpg")

        self.assertIsNone(name)
        self.assertIn("'evening'", problem)
        self.assertIn("lit, dim, dark", problem)

    def test_a_sequence_shorter_than_three_digits_is_rejected(self):
        name, problem = fsv.parse_frame_name("2026-09-22T14-31-08_booth-a_lit_near_front_7.jpg")

        self.assertIsNone(name)
        self.assertIn("at least 3 digits", problem)


class LabelParsingTests(unittest.TestCase):
    VOCAB = fsv.vocabulary()

    def parse(self, text: str):
        return fsv.parse_labels(text, subject="frame.jpg", vocab=self.VOCAB)

    def test_a_valid_row_is_parsed(self):
        rows, findings = self.parse("cell phone 0.744 0.612 0.058 0.091\n")

        self.assertEqual(findings, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].cls, "cell phone")
        self.assertAlmostEqual(rows[0].cx, 0.744)

    def test_a_multi_word_class_name_is_not_split_into_fields(self):
        """Fifteen COCO names contain a space, and splitting on whitespace breaks all.

        This was a real bug: `cell phone 0.7 0.6 0.06 0.09` read as six fields and was
        rejected as malformed, so any frame containing a phone failed validation with a
        message about field counts rather than about anything being wrong.
        """
        for name in ("cell phone", "stop sign", "hair drier", "potted plant", "hot dog"):
            rows, findings = self.parse(f"{name} 0.5 0.5 0.2 0.2\n")

            self.assertEqual(findings, [], f"{name} should parse")
            self.assertEqual(rows[0].cls, name)

    def test_a_class_index_is_resolved_through_the_vocabulary(self):
        rows, findings = self.parse("0 0.5 0.5 0.2 0.2\n")  # 0 is person

        self.assertEqual(findings, [])
        self.assertEqual(rows[0].cls, "person")

    def test_a_misspelled_class_is_rejected_with_the_reason_that_matters(self):
        rows, findings = self.parse("phone 0.5 0.5 0.2 0.2\n")

        self.assertEqual(rows, [])
        self.assertEqual(findings[0].code, "unknown-class")
        self.assertIn("exact spelling", findings[0].detail)

    def test_an_out_of_range_index_is_rejected(self):
        _rows, findings = self.parse("9000 0.5 0.5 0.2 0.2\n")

        self.assertEqual(findings[0].code, "unknown-class-index")

    def test_pixel_coordinates_are_caught_as_unnormalised(self):
        """The classic mistake: YOLO wants 0-1, people write pixels."""
        rows, findings = self.parse("person 320 240 100 50\n")

        self.assertEqual(rows, [])
        self.assertEqual(findings[0].code, "unnormalised-box")
        self.assertIn("not pixels", findings[0].detail)

    def test_a_zero_area_box_is_rejected(self):
        _rows, findings = self.parse("person 0.5 0.5 0.0 0.3\n")

        self.assertEqual(findings[0].code, "degenerate-box")

    def test_a_non_finite_coordinate_is_rejected(self):
        _rows, findings = self.parse("person nan 0.5 0.2 0.2\n")

        self.assertEqual(findings[0].code, "malformed-label")

    def test_the_wrong_field_count_is_rejected(self):
        _rows, findings = self.parse("person 0.5 0.5 0.2\n")

        self.assertEqual(findings[0].code, "malformed-label")
        self.assertIn("found 4", findings[0].detail)

    def test_a_box_past_the_edge_is_kept_but_flagged(self):
        # Spec §6c labels partial objects to their visible extent, so a box hanging
        # off the frame is a labelling decision worth surfacing, not a parse failure.
        rows, findings = self.parse("person 0.02 0.5 0.2 0.3\n")

        self.assertEqual(len(rows), 1)
        self.assertEqual(findings[0].code, "box-past-edge")
        self.assertEqual(findings[0].severity, fsv.SEVERITY_WARNING)

    def test_comments_and_blank_lines_are_ignored(self):
        rows, findings = self.parse("# a note\n\n   \nperson 0.5 0.5 0.2 0.2\n")

        self.assertEqual(len(rows), 1)
        self.assertEqual(findings, [])

    def test_every_problem_is_reported_not_just_the_first(self):
        _rows, findings = self.parse("phone 0.5 0.5 0.2 0.2\nperson 9 9 9 9\n")

        self.assertEqual(len(findings), 2)


class StructureTests(FrameSetTestCase):
    def test_a_well_formed_set_passes(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.4 0.5 0.2 0.4\ncell phone 0.8 0.6 0.05 0.09\n")
        builder.frame(_stem(3), "person 0.6 0.5 0.2 0.4\n")
        builder.frame(_stem(4), "")

        report = builder.validate(targets=["person", "cell phone"])

        self.assertTrue(report.ok, builder.errors(report))
        self.assertEqual(report.stats["frames"], 4)

    def test_a_missing_label_file_is_a_blocking_error(self):
        """Because it does not fail loudly downstream — it reads as 'nothing here'."""
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), label_file=False)

        report = builder.validate(targets=["person"])

        self.assertIn("missing-label", builder.errors(report))
        detail = next(row.detail for row in report.errors if row.code == "missing-label")
        self.assertIn("false positive", detail)

    def test_a_label_without_an_image_is_an_orphan(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        (builder.labels / f"{_stem(2)}.txt").write_text(
            "person 0.5 0.5 0.2 0.4\n", encoding="utf-8"
        )

        report = builder.validate(targets=["person"])

        self.assertIn("orphan-label", builder.errors(report))

    def test_a_bad_filename_is_reported_once_not_twice(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame("random-name", "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(3), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        self.assertIn("bad-filename", builder.errors(report))
        self.assertNotIn("orphan-label", builder.errors(report))

    def test_an_empty_directory_reports_no_frames(self):
        report = self.build().validate(targets=["person"])

        self.assertIn("no-frames", [row.code for row in report.errors])

    def test_a_flat_directory_is_accepted(self):
        """Most labelling tools write image and label side by side."""
        builder = self.build()
        flat = builder.root / "flat"
        flat.mkdir()
        (flat / f"{_stem(1)}.jpg").write_bytes(b"jpeg-bytes")
        (flat / f"{_stem(1)}.txt").write_text("person 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        (flat / f"{_stem(2)}.jpg").write_bytes(b"jpeg-bytes")
        (flat / f"{_stem(2)}.txt").write_text("", encoding="utf-8")

        report = fsv.validate_frame_set(flat, targets=["person"], probe=builder.probe)

        self.assertTrue(report.ok, [row.code for row in report.errors])

    def test_metadata_beside_the_frames_is_not_mistaken_for_a_label(self):
        """A flat layout puts targets.txt next to the labels, where it looks like one."""
        builder = self.build()
        flat = builder.root / "flat"
        flat.mkdir()
        (flat / f"{_stem(1)}.jpg").write_bytes(b"jpeg-bytes")
        (flat / f"{_stem(1)}.txt").write_text("person 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        (flat / "targets.txt").write_text("person\n", encoding="utf-8")

        report = fsv.validate_frame_set(flat, targets=["person"], probe=builder.probe)

        self.assertNotIn("orphan-label", [row.code for row in report.errors])

    def test_an_images_directory_without_a_labels_directory_is_named_as_the_problem(self):
        builder = self.build()
        (builder.labels / f"{_stem(1)}.txt").unlink(missing_ok=True)
        builder.labels.rmdir()
        (builder.images / f"{_stem(1)}.jpg").write_bytes(b"jpeg-bytes")

        report = builder.validate(targets=["person"])

        self.assertIn("no-labels-directory", builder.errors(report))


class FrameQualityTests(FrameSetTestCase):
    def test_a_frame_below_the_height_floor_is_a_blocking_error(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n", quality=(64, 150.0, 200.0))
        builder.frame(_stem(3), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        self.assertIn("too-small", builder.errors(report))
        self.assertEqual(report.stats["too_small"], [f"{_stem(2)}.jpg"])

    def test_a_dark_frame_outside_the_dark_state_is_a_warning(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n", quality=(480, 11.8, 200.0))
        builder.frame(_stem(3), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        dark = [row for row in report.findings if row.code == "below-brightness-floor"]
        self.assertEqual(dark[0].severity, fsv.SEVERITY_WARNING)
        self.assertIn("excluded from every benchmark run", dark[0].detail)

    def test_a_dark_frame_inside_the_dark_state_is_information_not_a_warning(self):
        """Capturing the dark state is the point; the filter is what to remember."""
        builder = self.build()
        builder.frame(_stem(1, lighting="dark"), "", quality=(480, 11.8, 200.0))
        builder.frame(_stem(2, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(3, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        dark = [row for row in report.findings if row.code == "below-brightness-floor"]
        self.assertEqual(dark[0].severity, fsv.SEVERITY_INFO)
        self.assertIn("--min-brightness", dark[0].detail)
        self.assertNotIn("below-brightness-floor", [row.code for row in report.warnings])

    def test_an_unreadable_frame_is_a_blocking_error(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n")
        builder.quality[_stem(2)] = None  # probe reports it as undecodable

        report = builder.validate(targets=["person"])

        self.assertIn("unreadable-frame", builder.errors(report))


class CoverageTests(FrameSetTestCase):
    def test_a_target_that_was_never_labelled_is_a_blocking_error(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person", "cell phone"])

        self.assertIn("target-never-labelled", builder.errors(report))

    def test_a_target_outside_the_model_vocabulary_is_a_blocking_error(self):
        """The class cannot be detected by any model, so capture cannot fix it."""
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person", "brochure"])

        self.assertIn("target-not-in-vocabulary", builder.errors(report))

    def test_a_lighting_state_with_no_frames_for_a_class_is_a_warning(self):
        builder = self.build()
        builder.frame(_stem(1, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(3, lighting="dim"), "")

        report = builder.validate(targets=["person"])

        holes = [row for row in report.findings if row.code == "coverage-hole"]
        self.assertEqual(len(holes), 1)
        self.assertIn("dim", holes[0].detail)

    def test_the_coverage_matrix_is_counted_per_class_and_lighting_state(self):
        builder = self.build()
        builder.frame(_stem(1, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2, lighting="dim"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(3, lighting="dim"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(4, lighting="lit"), "")

        report = builder.validate(targets=["person"])

        self.assertEqual(report.stats["coverage"]["person"], {"lit": 1, "dim": 2, "dark": 0})

    def test_an_undeclared_class_is_information_not_a_failure(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "chair 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        self.assertTrue(report.ok, builder.errors(report))
        self.assertIn("undeclared-classes", builder.codes(report))


class NegativeFrameTests(FrameSetTestCase):
    def test_a_set_with_no_target_free_frames_is_a_blocking_error(self):
        """Precision on such a set is 1.00 by construction and means nothing."""
        builder = self.build()
        for n in (1, 2, 3, 4):
            builder.frame(_stem(n), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        self.assertIn("no-negatives", builder.errors(report))
        detail = next(row.detail for row in report.errors if row.code == "no-negatives")
        self.assertIn("not computable", detail)

    def test_an_empty_label_file_counts_as_a_negative(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "")

        report = builder.validate(targets=["person"])

        self.assertEqual(report.stats["negative_frames"], 1)
        self.assertEqual(report.stats["empty_label_files"], 1)

    def test_a_missing_label_file_is_not_reported_as_an_empty_one(self):
        """Both are target-free; only one means anyone labelled anything.

        Found by running the validator over `memory/snapshots`, which has no label
        files at all. The summary said "59 empty label file(s)" while the findings
        said "no label file" 59 times -- turning a set with no labels into one that
        merely looked thoroughly annotated as background. The two need different
        fixes, so they are counted apart.
        """
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "", label_file=False)

        report = builder.validate(targets=["person"])

        self.assertEqual(report.stats["missing_label_files"], 1)
        self.assertEqual(report.stats["empty_label_files"], 0)
        # Still a negative: a missing file reads as "nothing here" too, which is
        # exactly why it is an error rather than a silent skip.
        self.assertEqual(report.stats["negative_frames"], 1)

    def test_a_distractor_only_frame_counts_as_a_negative(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "chair 0.5 0.5 0.2 0.4\n")

        report = builder.validate(targets=["person"])

        self.assertEqual(report.stats["distractor_only_frames"], 1)
        self.assertEqual(report.stats["empty_label_files"], 0)
        self.assertEqual(report.stats["negative_frames"], 1)

    def test_too_few_negatives_is_a_warning_not_an_error(self):
        builder = self.build()
        for n in range(1, 6):
            builder.frame(_stem(n), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(6), "")

        report = builder.validate(targets=["person"])

        self.assertIn("few-negatives", builder.codes(report))
        self.assertNotIn("few-negatives", builder.errors(report))

    def test_a_lighting_state_with_frames_but_no_negatives_is_flagged(self):
        builder = self.build()
        builder.frame(_stem(1, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2, lighting="dark"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(3, lighting="lit"), "")
        builder.frame(_stem(4, lighting="lit"), "")

        report = builder.validate(targets=["person"])

        flagged = [row for row in report.findings if row.code == "negatives-missing-state"]
        self.assertIn("dark", flagged[0].detail)

    def test_without_declared_targets_the_negative_checks_stay_silent(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "person 0.5 0.5 0.2 0.4\n")

        report = builder.validate()

        self.assertIn("no-targets-declared", builder.codes(report))
        self.assertNotIn("no-negatives", builder.codes(report))


class ReportTests(FrameSetTestCase):
    def test_the_report_renders_the_coverage_matrix(self):
        builder = self.build()
        builder.frame(_stem(1, lighting="lit"), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2, lighting="dim"), "")

        text = fsv.format_validation_report(builder.validate(targets=["person"]))

        self.assertIn("coverage (frames per target class x lighting):", text)
        self.assertIn("lit", text)
        self.assertIn("dark", text)

    def test_a_passing_set_says_explicitly_that_nothing_was_measured(self):
        builder = self.build()
        builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        builder.frame(_stem(2), "")

        text = fsv.format_validation_report(builder.validate(targets=["person"]))

        self.assertIn("No blocking problems found", text)
        self.assertIn("no precision has been measured", text)

    def test_a_failing_set_says_how_many_problems_block_it(self):
        builder = self.build()
        builder.frame(_stem(1), label_file=False)
        builder.frame(_stem(2), label_file=False)
        builder.frame(_stem(3), "")

        text = fsv.format_validation_report(builder.validate(targets=["person"]))

        self.assertIn("blocking problem(s)", text)

    def test_long_finding_lists_are_truncated_rather_than_flooding_the_terminal(self):
        builder = self.build()
        for n in range(1, 21):
            builder.frame(_stem(n), label_file=False)
        builder.frame(_stem(21), "person 0.5 0.5 0.2 0.4\n")

        text = fsv.format_validation_report(builder.validate(targets=["person"]), limit=5)

        self.assertIn("errors (20):", text)
        self.assertIn("... and 15 more", text)


class ValidateFramesCommandTests(unittest.TestCase):
    """The command, not the library: exit code and behaviour on a bad path."""

    def setUp(self) -> None:
        _install_stubs()
        self.main = importlib.import_module("main")
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.builder = _FrameSetBuilder(Path(holder.name))

    def run_command(self, **kwargs):
        output = io.StringIO()
        defaults = {
            "root": str(self.builder.root),
            "targets": None,
            "vocab": None,
            "min_brightness": 90.0,
            "json_out": None,
        }
        # The command uses the real image probe, which needs real JPEGs; the suite's
        # cv2 stub returns None from `imread`, so every frame would be reported
        # unreadable. What is under test here is the command -- exit codes and
        # argument plumbing -- not decoding, which the builder covers separately.
        with (
            mock.patch.object(
                self.main.frame_set_validation,
                "probe_frame",
                return_value=(480, 150.0, 200.0),
            ),
            redirect_stdout(output),
        ):
            self.main.cmd_validate_frames(**{**defaults, **kwargs})
        return output.getvalue()

    def test_a_missing_frame_set_exits_non_zero(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_command(root=str(self.builder.root / "absent"))

        self.assertEqual(caught.exception.code, 1)

    def test_validation_failures_exit_non_zero(self):
        self.builder.frame(_stem(1), label_file=False)
        self.builder.frame(_stem(2), "")

        with self.assertRaises(SystemExit) as caught:
            self.run_command()

        self.assertEqual(caught.exception.code, 1)

    def test_a_well_formed_set_exits_zero_and_reports_what_it_checked(self):
        self.builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        self.builder.frame(_stem(2), "")
        (self.builder.root / "targets.txt").write_text("person\n", encoding="utf-8")

        text = self.run_command()  # must not raise

        self.assertIn("Targets    : 1 declared", text)
        self.assertIn("Vocabulary : 80 classes", text)

    def test_a_declared_target_file_is_used_when_present(self):
        self.builder.frame(_stem(1), "chair 0.5 0.5 0.2 0.4\n")
        self.builder.frame(_stem(2), "")
        (self.builder.root / "targets.txt").write_text("# only chairs\nchair\n", encoding="utf-8")

        text = self.run_command()

        self.assertIn("1 declared", text)
        self.assertIn("chair", text)

    def test_an_override_vocabulary_replaces_coco(self):
        vocab_path = self.builder.root / "vocab.txt"
        vocab_path.write_text("widget\ngadget\n", encoding="utf-8")
        self.builder.frame(_stem(1), "widget 0.5 0.5 0.2 0.4\n")
        self.builder.frame(_stem(2), "")

        text = self.run_command(vocab=str(vocab_path), targets=None)

        self.assertIn("Vocabulary : 2 classes", text)

    def test_json_output_is_written_when_asked_for(self):
        self.builder.frame(_stem(1), "person 0.5 0.5 0.2 0.4\n")
        self.builder.frame(_stem(2), "")
        out = self.builder.root / "report.json"

        text = self.run_command(json_out=str(out), targets=None)

        self.assertIn("report.json", text)
        self.assertIn('"ok": true', out.read_text(encoding="utf-8"))
        self.assertIn('"frames": 2', out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
