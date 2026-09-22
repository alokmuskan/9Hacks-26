"""Tests for the detection review: sampling, verdict parsing, scoring and reporting.

The behaviour that matters is the arithmetic of an accuracy claim. An unreviewed
box must stay unreviewed rather than defaulting to correct, a partial review must
say it is a sample, and a miss somebody noticed must not become a recall figure.
"""

import json
import tempfile
import unittest
from pathlib import Path

import detection_review as dr


def _detection(
    index: int, label: str = "person", confidence: float = 0.5, frame: str = "frames/a.jpg"
) -> dr.Detection:
    return dr.Detection(
        index=index,
        frame=frame,
        label=label,
        confidence=confidence,
        bbox=(0.0, 0.0, 10.0, 10.0),
    )


class WilsonIntervalTests(unittest.TestCase):
    def test_a_perfect_small_sample_does_not_claim_certainty(self):
        """The normal approximation gives zero width at 100%, which is the common case here."""
        low, high = dr.wilson_interval(10, 10)

        self.assertLess(low, 1.0)
        self.assertAlmostEqual(high, 1.0, places=6)

    def test_a_half_sample_is_centred_near_a_half(self):
        low, high = dr.wilson_interval(5, 10)

        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual((low + high) / 2, 0.5, places=2)

    def test_the_interval_narrows_as_the_sample_grows(self):
        small = dr.wilson_interval(8, 10)
        large = dr.wilson_interval(80, 100)

        self.assertGreater(large[0], small[0])
        self.assertLess(large[1], small[1])

    def test_an_empty_sample_spans_everything_rather_than_raising(self):
        self.assertEqual(dr.wilson_interval(0, 0), (0.0, 1.0))


class SelectionTests(unittest.TestCase):
    def test_no_limit_keeps_every_detection(self):
        rows = [_detection(i) for i in range(1, 6)]

        self.assertEqual(len(dr.select_detections(rows, 0)), 5)
        self.assertEqual(len(dr.select_detections(rows, 5)), 5)
        self.assertEqual(len(dr.select_detections(rows, 99)), 5)

    def test_a_limited_review_spreads_across_confidence_not_just_the_top(self):
        """Reviewing only the most confident boxes would report the easiest precision."""
        rows = [_detection(i, confidence=1.0 - i / 10.0) for i in range(1, 11)]

        picked = dr.select_detections(rows, 3)

        self.assertEqual(len(picked), 3)
        self.assertLess(min(d.confidence for d in picked), 0.5)

    def test_the_picks_are_distinct(self):
        rows = [_detection(i, confidence=1.0 - i / 100.0) for i in range(1, 51)]

        picked = dr.select_detections(rows, 7)

        self.assertEqual(len({d.index for d in picked}), 7)


class VerdictParsingTests(unittest.TestCase):
    def test_the_page_output_is_read(self):
        payload = {"verdicts": {"1": "y", "2": "n"}, "missed": {"1": "bottle, cup"}}

        verdicts = dr.parse_verdicts(payload)

        self.assertEqual([(v.index, v.correct) for v in verdicts], [(1, True), (2, False)])
        self.assertEqual(verdicts[0].missed, ("bottle", "cup"))

    def test_a_hand_written_mapping_is_read_too(self):
        verdicts = dr.parse_verdicts({"3": True, "4": 0})

        self.assertEqual([(v.index, v.correct) for v in verdicts], [(3, True), (4, False)])

    def test_an_unreadable_answer_is_skipped_rather_than_assumed_correct(self):
        """Guessing here would inflate the score, which is the only thing it exists to prevent."""
        verdicts = dr.parse_verdicts({"1": "y", "2": "maybe", "x": "y"})

        self.assertEqual([(v.index, v.correct) for v in verdicts], [(1, True)])

    def test_a_non_mapping_is_not_read_as_a_success(self):
        for payload in ("y", ["1"], None, 5):
            with self.subTest(payload=payload):
                self.assertEqual(dr.parse_verdicts(payload), [])


class ScoringTests(unittest.TestCase):
    def test_a_full_review_is_a_census_with_no_interval(self):
        detections = [_detection(i) for i in range(1, 5)]
        verdicts = [dr.Verdict(i, i != 4) for i in range(1, 5)]

        score = dr.score(detections, verdicts)

        self.assertEqual(score.reviewed, 4)
        self.assertEqual(score.correct, 3)
        self.assertAlmostEqual(score.precision, 0.75, places=6)
        self.assertIsNone(score.interval)
        self.assertAlmostEqual(score.coverage, 1.0, places=6)

    def test_a_partial_review_is_reported_as_a_sample(self):
        detections = [_detection(i) for i in range(1, 11)]
        verdicts = [dr.Verdict(i, True) for i in range(1, 6)]

        score = dr.score(detections, verdicts)

        self.assertEqual(score.reviewed, 5)
        self.assertAlmostEqual(score.coverage, 0.5, places=6)
        self.assertIsNotNone(score.interval)
        assert score.interval is not None
        self.assertLess(score.interval[0], 1.0)

    def test_unreviewed_boxes_are_not_counted_as_correct(self):
        detections = [_detection(i) for i in range(1, 11)]
        verdicts = [dr.Verdict(1, True)]

        score = dr.score(detections, verdicts)

        self.assertEqual(score.reviewed, 1)
        self.assertEqual(score.correct, 1)
        self.assertEqual(score.total, 10)

    def test_precision_is_reported_per_class(self):
        detections = [
            _detection(1, "person"),
            _detection(2, "person"),
            _detection(3, "surfboard"),
        ]
        verdicts = [dr.Verdict(1, True), dr.Verdict(2, False), dr.Verdict(3, False)]

        score = dr.score(detections, verdicts)
        by_label = {entry.label: entry for entry in score.per_class}

        self.assertAlmostEqual(by_label["person"].precision, 0.5, places=6)
        self.assertEqual(by_label["person"].reviewed, 2)
        self.assertEqual(by_label["surfboard"].reviewed, 1)
        self.assertAlmostEqual(by_label["surfboard"].precision, 0.0, places=6)

    def test_a_verdict_for_an_unknown_box_is_ignored(self):
        detections = [_detection(1)]
        verdicts = [dr.Verdict(1, True), dr.Verdict(99, False)]

        score = dr.score(detections, verdicts)

        self.assertEqual(score.reviewed, 1)
        self.assertEqual(score.wrong, 0)

    def test_the_same_miss_on_several_boxes_of_one_frame_is_counted_once(self):
        detections = [_detection(1, frame="frames/a.jpg"), _detection(2, frame="frames/a.jpg")]
        verdicts = [
            dr.Verdict(1, True, missed=("bottle",)),
            dr.Verdict(2, True, missed=("Bottle",)),
        ]

        score = dr.score(detections, verdicts)

        self.assertEqual(len(score.misses), 1)

    def test_misses_on_different_frames_stay_separate(self):
        detections = [_detection(1, frame="frames/a.jpg"), _detection(2, frame="frames/b.jpg")]
        verdicts = [
            dr.Verdict(1, True, missed=("bottle",)),
            dr.Verdict(2, True, missed=("bottle",)),
        ]

        score = dr.score(detections, verdicts)

        self.assertEqual(len(score.misses), 2)


class ReportTests(unittest.TestCase):
    def test_an_unreviewed_list_gives_instructions_rather_than_a_score(self):
        text = dr.format_score(dr.score([_detection(1)], []))

        self.assertIn("Nothing reviewed yet", text)
        self.assertNotIn("%", text.split("\n")[0])

    def test_the_census_case_says_there_is_no_sampling_error(self):
        detections = [_detection(i) for i in range(1, 4)]
        score = dr.score(detections, [dr.Verdict(i, True) for i in range(1, 4)])

        text = dr.format_score(score)

        self.assertIn("census", text)
        self.assertIn("no sampling error", text)

    def test_a_thin_review_is_warned_about(self):
        detections = [_detection(i) for i in range(1, 11)]
        score = dr.score(detections, [dr.Verdict(1, True)])

        text = dr.format_score(score)

        self.assertIn("[warn]", text)
        self.assertIn("not a", text)

    def test_a_noted_miss_is_reported_without_becoming_a_recall_figure(self):
        detections = [_detection(1, frame="frames/a.jpg")]
        score = dr.score(detections, [dr.Verdict(1, True, missed=("bottle",))])

        text = dr.format_score(score)

        self.assertIn("a.jpg: bottle", text)
        self.assertIn("not a recall figure", text)
        self.assertNotIn("recall:", text)

    def test_the_scope_of_the_number_is_stated_on_the_output(self):
        detections = [_detection(1)]
        score = dr.score(detections, [dr.Verdict(1, True)])

        text = dr.format_score(score)

        self.assertIn("precision, not recall", text)
        self.assertIn("these frames only", text)


class FingerprintTests(unittest.TestCase):
    """A verdict set must not be scoreable against a list it was not recorded against."""

    def test_the_same_list_always_gives_the_same_id(self):
        rows = [_detection(1), _detection(2, label="bottle")]

        self.assertEqual(dr.fingerprint(rows), dr.fingerprint(list(rows)))

    def test_a_changed_confidence_changes_the_id(self):
        self.assertNotEqual(
            dr.fingerprint([_detection(1, confidence=0.5)]),
            dr.fingerprint([_detection(1, confidence=0.51)]),
        )

    def test_a_reordered_list_changes_the_id(self):
        """Reordering is what a rebuild does, and it silently renumbers every box."""
        self.assertNotEqual(
            dr.fingerprint([_detection(1, label="person"), _detection(2, label="bottle")]),
            dr.fingerprint([_detection(1, label="bottle"), _detection(2, label="person")]),
        )

    def test_matching_ids_are_accepted(self):
        rows = [_detection(1)]

        self.assertIsNone(
            dr.fingerprint_mismatch(dr.fingerprint(rows), {"fingerprint": dr.fingerprint(rows)})
        )

    def test_verdicts_from_another_list_are_refused_by_name(self):
        message = dr.fingerprint_mismatch("aaaaaaaaaaaaaaaa", {"fingerprint": "bbbbbbbbbbbbbbbb"})

        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("different detection list", message)

    def test_verdicts_with_no_id_are_allowed_through(self):
        """It cannot be checked, and refusing an unchecked file invents a problem."""
        self.assertIsNone(dr.fingerprint_mismatch("aaaaaaaaaaaaaaaa", {"verdicts": {"1": "y"}}))

    def test_a_list_with_no_id_never_blocks_a_score(self):
        self.assertIsNone(dr.fingerprint_mismatch(None, {"fingerprint": "zzzz"}))


class WriteReviewTests(unittest.TestCase):
    def test_the_written_list_carries_the_id_the_page_echoes(self):
        rows = [dr.ReviewRow(detection=_detection(1), thumb_b64="")]

        with tempfile.TemporaryDirectory() as tmp:
            detections_path, page_path = dr.write_review(rows, out_dir=tmp, source="frames/*.jpg")
            payload = json.loads(Path(detections_path).read_text(encoding="utf-8"))
            page = Path(page_path).read_text(encoding="utf-8")

        self.assertEqual(payload["fingerprint"], dr.fingerprint([_detection(1)]))
        self.assertIn(payload["fingerprint"], page)


class PreloadTests(unittest.TestCase):
    """A correction pass must start from the earlier answers, never from a guess."""

    def _page(self, existing: dict[int, bool]) -> str:
        rows = [dr.ReviewRow(detection=_detection(1), thumb_b64="")]
        return dr.render_page(rows, source="x", existing=existing)

    def test_recorded_verdicts_are_read_back_with_both_answers(self):
        payload = {"verdicts": {"1": "y", "2": "n", "3": "skip"}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verdicts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            existing = dr.load_existing(path)

        self.assertEqual(existing, {1: True, 2: False})

    def test_a_missing_or_broken_file_preloads_nothing_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dr.load_existing(Path(tmp) / "absent.json"), {})
            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            self.assertEqual(dr.load_existing(broken), {})

    def test_the_page_starts_with_the_previous_answers_already_marked(self):
        page = self._page({1: False})

        self.assertIn('const verdicts = {"1": false}', page)

    def test_a_blank_page_is_the_default_when_nothing_was_recorded(self):
        self.assertIn("const verdicts = {};", self._page({}))

    def test_write_review_passes_the_earlier_answers_through(self):
        rows = [dr.ReviewRow(detection=_detection(1), thumb_b64="")]

        with tempfile.TemporaryDirectory() as tmp:
            _, page_path = dr.write_review(rows, out_dir=tmp, source="x", existing={1: True})
            page = Path(page_path).read_text(encoding="utf-8")

        self.assertIn('const verdicts = {"1": true}', page)


class RoundTripTests(unittest.TestCase):
    def test_a_detection_survives_the_json_round_trip(self):
        original = _detection(7, label="cell phone", confidence=0.42)

        restored = dr.Detection.from_json(json.loads(json.dumps(original.to_json())))

        self.assertEqual(restored, original)

    def test_a_verdict_file_parses_end_to_end(self):
        payload = json.dumps({"verdicts": {"1": "n"}, "missed": {"1": "cup"}})

        verdicts = dr.parse_verdicts(json.loads(payload))
        score = dr.score([_detection(1)], verdicts)

        self.assertEqual(score.wrong, 1)
        self.assertAlmostEqual(score.precision, 0.0, places=6)


class PageTests(unittest.TestCase):
    def _page(self) -> str:
        rows = [
            dr.ReviewRow(detection=_detection(1, label="person", confidence=0.9), thumb_b64="AAA"),
            dr.ReviewRow(detection=_detection(2, label="<script>x</script>"), thumb_b64="BBB"),
        ]
        return dr.render_page(rows, source="memory/snapshots/*.jpg")

    def test_every_box_gets_an_index_a_reviewer_can_refer_to(self):
        page = self._page()

        self.assertIn('data-index="1"', page)
        self.assertIn('data-index="2"', page)

    def test_a_class_label_cannot_inject_markup_into_the_page(self):
        page = self._page()

        self.assertNotIn("<script>x</script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_the_page_states_what_it_measures_and_what_it_does_not(self):
        page = self._page()

        self.assertIn("precision only", page)
        self.assertIn("cannot measure recall", page)
        self.assertIn("never as correct", page)

    def test_the_page_is_marked_noindex_because_it_shows_a_camera(self):
        self.assertIn('content="noindex', self._page())

    def test_the_page_offers_the_detection_indices_it_was_built_for(self):
        page = self._page()

        self.assertIn("reviewed 0 / 2", page)

    def test_the_page_echoes_the_detection_list_id_into_the_payload(self):
        rows = [dr.ReviewRow(detection=_detection(1), thumb_b64="")]

        page = dr.render_page(rows, source="x", list_id="feedfacefeedface")

        self.assertIn("feedfacefeedface", page)


if __name__ == "__main__":
    unittest.main()
