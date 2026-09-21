"""Tests for the recorded-session frame budget.

The behaviour under test is mostly *refusal*: a stage that was never timed must
not become a ``0.0``, a session that cannot be measured must be named rather than
dropped, and a session average must not be presented as the cost of an ordinary
frame. Those are the failure modes that would turn this report into a
reassuring set of numbers.
"""

import unittest

import frame_budget as fb


def _row(**overrides: object) -> dict:
    """A well-formed session row: 100 frames in 100 s, face and gaze both running."""
    aggregate: dict = {
        "session_id": "monitor-test-000000",
        "frames_total": 100,
        "avg_fps": 1.0,
        "moving_avg_fps": 1.0,
        "min_fps": 0.5,
        "max_fps": 2.0,
        "frames_with_faces": 50,
        "face_recognition_enabled": True,
        "detection_calls": 100,
        "avg_detection_latency_ms": 50.0,
        "gaze_enabled": True,
        "gaze_model_loaded": True,
        "gaze_inference_calls": 50,
        "gaze_inference_avg_ms": 200.0,
    }
    aggregate.update(overrides)
    return {
        "event_type": fb.SESSION_EVENT_TYPE,
        "session_id": aggregate["session_id"],
        "duration_sec": 100.0,
        "aggregate": aggregate,
    }


def _budget(**overrides: object) -> fb.SessionBudget:
    budget, reason = fb.build_budget(_row(**overrides))
    assert budget is not None, reason
    return budget


class PeriodTests(unittest.TestCase):
    def test_the_period_comes_from_the_recorded_average(self):
        budget, reason = fb.build_budget(_row())

        self.assertEqual(reason, "")
        assert budget is not None
        self.assertEqual(budget.frames_total, 100)
        self.assertAlmostEqual(budget.frame_period_ms, 1000.0, places=6)

    def test_a_stage_cost_is_calls_times_average_over_frames(self):
        """50 ms/call over 100 calls and 100 frames is 50 ms of every frame."""
        budget, _ = fb.build_budget(_row())

        assert budget is not None
        self.assertAlmostEqual(budget.face.ms or 0.0, 50.0, places=6)
        self.assertAlmostEqual(budget.gaze.ms or 0.0, 100.0, places=6)
        self.assertAlmostEqual(budget.residual_ms, 850.0, places=6)
        self.assertAlmostEqual(budget.face_share or 0.0, 0.05, places=6)
        self.assertAlmostEqual(budget.residual_share, 0.85, places=6)

    def test_a_session_without_a_usable_period_is_skipped_by_name(self):
        for overrides in ({"frames_total": 0}, {"avg_fps": 0}):
            with self.subTest(overrides=overrides):
                budget, reason = fb.build_budget(_row(**overrides))

                self.assertIsNone(budget)
                self.assertIn("monitor-test-000000", reason)
                self.assertIn("no measurable period", reason)

    def test_a_row_without_an_aggregate_is_not_a_session_at_all(self):
        budget_set = fb.collect_budgets([{"event_type": "chat_query"}] * 3)

        self.assertEqual(len(budget_set), 0)
        self.assertEqual(budget_set.skipped, ())


class UntimedStageTests(unittest.TestCase):
    """A missing measurement must never read as a fast one."""

    def test_face_detection_with_no_recorded_latency_is_not_zero(self):
        budget, _ = fb.build_budget(
            _row(detection_calls=100, avg_detection_latency_ms=0.0)
        )

        assert budget is not None
        self.assertIsNone(budget.face.ms)
        self.assertEqual(budget.face.state, "untimed")
        self.assertEqual(budget.face.cell, fb.NOT_RECORDED)
        self.assertNotEqual(budget.face.cell, "0.0")
        self.assertTrue(any("no latency recorded" in note for note in budget.notes))

    def test_gaze_with_no_recorded_latency_is_not_zero(self):
        budget, _ = fb.build_budget(_row(gaze_inference_calls=50, gaze_inference_avg_ms=0.0))

        assert budget is not None
        self.assertEqual(budget.gaze.state, "untimed")
        self.assertEqual(budget.gaze.cell, fb.NOT_RECORDED)

    def test_a_stage_that_did_not_run_renders_as_off_not_as_missing(self):
        """Face recognition off still counts calls and still times a no-op."""
        budget, _ = fb.build_budget(
            _row(face_recognition_enabled=False, detection_calls=100, avg_detection_latency_ms=0.01)
        )

        assert budget is not None
        self.assertEqual(budget.face.state, "disabled")
        self.assertEqual(budget.face.cell, fb.DISABLED)
        # The timer measuring nothing is expected here, so it is not a finding.
        self.assertFalse(any("no latency recorded" in note for note in budget.notes))

    def test_an_untimed_stage_keeps_the_session_out_of_the_stage_cohort(self):
        budget, _ = fb.build_budget(_row(avg_detection_latency_ms=0.0))

        assert budget is not None
        self.assertFalse(budget.fully_recorded)

    def test_recorded_stages_exceeding_the_period_is_reported(self):
        budget, _ = fb.build_budget(
            _row(frames_total=10, avg_fps=100.0, detection_calls=10, avg_detection_latency_ms=50.0)
        )

        assert budget is not None
        self.assertLess(budget.residual_ms, 0)
        self.assertTrue(any("cannot be trusted" in note for note in budget.notes))


class GazeNoteTests(unittest.TestCase):
    def test_gaze_that_never_ran_because_its_weights_are_missing_says_so(self):
        budget, _ = fb.build_budget(
            _row(gaze_model_loaded=False, gaze_inference_calls=0, gaze_inference_avg_ms=0.0)
        )

        assert budget is not None
        self.assertTrue(any("weights never loaded" in note for note in budget.notes))

    def test_gaze_that_never_ran_because_no_face_was_seen_says_so(self):
        """Gaze is estimated per detected face, so no faces means no gaze cost."""
        budget, _ = fb.build_budget(
            _row(frames_with_faces=0, gaze_inference_calls=0, gaze_inference_avg_ms=0.0)
        )

        assert budget is not None
        self.assertTrue(any("no face was detected" in note for note in budget.notes))

    def test_gaze_that_never_ran_with_faces_present_is_a_finding(self):
        budget, _ = fb.build_budget(
            _row(frames_with_faces=50, gaze_inference_calls=0, gaze_inference_avg_ms=0.0)
        )

        assert budget is not None
        self.assertTrue(any("no call recorded" in note for note in budget.notes))


class SpreadTests(unittest.TestCase):
    def test_a_session_whose_frames_are_mostly_fast_is_flagged(self):
        """185854-shaped: EMA 4.07 against a 0.82 mean is a session of stalls."""
        budget, _ = fb.build_budget(_row(avg_fps=0.82, moving_avg_fps=4.07, max_fps=178.5))

        assert budget is not None
        self.assertTrue(budget.stall_dominated)
        self.assertTrue(any("dominated by stalls" in note for note in budget.notes))

    def test_a_session_whose_frames_are_genuinely_slow_is_not_flagged(self):
        budget, _ = fb.build_budget(_row(avg_fps=0.76, moving_avg_fps=0.81, max_fps=30.72))

        assert budget is not None
        self.assertFalse(budget.stall_dominated)
        self.assertFalse(any("dominated by stalls" in note for note in budget.notes))

    def test_an_impossible_instantaneous_rate_is_called_unusable(self):
        """One session recorded 29537 fps between frames; quoting it as a fact would mislead."""
        budget, _ = fb.build_budget(_row(avg_fps=0.68, moving_avg_fps=23926.9, max_fps=29537.35))

        assert budget is not None
        self.assertTrue(any("not usable" in note for note in budget.notes))
        self.assertFalse(any("dominated by stalls" in note for note in budget.notes))


class PacingTests(unittest.TestCase):
    def test_pacing_is_unknown_when_the_cap_was_not_recorded(self):
        budget, _ = fb.build_budget(_row())

        assert budget is not None
        self.assertEqual(budget.pacing, "not-recorded")

    def test_a_period_at_the_cap_is_cap_bound_and_idle_time_is_named(self):
        budget, _ = fb.build_budget(_row(avg_fps=12.0, fps_cap=12))

        assert budget is not None
        self.assertEqual(budget.pacing, "cap-bound")
        self.assertTrue(any("idle pacing" in note for note in budget.notes))

    def test_a_period_far_above_the_cap_is_throughput_bound(self):
        budget, _ = fb.build_budget(_row(avg_fps=0.8, fps_cap=12))

        assert budget is not None
        self.assertEqual(budget.pacing, "throughput-bound")


class SummaryTests(unittest.TestCase):
    def test_the_cohort_excludes_sessions_where_nothing_was_measured(self):
        """Including them would pull the stage split toward the toggles, not the pipeline."""
        summary = fb.summarize(
            [
                _budget(),
                _budget(
                    session_id="monitor-test-alloff",
                    face_recognition_enabled=False,
                    detection_calls=100,
                    avg_detection_latency_ms=0.0,
                    gaze_enabled=False,
                    gaze_inference_calls=0,
                    gaze_inference_avg_ms=0.0,
                ),
            ]
        )

        self.assertEqual(summary.sessions, 2)
        self.assertEqual(summary.stages.sessions, 1)
        self.assertEqual(summary.stages.measured_face, 1)
        self.assertAlmostEqual(summary.stages.median_residual_share, 0.85, places=6)

    def test_each_stage_carries_its_own_count(self):
        """Face detection runs whenever face recognition is on; gaze also needs faces."""
        summary = fb.summarize(
            [
                _budget(),
                _budget(
                    session_id="monitor-test-faceonly",
                    gaze_inference_calls=0,
                    gaze_inference_avg_ms=0.0,
                ),
            ]
        )

        self.assertEqual(summary.stages.sessions, 2)
        self.assertEqual(summary.stages.measured_face, 2)
        self.assertEqual(summary.stages.measured_gaze, 1)

    def test_medians_are_used_so_one_session_cannot_drag_the_result(self):
        summary = fb.summarize(
            [_budget(), _budget(session_id="monitor-test-fast", avg_fps=10.0), _budget()]
        )

        self.assertAlmostEqual(summary.median_avg_fps, 1.0, places=6)
        self.assertEqual(summary.fps_range, (1.0, 10.0))

    def test_counts_of_stall_dominated_and_unusable_clock_sessions_are_reported(self):
        summary = fb.summarize(
            [
                _budget(avg_fps=0.82, moving_avg_fps=4.07, max_fps=178.5),
                _budget(
                    session_id="monitor-test-clock",
                    avg_fps=0.68,
                    moving_avg_fps=23926.9,
                    max_fps=29537.35,
                ),
            ]
        )

        self.assertEqual(summary.stall_dominated, 2)
        self.assertEqual(summary.implausible_rate, 1)

    def test_an_empty_summary_is_zeroed_rather_than_raising(self):
        summary = fb.summarize([])

        self.assertEqual(summary.sessions, 0)
        self.assertEqual(summary.stages.sessions, 0)
        self.assertIsNone(summary.stages.median_face_ms)


class CollectionTests(unittest.TestCase):
    def test_the_limit_keeps_the_newest_sessions(self):
        rows = [_row(session_id=f"monitor-test-{n:02d}") for n in range(5)]

        budget_set = fb.collect_budgets(rows, limit=2)

        self.assertEqual([b.session_id for b in budget_set.budgets], ["monitor-test-03", "monitor-test-04"])

    def test_an_unmeasured_session_is_named_in_the_report(self):
        rows = [_row(), _row(session_id="monitor-test-empty", frames_total=0)]

        budget_set = fb.collect_budgets(rows)
        text = fb.format_report(budget_set)

        self.assertEqual(len(budget_set), 1)
        self.assertIn("Skipped rows", text)
        self.assertIn("monitor-test-empty", text)


class ReportTests(unittest.TestCase):
    def _report(self) -> str:
        rows = [_row(), _row(session_id="monitor-test-alloff", face_recognition_enabled=False, detection_calls=100, avg_detection_latency_ms=0.0, gaze_enabled=False, gaze_inference_calls=0, gaze_inference_avg_ms=0.0)]
        return fb.format_report(fb.collect_budgets(rows))

    def test_the_report_states_that_object_detection_is_not_recorded(self):
        """The single most misleading omission: 'detection' in the log means faces."""
        text = self._report()

        self.assertIn("NOT recorded", text)
        self.assertIn("detector.detect()", text)

    def test_the_report_says_the_unattributed_share_is_a_remainder(self):
        self.assertIn("a remainder, not a measurement", self._report())

    def test_the_report_states_the_pacing_cap_is_absent(self):
        self.assertIn("fps_cap", self._report())

    def test_off_and_not_recorded_render_differently(self):
        lines = [line for line in self._report().splitlines() if line.startswith("monitor-test")]
        by_session = {line.split()[0]: line for line in lines}

        row = by_session["monitor-test-alloff"]
        self.assertIn(f" {fb.DISABLED} ", row)
        self.assertNotIn(fb.NOT_RECORDED, row)

    def test_no_sessions_produces_a_statement_rather_than_an_empty_table(self):
        text = fb.format_report(fb.collect_budgets([]))

        self.assertIn("No session in the log carries a usable aggregate", text)


if __name__ == "__main__":
    unittest.main()
