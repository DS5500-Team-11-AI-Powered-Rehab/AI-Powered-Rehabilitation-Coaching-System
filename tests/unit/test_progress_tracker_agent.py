"""
Unit Tests — Progress Tracker Agent (pure logic only)

Covers:
  - progress_tracker: analyze_pain_trend, analyze_quality_trend,
                      analyze_mistake_trend, analyze_exercise_progression,
                      format_pain_trend
  - upstream_adapter: _build_exercise_record, validate_inputs,
                      merge_to_patient_context

LLM-coupled code (ProgressTrackerAgent.generate_progress_report) is
intentionally excluded.
"""

import pytest

from progress_tracker_agent.progress_tracker import (
    analyze_pain_trend,
    analyze_quality_trend,
    analyze_mistake_trend,
    analyze_exercise_progression,
    format_pain_trend,
)
from progress_tracker_agent.upstream_adapter import (
    merge_to_patient_context,
    validate_inputs,
    _build_exercise_record,
)
from progress_tracker_agent.schemas import RehabPhase, ConditionCategory


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_phases():
    """Three rehab phases spanning acute → early → mid with improving metrics."""
    return [
        {
            "phase_summary": {"rehab_phase": "acute", "pain_level": 8},
            "exercises": [
                {
                    "exercise_name": "Quad Set",
                    "avg_quality": 0.5,
                    "mistakes": [{"type": "forward lean", "occurrences": 5}],
                }
            ],
        },
        {
            "phase_summary": {"rehab_phase": "early", "pain_level": 5},
            "exercises": [
                {
                    "exercise_name": "Mini Squat",
                    "avg_quality": 0.65,
                    "mistakes": [
                        {"type": "forward lean", "occurrences": 4},
                        {"type": "knee valgus", "occurrences": 3},
                    ],
                }
            ],
        },
        {
            "phase_summary": {"rehab_phase": "mid", "pain_level": 3},
            "exercises": [
                {
                    "exercise_name": "Squat",
                    "avg_quality": 0.82,
                    "mistakes": [],
                }
            ],
        },
    ]


@pytest.fixture
def base_payload():
    return {
        "coaching_event": {
            "exercise": {"name": "squat", "confidence": 0.9},
            "mistake": {
                "type": "forward lean",
                "confidence": 0.5,
                "duration_seconds": 3.0,
                "persistence_rate": 0.3,
                "occurrences": 5,
            },
            "metrics": {"speed_rps": 0.9, "rom_level": 2},
            "quality_score": 0.45,
            "severity": "medium",
            "is_recoaching": False,
            "session_time_minutes": 1.5,
            "tier": "tier_2",
            "routing_reason": "test",
        },
        "session_id": "sess_001",
        "coaching_history": [],
    }


@pytest.fixture
def base_profile():
    return {
        "condition": "knee osteoarthritis",
        "condition_category": "knee",
        "rehab_phase": "mid",
        "pain_level": 4,
        "weeks_into_rehab": 10,
    }


# ---------------------------------------------------------------------------
# TestAnalyzePainTrend
# ---------------------------------------------------------------------------

class TestAnalyzePainTrend:
    """progress_tracker.analyze_pain_trend — delta = last − first."""

    def test_improving_pain(self, three_phases):
        result = analyze_pain_trend(three_phases)
        assert result["trend"] == "improving"
        assert result["first"] == 8
        assert result["latest"] == 3
        assert result["total_change"] == -5

    def test_worsening_pain(self):
        phases = [
            {"phase_summary": {"pain_level": 3}, "exercises": []},
            {"phase_summary": {"pain_level": 6}, "exercises": []},
        ]
        result = analyze_pain_trend(phases)
        assert result["trend"] == "worsening"

    def test_stable_pain(self):
        # delta = -1, which is NOT < -1, so stable
        phases = [
            {"phase_summary": {"pain_level": 5}, "exercises": []},
            {"phase_summary": {"pain_level": 4}, "exercises": []},
        ]
        result = analyze_pain_trend(phases)
        assert result["trend"] == "stable"


# ---------------------------------------------------------------------------
# TestAnalyzeQualityTrend
# ---------------------------------------------------------------------------

class TestAnalyzeQualityTrend:
    """progress_tracker.analyze_quality_trend."""

    def test_improving_quality(self, three_phases):
        result = analyze_quality_trend(three_phases)
        assert result["trend"] == "improving"
        assert result["first"] == pytest.approx(0.5, abs=0.01)
        assert result["latest"] == pytest.approx(0.82, abs=0.01)

    def test_no_exercises_gives_none_values(self):
        phases = [{"phase_summary": {"pain_level": 5}, "exercises": []}]
        result = analyze_quality_trend(phases)
        assert result["values_per_phase"] == [None]
        assert result["trend"] == "stable"
        assert result["first"] is None


# ---------------------------------------------------------------------------
# TestAnalyzeMistakeTrend
# ---------------------------------------------------------------------------

class TestAnalyzeMistakeTrend:
    """progress_tracker.analyze_mistake_trend."""

    def test_persistent_mistakes_appear_in_two_or_more_phases(self, three_phases):
        result = analyze_mistake_trend(three_phases)
        # "forward lean" appears in acute + early = 2 phases → persistent
        assert "forward lean" in result["persistent"]
        # "knee valgus" only in early = 1 phase → not persistent
        assert "knee valgus" not in result["persistent"]

    def test_ranked_by_total_occurrences(self, three_phases):
        result = analyze_mistake_trend(three_phases)
        # forward lean: 5+4 = 9, knee valgus: 3
        assert result["ranked"][0] == ("forward lean", 9)


# ---------------------------------------------------------------------------
# TestAnalyzeExerciseProgression
# ---------------------------------------------------------------------------

class TestAnalyzeExerciseProgression:
    """progress_tracker.analyze_exercise_progression."""

    def test_returns_phase_exercise_pairs(self, three_phases):
        result = analyze_exercise_progression(three_phases)
        assert result == ["acute: Quad Set", "early: Mini Squat", "mid: Squat"]


# ---------------------------------------------------------------------------
# TestFormatPainTrend
# ---------------------------------------------------------------------------

class TestFormatPainTrend:
    """progress_tracker.format_pain_trend."""

    def test_format_includes_values_trend_and_change(self):
        trend_data = {"values": [8, 5, 3], "trend": "improving", "total_change": -5}
        result = format_pain_trend(trend_data)
        assert "8" in result and "5" in result and "3" in result
        assert "improving" in result.lower()
        assert "-5" in result


# ---------------------------------------------------------------------------
# TestBuildExerciseRecord
# ---------------------------------------------------------------------------

class TestBuildExerciseRecord:
    """upstream_adapter._build_exercise_record — difficulty and completed flags."""

    def _make_event(self, quality, severity="medium", occurrences=5, duration=3.0):
        # _build_exercise_record receives the inner coaching_event dict directly
        return {
            "exercise": {"name": "squat", "confidence": 0.9},
            "mistake": {
                "type": "forward lean",
                "confidence": 0.5,
                "duration_seconds": duration,
                "persistence_rate": 0.3,
                "occurrences": occurrences,
            },
            "metrics": {"speed_rps": 0.8, "rom_level": 2},
            "quality_score": quality,
            "severity": severity,
            "is_recoaching": False,
            "session_time_minutes": 1.0,
            "tier": "tier_2",
            "routing_reason": "test",
        }

    def test_low_quality_is_too_hard(self):
        rec = _build_exercise_record(self._make_event(quality=0.45, severity="medium"))
        assert rec.difficulty_feedback.startswith("too hard")
        assert rec.completed is True

    def test_medium_quality_is_ok_minor(self):
        rec = _build_exercise_record(self._make_event(quality=0.6, severity="medium"))
        assert rec.difficulty_feedback.startswith("ok — minor") or "ok" in rec.difficulty_feedback

    def test_high_quality_is_ok(self):
        rec = _build_exercise_record(self._make_event(quality=0.8, severity="medium"))
        assert rec.difficulty_feedback.startswith("ok")

    def test_high_severity_low_quality_not_completed(self):
        rec = _build_exercise_record(self._make_event(quality=0.3, severity="high"))
        assert rec.completed is False

    def test_high_severity_acceptable_quality_still_completed(self):
        rec = _build_exercise_record(self._make_event(quality=0.45, severity="high"))
        assert rec.completed is True


# ---------------------------------------------------------------------------
# TestValidateInputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    """upstream_adapter.validate_inputs — returns list of warning strings."""

    def test_missing_coaching_event_key(self, base_profile):
        warnings = validate_inputs({}, base_profile)
        joined = " ".join(warnings)
        assert "MISSING" in joined or "missing" in joined.lower()
        assert "coaching_event" in joined

    def test_unknown_rehab_phase_warns(self, base_payload):
        bad_profile = {
            "condition": "knee osteoarthritis",
            "condition_category": "knee",
            "rehab_phase": "unknown",
            "pain_level": 4,
            "weeks_into_rehab": 10,
        }
        warnings = validate_inputs(base_payload, bad_profile)
        joined = " ".join(warnings)
        assert "unknown" in joined.lower()

    def test_clean_inputs_return_no_warnings(self, base_payload, base_profile):
        warnings = validate_inputs(base_payload, base_profile)
        assert warnings == []


# ---------------------------------------------------------------------------
# TestMergeToPatientContext
# ---------------------------------------------------------------------------

class TestMergeToPatientContext:
    """upstream_adapter.merge_to_patient_context — assembles PatientContext dataclass."""

    def test_context_fields(self, base_payload, base_profile):
        ctx = merge_to_patient_context(base_payload, base_profile)

        assert ctx.patient_id == "sess_001"
        assert ctx.rehab_phase == RehabPhase.MID
        assert ctx.condition_category == ConditionCategory.KNEE
        assert ctx.pain_level == 4

        # Exercise name should be title-cased
        assert ctx.recent_exercises[0].name == "Squat"

        # quality=0.45 < 0.5 → "too hard"
        assert ctx.recent_exercises[0].difficulty_feedback.startswith("too hard")

        # Patient message must include the mistake type
        assert "forward lean" in ctx.patient_message
