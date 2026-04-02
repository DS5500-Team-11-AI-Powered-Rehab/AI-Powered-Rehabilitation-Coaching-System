"""
Unit Tests — Coaching Agent (pure logic only)

Covers:
  - session_prompts: infer_quality_trend, infer_ok_aspects,
                     format_mistakes_text, format_ok_aspects_text
  - session_manager: ExerciseBuffer lifecycle and aggregation

LLM-coupled code (CoachingAgent.handle_event, SessionManager._generate_exercise_summary)
is intentionally excluded — no LangGraph/Ollama mocking needed.
"""

import pytest

from session_prompts import (
    infer_quality_trend,
    infer_ok_aspects,
    format_mistakes_text,
    format_ok_aspects_text,
)
from session_manager import ExerciseBuffer


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def one_event():
    return {
        "coaching_event": {
            "exercise": {"name": "squat", "confidence": 0.9},
            "mistake": {
                "type": "forward lean",
                "confidence": 0.4,
                "duration_seconds": 3.0,
                "persistence_rate": 0.3,
                "occurrences": 5,
            },
            "metrics": {"speed_rps": 0.9, "height_level": 3, "torso_rotation": 0.3},
            "quality_score": 0.7,
            "severity": "medium",
            "is_recoaching": False,
            "session_time_minutes": 1.0,
            "tier": "tier_2",
            "routing_reason": "test",
        }
    }


# ---------------------------------------------------------------------------
# TestInferQualityTrend
# ---------------------------------------------------------------------------

class TestInferQualityTrend:
    """session_prompts.infer_quality_trend — splits list in half, compares averages."""

    def test_empty_list_returns_stable(self):
        assert infer_quality_trend([]) == "stable"

    def test_single_element_returns_stable(self):
        assert infer_quality_trend([0.8]) == "stable"

    def test_improving_trend(self):
        # first half avg 0.4, second half avg 0.9 → delta +0.5 > 0.05
        assert infer_quality_trend([0.4, 0.4, 0.9, 0.9]) == "improving"

    def test_declining_trend(self):
        # first half avg 0.9, second half avg 0.4 → delta -0.5 < -0.05
        assert infer_quality_trend([0.9, 0.9, 0.4, 0.4]) == "declining"

    def test_within_threshold_is_stable(self):
        # delta = 0.62 - 0.60 = 0.02 → within ±0.05
        assert infer_quality_trend([0.6, 0.62]) == "stable"

    def test_just_above_threshold_is_improving(self):
        # delta = 0.56 - 0.50 = 0.06 → above +0.05
        assert infer_quality_trend([0.5, 0.56]) == "improving"


# ---------------------------------------------------------------------------
# TestInferOkAspects
# ---------------------------------------------------------------------------

class TestInferOkAspects:
    """session_prompts.infer_ok_aspects — checks speed/height/rotation thresholds."""

    def test_empty_events_returns_empty_list(self):
        assert infer_ok_aspects([]) == []

    def test_good_speed_appears_in_output(self):
        events = [{
            "coaching_event": {
                "metrics": {"speed_rps": 0.9, "height_level": 0, "torso_rotation": 0.6},
            }
        }]
        result = infer_ok_aspects(events)
        assert any("speed" in s.lower() or "movement" in s.lower() for s in result)

    def test_fallback_when_nothing_passes(self):
        events = [{
            "coaching_event": {
                "metrics": {"speed_rps": 0.0, "height_level": 0, "torso_rotation": 0.6},
            }
        }]
        result = infer_ok_aspects(events)
        assert len(result) == 1
        assert "effort" in result[0].lower() or "persistence" in result[0].lower()


# ---------------------------------------------------------------------------
# TestFormatHelpers
# ---------------------------------------------------------------------------

class TestFormatHelpers:
    """session_prompts format helpers — pure string formatting."""

    def test_format_mistakes_text_empty(self):
        assert format_mistakes_text([]) == "No significant mistakes detected."

    def test_format_mistakes_text_with_mistake(self):
        mistakes = [{
            "type": "knee valgus",
            "occurrences": 8,
            "avg_duration_s": 4.2,
            "severity": "high",
        }]
        result = format_mistakes_text(mistakes)
        assert "knee valgus" in result
        assert "8" in result          # occurrences
        assert "4.2" in result        # duration
        assert "high" in result       # severity

    def test_format_ok_aspects_text_empty(self):
        assert format_ok_aspects_text([]) == "Form was consistent overall."


# ---------------------------------------------------------------------------
# TestExerciseBuffer
# ---------------------------------------------------------------------------

class TestExerciseBuffer:
    """session_manager.ExerciseBuffer — lifecycle, empty summary, aggregation."""

    def test_lifecycle(self, one_event):
        buf = ExerciseBuffer()
        assert buf.is_empty() is True

        buf.add(one_event)
        assert buf.is_empty() is False

        buf.reset()
        assert buf.is_empty() is True
        assert buf.exercise_name == "Unknown"

    def test_summarise_empty_returns_empty_dict(self):
        buf = ExerciseBuffer()
        assert buf.summarise() == {}

    def test_summarise_aggregates_two_events(self):
        buf = ExerciseBuffer()

        event1 = {
            "coaching_event": {
                "exercise": {"name": "squat", "confidence": 0.9},
                "mistake": {
                    "type": "forward lean",
                    "confidence": 0.5,
                    "duration_seconds": 2.0,
                    "persistence_rate": 0.3,
                    "occurrences": 3,
                },
                "metrics": {"speed_rps": 0.9, "height_level": 3, "torso_rotation": 0.3},
                "quality_score": 0.4,
                "severity": "medium",
                "is_recoaching": False,
                "session_time_minutes": 0.5,
                "tier": "tier_2",
                "routing_reason": "test",
            }
        }
        event2 = {
            "coaching_event": {
                "exercise": {"name": "squat", "confidence": 0.9},
                "mistake": {
                    "type": "forward lean",
                    "confidence": 0.5,
                    "duration_seconds": 4.0,
                    "persistence_rate": 0.4,
                    "occurrences": 4,
                },
                "metrics": {"speed_rps": 0.9, "height_level": 3, "torso_rotation": 0.3},
                "quality_score": 0.9,
                "severity": "medium",
                "is_recoaching": False,
                "session_time_minutes": 1.5,
                "tier": "tier_2",
                "routing_reason": "test",
            }
        }

        buf.add(event1)
        buf.add(event2)
        result = buf.summarise()

        assert result["exercise_name"] == "Squat"
        assert result["event_count"] == 2
        assert result["avg_quality"] == pytest.approx(0.65, abs=0.01)
        assert result["quality_trend"] == "improving"
        assert len(result["mistakes"]) == 1
        assert result["mistakes"][0]["occurrences"] == 7
        assert result["mistakes"][0]["avg_duration_s"] == pytest.approx(3.0, abs=0.01)
        assert isinstance(result["ok_aspects"], list)
