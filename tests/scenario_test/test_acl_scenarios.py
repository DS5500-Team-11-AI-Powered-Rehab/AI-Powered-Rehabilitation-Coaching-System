"""
Scenario Tests — ACL Rehabilitation Patient

Clinical context:
  ACL (Anterior Cruciate Ligament) rehabilitation requires careful monitoring
  of knee alignment and range of motion. Common exercises include squats,
  lunges, and step-ups. The two most critical mistakes are:

    - Knee valgus (inward knee collapse): HIGH risk of re-injury → HIGH severity
    - Incomplete range of motion:          Slows recovery → MEDIUM severity

These tests verify the system detects and correctly prioritises these
mistakes during a simulated ACL rehab session.
"""

from conftest import make_scenario_frame, make_clean_frame, run_scenario


class TestACLScenarios:

    def test_knee_valgus_during_squat_generates_high_severity_event(self, layer):
        """
        Knee valgus is a critical ACL risk.
        A persistent valgus pattern during squatting should trigger an
        immediate HIGH severity coaching event.
        """
        frames = [make_scenario_frame("squat", "knee valgus", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for persistent knee valgus"
        assert event["mistake"]["type"] == "knee valgus"
        assert event["severity"] == "high"
        assert event["exercise"]["name"] == "squat"

    def test_incomplete_rom_during_lunge_generates_medium_severity_event(self, layer):
        """
        Incomplete range of motion slows ACL recovery but is not an
        immediate safety risk — expect MEDIUM severity coaching.
        """
        frames = [make_scenario_frame("lunge", "incomplete range of motion", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for incomplete ROM"
        assert event["mistake"]["type"] == "incomplete range of motion"
        assert event["severity"] == "medium"
        assert event["exercise"]["name"] == "lunge"

    def test_good_form_produces_no_coaching_event(self, layer):
        """
        A patient performing squats with correct form should receive
        no coaching interruption.
        """
        frames = [make_clean_frame("squat", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is None, "Expected no coaching event when form is correct"

    def test_knee_valgus_routes_to_correct_tier(self, layer):
        """
        Knee valgus is high severity. Without a cached response it should
        route to Tier 2 (RAG + LLM) by default, or Tier 3 if the pattern
        is also flagged as complex.
        """
        frames = [make_scenario_frame("squat", "knee valgus", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None
        assert event["tier"] in ("tier_2", "tier_3"), (
            f"High-severity ACL mistake should use LLM coaching, got: {event['tier']}"
        )
