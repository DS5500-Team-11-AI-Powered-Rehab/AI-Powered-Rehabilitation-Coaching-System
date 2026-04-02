"""
Scenario Tests — Back Pain Rehabilitation Patient

Clinical context:
  Back pain rehabilitation requires close monitoring of spinal alignment.
  Exercises include bridges, deadlifts, and back extensions. Key mistakes:

    - Lumbar instability:  Direct spinal risk → HIGH severity + immediate coaching
    - Trunk lean:          Compensatory lean stresses the lower back → MEDIUM severity
    - Moving too fast:     Rushing reps reduces effectiveness → MEDIUM severity

These tests also cover a degraded video quality scenario — common in
at-home rehab settings — where the system must still identify serious
spinal risks (lumbar) despite poor visibility.
"""

from conftest import make_scenario_frame, make_clean_frame, run_scenario


class TestBackPainScenarios:

    def test_lumbar_instability_generates_high_severity_event(self, layer):
        """
        'Lumbar' is a critical safety keyword. Any persistent lumbar instability
        pattern must immediately trigger a HIGH severity coaching event.
        """
        frames = [make_scenario_frame("deadlift", "lumbar instability", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for lumbar instability"
        assert event["mistake"]["type"] == "lumbar instability"
        assert event["severity"] == "high"
        assert event["exercise"]["name"] == "deadlift"

    def test_trunk_lean_during_bridge_generates_medium_severity_event(self, layer):
        """
        Trunk lean contains the FORM keyword 'lean' — should trigger
        MEDIUM severity coaching during a bridge exercise.
        """
        frames = [make_scenario_frame("bridge", "trunk lean", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for trunk lean"
        assert event["mistake"]["type"] == "trunk lean"
        assert event["severity"] == "medium"

    def test_moving_too_fast_generates_medium_severity_event(self, layer):
        """
        Rushing through back extension reps reduces effectiveness.
        'Fast' is a FORM keyword — should produce MEDIUM severity coaching.
        """
        frames = [make_scenario_frame("back extension", "moving too fast", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for moving too fast"
        assert event["severity"] == "medium"

    def test_lumbar_instability_routes_to_llm_tier(self, layer):
        """
        HIGH severity lumbar mistakes should always use LLM-based coaching
        (Tier 2 or Tier 3) rather than a simple cache lookup.
        """
        frames = [make_scenario_frame("deadlift", "lumbar instability", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None
        assert event["tier"] in ("tier_2", "tier_3"), (
            f"Lumbar instability requires LLM coaching, got: {event['tier']}"
        )

    def test_lumbar_instability_poor_video_quality_still_detected(self, layer):
        """
        Simulates a patient exercising in a poorly lit room (low video quality).
        Even with quality_score=0.1 the system must still detect the lumbar
        risk and flag it as HIGH severity — safety cannot depend on video quality.
        """
        frames = [
            make_scenario_frame("deadlift", "lumbar instability", i, quality_score=0.1)
            for i in range(15)
        ]
        event = run_scenario(layer, frames)

        assert event is not None, "Lumbar instability must be detected even in poor video quality"
        assert event["severity"] == "high"

    def test_correct_back_form_produces_no_coaching_event(self, layer):
        """
        A patient performing deadlifts with correct spinal alignment
        should receive no coaching interruption.
        """
        frames = [make_clean_frame("deadlift", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is None, "Expected no coaching event when back form is correct"
