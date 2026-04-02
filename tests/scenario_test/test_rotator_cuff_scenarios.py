"""
Scenario Tests — Rotator Cuff Rehabilitation Patient

Clinical context:
  Rotator cuff rehabilitation focuses on controlled shoulder movements,
  avoiding compensatory patterns and ensuring full range of motion.
  Common exercises include external rotation, wall slides, and arm raises.
  Key mistakes to monitor:

    - Incomplete arm raise:  Patient not reaching full elevation → MEDIUM severity
    - Arm twisting:          Compensatory rotation that strains the repair → MEDIUM severity

These tests verify the system detects shoulder-specific form errors
during a simulated rotator cuff rehab session.
"""

from conftest import make_scenario_frame, make_clean_frame, run_scenario


class TestRotatorCuffScenarios:

    def test_incomplete_arm_raise_generates_coaching_event(self, layer):
        """
        Patients compensate by not raising the arm fully, which limits
        recovery. A persistent incomplete arm raise should trigger coaching.
        """
        frames = [make_scenario_frame("arm raise", "incomplete arm raise", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for incomplete arm raise"
        assert event["mistake"]["type"] == "incomplete arm raise"
        assert event["exercise"]["name"] == "arm raise"

    def test_incomplete_arm_raise_is_medium_severity(self, layer):
        """
        Incomplete arm raise contains the FORM keywords 'incomplete' and 'arm raise'
        — should be classified as MEDIUM severity, not HIGH.
        """
        frames = [make_scenario_frame("arm raise", "incomplete arm raise", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None
        assert event["severity"] == "medium"

    def test_arm_twisting_during_external_rotation_generates_event(self, layer):
        """
        Twisting motion is a compensatory pattern during external rotation —
        the FORM keyword 'twisting' should flag this as MEDIUM severity.
        """
        frames = [make_scenario_frame("external rotation", "arm twisting", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is not None, "Expected a coaching event for arm twisting"
        assert event["mistake"]["type"] == "arm twisting"
        assert event["severity"] == "medium"

    def test_correct_shoulder_form_produces_no_event(self, layer):
        """
        A patient performing external rotation exercises with proper form
        should not receive any coaching interruption.
        """
        frames = [make_clean_frame("external rotation", i) for i in range(15)]
        event = run_scenario(layer, frames)

        assert event is None, "Expected no coaching event when shoulder form is correct"
