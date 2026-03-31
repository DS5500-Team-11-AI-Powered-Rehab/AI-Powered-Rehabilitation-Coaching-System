"""
scripts/smoke_test_integration.py
----------------------------------
End-to-end smoke test for the integration-layer → CoachingAgent pipeline.
No webcam or fake frame loops required.  Run from the repository root:

    python scripts/smoke_test_integration.py

What it does
------------
1. Defines 2 synthetic integration-layer JSON events matching the real format
   emitted by IntegrationLayer._create_coaching_event().
2. Feeds each event through SessionRunner.handle_integration_event().
3. Prints the CoachingEvent dataclass, the coaching cue, and the latency.
4. Calls end_session() and prints the result.
"""

import sys
import os
import json
import time

# Ensure the repo root is on sys.path so `src.*` imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.session_runner import SessionRunner
from src.integration.schemas import coachable_event_from_integration_json


# ── 1. Synthetic integration-layer JSON events ──────────────────────────────

EVENTS = [
    {
        "event_id": "test_001_event_0",
        "timestamp": 32.5,
        "frame_index": 487,
        "exercise": {"name": "single_leg_squat", "confidence": 0.91},
        "mistake": {
            "name": "knee_valgus",
            "type": "knee_valgus",
            "confidence": 0.72,
            "duration_seconds": 8.4,
            "persistence_rate": 0.65,
            "occurrences": 14,
        },
        "metrics": {
            "speed_rps": 0.8,
            "rom_level": 2,
            "height_level": 3,
            "torso_rotation": 1,
            "direction": "none",
            "no_obvious_issue_p": 0.08,
        },
        "quality_score": 0.28,
        "severity": "high",
        "is_recoaching": False,
        "session_time_minutes": 0.54,
        "tier": "tier_3",
        "cache_key": None,
        "routing_reason": "High severity needs RAG context",
    },
    {
        "event_id": "test_001_event_1",
        "timestamp": 58.1,
        "frame_index": 871,
        "exercise": {"name": "single_leg_squat", "confidence": 0.88},
        "mistake": {
            "name": "hip_drop",
            "type": "hip_drop",
            "confidence": 0.55,
            "duration_seconds": 5.2,
            "persistence_rate": 0.42,
            "occurrences": 8,
        },
        "metrics": {
            "speed_rps": 0.9,
            "rom_level": 2,
            "height_level": 2,
            "torso_rotation": 1,
            "direction": "none",
            "no_obvious_issue_p": 0.15,
        },
        "quality_score": 0.41,
        "severity": "medium",
        "is_recoaching": False,
        "session_time_minutes": 0.97,
        "tier": "tier_2",
        "cache_key": None,
        "routing_reason": "medium severity mistake needs RAG context",
    },
]


# ── 2. Run the pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SMOKE TEST: IntegrationLayer JSON → CoachingAgent")
    print("=" * 60)
    print()

    patient_profile = {
        "patient_id": "test_001",
        "injury": "ACL",
        "session_number": 1,
    }

    print(f"Patient profile: {patient_profile}")
    print()

    runner = SessionRunner(patient_profile=patient_profile)

    print(f"Events to process: {len(EVENTS)}")
    print()

    for i, event_json in enumerate(EVENTS):
        print(f"{'─' * 60}")
        print(f"[Event {i}]  Processing integration-layer JSON")
        print(f"  event_id  : {event_json['event_id']}")
        print(f"  exercise  : {event_json['exercise']['name']}")
        print(f"  mistake   : {event_json['mistake']['name']}")
        print(f"  severity  : {event_json['severity']}")
        print(f"  tier      : {event_json['tier']}")
        print()

        # Show the CoachingEvent that the mapper produces
        coaching_event = coachable_event_from_integration_json(event_json)
        print(f"  CoachingEvent:")
        print(f"    exercise           : {coaching_event.exercise}")
        print(f"    rep_number         : {coaching_event.rep_number}")
        print(f"    persistent_mistakes: {coaching_event.persistent_mistakes}")
        print(f"    severity_scores    : {coaching_event.severity_scores}")
        print(f"    priority           : {coaching_event.priority}")
        print(f"    session_id         : {coaching_event.session_id}")
        print()

        t0 = time.time()
        cue = runner.handle_integration_event(event_json)
        wall_ms = (time.time() - t0) * 1000

        print(f"  Coaching cue : {cue}")
        print(f"  Latency      : {wall_ms:.0f}ms")
        print()

    print("=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Total events processed : {len(EVENTS)}")
    print("=" * 60)
    print()

    print("Calling end_session()...")
    summary = runner.end_session()
    print()
    print("Session summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
