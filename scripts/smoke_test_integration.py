"""
scripts/smoke_test_integration.py
----------------------------------
End-to-end smoke test for the CV → EventProcessor → CoachingAgent pipeline.
No webcam required.  Run from the repository root:

    python scripts/smoke_test_integration.py

What it does
------------
1. Builds 80 synthetic CV frames:
   - Frames  0-19 : no mistakes
   - Frames 20-69 : knee_valgus (severity 0.85) — single_leg_squat, rep 7
   - Frames 70-79 : hip_drop    (severity 0.50) — single_leg_squat, rep 8
2. Feeds all 80 frames through SessionRunner.process_frame().
3. Prints which frames triggered a CoachingEvent and what the agent returned.
4. Calls end_session() and prints the summary.
"""

import sys
import os

# Ensure the repo root is on sys.path so `src.*` imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.session_runner import SessionRunner


# ── 1. Build synthetic CV frames ─────────────────────────────────────────────

def build_frames():
    frames = []

    # Frames 0-19: clean reps
    for i in range(20):
        frames.append({
            "exercise": "single_leg_squat",
            "rep_number": 6,
            "mistakes": [],
            "severity": {},
            "angles": {},
        })

    # Frames 20-69: persistent knee_valgus (50 frames = 71% of 70-frame window)
    for i in range(50):
        frames.append({
            "exercise": "single_leg_squat",
            "rep_number": 7,
            "mistakes": ["knee_valgus"],
            "severity": {"knee_valgus": 0.85},
            "angles": {"knee": 25.0},
        })

    # Frames 70-149: persistent hip_drop (80 frames = 100% of next window)
    # This tests deduplication — knee_valgus already coached, so hip_drop should fire
    for i in range(80):
        frames.append({
            "exercise": "single_leg_squat",
            "rep_number": 8,
            "mistakes": ["hip_drop"],
            "severity": {"hip_drop": 0.50},
            "angles": {"hip": 18.0},
        })

    return frames


# ── 2. Run the pipeline ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SMOKE TEST: CV → EventProcessor → CoachingAgent")
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

    frames = build_frames()
    print(f"Total frames to process: {len(frames)}")
    print()

    total_events = 0

    for frame_idx, frame in enumerate(frames):
        cue = runner.process_frame(frame)

        if cue is not None:
            total_events += 1
            print(f"{'─' * 60}")
            print(f"[Frame {frame_idx:>3}]  CoachingEvent fired!")
            print(f"  Exercise         : {frame['exercise']}")
            print(f"  Rep              : {frame['rep_number']}")
            print(f"  Mistakes         : {frame['mistakes']}")
            print(f"  Severity         : {frame['severity']}")
            print(f"  Angles           : {frame['angles']}")
            print(f"  Coaching cue     : \"{cue}\"")
            print()

    print("=" * 60)
    print("PROCESSING COMPLETE")
    print(f"  Total frames processed : {len(frames)}")
    print(f"  Total events fired     : {total_events}")
    print("=" * 60)
    print()

    print("Calling end_session()...")
    summary = runner.end_session()
    print()
    print("Session summary:")
    import json
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
