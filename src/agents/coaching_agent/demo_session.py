"""
demo_session.py — Coaching Agent Session Demo
Run: python demo_session.py

Simulates a full rehab session with two exercises.
Press Enter to proceed through each stage.

Timing rules:
  - Event arrives every 5s
  - 30s  silence → exercise ends → exercise feedback generated
  - 120s silence → phase ends   → phase report + JSON exported
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from session_manager import SessionManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def pause(msg: str = ""):
    """Wait for user to press Enter before continuing."""
    prompt = f"\n{'─'*60}\n  ▶  {msg}\n  Press Enter to continue...\n{'─'*60}"
    input(prompt)


def make_event(exercise_name, mistake_type, quality, severity,
               occurrences, rom_level, session_time):
    """Build a coaching_event payload."""
    return {
        "coaching_event": {
            "exercise": {"name": exercise_name, "confidence": 0.88},
            "mistake": {
                "type":             mistake_type,
                "confidence":       0.75,
                "duration_seconds": 3.5,
                "persistence_rate": 0.4,
                "occurrences":      occurrences,
            },
            "metrics": {
                "speed_rps":          0.9,
                "rom_level":          rom_level,
                "height_level":       3,
                "torso_rotation":     0,
                "direction":          "none",
                "no_obvious_issue_p": 0.1,
            },
            "quality_score":        quality,
            "severity":             severity,
            "is_recoaching":        False,
            "session_time_minutes": session_time,
            "tier":                 "tier_2",
            "cache_key":            None,
            "routing_reason":       "form issue detected",
        },
        "session_id":       "session_P001_mid",
        "coaching_history": [],
    }


# ── Patient & Events ──────────────────────────────────────────────────────────

patient_profile = {
    "patient_id":         "P001",
    "condition":          "knee osteoarthritis",
    "condition_category": "knee",
    "rehab_phase":        "mid",
    "pain_level":         4,
    "weeks_into_rehab":   10,
    "age":                58,
    "goals":              "Walk dog 30 minutes daily without pain",
}

# Exercise 1: Mini Squat — knee valgus, improving over 6 events
events_ex1 = [
    make_event("mini squat", "knee valgus", 0.28, "high",   31, 1, 0.0),
    make_event("mini squat", "knee valgus", 0.31, "high",   28, 1, 0.1),
    make_event("mini squat", "knee valgus", 0.35, "medium", 22, 2, 0.2),
    make_event("mini squat", "knee valgus", 0.40, "medium", 18, 2, 0.3),
    make_event("mini squat", "knee valgus", 0.44, "medium", 15, 2, 0.4),
    make_event("mini squat", "knee valgus", 0.50, "low",    10, 3, 0.5),
]

# Exercise 2: Leg Press — forward lean, stable quality
events_ex2 = [
    make_event("leg press", "forward lean", 0.55, "medium", 20, 2, 2.0),
    make_event("leg press", "forward lean", 0.52, "medium", 18, 2, 2.1),
    make_event("leg press", "forward lean", 0.58, "medium", 15, 3, 2.2),
    make_event("leg press", "forward lean", 0.56, "low",    12, 3, 2.3),
    make_event("leg press", "forward lean", 0.60, "low",    10, 3, 2.4),
    make_event("leg press", "forward lean", 0.62, "low",     8, 3, 2.5),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  COACHING AGENT — SESSION DEMO")
    print("=" * 60)
    print(f"  Patient  : {patient_profile['patient_id']}")
    print(f"  Condition: {patient_profile['condition']}")
    print(f"  Phase    : {patient_profile['rehab_phase']}")
    print(f"  Goal     : {patient_profile['goals']}")
    print()
    print("  Session layout:")
    print("    Exercise 1 — Mini Squat  (6 events, knee valgus)")
    print("    Exercise 2 — Leg Press   (6 events, forward lean)")
    print()
    print("  ⚠️  Make sure Ollama is running: ollama serve")
    print("  ⚠️  Total runtime ≈ 5–6 minutes")

    # ── Stage 1: Initialise ───────────────────────────────────────────────────
    pause("Stage 1 of 5 — Initialise SessionManager")

    sm = SessionManager(
        patient_profile=patient_profile,
        ollama_model="gemma3:4b",
        verbose=True,
    )
    sm.start()

    # ── Stage 2: Exercise 1 ───────────────────────────────────────────────────
    pause("Stage 2 of 5 — Send Exercise 1 events (Mini Squat, 6 × 5s = 30s)")

    print("--- Sending Exercise 1 events ---")
    for payload in events_ex1:
        sm.ingest(payload)
        time.sleep(5)

    print("\n[Waiting 35s for exercise timeout + LLM generation...]")
    time.sleep(35)

    # Print Exercise 1 feedback from main thread
    if sm.exercise_feedbacks:
        f = sm.exercise_feedbacks[-1]
        print(f"\n{'─'*60}")
        print(f"  🏋️  EXERCISE ENDED: {f['exercise_name']}")
        print(f"  Avg quality: {f['avg_quality']:.2f} | Trend: {f['quality_trend']}")
        print(f"{'─'*60}")
        print("\n  ❌ WHAT TO CORRECT:")
        for m in f["mistakes"]:
            print(f"    • {m['type']} ({m['severity']}) — {m['occurrences']} occurrences")
        print("\n  ✅ WHAT YOU DID WELL:")
        for ok in f["ok_aspects"]:
            print(f"    • {ok}")
        print(f"\n  💡 COACHING CUE:\n  {f['feedback']}\n")

    # ── Stage 3: Exercise 2 ───────────────────────────────────────────────────
    pause("Stage 3 of 5 — Send Exercise 2 events (Leg Press, 6 × 5s = 30s)")

    print("--- Sending Exercise 2 events ---")
    for payload in events_ex2:
        sm.ingest(payload)
        time.sleep(5)

    print("\n[Waiting 35s for exercise 2 timeout + LLM generation...]")
    time.sleep(35)

    # Print Exercise 2 feedback from main thread
    if len(sm.exercise_feedbacks) >= 2:
        f = sm.exercise_feedbacks[-1]
        print(f"\n{'─'*60}")
        print(f"  🏋️  EXERCISE ENDED: {f['exercise_name']}")
        print(f"  Avg quality: {f['avg_quality']:.2f} | Trend: {f['quality_trend']}")
        print(f"{'─'*60}")
        print("\n  ❌ WHAT TO CORRECT:")
        for m in f["mistakes"]:
            print(f"    • {m['type']} ({m['severity']}) — {m['occurrences']} occurrences")
        print("\n  ✅ WHAT YOU DID WELL:")
        for ok in f["ok_aspects"]:
            print(f"    • {ok}")
        print(f"\n  💡 COACHING CUE:\n  {f['feedback']}\n")

    # ── Stage 4: Phase timeout ────────────────────────────────────────────────
    pause("Stage 4 of 5 — Wait for phase timeout (120s) + phase report generation")

    print("[Waiting 125s for phase timeout + report generation...]")
    time.sleep(125)
    print("Session complete ✓")

    # ── Stage 5: Inspect JSON ─────────────────────────────────────────────────
    pause("Stage 5 of 5 — Inspect exported JSON")

    outputs_dir = Path(__file__).parent.parent / "phase_outputs"
    json_files  = sorted(outputs_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found — phase may still be generating.")
        return

    latest = json_files[-1]
    print(f"Latest output: {latest.name}\n")

    with open(latest, encoding="utf-8") as fp:
        data = json.load(fp)

    ps = data["phase_summary"]
    print(f"Patient    : {ps['patient_id']}")
    print(f"Condition  : {ps['condition']}")
    print(f"Phase      : {ps['rehab_phase']}")
    print(f"Duration   : {ps['phase_duration_s']:.0f}s")
    print(f"Pain level : {ps['pain_level']}/10")

    print(f"\nExercises completed: {len(data['exercises'])}")
    for ex in data["exercises"]:
        print(f"  • {ex['exercise_name']}: "
              f"avg quality {ex['avg_quality']:.2f}, "
              f"trend {ex['quality_trend']}")
        for m in ex["mistakes"]:
            print(f"      ↳ {m['type']} ({m['severity']}) — "
                  f"{m['occurrences']} occurrences")

    print(f"\nNext phase focus:")
    for focus in data["next_phase_focus"]:
        print(f"  → {focus}")

    print(f"\nOverall quality trend: {data['overall_quality_trend']}")
    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
