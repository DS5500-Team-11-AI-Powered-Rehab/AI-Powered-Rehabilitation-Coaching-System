"""
demo_progress_tracker.py — Progress Tracker Agent Demo
Run: python demo_progress_tracker.py

Reads all phase JSON files from phase_outputs/ and generates
a longitudinal progress report.
Press Enter to proceed through each stage.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from progress_tracker import ProgressTracker, load_phase_jsons, PHASE_OUTPUTS_DIR


# ── Helpers ───────────────────────────────────────────────────────────────────

def pause(msg: str = ""):
    """Wait for user to press Enter before continuing."""
    prompt = f"\n{'─'*60}\n  ▶  {msg}\n  Press Enter to continue...\n{'─'*60}"
    input(prompt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PROGRESS TRACKER AGENT — DEMO")
    print("=" * 60)
    print(f"  Reading phase files from: {PHASE_OUTPUTS_DIR}")
    print()
    print("  ⚠️  Make sure Ollama is running: ollama serve")
    print("  ⚠️  Make sure phase_outputs/ contains JSON files")
    print("      (run demo_session.py first if needed)")

    # ── Stage 1: Inspect available files ─────────────────────────────────────
    pause("Stage 1 of 3 — Inspect available phase files")

    json_files = sorted(PHASE_OUTPUTS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} phase file(s):\n")

    if not json_files:
        print("  No JSON files found in phase_outputs/")
        print("  Run demo_session.py first to generate phase data.")
        return

    for f in json_files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        ps = d.get("phase_summary", {})
        print(f"  • {f.name}")
        print(f"    Patient  : {ps.get('patient_id', '?')}")
        print(f"    Phase    : {ps.get('rehab_phase', '?')}")
        print(f"    Pain     : {ps.get('pain_level', '?')}/10")
        print(f"    Exercises: {[ex['exercise_name'] for ex in d.get('exercises', [])]}")
        print()

    # ── Stage 2: Run Progress Tracker ────────────────────────────────────────
    pause("Stage 2 of 3 — Run Progress Tracker (LLM report generation)")

    tracker = ProgressTracker(
        ollama_model="gemma3:4b",
        verbose=True,
    )

    # Pass patient_id to filter, or None to analyze all files
    result = tracker.run(patient_id=None)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # ── Stage 3: Print report ─────────────────────────────────────────────────
    pause("Stage 3 of 3 — Print progress report")

    print("=" * 65)
    print("  LONGITUDINAL PROGRESS REPORT")
    print("=" * 65)
    print(f"  Patient   : {result['patient_id']}")
    print(f"  Condition : {result['condition']}")
    print(f"  Phases    : {result['phases_analyzed']} ({result['phase_progression']})")
    print()

    # Trend summary
    pain = result["pain_analysis"]
    qual = result["quality_analysis"]
    mist = result["mistake_analysis"]

    print("  📉 PAIN TREND")
    pain_values = " → ".join(str(v) for v in pain["values"])
    print(f"     {pain_values}/10  ({pain['trend']})")

    print("\n  📈 QUALITY TREND")
    qual_values = " → ".join(
        str(v) if v is not None else "N/A"
        for v in qual["values_per_phase"]
    )
    print(f"     {qual_values}  ({qual['trend']})")

    print("\n  ⚠️  TOP MISTAKES")
    for mtype, total in mist["ranked"][:3]:
        flag = " ← PERSISTENT" if mtype in mist["persistent"] else ""
        print(f"     • {mtype}: {total} occurrences{flag}")

    print("\n  🏋️  EXERCISE PROGRESSION")
    for line in result["exercise_progression"]:
        print(f"     {line}")

    print()
    print("=" * 65)
    print("  FULL REPORT")
    print("=" * 65)
    print()
    for line in result["progress_report"].strip().split("\n"):
        print(f"  {line}")
    print()

    # Raw structured output (without long report text)
    print("=" * 65)
    print("  RAW STRUCTURED OUTPUT")
    print("=" * 65)
    summary = {k: v for k, v in result.items() if k != "progress_report"}
    print(json.dumps(summary, indent=2, default=str))

    print(f"\n{'='*65}")
    print("  Demo complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()