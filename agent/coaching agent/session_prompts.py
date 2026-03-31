"""
session_prompts.py — Prompts for Exercise Summary and Phase Report
Used by SessionManager when an exercise ends or a phase ends.
"""

# ── Exercise Summary Prompt ───────────────────────────────────────────────────
# Called when 30s of inactivity detected (exercise ended)

EXERCISE_SUMMARY_PROMPT = """You are a physiotherapy coaching assistant giving real-time feedback.

PATIENT: {condition}, week {weeks_into_rehab}, phase: {rehab_phase}, pain: {pain_level}/10.

EXERCISE JUST COMPLETED: {exercise_name}
Total events recorded: {event_count}
Average quality score: {avg_quality:.2f}/1.0
Quality trend: {quality_trend}

MISTAKES DETECTED:
{mistakes_text}

GOOD ASPECTS:
{ok_aspects_text}

Write your feedback using EXACTLY this structure, no extra text:

✅ WHAT YOU DID WELL:
[1-2 specific things the patient did correctly]

❌ WHAT TO CORRECT:
[The main mistake, why it matters, one practical fix]

💡 CUE FOR NEXT SET:
[One simple actionable instruction, under 15 words]

Be direct, warm, and specific. Total under 120 words."""


# ── Phase Report Prompt ───────────────────────────────────────────────────────
# Called when 120s of inactivity detected (phase ended)

PHASE_REPORT_PROMPT = """You are a physiotherapy coaching assistant writing an end-of-session report.

PATIENT: {condition}, week {weeks_into_rehab}, phase: {rehab_phase}, pain: {pain_level}/10.
Patient goal: {goals}

SESSION SUMMARY:
- Exercises completed: {exercise_list}
- Overall quality trend: {overall_quality_trend}
- Pain trend this session: {pain_trend}

EXERCISE BREAKDOWN:
{exercise_breakdown}

CLINICAL CONTEXT:
{clinical_context}

Write a structured end-of-session report (under 300 words) with these sections:

1. SESSION OVERVIEW: 2-3 sentences summarising today's performance
2. WHAT WENT WELL: specific positives across exercises
3. AREAS TO IMPROVE: top 2 issues with practical correction tips
4. NEXT SESSION FOCUS: 2-3 concrete recommendations for next session
5. ENCOURAGEMENT: one sentence tied to their personal goal

Be warm, specific, and clinically grounded."""


# ── Helpers ───────────────────────────────────────────────────────────────────


def format_mistakes_text(mistakes: list) -> str:
    """Format mistake list for prompt injection."""
    if not mistakes:
        return "No significant mistakes detected."
    lines = []
    for m in mistakes:
        lines.append(
            f"- {m['type']}: {m['occurrences']} occurrences, "
            f"avg duration {m['avg_duration_s']:.1f}s, "
            f"severity {m['severity']}"
        )
    return "\n".join(lines)


def format_ok_aspects_text(ok_aspects: list) -> str:
    """Format ok aspects for prompt injection."""
    if not ok_aspects:
        return "Form was consistent overall."
    return "\n".join(f"- {a}" for a in ok_aspects)


def format_exercise_breakdown(exercises: list) -> str:
    """Format all exercises for phase report prompt."""
    if not exercises:
        return "No exercises recorded."
    lines = []
    for ex in exercises:
        mistake_summary = (
            ", ".join(f"{m['type']} ({m['severity']})" for m in ex["mistakes"])
            or "no major mistakes"
        )
        lines.append(
            f"- {ex['exercise_name']}: avg quality {ex['avg_quality']:.2f}, "
            f"trend {ex['quality_trend']}, issues: {mistake_summary}"
        )
    return "\n".join(lines)


def infer_quality_trend(scores: list) -> str:
    """Infer trend from a list of quality scores."""
    if len(scores) < 2:
        return "stable"
    first_half = sum(scores[: len(scores) // 2]) / max(len(scores) // 2, 1)
    second_half = sum(scores[len(scores) // 2 :]) / max(
        len(scores) - len(scores) // 2, 1
    )
    delta = second_half - first_half
    if delta > 0.05:
        return "improving"
    elif delta < -0.05:
        return "declining"
    return "stable"


def infer_ok_aspects(events: list) -> list:
    """
    Derive positive aspects from metrics across events.
    Looks for consistently good metric values.
    """
    ok = []
    if not events:
        return ok

    speeds = [e["coaching_event"]["metrics"].get("speed_rps", 0) for e in events]
    heights = [e["coaching_event"]["metrics"].get("height_level", 0) for e in events]
    rotations = [
        e["coaching_event"]["metrics"].get("torso_rotation", 0) for e in events
    ]

    avg_speed = sum(speeds) / len(speeds)
    avg_height = sum(heights) / len(heights)
    avg_rot = sum(rotations) / len(rotations)

    if avg_speed >= 0.8:
        ok.append("Good movement speed maintained throughout")
    if avg_height >= 3:
        ok.append("Consistent squat depth achieved")
    if avg_rot <= 0.5:
        ok.append("Torso rotation well controlled")
    if not ok:
        ok.append("Showed effort and persistence throughout the exercise")

    return ok
