"""
upstream_adapter.py — Upstream Adapter for Coaching Agent
Merges two upstream inputs into a PatientContext:
  1. coaching_event dict  (from upstream LLM agent — real-time exercise analysis)
  2. patient_profile dict (from patient data source — basic patient info)
"""

from typing import Optional
from progress_tracker_agent.schemas import (
    PatientContext, ExerciseRecord,
    RehabPhase, ConditionCategory
)


# ── Phase string → RehabPhase enum ──────────────────────────────────────────

PHASE_MAP = {
    "acute":        RehabPhase.ACUTE,
    "early":        RehabPhase.EARLY,
    "mid":          RehabPhase.MID,
    "late":         RehabPhase.LATE,
    "maintenance":  RehabPhase.MAINTENANCE,
}

CATEGORY_MAP = {
    "knee":       ConditionCategory.KNEE,
    "hip":        ConditionCategory.HIP,
    "shoulder":   ConditionCategory.SHOULDER,
    "lower_back": ConditionCategory.LOWER_BACK,
    "mid_back":   ConditionCategory.MID_BACK,
    "general_msk":ConditionCategory.GENERAL_MSK,
}


# ── Core adapter function ────────────────────────────────────────────────────

def merge_to_patient_context(
    coaching_event_payload: dict,
    patient_profile: dict,
) -> PatientContext:
    """
    Merge upstream coaching_event payload + patient_profile into PatientContext.

    Args:
        coaching_event_payload: Full dict from upstream agent, containing
                                 'coaching_event', 'session_id', 'coaching_history'
        patient_profile:         Patient basic info dict from patient data source

    Returns:
        PatientContext ready to pass to CoachingAgent.generate_coaching()
    """

    event      = coaching_event_payload["coaching_event"]
    session_id = coaching_event_payload.get("session_id", "unknown_session")
    history    = coaching_event_payload.get("coaching_history", [])

    # ── Patient ID: use session_id since patient_id is not provided ──────────
    patient_id = session_id

    # ── Parse patient profile fields ─────────────────────────────────────────
    condition          = patient_profile.get("condition", "musculoskeletal condition")
    condition_category = CATEGORY_MAP.get(
        patient_profile.get("condition_category", "general_msk"),
        ConditionCategory.GENERAL_MSK
    )
    rehab_phase = PHASE_MAP.get(
        patient_profile.get("rehab_phase", "early"),
        RehabPhase.EARLY
    )
    pain_level       = int(patient_profile.get("pain_level", 5))
    weeks_into_rehab = int(patient_profile.get("weeks_into_rehab", 1))
    age              = patient_profile.get("age")
    goals            = patient_profile.get("goals")

    # ── Build ExerciseRecord from coaching_event ─────────────────────────────
    exercise_record = _build_exercise_record(event)

    # ── Build patient_message from mistake + history ─────────────────────────
    patient_message = _build_patient_message(event, history)

    return PatientContext(
        patient_id         = patient_id,
        condition          = condition,
        condition_category = condition_category,
        rehab_phase        = rehab_phase,
        pain_level         = pain_level,
        weeks_into_rehab   = weeks_into_rehab,
        recent_exercises   = [exercise_record],
        patient_message    = patient_message,
        age                = age,
        goals              = goals,
    )


# ── Exercise record builder ──────────────────────────────────────────────────

def _build_exercise_record(event: dict) -> ExerciseRecord:
    """
    Convert coaching_event fields into an ExerciseRecord.

    Maps:
      exercise.name       → name
      quality_score       → difficulty_feedback (thresholded)
      mistake.type        → embedded in difficulty_feedback
      session_time_minutes → used to flag if exercise just started
    """
    exercise    = event.get("exercise", {})
    mistake     = event.get("mistake", {})
    quality     = event.get("quality_score", 1.0)   # 0–1, lower = worse form
    severity    = event.get("severity", "low")

    name = exercise.get("name", "unknown exercise").title()

    # quality_score → difficulty_feedback
    # Invert: low quality = patient is struggling = "too hard"
    if quality < 0.5:
        difficulty = f"too hard — {mistake.get('type', 'form issue')} detected"
    elif quality < 0.75:
        difficulty = f"ok — minor {mistake.get('type', 'form issue')}"
    else:
        difficulty = "ok"

    # completed: True unless severity is high and quality is very low
    completed = not (severity == "high" and quality < 0.4)

    return ExerciseRecord(
        name                = name,
        sets                = None,   # not available from upstream
        reps                = None,   # not available from upstream
        completed           = completed,
        difficulty_feedback = difficulty,
    )


# ── Patient message builder ───────────────────────────────────────────────────

def _build_patient_message(event: dict, history: list) -> str:
    """
    Synthesise a natural-language patient message from:
      - The current mistake type and severity
      - Movement metrics (rom_level, speed_rps)
      - coaching_history (to surface recurring issues)
    """
    exercise  = event.get("exercise", {}).get("name", "the exercise")
    mistake   = event.get("mistake", {})
    metrics   = event.get("metrics", {})
    severity  = event.get("severity", "low")
    quality   = event.get("quality_score", 1.0)

    mistake_type = mistake.get("type", "")
    occurrences  = mistake.get("occurrences", 0)
    duration     = mistake.get("duration_seconds", 0)

    # -- Base message from current event
    if mistake_type:
        msg = (
            f"During {exercise}, I'm having trouble with {mistake_type}. "
            f"It happened {occurrences} times over {duration:.0f} seconds."
        )
    else:
        msg = f"I completed {exercise} but my form score was low ({quality:.0f}/1.0)."

    # -- Append severity context
    if severity == "high":
        msg += " It feels quite difficult to correct."
    elif severity == "medium":
        msg += " I'm aware of the issue but struggling to fix it consistently."

    # -- Append ROM note if low
    rom = metrics.get("rom_level", 3)
    if rom <= 1:
        msg += " My range of motion feels very restricted."
    elif rom == 2:
        msg += " My range of motion is somewhat limited."

    # -- Append recurring issue note from history
    if history:
        past_mistakes = [
            h.get("coaching_event", {}).get("mistake", {}).get("type", "")
            for h in history
            if h.get("coaching_event", {}).get("mistake", {}).get("type")
        ]
        recurring = [m for m in past_mistakes if m == mistake_type]
        if len(recurring) >= 2:
            msg += f" This {mistake_type} issue has come up {len(recurring)} times this session already."

    return msg


# ── Validation helper ────────────────────────────────────────────────────────

def validate_inputs(coaching_event_payload: dict, patient_profile: dict) -> list[str]:
    """
    Check for missing required fields before merging.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []

    # coaching_event_payload checks
    if "coaching_event" not in coaching_event_payload:
        warnings.append("MISSING: 'coaching_event' key in payload")
    if "session_id" not in coaching_event_payload:
        warnings.append("WARNING: 'session_id' missing — patient_id will be 'unknown_session'")

    event = coaching_event_payload.get("coaching_event", {})
    if not event.get("exercise", {}).get("name"):
        warnings.append("WARNING: exercise name missing from coaching_event")
    if "quality_score" not in event:
        warnings.append("WARNING: quality_score missing — difficulty_feedback will default to 'ok'")

    # patient_profile checks
    for required in ["condition", "condition_category", "rehab_phase", "pain_level", "weeks_into_rehab"]:
        if required not in patient_profile:
            warnings.append(f"MISSING: '{required}' in patient_profile")

    if patient_profile.get("condition_category") not in CATEGORY_MAP:
        warnings.append(
            f"WARNING: unknown condition_category '{patient_profile.get('condition_category')}' "
            f"— will default to GENERAL_MSK. Valid values: {list(CATEGORY_MAP.keys())}"
        )
    if patient_profile.get("rehab_phase") not in PHASE_MAP:
        warnings.append(
            f"WARNING: unknown rehab_phase '{patient_profile.get('rehab_phase')}' "
            f"— will default to EARLY. Valid values: {list(PHASE_MAP.keys())}"
        )

    return warnings


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Sample upstream payload (exactly as provided)
    sample_payload = {
        "coaching_event": {
            "event_id": "test_session_123_event_1",
            "timestamp": 10.33,
            "frame_index": 155,
            "exercise": {"name": "squat", "confidence": 0.85},
            "mistake": {
                "type": "forward lean",
                "confidence": 0.38,
                "duration_seconds": 4.1,
                "persistence_rate": 0.31,
                "occurrences": 47,
            },
            "metrics": {
                "speed_rps": 1.0,
                "rom_level": 2,
                "height_level": 3,
                "torso_rotation": 1,
                "direction": "none",
                "no_obvious_issue_p": 0.1,
            },
            "quality_score": 0.35,
            "severity": "medium",
            "is_recoaching": False,
            "session_time_minutes": 0.17,
            "tier": "tier_2",
            "cache_key": None,
            "routing_reason": "medium severity mistake needs RAG context",
        },
        "session_id": "test_session_123",
        "coaching_history": [],
    }

    sample_profile = {
        "condition": "knee osteoarthritis",
        "condition_category": "knee",
        "rehab_phase": "mid",
        "pain_level": 4,
        "weeks_into_rehab": 10,
        "age": 58,
        "goals": "Walk dog 30 minutes daily without pain",
    }

    # Validate
    warnings = validate_inputs(sample_payload, sample_profile)
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("Validation passed ✓")

    # Merge
    context = merge_to_patient_context(sample_payload, sample_profile)

    print(f"\nPatientContext built:")
    print(f"  patient_id:      {context.patient_id}")
    print(f"  condition:       {context.condition}")
    print(f"  rehab_phase:     {context.rehab_phase.value}")
    print(f"  pain_level:      {context.pain_level}")
    print(f"  exercise:        {context.recent_exercises[0].name}")
    print(f"  difficulty:      {context.recent_exercises[0].difficulty_feedback}")
    print(f"  patient_message: {context.patient_message}")
