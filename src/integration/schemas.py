"""
CoachingEvent dataclass — canonical representation of a processed CV event
that is passed to the coaching agent.

Also contains coachable_event_from_integration_json() which maps the real
integration layer's JSON dict into a CoachingEvent.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CoachingEvent:
    """
    A structured coaching event produced by the integration layer.

    Fields
    ------
    exercise          : name of the exercise being performed
    rep_number        : current repetition number
    persistent_mistakes: mistake keys that appear in >= 60 % of the window
    severity_scores   : average severity per persistent mistake key
    priority          : "safety" if any avg severity > 0.8, else "form"
    angles            : latest joint-angle readings from the CV frame
    session_id        : optional session identifier
    coaching_latency_ms: (optional) time in ms from event detection to coaching cue
    """

    exercise: str
    rep_number: int
    persistent_mistakes: List[str]
    severity_scores: Dict[str, float]
    priority: str  # "safety" | "form" | "optimization"
    angles: Dict[str, float]
    session_id: str = ""
    coaching_latency_ms: float = 0.0


def coachable_event_from_integration_json(event_json: dict) -> CoachingEvent:
    """
    Map a real integration layer JSON event dict into a CoachingEvent dataclass.

    The integration layer (src/integration/integration_layer.py) emits dicts
    with nested structures like:

        {
            "event_id": "session_123_event_5",
            "exercise": {"name": "squat", "confidence": 0.85},
            "mistake":  {"name": "knee valgus", "type": "knee valgus",
                         "occurrences": 10, ...},
            "severity": "high",
            "quality_score": 0.234,
            "metrics":  {"speed_rps": 1.2, ...},
            ...
        }

    This function extracts the fields the coaching agent needs and returns a
    flat CoachingEvent dataclass.  Missing or None values are replaced with
    safe defaults rather than raising KeyError.
    """
    exercise_data = event_json.get("exercise") or {}
    exercise = exercise_data.get("name", "") if isinstance(exercise_data, dict) else str(exercise_data)

    mistake_data = event_json.get("mistake") or {}
    mistake_name = mistake_data.get("name") or mistake_data.get("type", "")

    occurrences = mistake_data.get("occurrences", 0) if isinstance(mistake_data, dict) else 0

    quality_score = event_json.get("quality_score", 0.0)
    severity_scores = {mistake_name: quality_score} if mistake_name else {}

    severity = event_json.get("severity", "medium")
    priority = "safety" if severity == "high" else "form"

    metrics = event_json.get("metrics") or {}

    event_id = event_json.get("event_id", "")
    session_id = event_id.split("_event_")[0] if "_event_" in event_id else event_id

    return CoachingEvent(
        exercise=exercise,
        rep_number=occurrences,
        persistent_mistakes=[mistake_name] if mistake_name else [],
        severity_scores=severity_scores,
        priority=priority,
        angles=metrics,
        session_id=session_id,
    )
