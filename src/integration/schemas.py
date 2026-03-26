"""
CoachingEvent dataclass — canonical representation of a processed CV event
that is passed to the coaching agent.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CoachingEvent:
    """
    A structured coaching event produced by EventProcessor from raw CV frames.

    Fields
    ------
    exercise          : name of the exercise being performed
    rep_number        : current repetition number
    persistent_mistakes: mistake keys that appear in >= 60 % of the window
    severity_scores   : average severity per persistent mistake key
    priority          : "safety" if any avg severity > 0.8, else "form"
    angles            : latest joint-angle readings from the CV frame
    session_id        : optional session identifier
    """

    exercise: str
    rep_number: int
    persistent_mistakes: List[str]
    severity_scores: Dict[str, float]
    priority: str  # "safety" | "form" | "optimization"
    angles: Dict[str, float]
    session_id: str = ""
