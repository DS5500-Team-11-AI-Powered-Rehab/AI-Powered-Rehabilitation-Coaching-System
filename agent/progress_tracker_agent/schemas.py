"""
schemas.py — Data structures for Coaching Agent
Defines PatientContext (upstream input) and CoachingOutput (downstream delivery)
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class RehabPhase(str, Enum):
    ACUTE = "acute"           # 0-2 weeks: pain management, gentle ROM
    EARLY = "early"           # 2-8 weeks: strength building begins
    MID = "mid"               # 8-16 weeks: progressive loading
    LATE = "late"             # 16+ weeks: return to function
    MAINTENANCE = "maintenance"  # ongoing: self-management


class ConditionCategory(str, Enum):
    KNEE = "knee"
    HIP = "hip"
    SHOULDER = "shoulder"
    LOWER_BACK = "lower_back"
    MID_BACK = "mid_back"
    GENERAL_MSK = "general_msk"


@dataclass
class ExerciseRecord:
    """Record of a completed or assigned exercise"""
    name: str
    sets: Optional[int] = None
    reps: Optional[int] = None
    completed: bool = False
    difficulty_feedback: Optional[str] = None  # "too easy", "ok", "too hard"


@dataclass
class PatientContext:
    """
    Upstream context passed to the Coaching Agent.
    
    In production this comes from an upstream agent or DB.
    For testing, can be manually constructed.
    """
    # --- Core identifiers ---
    patient_id: str
    condition: str                        # e.g. "knee osteoarthritis", "ACL recovery"
    condition_category: ConditionCategory

    # --- Current status ---
    rehab_phase: RehabPhase
    pain_level: int                       # 0-10 NRS scale
    weeks_into_rehab: int

    # --- Exercise history (last session) ---
    recent_exercises: List[ExerciseRecord] = field(default_factory=list)
    
    # --- Patient's expressed concerns / questions ---
    patient_message: str = ""             # e.g. "My knee still hurts going upstairs"

    # --- Optional enrichment ---
    age: Optional[int] = None
    prior_conditions: List[str] = field(default_factory=list)
    goals: Optional[str] = None          # e.g. "return to hiking by summer"
    mobility_aids: bool = False


@dataclass
class CoachingOutput:
    """
    Final coaching feedback delivered to the patient.
    """
    patient_id: str
    coaching_feedback: str                # Main personalized coaching text
    suggested_exercises: List[str]        # Exercise recommendations for next session
    safety_notes: List[str]               # Important safety reminders
    motivational_note: str                # Closing encouragement
    retrieved_sources: List[str]          # Which documents were used (transparency)
    confidence_score: float               # Internal quality estimate 0-1


# ── Quick test fixtures ─────────────────────────────────────────────────────

def make_sample_context(scenario: str = "knee") -> PatientContext:
    """Generate sample PatientContext for testing."""

    if scenario == "knee":
        return PatientContext(
            patient_id="P001",
            condition="knee osteoarthritis",
            condition_category=ConditionCategory.KNEE,
            rehab_phase=RehabPhase.MID,
            pain_level=4,
            weeks_into_rehab=10,
            recent_exercises=[
                ExerciseRecord("Quad sets", sets=3, reps=10, completed=True, difficulty_feedback="ok"),
                ExerciseRecord("Straight leg raises", sets=3, reps=12, completed=True, difficulty_feedback="too easy"),
                ExerciseRecord("Mini squats", sets=2, reps=8, completed=False, difficulty_feedback="too hard"),
            ],
            patient_message="I finished most exercises but the mini squats hurt my knee going down. "
                            "Also I'm struggling with stairs at home.",
            age=58,
            goals="Walk my dog 30 minutes daily without pain",
        )

    elif scenario == "shoulder":
        return PatientContext(
            patient_id="P002",
            condition="rotator cuff tendinopathy",
            condition_category=ConditionCategory.SHOULDER,
            rehab_phase=RehabPhase.EARLY,
            pain_level=5,
            weeks_into_rehab=3,
            recent_exercises=[
                ExerciseRecord("Pendulum swings", sets=3, reps=15, completed=True, difficulty_feedback="ok"),
                ExerciseRecord("Shoulder external rotation", sets=2, reps=10, completed=True, difficulty_feedback="too hard"),
            ],
            patient_message="My shoulder feels tight in the morning and the external rotation exercise causes a sharp pain at the end range.",
            age=45,
            goals="Return to swimming 3x per week",
        )

    elif scenario == "acl":
        return PatientContext(
            patient_id="P003",
            condition="ACL reconstruction recovery",
            condition_category=ConditionCategory.KNEE,
            rehab_phase=RehabPhase.LATE,
            pain_level=2,
            weeks_into_rehab=20,
            recent_exercises=[
                ExerciseRecord("Single leg squats", sets=3, reps=8, completed=True, difficulty_feedback="ok"),
                ExerciseRecord("Box jumps", sets=2, reps=6, completed=True, difficulty_feedback="ok"),
            ],
            patient_message="Feeling much stronger. When can I start jogging again?",
            age=26,
            goals="Return to recreational football",
        )

    return make_sample_context("knee")  # default
