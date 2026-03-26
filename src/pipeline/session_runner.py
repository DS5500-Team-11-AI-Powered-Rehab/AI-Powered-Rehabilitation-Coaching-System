"""
SessionRunner — top-level coordinator that wires EventProcessor → CoachingAgent.

Error handling
--------------
If langgraph, langchain, or chromadb are not installed the import of CoachingAgent
will raise an ImportError.  SessionRunner detects this and re-raises with a clear
human-readable message so the caller knows exactly what to install.

Usage
-----
    runner = SessionRunner(patient_profile={"patient_id": "p001", "injury": "ACL"})
    for frame in cv_stream:
        cue = runner.process_frame(frame)
        if cue:
            print("Coach says:", cue)
    summary = runner.end_session()
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict

# ── Dependency check ─────────────────────────────────────────────────────────
_MISSING_DEPS: Optional[str] = None
try:
    import langgraph  # noqa: F401
except ImportError:
    _MISSING_DEPS = "langgraph"

if _MISSING_DEPS is None:
    try:
        import langchain  # noqa: F401
    except ImportError:
        _MISSING_DEPS = "langchain"

if _MISSING_DEPS is None:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        _MISSING_DEPS = "chromadb"

# ── Internal imports ─────────────────────────────────────────────────────────
from src.integration.event_processor import EventProcessor
from src.integration.schemas import CoachingEvent

LOG_FILE = os.path.join("logs", "session_events.jsonl")


class SessionRunner:
    """
    Orchestrates one rehabilitation session:
      CV frame → EventProcessor → CoachingAgent → coaching cue string.

    Parameters
    ----------
    patient_profile : dict
        Minimal patient metadata.  At minimum include ``patient_id``.
    window_size : int
        Number of CV frames in the EventProcessor sliding window (default 70).
    """

    def __init__(self, patient_profile: Dict, window_size: int = 70):
        if _MISSING_DEPS:
            raise ImportError(
                f"Required package '{_MISSING_DEPS}' is not installed.\n"
                "Install all dependencies with:\n"
                "  pip install langgraph langchain langchain-anthropic chromadb\n"
                "Then re-run this script."
            )

        self.patient_profile = patient_profile
        session_id = str(patient_profile.get("patient_id", "session"))

        # Clear the session log at session start to avoid cross-run accumulation
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        open(LOG_FILE, "w").close()  # Truncate

        self._processor = EventProcessor(
            session_id=session_id,
            window_size=window_size,
        )

        # CoachingAgent wraps the LangGraph graph
        from src.agents.coaching_agent.coaching_agent import CoachingAgent
        self._coaching_agent = CoachingAgent()

        # Initialize ProgressTrackingAgent with knowledge base
        try:
            try:
                from src.agents.progress_tracker_agent.progress_tracker_agent import ProgressTrackerAgent
                from src.agents.progress_tracker_agent.rag_retriever import CoachingKnowledgeBase
            except ImportError:
                # Fallback for when running from scripts/ or other locations
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from src.agents.progress_tracker_agent.progress_tracker_agent import ProgressTrackerAgent
                from src.agents.progress_tracker_agent.rag_retriever import CoachingKnowledgeBase
            self._progress_kb = CoachingKnowledgeBase(
                data_dir="dataset/clean",
                persist_dir="./src/agents/progress_tracker_agent/chroma_coaching_db"
            ).load_or_build()
            self._progress_agent = ProgressTrackerAgent(
                knowledge_base=self._progress_kb,
                verbose=False
            )
        except Exception as e:
            print(f"[SessionRunner] Warning: ProgressTrackerAgent not available ({e})")
            self._progress_agent = None

    # ── Frame-level API ──────────────────────────────────────────────────────

    def process_frame(self, cv_frame: dict) -> Optional[str]:
        """
        Feed one CV frame into the pipeline.

        Returns the coaching cue string if a CoachingEvent was emitted,
        or None if no event fired this frame.
        """
        import time

        event: Optional[CoachingEvent] = self._processor.process(cv_frame)
        if event is None:
            return None

        # Time the coaching agent
        t0 = time.time()
        cue = self._coaching_agent.handle_event(event)
        latency_ms = (time.time() - t0) * 1000

        # Update event with latency and log it
        event.coaching_latency_ms = latency_ms
        self._log_event(event)

        return cue

    # ── Session-level API ────────────────────────────────────────────────────

    def end_session(self) -> dict:
        """
        Finalise the session.

        Resets the EventProcessor, generates progress report if agent available,
        then returns a summary dict with session events and coaching feedback.
        """
        self._processor.reset()

        events = _read_session_events()
        summary = {
            "patient_id": self.patient_profile.get("patient_id"),
            "total_events_logged": len(events),
            "events": events,
        }

        # Call ProgressTrackerAgent to generate session summary and coaching feedback
        if self._progress_agent:
            try:
                try:
                    from src.agents.progress_tracker_agent.schemas import (
                        PatientContext, RehabPhase, ConditionCategory, ExerciseRecord
                    )
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                    from src.agents.progress_tracker_agent.schemas import (
                        PatientContext, RehabPhase, ConditionCategory, ExerciseRecord
                    )
                import dataclasses

                # Map injury to condition_category
                injury = self.patient_profile.get("injury", "general_msk").lower()
                condition_map = {
                    "acl": ConditionCategory.KNEE,
                    "knee": ConditionCategory.KNEE,
                    "hip": ConditionCategory.HIP,
                    "shoulder": ConditionCategory.SHOULDER,
                    "back": ConditionCategory.LOWER_BACK,
                }
                condition_category = condition_map.get(
                    next((k for k in condition_map if k in injury), None),
                    ConditionCategory.GENERAL_MSK
                )

                # Build PatientContext from session data
                context = PatientContext(
                    patient_id=self.patient_profile.get("patient_id", "unknown"),
                    condition=self.patient_profile.get("injury", "general_rehab"),
                    condition_category=condition_category,
                    rehab_phase=RehabPhase.EARLY,  # Default to early phase
                    pain_level=self.patient_profile.get("pain_level", 3),
                    weeks_into_rehab=self.patient_profile.get("weeks_into_rehab", 1),
                    age=self.patient_profile.get("age"),
                    goals=self.patient_profile.get("goals", "Return to normal function"),
                    recent_exercises=[
                        ExerciseRecord(
                            name=event.get("exercise", "unknown"),
                            reps=event.get("rep_number", 0),
                            completed=True,
                            difficulty_feedback=f"Addressed: {', '.join(event.get('persistent_mistakes', []))}",
                        )
                        for event in events
                    ],
                    patient_message="End of session - please provide progress feedback.",
                )

                # Generate progress report
                report = self._progress_agent.generate_progress_report(context)
                summary["progress_report"] = dataclasses.asdict(report)
            except Exception as e:
                summary["progress_report"] = {
                    "error": str(e),
                    "fallback": f"Session complete. {len(events)} coaching events delivered."
                }
        else:
            summary["progress_report"] = None

        return summary


    # ── Private helpers ─────────────────────────────────────────────────────

    def _log_event(self, event: CoachingEvent) -> None:
        """Log a CoachingEvent (with latency) to the session log."""
        import dataclasses
        record = dataclasses.asdict(event)
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# ── Module-level helpers ────────────────────────────────────────────────────

def _read_session_events() -> list:
    """Return all events written to the JSONL log during this process run."""
    if not os.path.exists(LOG_FILE):
        return []
    records = []
    with open(LOG_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
