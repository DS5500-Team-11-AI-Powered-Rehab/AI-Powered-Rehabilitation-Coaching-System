"""
SessionRunner — top-level coordinator that wires integration layer events
to the CoachingAgent via the CoachingEvent dataclass.

Error handling
--------------
If langgraph, langchain, or chromadb are not installed the import of CoachingAgent
will raise an ImportError.  SessionRunner detects this and re-raises with a clear
human-readable message so the caller knows exactly what to install.

Usage
-----
    runner = SessionRunner(patient_profile={"patient_id": "p001", "injury": "ACL"})
    cue = runner.handle_integration_event(event_json)
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

if _MISSING_DEPS is None:
    try:
        import langchain_anthropic  # noqa: F401
    except ImportError:
        _MISSING_DEPS = "langchain_anthropic"

# ── Internal imports ─────────────────────────────────────────────────────────
from src.integration.schemas import CoachingEvent, coachable_event_from_integration_json

LOG_FILE = os.path.join("logs", "session_events.jsonl")


class SessionRunner:
    """
    Orchestrates one rehabilitation session:
      Integration-layer event JSON → CoachingEvent → CoachingAgent → cue string.

    Parameters
    ----------
    patient_profile : dict
        Minimal patient metadata.  At minimum include ``patient_id``.
    """

    def __init__(self, patient_profile: Dict):
        if _MISSING_DEPS:
            raise ImportError(
                f"Required package '{_MISSING_DEPS}' is not installed.\n"
                "Install all dependencies with:\n"
                "  pip install langgraph langchain langchain-anthropic chromadb\n"
                "Then re-run this script."
            )

        self.patient_profile = patient_profile

        # Clear the session log at session start to avoid cross-run accumulation
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        open(LOG_FILE, "w").close()  # Truncate

        # Load ground truth library for quality gate fallback
        try:
            from src.integration.ground_truth_library import GroundTruthLibrary
            self._ground_truth_library = GroundTruthLibrary("data/ground_truth_coaching_cues.json")
        except Exception as e:
            print(f"[SessionRunner] Warning: Ground truth library not available ({e})")
            self._ground_truth_library = None

        # CoachingAgent wraps the LangGraph graph
        from src.agents.coaching_agent.coaching_agent import CoachingAgent
        self._coaching_agent = CoachingAgent(ground_truth_library=self._ground_truth_library)

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

    # ── Event-level API ──────────────────────────────────────────────────────

    def handle_integration_event(self, event_json: dict) -> str:
        """
        Accept a single integration-layer JSON event and return a coaching cue.

        Parameters
        ----------
        event_json : dict
            A coaching event dict as emitted by IntegrationLayer.process_frame().

        Returns
        -------
        str
            The coaching cue produced by the LangGraph coaching agent.
        """
        event: CoachingEvent = coachable_event_from_integration_json(event_json)

        # Forward integration layer routing info so the graph respects it
        routing_override = {
            "tier": event_json.get("tier"),
            "cache_key": event_json.get("cache_key"),
            "routing_reason": event_json.get("routing_reason"),
        }
        cue, latency_ms = self._coaching_agent.handle_event(
            event, routing_override=routing_override
        )

        event.coaching_latency_ms = latency_ms
        self._log_event(event)

        return cue

    # ── Session-level API ────────────────────────────────────────────────────

    def end_session(self) -> dict:
        """
        Finalise the session.

        Generates progress report if agent available,
        then returns a summary dict with session events and coaching feedback.
        """
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
