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

        self._processor = EventProcessor(
            session_id=session_id,
            window_size=window_size,
        )

        # CoachingAgent wraps the LangGraph graph
        from src.agents.coaching_agent.coaching_agent import CoachingAgent
        self._coaching_agent = CoachingAgent()

        # ProgressTrackingAgent — load if interface is straightforward
        # TODO: Wire ProgressTrackerAgent once a session-level (non-LLM-gated)
        #       interface is available.  Currently it requires PatientContext
        #       built from phase JSON files, which are not available mid-session.
        #       See src/agents/progress_tracker_agent/progress_tracker_agent.py.
        self._progress_agent = None

    # ── Frame-level API ──────────────────────────────────────────────────────

    def process_frame(self, cv_frame: dict) -> Optional[str]:
        """
        Feed one CV frame into the pipeline.

        Returns the coaching cue string if a CoachingEvent was emitted,
        or None if no event fired this frame.
        """
        event: Optional[CoachingEvent] = self._processor.process(cv_frame)
        if event is None:
            return None
        return self._coaching_agent.handle_event(event)

    # ── Session-level API ────────────────────────────────────────────────────

    def end_session(self) -> dict:
        """
        Finalise the session.

        Resets the EventProcessor, then returns a summary dict that includes
        all events logged to logs/session_events.jsonl during this session.
        """
        self._processor.reset()

        # TODO: Call self._progress_agent once it supports session-level input.
        #       Replace the lines below with:
        #           report = self._progress_agent.generate_progress_report(context)
        #           return dataclasses.asdict(report)

        events = _read_session_events()
        return {
            "patient_id": self.patient_profile.get("patient_id"),
            "total_events_logged": len(events),
            "events": events,
            "progress_report": "(TODO: wire ProgressTrackerAgent)",
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

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
