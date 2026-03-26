"""
EventProcessor — converts a stream of raw CV frames into CoachingEvent objects.

Design
------
- Maintains a sliding deque window of raw CV frames (default 70 frames ≈ 7 s at 10 fps).
- On each call to process(cv_frame) it returns a CoachingEvent when a new,
  persistent mistake is detected, otherwise returns None.
- Emitted events are written as JSON-lines to logs/session_events.jsonl.

Expected cv_frame format
------------------------
{
    "exercise":   str,               # exercise name
    "rep_number": int,               # current rep
    "mistakes":   List[str],         # mistake keys present in this frame
    "severity":   Dict[str, float],  # severity per mistake key (0-1)
    "angles":     Dict[str, float],  # joint angles
}
"""

import dataclasses
import json
import os
from collections import deque
from typing import Optional, Dict, Set

try:
    from .schemas import CoachingEvent
except ImportError:
    from schemas import CoachingEvent

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SIZE = 70          # frames (≈ 7 s at 10 fps)
PERSISTENCE_THRESHOLD = 0.60     # mistake must appear in >= 60 % of frames
SAFETY_SEVERITY_THRESHOLD = 0.80 # avg severity above this → "safety" priority

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "session_events.jsonl")


# ── EventProcessor ───────────────────────────────────────────────────────────

class EventProcessor:
    """
    Sliding-window CV frame processor that emits CoachingEvent dataclasses.

    Usage
    -----
        ep = EventProcessor(session_id="session_001")
        for frame in cv_stream:
            event = ep.process(frame)
            if event:
                cue = coaching_agent.handle_event(event)
    """

    def __init__(
        self,
        session_id: str = "",
        window_size: int = DEFAULT_WINDOW_SIZE,
    ):
        self.session_id = session_id
        self._window: deque = deque(maxlen=window_size)
        self.coached_this_session: Set[str] = set()

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, cv_frame: dict) -> Optional[CoachingEvent]:
        """
        Ingest one CV frame and return a CoachingEvent if appropriate.

        Returns None when:
        - The window is not yet full
        - No mistake persists in >= 60 % of frames
        - All persistent mistakes have already been coached this session
        """
        self._window.append(cv_frame)

        if len(self._window) < self._window.maxlen:
            return None

        # Count how often each mistake key appears across the window
        mistake_counts: Dict[str, int] = {}
        severity_sums: Dict[str, float] = {}
        window_size = len(self._window)

        for frame in self._window:
            for key in frame.get("mistakes", []):
                mistake_counts[key] = mistake_counts.get(key, 0) + 1
                sev = frame.get("severity", {}).get(key, 0.0)
                severity_sums[key] = severity_sums.get(key, 0.0) + sev

        # Keep only mistakes present in >= 60 % of frames
        persistent = [
            key for key, count in mistake_counts.items()
            if count / window_size >= PERSISTENCE_THRESHOLD
        ]

        # Remove mistakes already coached this session
        new_persistent = [m for m in persistent if m not in self.coached_this_session]

        if not new_persistent:
            return None

        # Compute average severity per persistent mistake
        severity_scores: Dict[str, float] = {
            key: round(severity_sums[key] / mistake_counts[key], 4)
            for key in new_persistent
        }

        # Assign priority
        priority = (
            "safety"
            if any(v > SAFETY_SEVERITY_THRESHOLD for v in severity_scores.values())
            else "form"
        )

        # Record so we do not re-coach the same mistakes
        self.coached_this_session.update(new_persistent)

        # Pull metadata from the latest frame
        latest = self._window[-1]
        event = CoachingEvent(
            exercise=latest.get("exercise", "unknown"),
            rep_number=latest.get("rep_number", 0),
            persistent_mistakes=new_persistent,
            severity_scores=severity_scores,
            priority=priority,
            angles=latest.get("angles", {}),
            session_id=self.session_id,
        )

        # NOTE: Logging is deferred to SessionRunner.process_frame() so that
        # coaching_latency_ms can be captured and included in the log
        return event

    def reset(self) -> None:
        """Clear the window and the coached-mistakes set (call between sessions)."""
        self._window.clear()
        self.coached_this_session.clear()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _log_event(self, event: CoachingEvent) -> None:
        """Append the event as a JSON line to logs/session_events.jsonl."""
        os.makedirs(LOG_DIR, exist_ok=True)
        record = dataclasses.asdict(event)
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
