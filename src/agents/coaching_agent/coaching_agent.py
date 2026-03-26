"""
CoachingAgent — thin class wrapper around the existing LangGraph graph.

Responsibilities
----------------
- Owns a compiled LangGraph instance (create_coaching_graph()).
- Exposes handle_event(event: CoachingEvent) -> str which translates a
  CoachingEvent dataclass into the exact state dict the graph already expects,
  invokes the graph, and returns the feedback_audio string.

Everything inside create_coaching_graph() is unchanged.
"""

from typing import List, Dict


class CoachingAgent:
    """
    Wraps the LangGraph coaching workflow for event-driven invocation.

    Usage
    -----
        agent = CoachingAgent()
        cue = agent.handle_event(event)   # event: CoachingEvent
    """

    def __init__(self):
        from src.integration.graph import create_coaching_graph
        self.graph = create_coaching_graph()
        self.coaching_history: List[Dict] = []
        self._event_counter: int = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def handle_event(self, event) -> str:
        """
        Translate a CoachingEvent dataclass into graph state and invoke the graph.

        Parameters
        ----------
        event : CoachingEvent
            Produced by EventProcessor.

        Returns
        -------
        str
            The polished coaching cue (state["feedback_audio"]).
        """
        from src.integration.schemas import CoachingEvent  # local import to avoid cycles

        if not isinstance(event, CoachingEvent):
            raise TypeError(f"Expected CoachingEvent, got {type(event)}")

        # ── Map CoachingEvent → coaching_event dict expected by the graph ──
        # The graph reads: coaching_event["exercise"]["name"]
        #                  coaching_event["mistake"]["type"]
        #                  coaching_event["severity"]
        #                  coaching_event["tier"]
        #                  coaching_event["cache_key"]
        #                  coaching_event["routing_reason"]

        mistake_type = (
            event.persistent_mistakes[0] if event.persistent_mistakes else "unknown"
        )
        avg_severity = event.severity_scores.get(mistake_type, 0.5)

        severity_str = (
            "high"   if event.priority == "safety"       else
            "medium" if event.priority == "form"         else
            "low"
        )
        # Safety issues → tier_3 (full reasoning); form/optimization → tier_2 (RAG)
        tier = "tier_3" if event.priority == "safety" else "tier_2"

        coaching_event_dict = {
            "event_id": f"{event.session_id}_event_{self._event_counter}",
            "timestamp": 0,
            "frame_index": 0,
            "exercise": {
                "name": event.exercise,
                "confidence": 1.0,
            },
            "mistake": {
                "type": mistake_type,
                "confidence": avg_severity,
                "duration_seconds": 0,
                "persistence_rate": 1.0,
                "occurrences": 1,
            },
            "metrics": event.severity_scores,
            "quality_score": 0.5,
            "severity": severity_str,
            "is_recoaching": False,
            "session_time_minutes": 0,
        }

        initial_state = {
            "coaching_event": coaching_event_dict,
            "session_id": event.session_id or "default_session",
            "coaching_history": self.coaching_history,
            # cache and ground_truth_library are None; the graph handles this gracefully
            "cache": None,
            "ground_truth_library": None,
            "tier": tier,
            "cache_key": None,
            "routing_reason": f"Priority: {event.priority}",
        }

        final_state = self.graph.invoke(initial_state)
        self._event_counter += 1

        cue: str = final_state.get("feedback_audio", "")
        self.coaching_history.append({
            "timestamp": 0,
            "mistake_type": mistake_type,
            "response": cue,
            "tier_used": final_state.get("tier_used", tier),
            "severity": severity_str,
            "event_id": coaching_event_dict["event_id"],
        })
        return cue
