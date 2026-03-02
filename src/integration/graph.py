"""
LangGraph workflow for coaching orchestration
Handles: Tier routing → LLM processing → Response delivery
"""

from langgraph.graph import StateGraph, END
from state import CoachingState
import time

def enrich_context_node(state: CoachingState) -> CoachingState:
    """
    STEP 1: Enrich coaching event with session context
    
    Loads:
    - Patient profile from database
    - Coaching history from current session
    - Any patient-specific preferences/limitations
    """
    
    # Extract session metadata
    coaching_event = state.get("coaching_event", {})
    session_id = coaching_event.get("event_id", "").split("_event_")[0] if coaching_event else "unknown"
    
    # Load patient profile from database
    # In production: Replace with actual database query
    # Example:
    #   from your_db import get_patient_by_session
    #   profile = get_patient_by_session(session_id)
    
    # Mock data for now
    state["patient_profile"] = {
        "session_id": session_id,
        "name": "Test Patient",
        "age": 35,
        "known_limitations": [],
        "past_injuries": [],
        "preferences": {
            "coaching_style": "encouraging",
            "audio_enabled": True,
            "detailed_explanations": False
        }
    }
    
    # Coaching history comes from IntegrationLayer
    # (already passed in state["coaching_history"] from main.py)
    
    print(f"[Enrich Context] Loaded profile for session: {session_id}")
    
    return state


def tier_1_cache_node(state: CoachingState) -> CoachingState:
    """
    TIER 1: Cache lookup (fastest path ~50ms)
    
    Looks up pre-computed response from cache
    """
    
    start_time = time.time()
    cache_key = state.get("cache_key")
    
    # Lookup cached response
    # Note: Cache instance should be passed from main.py via state["cache"]
    # Or access via: from integration_layer import ResponseCache
    
    cached_data = None
    if "cache" in state:
        # Cache passed from main.py
        cached_data = state["cache"].get(cache_key)
    
    if cached_data:
        # Use cached response
        state["coaching_response"] = cached_data["response"]
        state["delivery_timing"] = cached_data.get("timing", "immediate")
        print(f"[Tier 1] Cache hit: {cache_key}")
    else:
        # Fallback if cache miss (shouldn't happen with proper routing)
        state["coaching_response"] = f"Maintain proper form"
        state["delivery_timing"] = "immediate"
        print(f"[Tier 1] Cache miss (unexpected): {cache_key}")
    
    state["tier_used"] = "tier_1"
    latency = (time.time() - start_time) * 1000
    state["latency_ms"] = latency
    
    return state


def tier_2_rag_node(state: CoachingState) -> CoachingState:
    """
    TIER 2: RAG + Simple LLM (~1-2 seconds)
    
    Process:
    1. Retrieve 1-2 relevant docs from RAG
    2. Simple focused LLM prompt
    3. Generate brief coaching cue (15-20 words)
    """
    
    start_time = time.time()
    
    # TODO: Implement RAG retrieval
    # Hint: Use ChromaDB or similar vector store
    # Example:
    # vectorstore = Chroma(collection_name="pt_guidelines")
    # docs = vectorstore.similarity_search(query, k=2)
    
    coaching_event = state["coaching_event"]
    mistake_type = coaching_event["mistake"]["type"]
    exercise = coaching_event["exercise"]["name"]
    
    # Mock RAG retrieval
    query = f"How to correct {mistake_type} during {exercise}"
    print(f"[Tier 2] RAG Query: {query}")
    
    # TODO: Replace with actual RAG
    mock_docs = [
        f"Physical therapy guidelines for {mistake_type}: Ensure proper alignment and form...",
        f"Common corrections for {exercise}: Focus on controlled movement..."
    ]
    state["retrieved_docs"] = mock_docs
    
    # TODO: Implement LLM generation
    # Hint: Use langchain_anthropic.ChatAnthropic
    # Example:
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=100)
    # prompt = f"Generate brief coaching cue for: {mistake_type}"
    # response = llm.invoke(prompt)
    
    # Mock LLM response
    state["coaching_response"] = f"Focus on correcting {mistake_type} - maintain proper form throughout the movement."
    state["delivery_timing"] = "rep_end"
    state["tier_used"] = "tier_2"
    
    latency = (time.time() - start_time) * 1000
    state["latency_ms"] = latency
    
    print(f"[Tier 2] RAG + LLM generation complete ({latency:.0f}ms)")
    
    return state


def tier_3_reasoning_node(state: CoachingState) -> CoachingState:
    """
    TIER 3: Full reasoning with tools (~3-5 seconds)
    
    Process:
    1. Retrieve 3-5 docs from RAG
    2. Use Movement Analysis Agent with tools
    3. Chain-of-thought reasoning
    4. Generate detailed coaching with explanation
    """
    
    start_time = time.time()
    
    # TODO: Implement full reasoning workflow
    # Hint: Use LangChain agent with tools
    # Example tools:
    # - analyze_compensation_pattern()
    # - check_patient_limitations()
    # - recommend_modification()
    
    coaching_event = state["coaching_event"]
    
    print(f"[Tier 3] Starting complex reasoning...")
    
    # Mock complex reasoning
    state["movement_analysis"] = "Detected potential compensation pattern due to fatigue"
    state["coaching_response"] = (
        "I notice this mistake is persisting. This often happens when certain muscles "
        "fatigue faster than others. Let's modify the exercise slightly to maintain "
        "good form while building the strength you need."
    )
    state["delivery_timing"] = "rest_period"
    state["tier_used"] = "tier_3"
    
    latency = (time.time() - start_time) * 1000
    state["latency_ms"] = latency
    
    print(f"[Tier 3] Complex reasoning complete ({latency:.0f}ms)")
    
    return state


def coaching_agent_node(state: CoachingState) -> CoachingState:
    """
    COACHING AGENT: Polish the response
    
    Takes tier output and:
    - Makes it conversational and encouraging
    - Adjusts tone based on patient preferences
    - Prepares for audio delivery
    """
    
    # TODO: Implement response polishing
    # Hint: Use LLM to refine the coaching_response
    # Make it conversational, encouraging, and natural
    
    raw_response = state["coaching_response"]
    patient_name = state["patient_profile"].get("name", "")
    
    # Mock polishing (in production, use LLM)
    polished = raw_response  # For now, pass through
    
    state["feedback_audio"] = polished
    
    print(f"[Coaching Agent] Polished response ready")
    
    return state


def format_feedback_node(state: CoachingState) -> CoachingState:
    """
    FORMAT & DELIVER: Prepare feedback for delivery
    
    Formats for:
    - Audio TTS
    - UI display
    - Logging
    """
    
    feedback = state["feedback_audio"]
    timing = state["delivery_timing"]
    tier = state["tier_used"]
    coaching_event = state["coaching_event"]
    
    # Format delivery package
    delivery_package = {
        "message": feedback,
        "timing": timing,
        "tier": tier,
        "timestamp": coaching_event["timestamp"],
        "event_id": coaching_event["event_id"],
        "latency_ms": state.get("latency_ms", 0),
        "audio_enabled": state["patient_profile"].get("preferences", {}).get("audio_enabled", True)
    }
    
    # Log delivery
    print(f"\n{'='*60}")
    print(f"[FEEDBACK DELIVERY]")
    print(f"Event: {delivery_package['event_id']}")
    print(f"Timing: {timing}")
    print(f"Tier: {tier}")
    print(f"Message: {feedback}")
    print(f"Latency: {delivery_package['latency_ms']:.0f}ms")
    print(f"{'='*60}\n")
    
    # Store formatted delivery for output
    state["delivery_package"] = delivery_package
    
    # In production, send to:
    # - TTS engine: text_to_speech(feedback)
    # - UI display: websocket_send(delivery_package)
    # - Database: log_coaching_event(delivery_package)
    
    return state


def progress_tracking_node(state: CoachingState) -> CoachingState:
    """
    PROGRESS TRACKING: Update session metrics (async)
    
    This runs in background, doesn't block feedback delivery
    """
    
    coaching_event = state["coaching_event"]
    tier_used = state["tier_used"]
    
    # Build tracking record
    tracking_record = {
        "event_id": coaching_event["event_id"],
        "timestamp": coaching_event["timestamp"],
        "mistake_type": coaching_event["mistake"]["type"],
        "exercise": coaching_event["exercise"]["name"],
        "tier_used": tier_used,
        "severity": coaching_event["severity"],
        "latency_ms": state.get("latency_ms", 0),
        "response": state["coaching_response"]
    }
    
    # In production: Send to IntegrationLayer for session tracking
    # if "integration_layer" in state:
    #     state["integration_layer"].record_coaching_complete(
    #         coaching_event,
    #         state["coaching_response"],
    #         tier_used
    #     )
    
    # Update session summary
    # This would normally aggregate across all events in session
    state["session_summary"] = {
        "latest_event": tracking_record["event_id"],
        "total_events": state.get("coaching_history", []).count + 1 if "coaching_history" in state else 1,
        "mistakes_coached": [tracking_record["mistake_type"]],
        "tier_breakdown": {
            tier_used: 1
        }
    }
    
    print(f"[Progress] Tracked event {tracking_record['event_id']} ({tier_used}, {tracking_record['latency_ms']:.0f}ms)")
    
    return state

# Graph Construction
def route_by_tier(state: CoachingState) -> str:
    """
    Conditional routing function
    Reads state["tier"] and returns which node to go to
    """
    tier = state.get("tier", "tier_2")
    return tier


def create_coaching_graph():
    """
    BUILD LANGGRAPH WORKFLOW
    
    Flow:
    1. Enrich context (load patient data)
    2. Route to tier based on IntegrationLayer decision
    3. [CONDITIONAL] Execute Tier 1, 2, or 3
    4. Coaching Agent (polish response)
    5. Format & deliver
    6. Progress tracking (async)
    """
    
    workflow = StateGraph(CoachingState)
    
    # === ADD NODES ===
    workflow.add_node("enrich_context", enrich_context_node)
    workflow.add_node("tier_1_cache", tier_1_cache_node)
    workflow.add_node("tier_2_rag", tier_2_rag_node)
    workflow.add_node("tier_3_reasoning", tier_3_reasoning_node)
    workflow.add_node("coaching_agent", coaching_agent_node)
    workflow.add_node("format_feedback", format_feedback_node)
    workflow.add_node("progress_tracking", progress_tracking_node)
    
    # === DEFINE FLOW ===
    
    # Start: Enrich with patient context
    workflow.set_entry_point("enrich_context")
    
    # After enrichment, route to appropriate tier
    workflow.add_conditional_edges(
        "enrich_context",
        route_by_tier,  # Function that reads state["tier"]
        {
            "tier_1": "tier_1_cache",
            "tier_2": "tier_2_rag",
            "tier_3": "tier_3_reasoning"
        }
    )
    
    # All tiers converge to coaching agent
    workflow.add_edge("tier_1_cache", "coaching_agent")
    workflow.add_edge("tier_2_rag", "coaching_agent")
    workflow.add_edge("tier_3_reasoning", "coaching_agent")
    
    # Coaching agent → Format & deliver
    workflow.add_edge("coaching_agent", "format_feedback")
    
    # Format → Progress tracking (async, doesn't block)
    workflow.add_edge("format_feedback", "progress_tracking")
    
    # Progress tracking → END
    workflow.add_edge("progress_tracking", END)
    
    return workflow.compile()