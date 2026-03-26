"""
LangGraph workflow for coaching orchestration
Handles: Tier routing → LLM processing → Response delivery
"""

from langgraph.graph import StateGraph, END
try:
    from .state import CoachingState
except ImportError:
    from state import CoachingState
import time

# Failure strings produced by tier fallbacks — these trigger the quality gate
_KNOWN_FAILURE_PATTERNS = [
    "i apologize", "i'm sorry", "i cannot", "i don't know",
    "focus on correcting",        # Tier 2 exception fallback
    "let's focus on correcting",  # Tier 3 exception fallback
    "maintain proper form",       # Tier 1 cache-miss fallback
    "as an ai", "i am an ai",
]

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
    
    coaching_event = state["coaching_event"]
    mistake_type = coaching_event["mistake"]["type"]
    exercise = coaching_event["exercise"]["name"]
    
    # Build RAG query
    query = f"How to correct {mistake_type} during {exercise}"
    print(f"[Tier 2] RAG Query: {query}")
    
    # RAG Retrieval using ChromaDB
    retrieved_docs = []
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Suppress harmless BERT model loading warnings (position_ids mismatch is safe)
        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)

        # Use sentence-transformers for embeddings
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name="pt_guidelines",
            embedding_function=embedding_fn
        )
        
        # Query for top 2 most relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        # Extract document texts
        if results and results['documents'] and len(results['documents']) > 0:
            retrieved_docs = results['documents'][0]
            print(f"[Tier 2] Retrieved {len(retrieved_docs)} documents from RAG")
        else:
            print(f"[Tier 2] No documents found in ChromaDB, using fallback")
            retrieved_docs = [
                f"Physical therapy guidelines for {mistake_type}: Ensure proper alignment and form...",
                f"Common corrections for {exercise}: Focus on controlled movement..."
            ]
            
    except Exception as e:
        print(f"[Tier 2] RAG retrieval failed: {e}. Using fallback.")
        retrieved_docs = [
            f"Physical therapy guidelines for {mistake_type}: Ensure proper alignment and form...",
            f"Common corrections for {exercise}: Focus on controlled movement..."
        ]
    
    state["retrieved_docs"] = retrieved_docs
    
    # LLM Generation with Claude
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        
        # Initialize Claude Sonnet 4
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            temperature=0.3
        )
        
        # Create focused prompt for brief coaching cue
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a physical therapy coach. Generate a brief, actionable coaching cue (15-20 words max) based on the guidelines provided. Be concise and direct."),
            ("user", """Context from PT guidelines:
{context}

Mistake detected: {mistake_type}
Exercise: {exercise}

Generate a brief coaching cue to correct this mistake:""")
        ])
        
        # Prepare context from retrieved docs
        context = "\n\n".join(retrieved_docs)
        
        # Generate response
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "mistake_type": mistake_type,
            "exercise": exercise
        })
        
        state["coaching_response"] = response.content.strip()
        print(f"[Tier 2] LLM generated coaching cue")
        
    except Exception as e:
        print(f"[Tier 2] LLM generation failed: {e}. Using fallback.")
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
    
    coaching_event = state["coaching_event"]
    mistake_type = coaching_event["mistake"]["type"]
    exercise = coaching_event["exercise"]["name"]
    severity = coaching_event.get("severity", "medium")
    coaching_history = state.get("coaching_history", [])
    patient_profile = state.get("patient_profile", {})
    
    print(f"[Tier 3] Starting complex reasoning...")
    
    # === STEP 1: RAG Retrieval (3-5 documents) ===
    query = f"Detailed analysis and correction strategies for {mistake_type} during {exercise}"
    print(f"[Tier 3] RAG Query: {query}")
    
    retrieved_docs = []
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Suppress harmless BERT model loading warnings (position_ids mismatch is safe)
        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)

        # Use sentence-transformers for embeddings
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name="pt_guidelines",
            embedding_function=embedding_fn
        )
        
        # Query for top 5 most relevant documents
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        # Extract document texts
        if results and results['documents'] and len(results['documents']) > 0:
            retrieved_docs = results['documents'][0]
            print(f"[Tier 3] Retrieved {len(retrieved_docs)} documents from RAG")
        else:
            print(f"[Tier 3] No documents found in ChromaDB, using fallback")
            retrieved_docs = [
                f"Physical therapy guidelines for {mistake_type}: Detailed biomechanical analysis...",
                f"Compensation patterns in {exercise}: Common causes and corrections...",
                f"Progressive modification strategies for movement quality...",
                f"Patient-centered coaching approaches for persistent mistakes...",
                f"Evidence-based cues for {exercise} technique..."
            ]
            
    except Exception as e:
        print(f"[Tier 3] RAG retrieval failed: {e}. Using fallback.")
        retrieved_docs = [
            f"Physical therapy guidelines for {mistake_type}: Detailed biomechanical analysis...",
            f"Compensation patterns in {exercise}: Common causes and corrections...",
            f"Progressive modification strategies for movement quality...",
            f"Patient-centered coaching approaches for persistent mistakes...",
            f"Evidence-based cues for {exercise} technique..."
        ]
    
    state["retrieved_docs"] = retrieved_docs
    
    # === STEP 2: Define Analysis Tools ===
    from langchain.tools import tool
    
    @tool
    def analyze_compensation_pattern(mistake_history: str) -> str:
        """Analyze if mistake indicates a compensation pattern based on patient history."""
        # Check if mistake is recurring
        if coaching_history and len(coaching_history) > 2:
            recent_mistakes = [h.get("mistake", {}).get("type") for h in coaching_history[-3:]]
            if recent_mistakes.count(mistake_type) >= 2:
                return f"PATTERN DETECTED: {mistake_type} has occurred {recent_mistakes.count(mistake_type)} times in last 3 events. Likely compensation due to fatigue or weakness."
        return f"No clear compensation pattern detected yet for {mistake_type}."
    
    @tool
    def check_patient_limitations(limitation_query: str) -> str:
        """Check patient profile for known limitations or past injuries."""
        limitations = patient_profile.get("known_limitations", [])
        injuries = patient_profile.get("past_injuries", [])
        
        result = []
        if limitations:
            result.append(f"Known limitations: {', '.join(limitations)}")
        if injuries:
            result.append(f"Past injuries: {', '.join(injuries)}")
        
        if result:
            return " | ".join(result)
        return "No documented limitations or past injuries in patient profile."
    
    @tool
    def recommend_modification(exercise_name: str) -> str:
        """Recommend exercise modification to maintain quality while building strength."""
        modifications = {
            "squat": "Reduce depth to parallel, add tempo control (3-1-3), or use box squat for feedback",
            "lunge": "Shorten stride length, add support rail, or perform static split squat",
            "push-up": "Elevate hands on bench, reduce range of motion, or perform on knees",
            "deadlift": "Reduce weight, use trap bar, or perform Romanian deadlift with lighter load",
            "plank": "Reduce hold time, perform on knees, or do incline plank"
        }
        
        exercise_lower = exercise_name.lower()
        for key in modifications:
            if key in exercise_lower:
                return f"Modification for {exercise_name}: {modifications[key]}"
        
        return f"General modification: Reduce load/intensity, slow tempo, or simplify range of motion"
    
    tools = [analyze_compensation_pattern, check_patient_limitations, recommend_modification]
    
    # === STEP 3: Agent with Chain-of-Thought Reasoning ===
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage

        # Initialize Claude Sonnet 4 with tool use
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.4
        )

        system_prompt = """You are an expert physical therapy movement analysis agent.
Your role is to perform chain-of-thought reasoning to understand WHY a patient is making a mistake and provide detailed, actionable coaching.

Available context:
- Patient profile with limitations and history
- Physical therapy guidelines from knowledge base
- Exercise-specific biomechanics

Use the available tools to:
1. Analyze if this is a compensation pattern
2. Check for patient-specific limitations
3. Recommend appropriate modifications if needed

Then synthesize a detailed coaching response (30-50 words) that:
- Explains the root cause of the mistake
- Provides specific correction cues
- Offers encouragement and context
- Suggests modifications if appropriate"""

        # Create agent (LangChain 1.x unified API — returns a compiled graph directly)
        agent = create_agent(llm, tools, system_prompt=system_prompt)

        # Prepare context
        context = "\n\n".join(retrieved_docs[:5])
        history_summary = f"{len(coaching_history)} events in session" if coaching_history else "First event in session"

        user_message = (
            f"Physical Therapy Guidelines:\n{context}\n\n"
            f"Current Situation:\n"
            f"- Exercise: {exercise}\n"
            f"- Mistake: {mistake_type}\n"
            f"- Severity: {severity}\n"
            f"- Patient: {patient_profile.get('name', 'Patient')}, Age {patient_profile.get('age', 'unknown')}\n\n"
            f"Recent coaching history: {history_summary}\n\n"
            f"Perform your analysis and generate a detailed coaching response."
        )

        # Run agent and extract the last message content
        result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
        agent_output = result["messages"][-1].content
        state["movement_analysis"] = f"Agent reasoning completed with {len(tools)} tools"
        state["coaching_response"] = agent_output
        
        print(f"[Tier 3] Agent reasoning complete - generated detailed coaching")
        
    except Exception as e:
        print(f"[Tier 3] Agent execution failed: {e}. Using fallback reasoning.")
        
        # Fallback: Manual reasoning without agent
        context = "\n".join(retrieved_docs[:3])
        
        # Simple pattern detection
        is_recurring = False
        if coaching_history and len(coaching_history) > 2:
            recent_mistakes = [h.get("mistake", {}).get("type") for h in coaching_history[-3:]]
            is_recurring = recent_mistakes.count(mistake_type) >= 2
        
        if is_recurring:
            state["movement_analysis"] = "Detected recurring mistake pattern - likely fatigue or compensation"
            state["coaching_response"] = (
                f"I notice {mistake_type} is persisting. This often happens when certain muscles "
                f"fatigue before others. Let's modify the {exercise} slightly - try reducing the "
                f"range of motion or slowing the tempo to maintain quality form while building the strength you need."
            )
        else:
            state["movement_analysis"] = "Single occurrence of mistake - providing detailed correction"
            state["coaching_response"] = (
                f"Let's focus on correcting your {mistake_type} in the {exercise}. The key is to "
                f"maintain alignment throughout the movement. Focus on controlled tempo and proper "
                f"positioning. This will help prevent compensation patterns as you progress."
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
    
    raw_response = state["coaching_response"]
    patient_profile = state.get("patient_profile", {})
    patient_name = patient_profile.get("name", "")
    coaching_style = patient_profile.get("preferences", {}).get("coaching_style", "encouraging")
    tier_used = state.get("tier_used", "tier_2")
    
    # Skip polishing for Tier 1 (cached responses are already polished)
    if tier_used == "tier_1":
        state["feedback_audio"] = raw_response
        print(f"[Coaching Agent] Tier 1 response - skipping polish")
        return state
    
    # Polish response with LLM
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        
        # Initialize Claude for response polishing
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            temperature=0.5  # Slightly higher for more natural variation
        )
        
        # Define coaching styles
        style_instructions = {
            "encouraging": "Be warm, supportive, and motivating. Use positive reinforcement.",
            "direct": "Be clear, concise, and straightforward. Focus on actionable instructions.",
            "detailed": "Provide thorough explanations with context. Be educational and informative.",
            "friendly": "Be casual, conversational, and personable. Use a warm, approachable tone."
        }
        
        style_instruction = style_instructions.get(coaching_style, style_instructions["encouraging"])
        
        # Create polishing prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a physical therapy coaching assistant that makes feedback natural and encouraging for audio delivery.

Your task: Refine the coaching feedback to be conversational, natural-sounding when spoken aloud, and appropriate for the patient's preferences.

Coaching style preference: {coaching_style}
Style guidance: {style_instruction}

Guidelines:
- Keep the core technical content and corrections
- Make it sound natural when spoken aloud (avoid robotic phrasing)
- Use contractions and conversational language
- Be encouraging and supportive
- Keep it concise (don't make it significantly longer)
- Remove any awkward phrasing or overly formal language
- Address the patient directly using "you" and "your"
{"- Use the patient's name naturally: " + patient_name if patient_name and patient_name != "Test Patient" else ""}

Output ONLY the polished coaching message, nothing else."""),
            ("user", "Raw coaching feedback to polish:\n\n{raw_response}")
        ])
        
        # Generate polished response
        chain = prompt | llm
        result = chain.invoke({"raw_response": raw_response})
        
        polished = result.content.strip()
        
        # Validation: Ensure polished response isn't empty or too different
        if not polished or len(polished) < 10:
            print(f"[Coaching Agent] Polish result too short, using original")
            polished = raw_response
        elif len(polished) > len(raw_response) * 2:
            print(f"[Coaching Agent] Polish result too long, using original")
            polished = raw_response
        else:
            print(f"[Coaching Agent] Response polished successfully ({coaching_style} style)")
        
        state["feedback_audio"] = polished
        
    except Exception as e:
        print(f"[Coaching Agent] Polishing failed: {e}. Using original response.")
        state["feedback_audio"] = raw_response
    
    return state


def quality_gate_node(state: CoachingState) -> CoachingState:
    """
    QUALITY GATE: Validate coaching response; fall back to ground truth if poor.

    Sits between coaching_agent and format_feedback. Evaluates state["feedback_audio"]
    using three lightweight heuristics and replaces it with a curated cue from
    GroundTruthLibrary when the response is clearly bad.

    Signals:
      1. Length — response must be within expected word-count range for the tier
      2. Relevance — response must mention at least one token from exercise/mistake
      3. Refusal — response must not contain known failure/fallback strings

    Falls back if fewer than 2 out of 3 signals pass.
    Tier 1 (cached responses) is always trusted and skipped.
    """
    # Tier 1 cache responses are pre-curated — skip evaluation
    if state.get("tier_used") == "tier_1":
        state["used_fallback"] = False
        state["fallback_source"] = None
        return state

    response = state.get("feedback_audio", "")
    coaching_event = state["coaching_event"]
    exercise = coaching_event["exercise"]["name"]
    mistake = coaching_event["mistake"]["type"]
    tier = state.get("tier_used", "tier_2")

    # Signal 1: Length gate
    word_count = len(response.split())
    min_words = {"tier_2": 8, "tier_3": 15}.get(tier, 8)
    max_words = {"tier_2": 60, "tier_3": 100}.get(tier, 60)
    length_ok = min_words <= word_count <= max_words

    # Signal 2: Relevance — response should reference exercise or mistake context
    context_tokens = (
        set(exercise.lower().split())
        | set(mistake.lower().replace("-", " ").split())
    )
    response_tokens = set(response.lower().split())
    relevance_ok = bool(context_tokens & response_tokens)

    # Signal 3: No known failure / refusal patterns
    response_lower = response.lower()
    refusal_ok = not any(p in response_lower for p in _KNOWN_FAILURE_PATTERNS)

    quality_score = sum([length_ok, relevance_ok, refusal_ok])
    use_fallback = quality_score < 2

    if use_fallback and "ground_truth_library" in state and state["ground_truth_library"] is not None:
        gt_lib = state["ground_truth_library"]
        cue = gt_lib.lookup(exercise, mistake)
        if cue:
            state["feedback_audio"] = cue
            state["used_fallback"] = True
            state["fallback_source"] = "ground_truth_library"
        else:
            state["feedback_audio"] = gt_lib.template_fallback(exercise, mistake)
            state["used_fallback"] = True
            state["fallback_source"] = "template"
        print(
            f"[Quality Gate] Fallback triggered (score={quality_score}/3, "
            f"length={length_ok}, relevance={relevance_ok}, refusal={refusal_ok}) "
            f"→ {state['fallback_source']}"
        )
    else:
        state["used_fallback"] = False
        state["fallback_source"] = None
        if use_fallback:
            print(f"[Quality Gate] Low quality (score={quality_score}/3) but no library available — keeping response")
        else:
            print(f"[Quality Gate] Response passed (score={quality_score}/3)")

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
        "total_events": len(state.get("coaching_history", [])) + 1,
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
    workflow.add_node("quality_gate", quality_gate_node)
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
    
    # Coaching agent → Quality gate → Format & deliver
    workflow.add_edge("coaching_agent", "quality_gate")
    workflow.add_edge("quality_gate", "format_feedback")
    
    # Format → Progress tracking (async, doesn't block)
    workflow.add_edge("format_feedback", "progress_tracking")
    
    # Progress tracking → END
    workflow.add_edge("progress_tracking", END)
    
    return workflow.compile()