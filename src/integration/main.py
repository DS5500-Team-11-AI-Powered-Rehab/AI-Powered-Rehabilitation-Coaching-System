"""
Main integration loop: CV Stream → IntegrationLayer → LangGraph → Delivery
This is the entry point that ties everything together
"""

import json
from integration_layer import IntegrationLayer, Config
from graph import create_coaching_graph


def main():
    """
    Main integration loop
    
    Process:
    1. Initialize IntegrationLayer (preprocessing)
    2. Initialize LangGraph (LLM orchestration)
    3. For each CV frame:
       a. IntegrationLayer filters and routes
       b. If coaching event → LangGraph processes
       c. Record completion
    """
    
    print("="*60)
    print("AI Rehabilitation Coaching System")
    print("="*60)
    print()
    
    # ==========================================
    # STEP 1: INITIALIZATION
    # ==========================================
    
    print("[Setup] Initializing components...")
    
    # Initialize config
    config = Config()
    
    # Initialize IntegrationLayer
    session_id = "test_session_123"
    integration_layer = IntegrationLayer(session_id=session_id, config=config)
    
    # Populate cache with default patterns
    print("[Setup] Populating Tier 1 cache...")
    integration_layer.cache.populate_defaults()
    cached_patterns = integration_layer.list_cached_patterns()
    print(f"[Setup] Cached {len(cached_patterns)} common patterns")
    
    # Initialize LangGraph
    print("[Setup] Building LangGraph workflow...")
    coaching_graph = create_coaching_graph()
    
    print("[Setup] ✅ All components ready!\n")
    
    # ==========================================
    # STEP 2: LOAD CV OUTPUT
    # ==========================================
    
    # TODO: Replace with your actual CV output
    # For now, using mock data
    
    print("[CV] Loading CV output stream...")
    
    # Option A: Load from file
    # with open('path/to/cv_output.jsonl') as f:
    #     cv_frames = [json.loads(line) for line in f]
    
    # Option B: Mock data for testing
    cv_frames = generate_mock_cv_frames()
    
    print(f"[CV] Loaded {len(cv_frames)} frames\n")
    
    # ==========================================
    # STEP 3: MAIN PROCESSING LOOP
    # ==========================================
    
    print("="*60)
    print("STARTING REAL-TIME PROCESSING")
    print("="*60)
    print()
    
    coaching_count = 0
    
    for i, cv_frame in enumerate(cv_frames):
        
        # STEP 3A: IntegrationLayer preprocessing
        coaching_event = integration_layer.process_frame(cv_frame)
        
        if coaching_event is None:
            # No coaching needed this frame
            if i % 50 == 0:  # Progress indicator
                print(f"[Frame {i}] No coaching needed (monitoring...)")
            continue
        
        # STEP 3B: Coaching event created!
        coaching_count += 1
        timestamp = cv_frame['timestamp_s']
        
        print(f"\n{'='*60}")
        print(f"[t={timestamp:.1f}s] COACHING EVENT #{coaching_count}")
        print(f"{'='*60}")
        print(f"Exercise: {coaching_event['exercise']['name']}")
        print(f"Mistake: {coaching_event['mistake']['type']}")
        print(f"Severity: {coaching_event['severity']}")
        print(f"Duration: {coaching_event['mistake']['duration_seconds']:.1f}s")
        print(f"Persistence: {coaching_event['mistake']['persistence_rate']:.0%}")
        print(f"Routing: {coaching_event['tier']} ({coaching_event['routing_reason']})")
        print()
        
        # STEP 3C: LangGraph processing
        print(f"[LangGraph] Starting orchestration...")
        
        initial_state = {
            "coaching_event": coaching_event,
            "session_id": integration_layer.session_id,
            "coaching_history": integration_layer.coaching_history
        }
        
        # Invoke LangGraph
        final_state = coaching_graph.invoke(initial_state)
        
        # STEP 3D: Record completion in IntegrationLayer
        integration_layer.record_coaching_complete(
            coaching_event,
            final_state['feedback_audio'],
            final_state['tier_used']
        )
        
        print(f"[Result] ✅ Coaching delivered")
        print(f"  Tier: {final_state['tier_used']}")
        print(f"  Latency: {final_state.get('latency_ms', 0):.0f}ms")
        print(f"  Message: \"{final_state['feedback_audio']}\"")
        print()
    
    # ==========================================
    # STEP 4: SESSION SUMMARY
    # ==========================================
    
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
    
    summary = integration_layer.get_session_summary()
    
    print(f"\nSession ID: {summary['session_id']}")
    print(f"Duration: {summary['session_duration_seconds']:.1f} seconds")
    print(f"Total Coaching Events: {summary['total_events']}")
    print(f"Unique Mistakes Addressed: {len(summary['coached_mistakes'])}")
    
    if summary['coached_mistakes']:
        print("\nMistakes Coached:")
        for mistake in summary['coached_mistakes']:
            print(f"  - {mistake}")
    
    from collections import Counter
    
    # Tier breakdown
    tier_counts = Counter(entry['tier_used'] for entry in summary['coaching_history'])
    tier_breakdown = {
        "tier_1": tier_counts.get("tier_1", 0),
        "tier_2": tier_counts.get("tier_2", 0),
        "tier_3": tier_counts.get("tier_3", 0)
    }
    
    print(f"\nTier Breakdown:")
    print(f"  Tier 1 (Cache): {tier_breakdown['tier_1']}")
    print(f"  Tier 2 (RAG): {tier_breakdown['tier_2']}")
    print(f"  Tier 3 (Reasoning): {tier_breakdown['tier_3']}")
    
    cache_hit_rate = (tier_breakdown['tier_1'] / summary['total_events'] * 100) if summary['total_events'] > 0 else 0
    print(f"\nCache Hit Rate: {cache_hit_rate:.0f}%")
    
    print("\n✅ Session complete!")


def generate_mock_cv_frames():
    """
    Generate mock CV frames for testing
    
    TODO: Replace with actual CV output loader
    """
    
    frames = []
    
    # Generate 200 frames (simulating ~13 seconds at 15fps)
    for i in range(200):
        timestamp = i / 15.0  # 15 fps
        
        # Simulate a persistent mistake appearing from frame 75-120
        mistakes = []
        if 75 <= i <= 120:
            mistakes.append({
                'name': 'knee valgus',
                'p': 0.45 + (i - 75) * 0.002  # Gradually increasing confidence
            })
        
        # Add occasional other mistakes
        if 150 <= i <= 160:
            mistakes.append({
                'name': 'forward lean',
                'p': 0.38
            })
        
        frame = {
            'timestamp_s': timestamp,
            'frame_index': i,
            'source_fps': 15.0,
            'exercise': {
                'name': 'squat',
                'p': 0.85
            },
            'mistakes': mistakes,
            'metrics': {
                'speed_rps': 1.0,
                'rom_level': 2,
                'height_level': 3,
                'torso_rotation': 1,
                'direction': 'none',
                'no_obvious_issue_p': 0.1 if mistakes else 0.8
            },
            'quality_score': 0.35 if mistakes else 0.75,
            'speak_now': 0.0
        }
        
        frames.append(frame)
    
    return frames


def load_cv_output_from_file(filepath: str):
    """
    Load CV output from JSONL file
    
    Args:
        filepath: Path to CV output file (one JSON object per line)
        
    Returns:
        List of CV frame dictionaries
    """
    frames = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                frame = json.loads(line)
                frames.append(frame)
    
    return frames

def example_add_cached_pattern():
    """
    Example: Dynamically add a new cached pattern
    """
    
    config = Config()
    layer = IntegrationLayer(session_id="demo", config=config)
    
    # Add new pattern discovered during testing
    layer.add_cached_response(
        exercise="lunge",
        mistake="forward lean",
        response="Keep your torso upright - imagine a string pulling you up from the top of your head.",
        timing="immediate"
    )
    
    print("✅ Added new cached pattern: lunge_forward_lean")
    print(f"Total cached patterns: {len(layer.list_cached_patterns())}")


def example_process_single_event():
    """
    Example: Process a single coaching event
    """
    
    from integration_layer import IntegrationLayer, Config
    from graph import create_coaching_graph
    
    # Setup
    layer = IntegrationLayer(session_id="demo", config=Config())
    layer.cache.populate_defaults()
    graph = create_coaching_graph()
    
    # Mock coaching event (would come from layer.process_frame())
    coaching_event = {
        "event_id": "demo_event_1",
        "timestamp": 10.5,
        "exercise": {"name": "squat", "confidence": 0.85},
        "mistake": {
            "type": "knee valgus",
            "confidence": 0.45,
            "duration_seconds": 4.2,
            "persistence_rate": 0.45
        },
        "severity": "high",
        "tier": "tier_2",
        "cache_key": None,
        "routing_reason": "High severity needs RAG context"
    }
    
    # Process with LangGraph
    initial_state = {
        "coaching_event": coaching_event,
        "session_id": "demo",
        "coaching_history": []
    }
    
    final_state = graph.invoke(initial_state)
    
    print(f"Response: {final_state['feedback_audio']}")
    print(f"Tier: {final_state['tier_used']}")
    print(f"Latency: {final_state.get('latency_ms', 0):.0f}ms")



if __name__ == "__main__":
    """
    Run the main integration loop
    
    Usage:
        python main.py
    """
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Exit] Interrupted by user")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()