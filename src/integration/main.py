"""Main integration loop: CV Stream -> IntegrationLayer -> SessionRunner / LangGraph."""

import argparse
import gzip
import json
import time
from pathlib import Path

try:
    from .integration_layer import IntegrationLayer, Config
except ImportError:
    from integration_layer import IntegrationLayer, Config

try:
    from .ground_truth_library import GroundTruthLibrary
except ImportError:
    from ground_truth_library import GroundTruthLibrary

# Module-level SessionRunner singleton — initialised in main() when --session-runner is set
_session_runner = None


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
    args = parse_args()
    
    # Initialize config
    config = Config()

    if args.debug_fast:
        # Relax thresholds for live demo runs where CV emits sparse JSON events.
        config.WINDOW_SIZE_FRAMES = 30
        config.MIN_FRAMES = 10
        config.MIN_PERSISTENCE_RATE = 0.15
        config.MIN_CONFIDENCE = 0.25
        config.MIN_DURATION_SECONDS = 0.5
        config.MIN_COACHING_INTERVAL = 2
        config.RE_COACHING_THRESHOLD = 15
    
    print(
        "[Setup] Thresholds "
        f"(window_frames={config.WINDOW_SIZE_FRAMES}, "
        f"min_frames={config.MIN_FRAMES}, "
        f"min_persistence={config.MIN_PERSISTENCE_RATE}, "
        f"min_confidence={config.MIN_CONFIDENCE}, "
        f"min_duration_s={config.MIN_DURATION_SECONDS})"
    )

    # Initialize ground-truth fallback library
    print("[Setup] Loading ground-truth coaching library...")
    gt_library_path = "data/ground_truth_coaching_cues.json"
    gt_library = GroundTruthLibrary(gt_library_path)
    print(f"[Setup] Ground-truth library: {len(gt_library)} pairs loaded")

    # Initialize IntegrationLayer with ground-truth for dynamic cache promotion
    session_id = "test_session_123"
    integration_layer = IntegrationLayer(
        session_id=session_id, config=config, gt_library=gt_library
    )

    # Populate cache with default patterns
    print("[Setup] Populating Tier 1 cache...")
    integration_layer.cache.populate_defaults()
    cached_patterns = integration_layer.list_cached_patterns()
    print(f"[Setup] Cached {len(cached_patterns)} common patterns")

    print("[Setup] ✅ Core components ready!\n")

    # ==========================================
    # STEP 2: LOAD CV OUTPUT
    # ==========================================

    print("[CV] Loading CV output stream...")

    if args.cv_jsonl:
        print(f"[CV] Source: {args.cv_jsonl}")
        if args.follow:
            print("[CV] Mode: live follow (waiting for appended frames)")
            frame_source = iter_cv_output_from_file(
                filepath=args.cv_jsonl,
                follow=True,
                poll_interval=args.poll_interval,
            )
        else:
            cv_frames = load_cv_output_from_file(args.cv_jsonl)
            if args.max_frames > 0:
                cv_frames = cv_frames[:args.max_frames]
            print(f"[CV] Loaded {len(cv_frames)} frames")
            frame_source = cv_frames
    else:
        cv_frames = generate_mock_cv_frames()
        if args.max_frames > 0:
            cv_frames = cv_frames[:args.max_frames]
        print("[CV] Source: built-in mock frames")
        print(f"[CV] Loaded {len(cv_frames)} frames")
        frame_source = cv_frames

    print()

    coaching_graph = None
    if args.use_langgraph:
        print("[Setup] Building LangGraph workflow...")
        try:
            try:
                from .graph import create_coaching_graph
            except ImportError:
                from graph import create_coaching_graph
            coaching_graph = create_coaching_graph()
            print("[Setup] LangGraph enabled")
        except Exception as e:
            print(f"[Setup] LangGraph unavailable ({e}); falling back to integration-only mode")
            args.use_langgraph = False

    # ── SessionRunner (preferred full-stack path) ───────────────────────────
    global _session_runner
    if args.session_runner:
        print("[Setup] Initialising SessionRunner...")
        try:
            try:
                from src.pipeline.session_runner import SessionRunner
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from src.pipeline.session_runner import SessionRunner
            patient_profile = {
                "patient_id": args.patient_id,
                "injury": args.injury,
                "session_number": 1,
            }
            _session_runner = SessionRunner(patient_profile=patient_profile)
            # Share integration layer cache with the coaching agent for Tier 1 lookups
            _session_runner._coaching_agent.cache = integration_layer.cache
            print(f"[Setup] SessionRunner ready (patient={args.patient_id}, injury={args.injury})")
        except Exception as e:
            print(f"[Setup] SessionRunner unavailable ({e}); falling back to --use-langgraph / integration-only mode")
            args.session_runner = False
    
    # ==========================================
    # STEP 3: MAIN PROCESSING LOOP
    # ==========================================
    
    print("="*60)
    print("STARTING REAL-TIME PROCESSING")
    print("="*60)
    print()
    
    coaching_count = 0
    
    try:
        for i, cv_frame in enumerate(frame_source):
            if args.max_frames > 0 and i >= args.max_frames:
                print(f"[Run] Reached --max-frames={args.max_frames}; stopping.")
                break

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

            # STEP 3C: Route event to coaching backend
            if args.session_runner and _session_runner is not None:
                # ── Full-stack path: SessionRunner (CoachingAgent + ProgressTrackerAgent) ──
                try:
                    cue = _session_runner.handle_integration_event(coaching_event)
                    integration_layer.record_coaching_complete(
                        coaching_event,
                        response=cue,
                        tier=coaching_event.get("tier", "tier_2"),
                    )
                    print("[LIVE COACHING CUE]")
                    print(f"  {cue}")
                except Exception as e:
                    print(f"[SessionRunner] Error processing event: {e}")
                    import traceback
                    traceback.print_exc()

            elif args.use_langgraph and coaching_graph is not None:
                # ── Legacy path: direct LangGraph invocation ──
                print("[LangGraph] Starting orchestration...")

                initial_state = {
                    "coaching_event": coaching_event,
                    "session_id": integration_layer.session_id,
                    "coaching_history": integration_layer.coaching_history,
                    "cache": integration_layer.cache,
                    "ground_truth_library": gt_library,
                    "tier": coaching_event.get("tier"),
                    "cache_key": coaching_event.get("cache_key"),
                    "routing_reason": coaching_event.get("routing_reason"),
                }

                final_state = coaching_graph.invoke(initial_state)

                integration_layer.record_coaching_complete(
                    coaching_event,
                    final_state["feedback_audio"],
                    final_state["tier_used"],
                )

                print("[Result] ✅ Coaching delivered")
                print(f"  Tier: {final_state['tier_used']}")
                print(f"  Latency: {final_state.get('latency_ms', 0):.0f}ms")
                if final_state.get("used_fallback"):
                    print(f"  Fallback: {final_state.get('fallback_source', 'unknown')} (ground-truth cue used)")
                print(f"  Message: \"{final_state['feedback_audio']}\"")

            else:
                # ── Integration-only mode: show what the integration layer emits ──
                integration_layer.record_coaching_complete(
                    coaching_event,
                    response="(integration-only mode)",
                    tier=coaching_event["tier"],
                )
                print("[Result] ✅ Integration event emitted (no LangGraph)")
                print(f"  Event JSON: {json.dumps(coaching_event)}")
            print()

    except KeyboardInterrupt:
        print("\n[Session] Ctrl+C received — flushing session...")

    finally:
        # ==========================================
        # STEP 4: SESSION SUMMARY (always runs)
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

        fallback_count = sum(
            1 for entry in summary.get("coaching_history", [])
            if entry.get("used_fallback", False)
        )
        if summary['total_events'] > 0:
            fallback_rate = fallback_count / summary['total_events'] * 100
            print(f"Ground Truth Fallbacks: {fallback_count} ({fallback_rate:.0f}%)")

        # ── SessionRunner end-of-session report ──────────────────────────
        if args.session_runner and _session_runner is not None:
            try:
                print("\n[SessionRunner] Generating end-of-session report...")
                sr_summary = _session_runner.end_session()
                print(f"[SessionRunner] Events logged: {sr_summary.get('total_events_logged', 0)}")
                progress = sr_summary.get("progress_report")
                if progress and not progress.get("error"):
                    print(f"\n{'='*60}")
                    print("PROGRESS REPORT")
                    print(f"{'='*60}")
                    print(f"Confidence: {progress.get('confidence_score', 'N/A')}")
                    print(f"\nCoaching Feedback:\n  {progress.get('coaching_feedback', '')}")
                    exercises = progress.get("suggested_exercises", [])
                    if exercises:
                        print("\nSuggested Exercises for Next Session:")
                        for ex in exercises:
                            print(f"  - {ex}")
                    safety = progress.get("safety_notes", [])
                    if safety:
                        print("\nSafety Notes:")
                        for note in safety:
                            print(f"  - {note}")
                    motivational = progress.get("motivational_note", "")
                    if motivational:
                        print(f"\nMotivational Note:\n  {motivational}")
                    sources = progress.get("retrieved_sources", [])
                    if sources:
                        print(f"\nSources ({len(sources)}):")
                        for src in sources:
                            print(f"  - {src[:80]}{'...' if len(src) > 80 else ''}")
                    print(f"{'='*60}")
                elif progress and progress.get("error"):
                    print(f"[SessionRunner] Progress report error: {progress['error']}")
            except Exception as e:
                print(f"[SessionRunner] end_session() failed: {e}")

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
    
    path = Path(filepath)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, 'rt') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                frame = json.loads(line)
                frames.append(frame)
    
    return frames


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CV -> integration pipeline")
    ap.add_argument(
        "--cv-jsonl",
        type=str,
        default="",
        help="Path to infer_stream output (.jsonl or .jsonl.gz). If omitted, uses mock frames.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of input frames for quick tests (0 = all).",
    )
    ap.add_argument(
        "--use-langgraph",
        action="store_true",
        help="Run full LangGraph orchestration after integration filtering.",
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="Follow a growing .jsonl file in real time (for live CV -> integration).",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Polling interval in seconds when --follow is enabled.",
    )
    ap.add_argument(
        "--debug-fast",
        action="store_true",
        help="Relax temporal thresholds for quick live webcam debugging.",
    )
    ap.add_argument(
        "--session-runner",
        action="store_true",
        help="Route coaching events through SessionRunner (CoachingAgent + ProgressTrackerAgent).",
    )
    ap.add_argument(
        "--patient-id",
        type=str,
        default="live_patient",
        help="Patient ID passed to SessionRunner (default: live_patient).",
    )
    ap.add_argument(
        "--injury",
        type=str,
        default="general_rehab",
        help="Injury/condition label passed to SessionRunner (default: general_rehab).",
    )
    return ap.parse_args()


def iter_cv_output_from_file(filepath: str, follow: bool = False, poll_interval: float = 0.1):
    """Yield CV frames from JSONL. Optionally follow appended lines in real time."""
    path = Path(filepath)

    if path.suffix == ".gz":
        # Gzip streams are suitable for offline replay, not for tailing live writes.
        if follow:
            raise ValueError("--follow is not supported with .gz files. Use a plain .jsonl output path.")
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        f.seek(0)
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data.get("eof"):
                        # CV stream wrote its EOF sentinel — exit follow loop cleanly
                        print("[CV] Stream ended (EOF sentinel received)")
                        return
                    yield data
                continue

            if not follow:
                break
            time.sleep(max(0.01, poll_interval))

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