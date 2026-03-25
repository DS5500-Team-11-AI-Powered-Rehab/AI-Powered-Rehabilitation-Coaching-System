# Integration Layer

**Intelligent temporal filtering between CV model and LLM coaching pipeline**

---

## Overview

The Integration Layer acts as a preprocessing filter that transforms noisy, high-frequency CV predictions into actionable coaching events. It sits between the computer vision model (15 FPS) and the LLM-based coaching system, reducing noise by ~95% while preserving all clinically significant patterns.

### The Problem

- CV model outputs **15 frames/second** = 900 predictions/minute
- Raw coaching on every frame would overwhelm users
- Need intelligent filtering to identify **persistent** mistakes worth coaching

### The Solution

Four-stage pipeline that processes CV frames in real-time:

1. **Temporal Aggregation** - Sliding window analysis to detect persistent patterns
2. **Deduplication** - Session memory to avoid repetitive coaching
3. **Severity Classification** - Risk assessment using keyword matching and quality scores
4. **Intelligent Routing** - Three-tier system for optimal response latency

---

## Architecture

```
CV Stream (15 FPS) → Integration Layer → Coaching Event → LangGraph → User
                            ↓
                    [Cache | RAG | Reasoning]
```

### Core Components

#### 1. Temporal Aggregation (`_find_persistent_mistakes`)
- **10-second sliding window** (150 frames at 15 FPS)
- Filters for mistakes that:
  - Persist in **≥30% of frames**
  - Maintain **≥0.35 confidence**
  - Last **≥3 seconds**

#### 2. Deduplication (`_should_coach`)
- Tracks coaching history per session
- **10-second cooldown** between any coaching
- **Re-coaches** only if mistake persists 20+ seconds after first coaching

#### 3. Severity Classification (`_calculate_severity`)
- **High:** Safety-critical keywords (knee valgus, lumbar, pain)
- **Medium:** Form issues (twisting, incomplete range)
- **Low:** Optimization opportunities

#### 4. Three-Tier Routing (`_route_to_tier`)
- **Tier 1 (Cache)**: Common patterns → 50ms response
- **Tier 2 (RAG)**: Standard cases → 1-2s response  
- **Tier 3 (Reasoning)**: Complex situations → 3-5s response

---

## Usage

### Basic Usage

```python
from integration.integration_layer import IntegrationLayer

# Initialize for a session
layer = IntegrationLayer(session_id="patient_123")
layer.cache.populate_defaults()

# Process CV frames
for cv_frame in cv_stream:
    coaching_event = layer.process_frame(cv_frame)
    
    if coaching_event:
        # Route to LangGraph for response generation
        response = langgraph.process(coaching_event)
        deliver_feedback(response)
```

### Input Format (CV Frame)

```python
{
    "timestamp_s": 45.2,
    "frame_index": 678,
    "exercise": {"name": "squat", "p": 0.92},
    "mistakes": [
        {"name": "knee_valgus", "p": 0.81},
        {"name": "incomplete_depth", "p": 0.62}
    ],
    "metrics": {...},
    "quality_score": 0.75
}
```

### Output Format (Coaching Event)

```python
{
    "event_id": "session_123_event_5",
    "timestamp": 45.2,
    "exercise": {"name": "squat", "confidence": 0.92},
    "mistake": {
        "type": "knee_valgus",
        "confidence": 0.81,
        "duration_seconds": 4.2,
        "persistence_rate": 0.42
    },
    "severity": "high",
    "tier": "tier_2",
    "routing_reason": "high severity mistake needs RAG context",
    "is_recoaching": false
}
```

---

## Configuration

Adjust thresholds via `Config` class:

```python
from integration.integration_layer import IntegrationLayer, Config

config = Config()
config.MIN_PERSISTENCE_RATE = 0.25  # Relax to 25%
config.MIN_DURATION_SECONDS = 2.5   # Reduce to 2.5s
config.MIN_COACHING_INTERVAL = 15   # Increase cooldown

layer = IntegrationLayer(session_id="test", config=config)
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MIN_PERSISTENCE_RATE` | 0.30 | Minimum % of frames mistake appears |
| `MIN_CONFIDENCE` | 0.35 | Average confidence threshold |
| `MIN_DURATION_SECONDS` | 3.0 | Minimum time mistake persists |
| `MIN_COACHING_INTERVAL` | 10 | Cooldown between coaching (seconds) |
| `RE_COACHING_THRESHOLD` | 20 | When to re-coach same mistake (seconds) |

---

## Testing

Comprehensive test suite validates temporal filtering logic on both real CV event logs and synthetic continuous frame data.

### Quick Test
```bash
cd tests/integration_testing
conda activate rehab_ai_env
python test_integration.py --dataset train --max-videos 10 --config tuned_config.json
```

### Full Test Suite
```bash
# Run all tests + analysis + visualization
./quick_start_test.sh
```

### Test Results (1,509 Videos)
- **Event rate:** 0.3 events/video average
- **Noise reduction:** ~95% filtering of transient signals
- **Coverage:** 27-32% of videos generate coaching events
- **Tier distribution:** Balanced routing across all three tiers

See [tests/integration_testing/TEST_README.md](../../tests/integration_testing/TEST_README.md) for detailed testing documentation.

---

## File Structure

```
src/integration/
├── integration_layer.py    # Core filtering logic
├── graph.py                # LangGraph workflow (future)
├── state.py                # State definitions (future)
├── main.py                 # Entry point (future)
└── README.md               # This file
```

---

## Key Design Decisions

### Why 10-second window?
- Balances responsiveness vs. noise filtering
- Captures 3-5 exercise repetitions in typical movements
- Sufficient history for statistical confidence

### Why 30% persistence threshold?
- Eliminates transient false positives from CV jitter
- Requires consistent detection across multiple frames
- Validated through empirical testing on 1,500+ videos

### Why three tiers?
- **Tier 1:** Sub-second feedback for common patterns (user experience)
- **Tier 2:** RAG retrieval for context-aware coaching (quality)
- **Tier 3:** Deep reasoning for complex situations (safety)

### Why cooldown periods?
- Prevents feedback spam during continuous movement
- Allows user time to process and correct
- Mimics natural human coaching cadence

---

## Performance

- **Frame processing:** <100ms per frame (target)
- **Memory footprint:** ~1MB per session (150-frame buffer)
- **Cache lookups:** <5ms via diskcache
- **Event generation rate:** ~0.3 events/video (95% noise reduction)

---

## Future Enhancements

- [ ] Adaptive thresholds based on patient history
- [ ] Multi-mistake pattern detection (compensatory movements)
- [ ] Exercise-specific threshold profiles
- [ ] Real-time threshold adjustment via reinforcement learning
- [ ] Integration with patient progress tracking

---

## Contributing

When modifying filtering logic:

1. **Update tests** - Add cases to `tests/integration_testing/test_integration.py`
2. **Validate thresholds** - Run full test suite on both datasets
3. **Document changes** - Update this README and inline comments
4. **Verify performance** - Check event rate stays within target range

---

## References

- **Testing Guide:** [tests/integration_testing/TEST_README.md](../../tests/integration_testing/TEST_README.md)
- **Datasets:** CV event logs (`rag_infer_logs_train`, `rag_infer_logs_test`, `rag_infer_logs_val`) plus synthetic evaluation sets
- **LangGraph Integration:** Coming in Week 8

---

**Author:** David Ryan  
**Last Updated:** March 20, 2026  
**Status:** Testing Complete, Ready for LangGraph Integration
