"""
Build Tier 1 cache defaults from the ground-truth coaching cues library.

Reads data/ground_truth_coaching_cues.json, filters to high-confidence
entries, selects top common mistake types, and writes cache/tier1_defaults.json.

Usage:
    python scripts/build_tier1_cache.py [--top-n 50] [--min-confidence 0.8]
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Safety-critical keywords → immediate timing
CRITICAL_KEYWORDS = [
    "not moving", "knee valgus", "lumbar", "pain", "dangerous",
]

# Form-correction keywords → rep_end timing
FORM_KEYWORDS = [
    "twisting", "fast", "incomplete", "range", "arm raise", "lean",
]


def norm(s: str) -> str:
    """Normalize string to cache-key component (matches GroundTruthLibrary._make_key)."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")


def make_cache_key(exercise: str, mistake: str) -> str:
    """Double-underscore separated cache key."""
    return f"{norm(exercise)}__{norm(mistake)}"


def assign_timing(mistake: str) -> str:
    """Assign delivery timing based on mistake keywords."""
    lower = mistake.lower()
    if any(kw in lower for kw in CRITICAL_KEYWORDS):
        return "immediate"
    return "rep_end"


def main():
    parser = argparse.ArgumentParser(description="Build Tier 1 cache defaults")
    parser.add_argument(
        "--gt-path",
        default="data/ground_truth_coaching_cues.json",
        help="Path to ground truth coaching cues JSON",
    )
    parser.add_argument(
        "--output",
        default="cache/tier1_defaults.json",
        help="Output path for tier 1 cache defaults",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top mistake types to include",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for cache entries",
    )
    args = parser.parse_args()

    # Load ground truth
    gt_path = Path(args.gt_path)
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        return

    with open(gt_path) as f:
        data = json.load(f)

    pairs = data.get("pairs", {})
    print(f"[INFO] Loaded {len(pairs)} ground-truth pairs")

    # Filter to high-confidence entries
    high_conf = {}
    for key, entry in pairs.items():
        confidence = entry.get("confidence", 0)
        source = entry.get("source", "")
        if confidence >= args.min_confidence and source == "exact_keyword":
            high_conf[key] = entry

    print(f"[INFO] {len(high_conf)} pairs pass confidence >= {args.min_confidence} + exact_keyword filter")

    # Count mistake frequency to find top-N
    mistake_counts = Counter()
    for entry in high_conf.values():
        mistake_counts[entry["mistake"]] += 1

    top_mistakes = set(m for m, _ in mistake_counts.most_common(args.top_n))
    print(f"[INFO] Top {len(top_mistakes)} mistake types selected")

    # Build cache entries
    cache = {}
    for entry in high_conf.values():
        if entry["mistake"] not in top_mistakes:
            continue

        exercise = entry.get("exercise", "")
        mistake = entry.get("mistake", "")
        cue = entry.get("cue", "")

        if not exercise or not mistake or not cue:
            continue

        cache_key = make_cache_key(exercise, mistake)
        # Keep first (highest confidence) entry per key
        if cache_key not in cache:
            cache[cache_key] = {
                "response": cue,
                "timing": assign_timing(mistake),
            }

    print(f"[INFO] Built {len(cache)} cache entries")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"[INFO] Cache written to {output_path}")

    # Summary stats
    timing_counts = Counter(v["timing"] for v in cache.values())
    print(f"[INFO] Timing distribution: {dict(timing_counts)}")


if __name__ == "__main__":
    main()
