"""
Build the ground-truth coaching cues library from QEVD dataset.

Scans QEVD test/val JSONL files to extract all unique (exercise, mistake) pairs,
then generates coaching cues using a three-level rule engine:
  Level 1 - Exact keyword table (clinician-written cues for ~40 common mistakes)
  Level 2 - Regex pattern rules (handles mistake families)
  Level 3 - Template fallback (always produces something usable)

Optionally, --annotate calls the Anthropic API offline to enrich rule-generated
cues with higher-quality LLM responses (one-time cost; commit the result).

Usage:
    python scripts/build_ground_truth_library.py \
        --test-dir tests/integration_testing/rag_infer_logs_test \
        --val-dir  tests/integration_testing/rag_infer_logs_val \
        --output   data/ground_truth_coaching_cues.json

    # With LLM enrichment:
    python scripts/build_ground_truth_library.py ... --annotate
"""

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Level 1: Exact keyword table
# Keys are lowercased mistake names (or substrings).
# Priority: longest matching key wins.
# ---------------------------------------------------------------------------
EXACT_CUES: Dict[str, str] = {
    # Knee / lower limb
    "knee valgus":           "Drive your knees outward to track over your toes throughout the movement.",
    "knee pain":             "Stop and rest — knee pain is a signal to check your alignment before continuing.",
    "valgus":                "Push your knees out in line with your toes; avoid letting them cave inward.",
    "heel rise":             "Keep your heels pressed firmly into the floor throughout the movement.",
    "heel off":              "Root your heels down — drive through the whole foot, not just your toes.",
    "toes out":              "Point your toes forward or slightly out — avoid excessive flare.",
    "toes in":               "Turn your toes out slightly to align with your knees.",
    "foot flat":             "Keep your entire foot in contact with the floor for a stable base.",
    "ankle":                 "Keep your ankles stable and avoid rolling in or out.",

    # Hip / glutes
    "hip drop":              "Level your hips — engage your glutes to stop the unsupported side from sagging.",
    "hip shift":             "Keep your weight centered; avoid shifting sideways as you move.",
    "hip rotation":          "Square your hips forward and resist any rotation through the movement.",
    "anterior pelvic tilt":  "Tuck your pelvis slightly and brace your core to flatten your lower back.",
    "posterior pelvic tilt": "Allow a gentle natural arch in your lower back; avoid tucking too much.",
    "glute":                 "Squeeze your glutes at the top to complete the movement fully.",

    # Spine / torso
    "lumbar":                "Brace your core and maintain a neutral spine — avoid rounding your lower back.",
    "rounded back":          "Lift your chest and pull your shoulder blades back to straighten your spine.",
    "rounded lower back":    "Engage your core and lift your chest — keep your lumbar spine neutral.",
    "hunched":               "Open your chest and pull your shoulders back and down away from your ears.",
    "forward lean":          "Keep your torso upright — engage your core to stay tall throughout the rep.",
    "leaning forward":       "Keep your torso upright — engage your core to stay tall throughout the rep.",
    "leaning back":          "Bring your chest slightly forward and brace your core to stay balanced.",
    "torso rotation":        "Keep your chest square and resist rotating your torso during the movement.",
    "twisting":              "Stabilize your core and keep your spine facing forward — no rotation.",

    # Shoulder / upper body
    "shoulder":              "Keep your shoulders relaxed and pulled back — avoid shrugging or rounding.",
    "arm":                   "Keep your arms in the correct position throughout the movement.",
    "elbow":                 "Maintain proper elbow alignment — avoid flaring or collapsing inward.",
    "wrist":                 "Keep your wrists neutral and aligned with your forearms.",
    "head":                  "Keep your head in a neutral position — gaze forward, not up or down.",
    "neck":                  "Relax your neck and keep it aligned with your spine.",
    "chin":                  "Keep your chin tucked slightly — avoid jutting it forward.",

    # Range of motion / tempo
    "not moving":            "Keep moving through the full range — don't pause or stop mid-rep.",
    "low range of motion":   "Aim for a fuller range — push deeper into the movement for maximum benefit.",
    "insufficient":          "Extend further through the full range of motion on each repetition.",
    "incomplete":            "Complete each rep fully before returning to the start position.",
    "too fast":              "Slow down and control the movement — quality beats speed every time.",
    "moving slow":           "Maintain a steady, consistent tempo — avoid pausing during the movement.",
    "no movement":           "Keep the movement continuous — push through the full range of each rep.",

    # Balance / weight distribution
    "weight distribution":   "Balance your weight evenly across both feet for a stable base.",
    "off balance":           "Find your balance point before moving — engage your core for stability.",

    # General form
    "pain":                  "Ease off — pain is a signal. Reduce range or intensity, and check your alignment.",
    "dangerous":             "Stop and reset your form — safety first before continuing.",
    "compensation":          "Focus on the target muscles; avoid compensating with other body parts.",
    "asymmetry":             "Both sides should move equally — check that you're not favouring one side.",
    "wrong order":           "Focus on the correct movement sequence — watch the demonstration if needed.",
}

# ---------------------------------------------------------------------------
# Level 2: Regex pattern rules
# Each entry is (compiled_regex, cue_template).
# Use {exercise} and {direction} as placeholders filled at generation time.
# ---------------------------------------------------------------------------
PATTERN_RULES = [
    # "not moving - up/down/left/right"
    (re.compile(r"not moving\s*[-–]\s*(up|down|left|right|forward|back)", re.I),
     "Drive all the way {direction} on each rep — keep the movement continuous."),

    # "too fast" / "moving too fast"
    (re.compile(r"(moving\s+)?too\s+fast", re.I),
     "Slow down — control each rep with a steady, deliberate tempo."),

    # "too slow" / "moving too slow"
    (re.compile(r"(moving\s+)?too\s+slow", re.I),
     "Keep a consistent tempo — avoid pausing mid-movement."),

    # "low range of motion" or "insufficient range"
    (re.compile(r"(low|insufficient|limited)\s+range(\s+of\s+motion)?", re.I),
     "Push through a fuller range of motion — aim for complete extension and flexion."),

    # "not bending (back|left|right) leg" or "leg not bending"
    (re.compile(r"(not bending|leg not bending)", re.I),
     "Bend your knee fully to load the working muscles through the complete range."),

    # "leaning (too far) (forward|back|left|right)"
    (re.compile(r"leaning\s+(too\s+far\s+)?(forward|back|left|right)", re.I),
     "Keep your torso upright — brace your core to resist leaning."),

    # "kicking (too) (high|low)"
    (re.compile(r"kicking\s+(too\s+)?(high|low)", re.I),
     "Control your kick height — aim for the target range with smooth, controlled motion."),

    # "shoulders off the ground"
    (re.compile(r"shoulders?\s+off\s+(the\s+)?ground", re.I),
     "Press your shoulders gently into the mat — keep them grounded throughout."),

    # "hips off the ground"
    (re.compile(r"hips?\s+off\s+(the\s+)?ground", re.I),
     "Keep your hips down — engage your core to prevent them from lifting."),

    # "non-working leg (bent|straight)"
    (re.compile(r"non.?working\s+leg\s+(bent|straight)", re.I),
     "Keep your non-working leg in position — focus on isolating the working side."),

    # "looking down" / "looking up" / "head (too) (high|low)"
    (re.compile(r"(looking\s+(down|up)|head\s+(too\s+)?(high|low|forward|back))", re.I),
     "Keep your gaze forward and your head in line with your spine."),

    # "arms not straight" / "legs not straight"
    (re.compile(r"(arms?|legs?)\s+not\s+straight", re.I),
     "Extend fully — straighten the joint at the end of each rep."),

    # "below 90 degrees" / "legs 90 degrees"
    (re.compile(r"(below|above|at)\s+90\s+degrees?", re.I),
     "Aim for the correct joint angle — check your depth against the target position."),

    # "depth=N" or "squat_depth=N" (metric labels — skip)
    (re.compile(r"(depth|squat_depth|jump_height|rom_level|height_level|speed_rps)\s*[=:]\s*\d", re.I),
     None),  # None means: skip this pair (non-actionable metric label)
]

# ---------------------------------------------------------------------------
# Metric label detector (for filtering non-actionable CV model outputs)
# ---------------------------------------------------------------------------
METRIC_LABEL_RE = re.compile(
    r"^(depth|squat_depth|jump_height|rom_level|height_level|speed_rps|torso_rotation"
    r"|direction|no_obvious_issue|quality)\s*[=:]\s*",
    re.I
)

def is_metric_label(mistake: str) -> bool:
    return bool(METRIC_LABEL_RE.match(mistake.strip()))


# ---------------------------------------------------------------------------
# Three-level cue generator
# ---------------------------------------------------------------------------

def generate_cue(exercise: str, mistake: str) -> Tuple[Optional[str], str]:
    """
    Returns (cue, source) where source is one of:
      "exact_keyword", "pattern_rule", "template", or None (metric label → skip)
    """
    mistake_lower = mistake.lower().strip()

    # Pre-filter: metric labels produce no actionable cue
    if is_metric_label(mistake):
        return None, "metric_label"

    # Level 1: exact keyword match (longest key wins)
    matched_key = None
    for key in sorted(EXACT_CUES, key=len, reverse=True):
        if key in mistake_lower:
            matched_key = key
            break
    if matched_key:
        return EXACT_CUES[matched_key], "exact_keyword"

    # Level 2: regex pattern rules
    for pattern, template in PATTERN_RULES:
        m = pattern.search(mistake_lower)
        if m:
            if template is None:
                return None, "metric_label"
            # Fill direction placeholder if captured
            direction = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
            cue = template.format(direction=direction, exercise=exercise)
            return cue, "pattern_rule"

    # Level 3: template fallback (always succeeds)
    cue = (
        f"Pay attention to your {mistake} — "
        f"maintain control and proper alignment throughout each {exercise} repetition."
    )
    return cue, "template"


# ---------------------------------------------------------------------------
# QEVD scanner
# ---------------------------------------------------------------------------

def scan_directory(directory: str, pairs: Dict) -> int:
    """
    Walk .jsonl.gz files in directory, extract (exercise, mistake) pairs.
    pairs dict is updated in-place: key → best-confidence record.
    Returns count of files processed.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"  [warn] Directory not found, skipping: {directory}")
        return 0

    files = list(dir_path.glob("**/*.jsonl.gz")) + list(dir_path.glob("**/*.jsonl"))
    print(f"  Scanning {len(files)} files in {directory} ...")

    processed = 0
    for fpath in files:
        opener = gzip.open if fpath.suffix == ".gz" else open
        try:
            with opener(fpath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Skip metadata lines
                    if "__meta__" in obj:
                        continue

                    exercise_info = obj.get("exercise", {})
                    ex_name = exercise_info.get("name", "").strip()
                    ex_p = float(exercise_info.get("p", 0.0))

                    if not ex_name or ex_p < 0.3:
                        continue

                    mistakes = obj.get("mistakes", [])
                    for m in mistakes:
                        mk_name = m.get("name", "").strip()
                        mk_p = float(m.get("p", 0.0))

                        if not mk_name or mk_p < 0.3:
                            continue
                        if is_metric_label(mk_name):
                            continue

                        key = _make_key(ex_name, mk_name)
                        score = ex_p * mk_p

                        if key not in pairs or score > pairs[key]["score"]:
                            pairs[key] = {
                                "exercise": ex_name,
                                "mistake": mk_name,
                                "score": score,
                                "ex_p": ex_p,
                                "mk_p": mk_p,
                            }
            processed += 1
        except Exception as e:
            print(f"  [warn] Could not read {fpath}: {e}")

    return processed


def _make_key(exercise: str, mistake: str) -> str:
    """Canonical key: exercise__mistake (double underscore separator)."""
    def norm(s):
        return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")
    return f"{norm(exercise)}__{norm(mistake)}"


# ---------------------------------------------------------------------------
# Optional LLM annotation pass
# ---------------------------------------------------------------------------

def annotate_with_llm(pairs_output: Dict) -> Dict:
    """
    Upgrade rule-generated cues to LLM quality using the Anthropic API.
    Only processes entries with source != "llm_offline".
    Batch-processes all at once; updates in-place and returns the dict.
    """
    try:
        import anthropic
    except ImportError:
        print("[annotate] anthropic package not installed. Run: pip install anthropic")
        return pairs_output

    client = anthropic.Anthropic()
    model = "claude-haiku-4-5-20251001"  # Fast + cheap for offline batch

    to_upgrade = [
        (k, v) for k, v in pairs_output.items()
        if v.get("source") != "llm_offline"
    ]

    print(f"[annotate] Upgrading {len(to_upgrade)} cues via {model} ...")

    for i, (key, entry) in enumerate(to_upgrade):
        exercise = entry["exercise"]
        mistake = entry["mistake"]
        prompt = (
            f"Generate exactly one actionable coaching cue (12-20 words) for a rehabilitation patient.\n"
            f"Exercise: {exercise}\n"
            f"Mistake detected: {mistake}\n\n"
            f"Rules:\n"
            f"- Imperative voice (start with a verb)\n"
            f"- Specific body part or movement instruction\n"
            f"- No filler words, no preamble\n"
            f"- Output ONLY the cue, nothing else"
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            cue = response.content[0].text.strip().strip('"').strip("'")
            if cue and len(cue.split()) >= 6:
                pairs_output[key]["cue"] = cue
                pairs_output[key]["source"] = "llm_offline"
                pairs_output[key]["confidence"] = 0.9
        except Exception as e:
            print(f"  [warn] LLM call failed for {key}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(to_upgrade)} done")

    print(f"[annotate] Done.")
    return pairs_output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build ground-truth coaching cues from QEVD data")
    ap.add_argument("--test-dir", default="tests/integration_testing/rag_infer_logs_test",
                    help="Path to QEVD test split directory")
    ap.add_argument("--val-dir", default="tests/integration_testing/rag_infer_logs_val",
                    help="Path to QEVD val split directory")
    ap.add_argument("--train-dir", default="tests/integration_testing/rag_infer_logs_train",
                    help="Path to QEVD train split directory")
    ap.add_argument("--output", default="data/ground_truth_coaching_cues.json",
                    help="Output JSON path")
    ap.add_argument("--annotate", action="store_true",
                    help="Upgrade rule-generated cues via Anthropic API (one-time, offline)")
    ap.add_argument("--min-confidence", type=float, default=0.3,
                    help="Minimum exercise*mistake confidence score to include a pair")
    args = ap.parse_args()

    print("=" * 60)
    print("Building Ground-Truth Coaching Library from QEVD Data")
    print("=" * 60)

    # --- Scan QEVD files ---
    raw_pairs: Dict = {}

    for label, directory in [("test", args.test_dir), ("val", args.val_dir), ("train", args.train_dir)]:
        if not directory:
            continue
        print(f"\n[Scan] {label} split:")
        n = scan_directory(directory, raw_pairs)
        print(f"  → {n} files processed, {len(raw_pairs)} unique pairs so far")

    if not raw_pairs:
        print("\n[Error] No pairs extracted. Check that --test-dir / --val-dir exist and contain .jsonl.gz files.")
        sys.exit(1)

    print(f"\n[Build] {len(raw_pairs)} unique (exercise, mistake) pairs extracted")

    # --- Generate cues ---
    output_pairs: Dict = {}
    skipped_metric = 0
    source_counts = {"exact_keyword": 0, "pattern_rule": 0, "template": 0}

    for key, record in sorted(raw_pairs.items()):
        exercise = record["exercise"]
        mistake = record["mistake"]

        cue, source = generate_cue(exercise, mistake)

        if cue is None:
            skipped_metric += 1
            continue

        confidence = {"exact_keyword": 0.85, "pattern_rule": 0.75, "template": 0.6}.get(source, 0.6)

        output_pairs[key] = {
            "exercise": exercise,
            "mistake": mistake,
            "cue": cue,
            "source": source,
            "confidence": confidence,
        }
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"[Build] {len(output_pairs)} actionable pairs generated")
    print(f"  exact_keyword : {source_counts['exact_keyword']}")
    print(f"  pattern_rule  : {source_counts['pattern_rule']}")
    print(f"  template      : {source_counts['template']}")
    print(f"  skipped (metric labels): {skipped_metric}")

    # --- Optional LLM annotation ---
    if args.annotate:
        print("\n[Annotate] Starting LLM enrichment pass ...")
        output_pairs = annotate_with_llm(output_pairs)

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "version": "1.0",
        "generated_from": "qevd_scan",
        "total_pairs": len(output_pairs),
        "pairs": output_pairs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Wrote {len(output_pairs)} pairs to {output_path}")
    print("       Commit this file to the repo as a static artifact.")


if __name__ == "__main__":
    main()
