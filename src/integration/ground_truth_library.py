"""
GroundTruthLibrary: fast in-memory lookup for curated coaching cues.

Loaded once at startup from data/ground_truth_coaching_cues.json (built by
scripts/build_ground_truth_library.py).  All methods are O(1) dict lookups —
no disk I/O at inference time.

The library is used by quality_gate_node in graph.py as a fallback when the
Claude coaching agent produces a low-quality response.
"""

import json
import re
from pathlib import Path
from typing import Optional

# Regex for metric-label mistake names that are non-actionable CV model outputs
# e.g. "depth=1", "squat_depth=3", "jump_height=2", "rom_level=4"
_METRIC_LABEL_RE = re.compile(
    r"^(depth|squat_depth|jump_height|rom_level|height_level|speed_rps"
    r"|torso_rotation|direction|no_obvious_issue|quality)\s*[=:]\s*",
    re.IGNORECASE,
)


class GroundTruthLibrary:
    """
    In-memory lookup table: (exercise, mistake) → curated coaching cue.

    Usage:
        lib = GroundTruthLibrary("data/ground_truth_coaching_cues.json")
        cue = lib.lookup("squat", "knee valgus")
        # → "Drive your knees outward to track over your toes throughout the movement."
    """

    def __init__(self, json_path: str):
        """
        Load the ground-truth library from a JSON file.

        Args:
            json_path: Path to ground_truth_coaching_cues.json.
                       If the file does not exist, the library starts empty
                       (template_fallback still works).
        """
        self._pairs: dict = {}
        path = Path(json_path)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._pairs = data.get("pairs", {})
            print(f"[GroundTruthLibrary] Loaded {len(self._pairs)} pairs from {path}")
        else:
            print(f"[GroundTruthLibrary] File not found: {path}. "
                  "Run scripts/build_ground_truth_library.py to generate it. "
                  "Falling back to template-only mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, exercise: str, mistake: str) -> Optional[str]:
        """
        Return the curated coaching cue for this (exercise, mistake) pair,
        or None if not found or if the mistake is a non-actionable metric label.

        Args:
            exercise: Exercise name from coaching event (e.g. "squat").
            mistake:  Mistake type from coaching event (e.g. "knee valgus").

        Returns:
            Coaching cue string, or None.
        """
        if self._is_metric_label(mistake):
            return None

        key = self._make_key(exercise, mistake)
        entry = self._pairs.get(key)
        if entry:
            return entry.get("cue")

        # Try exercise-agnostic lookup (mistake only, exercise="any")
        generic_key = self._make_key("any", mistake)
        entry = self._pairs.get(generic_key)
        if entry:
            return entry.get("cue")

        return None

    def template_fallback(self, exercise: str, mistake: str) -> str:
        """
        Always returns a usable coaching cue via a simple template.
        Called when lookup() returns None.

        Args:
            exercise: Exercise name.
            mistake:  Mistake type.

        Returns:
            Template-generated coaching cue string.
        """
        # Clean up display names
        ex_display = exercise.replace("_", " ").strip()
        mk_display = mistake.replace("_", " ").strip()
        return (
            f"Pay attention to your {mk_display} — "
            f"maintain control and proper alignment throughout each {ex_display} repetition."
        )

    def __len__(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(exercise: str, mistake: str) -> str:
        """
        Canonical lookup key matching the format used by build_ground_truth_library.py.
        Double-underscore separates exercise from mistake to avoid collision with
        hyphenated or multi-word names.
        """
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")
        return f"{norm(exercise)}__{norm(mistake)}"

    @staticmethod
    def _is_metric_label(mistake: str) -> bool:
        """
        Return True for non-actionable CV model ordinal outputs
        like "depth=1", "squat_depth=3", "jump_height=2".
        These should pass through the quality gate unchanged.
        """
        return bool(_METRIC_LABEL_RE.match(mistake.strip()))
