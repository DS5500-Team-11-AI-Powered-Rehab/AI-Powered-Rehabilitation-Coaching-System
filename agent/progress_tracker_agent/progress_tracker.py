"""
progress_tracker.py — Progress Tracker Agent Core
Reads all phase JSON files from phase_outputs/ and generates
a longitudinal progress report using LLM.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# ── Phase outputs directory ───────────────────────────────────────────────────
PHASE_OUTPUTS_DIR = Path(__file__).parent.parent / "phase_outputs"


# ── Prompt ────────────────────────────────────────────────────────────────────

PROGRESS_REPORT_PROMPT = """You are a physiotherapy progress analyst reviewing a patient's rehabilitation journey across multiple sessions.

PATIENT: {condition}
Goal: {goals}
Total phases analyzed: {num_phases}
Phase progression: {phase_progression}

PAIN TREND ACROSS PHASES:
{pain_trend}

QUALITY SCORE TREND ACROSS PHASES:
{quality_trend}

MISTAKES ACROSS PHASES (frequency):
{mistake_trend}

EXERCISE PROGRESSION:
{exercise_progression}

PHASE REPORTS SUMMARY:
{phase_reports_summary}

Write a longitudinal progress report (under 400 words) with these sections:

1. OVERALL PROGRESS: 2-3 sentences summarising the patient's journey so far
2. PAIN & FUNCTION TREND: how pain and movement quality have changed
3. PERSISTENT ISSUES: mistakes that keep appearing across phases
4. KEY IMPROVEMENTS: what the patient has genuinely gotten better at
5. RECOMMENDATIONS FOR NEXT PHASE: 3 specific, actionable focus areas
6. LONG-TERM OUTLOOK: one paragraph connecting progress to the patient's goal

Be clinically grounded, specific, and encouraging."""


# ── JSON Loader ───────────────────────────────────────────────────────────────

def load_phase_jsons(patient_id: str = None) -> List[Dict]:
    """
    Load all phase JSON files from phase_outputs/.
    Optionally filter by patient_id.
    Returns list sorted by phase_start_ts (chronological order).
    """
    if not PHASE_OUTPUTS_DIR.exists():
        print(f"phase_outputs/ not found at {PHASE_OUTPUTS_DIR}")
        return []

    json_files = sorted(PHASE_OUTPUTS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found in phase_outputs/")
        return []

    phases = []
    for f in json_files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            # Filter by patient_id if specified
            if patient_id:
                pid = data.get("phase_summary", {}).get("patient_id", "")
                if pid != patient_id:
                    continue

            data["_filename"] = f.name
            phases.append(data)
        except Exception as e:
            print(f"  Warning: could not load {f.name}: {e}")

    # Sort chronologically
    phases.sort(key=lambda d: d.get("phase_summary", {}).get("phase_start_ts", 0))
    print(f"Loaded {len(phases)} phase file(s)")
    return phases


# ── Trend Analyzers ───────────────────────────────────────────────────────────

def analyze_pain_trend(phases: List[Dict]) -> Dict:
    """Extract pain level per phase and compute trend."""
    pain_values = [
        p["phase_summary"].get("pain_level", 0) for p in phases
    ]
    trend = "stable"
    if len(pain_values) >= 2:
        delta = pain_values[-1] - pain_values[0]
        if delta < -1:
            trend = "improving"
        elif delta > 1:
            trend = "worsening"

    return {
        "values":       pain_values,
        "trend":        trend,
        "first":        pain_values[0] if pain_values else None,
        "latest":       pain_values[-1] if pain_values else None,
        "total_change": pain_values[-1] - pain_values[0] if len(pain_values) >= 2 else 0,
    }


def analyze_quality_trend(phases: List[Dict]) -> Dict:
    """Extract avg quality score per phase and compute trend."""
    quality_per_phase = []
    for p in phases:
        exercises = p.get("exercises", [])
        if exercises:
            avg = sum(ex.get("avg_quality", 0) for ex in exercises) / len(exercises)
            quality_per_phase.append(round(avg, 2))
        else:
            quality_per_phase.append(None)

    valid = [q for q in quality_per_phase if q is not None]
    trend = "stable"
    if len(valid) >= 2:
        delta = valid[-1] - valid[0]
        if delta > 0.05:
            trend = "improving"
        elif delta < -0.05:
            trend = "declining"

    return {
        "values_per_phase": quality_per_phase,
        "trend":            trend,
        "first":            valid[0] if valid else None,
        "latest":           valid[-1] if valid else None,
    }


def analyze_mistake_trend(phases: List[Dict]) -> Dict:
    """Aggregate mistakes across all phases, ranked by total occurrences."""
    mistake_totals: Dict[str, int] = {}
    mistake_phases: Dict[str, List[str]] = {}

    for p in phases:
        phase_label = p["phase_summary"].get("rehab_phase", "?")
        for ex in p.get("exercises", []):
            for m in ex.get("mistakes", []):
                mtype = m.get("type", "unknown")
                occ   = m.get("occurrences", 0)
                mistake_totals[mtype] = mistake_totals.get(mtype, 0) + occ
                if mtype not in mistake_phases:
                    mistake_phases[mtype] = []
                if phase_label not in mistake_phases[mtype]:
                    mistake_phases[mtype].append(phase_label)

    ranked = sorted(mistake_totals.items(), key=lambda x: x[1], reverse=True)
    return {
        "ranked":         ranked,            # [(mistake_type, total_occurrences), ...]
        "mistake_phases": mistake_phases,    # {mistake_type: [phases it appeared in]}
        "persistent":     [m for m, _ in ranked if len(mistake_phases[m]) >= 2],
    }


def analyze_exercise_progression(phases: List[Dict]) -> List[str]:
    """List exercises per phase to show progression."""
    lines = []
    for p in phases:
        phase_label = p["phase_summary"].get("rehab_phase", "?")
        exercises   = [ex["exercise_name"] for ex in p.get("exercises", [])]
        lines.append(f"{phase_label}: {', '.join(exercises) if exercises else 'none recorded'}")
    return lines


# ── Prompt Formatters ─────────────────────────────────────────────────────────

def format_pain_trend(pain: Dict) -> str:
    values = " → ".join(str(v) for v in pain["values"])
    return (
        f"Pain levels per phase: {values}/10\n"
        f"Overall trend: {pain['trend']} "
        f"(change: {pain['total_change']:+d} pts)"
    )


def format_quality_trend(quality: Dict) -> str:
    values = " → ".join(
        str(v) if v is not None else "N/A"
        for v in quality["values_per_phase"]
    )
    return (
        f"Avg quality scores per phase: {values}\n"
        f"Overall trend: {quality['trend']}"
    )


def format_mistake_trend(mistakes: Dict) -> str:
    if not mistakes["ranked"]:
        return "No mistakes recorded."
    lines = []
    for mtype, total in mistakes["ranked"][:5]:   # top 5
        phases_seen = mistakes["mistake_phases"].get(mtype, [])
        persistent  = " ⚠️ PERSISTENT" if mtype in mistakes["persistent"] else ""
        lines.append(f"- {mtype}: {total} total occurrences across {phases_seen}{persistent}")
    return "\n".join(lines)


def format_phase_reports_summary(phases: List[Dict]) -> str:
    lines = []
    for p in phases:
        phase_label = p["phase_summary"].get("rehab_phase", "?")
        report_text = p.get("phase_report", "")
        # Take first 200 chars of each phase report as summary
        snippet = report_text[:200].replace("\n", " ").strip()
        lines.append(f"[{phase_label.upper()} PHASE]: {snippet}...")
    return "\n\n".join(lines)


# ── Main ProgressTracker Class ────────────────────────────────────────────────

class ProgressTracker:
    """
    Reads all phase JSON files from phase_outputs/ and generates
    a longitudinal progress report using gemma3:4b.
    """

    def __init__(
        self,
        ollama_model:    str  = "gemma3:4b",
        ollama_base_url: str  = "http://localhost:11434",
        verbose:         bool = True,
    ):
        self.verbose = verbose
        self.llm = OllamaLLM(
            model       = ollama_model,
            base_url    = ollama_base_url,
            temperature = 0.5,
            num_predict = 600,
            num_ctx     = 2048,
        )
        self.parser = StrOutputParser()
        if verbose:
            print("ProgressTracker ready ✓")

    def run(self, patient_id: str = None) -> Dict:
        """
        Full pipeline:
        Load JSONs → Analyze trends → Generate LLM report → Return structured result.
        """

        # 1. Load
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  PROGRESS TRACKER")
            print(f"  Reading from: {PHASE_OUTPUTS_DIR}")
            print(f"{'='*60}")

        phases = load_phase_jsons(patient_id)
        if not phases:
            return {"error": "No phase data found."}

        # 2. Analyze
        pain_analysis     = analyze_pain_trend(phases)
        quality_analysis  = analyze_quality_trend(phases)
        mistake_analysis  = analyze_mistake_trend(phases)
        exercise_progress = analyze_exercise_progression(phases)

        condition = phases[0]["phase_summary"].get("condition", "musculoskeletal condition")
        goals     = phases[0]["phase_summary"].get("goals", "improve function")
        phase_progression = " → ".join(
            p["phase_summary"].get("rehab_phase", "?") for p in phases
        )

        if self.verbose:
            print(f"\n  Patient condition : {condition}")
            print(f"  Phases analyzed   : {len(phases)}")
            print(f"  Phase progression : {phase_progression}")
            print(f"  Pain trend        : {pain_analysis['trend']}")
            print(f"  Quality trend     : {quality_analysis['trend']}")
            print(f"  Persistent issues : {mistake_analysis['persistent']}")
            print(f"\n  Generating progress report...")

        # 3. Build prompt
        prompt = PROGRESS_REPORT_PROMPT.format(
            condition            = condition,
            goals                = goals,
            num_phases           = len(phases),
            phase_progression    = phase_progression,
            pain_trend           = format_pain_trend(pain_analysis),
            quality_trend        = format_quality_trend(quality_analysis),
            mistake_trend        = format_mistake_trend(mistake_analysis),
            exercise_progression = "\n".join(exercise_progress),
            phase_reports_summary= format_phase_reports_summary(phases),
        )

        # 4. Generate
        try:
            result      = self.llm.invoke(prompt)
            report_text = result if isinstance(result, str) else self.parser.invoke(result)
        except Exception as e:
            report_text = f"Report generation failed: {e}"

        # 5. Return structured output
        return {
            "patient_id":         patient_id or "all",
            "condition":          condition,
            "phases_analyzed":    len(phases),
            "phase_progression":  phase_progression,
            "pain_analysis":      pain_analysis,
            "quality_analysis":   quality_analysis,
            "mistake_analysis":   {
                "ranked":     mistake_analysis["ranked"],
                "persistent": mistake_analysis["persistent"],
            },
            "exercise_progression": exercise_progress,
            "progress_report":    report_text,
            "generated_at":       datetime.utcnow().isoformat() + "Z",
        }
