"""
session_manager.py — Session State Machine for Coaching Agent
Manages the lifecycle of exercises and rehab phases.

Timing rules:
  - 30s  no new event → exercise ended  → generate exercise summary
  - 120s no new event → phase ended     → generate phase report + export JSON
"""

import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path setup: import from progress_tracker_agent ───────────────────────────
# Add src/agents/ to sys.path so progress_tracker_agent can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

from session_prompts import (
    EXERCISE_SUMMARY_PROMPT,
    PHASE_REPORT_PROMPT,
    format_mistakes_text,
    format_ok_aspects_text,
    format_exercise_breakdown,
    infer_quality_trend,
    infer_ok_aspects,
)

# Phase outputs directory (sibling of both agent folders)
PHASE_OUTPUTS_DIR = Path(__file__).parent.parent / "phase_outputs"


# ── Exercise Buffer ───────────────────────────────────────────────────────────


class ExerciseBuffer:
    """Accumulates coaching_events for one exercise."""

    def __init__(self):
        self.events: list = []
        self.exercise_name = "Unknown"
    def add(self, payload: dict):
        event = payload.get("coaching_event", {})
        name = event.get("exercise", {}).get("name", "Unknown").title()
        if self.events:
            # keep the most common exercise name seen so far
            pass
        else:
            self.exercise_name = name
        self.events.append(payload)

    def is_empty(self) -> bool:
        return len(self.events) == 0

    def summarise(self) -> dict:
        """Aggregate all events into a single exercise summary dict."""
        if not self.events:
            return {}

        quality_scores = [
            e["coaching_event"].get("quality_score", 0.5) for e in self.events
        ]

        # Aggregate mistakes by type
        mistake_map: dict = {}
        for e in self.events:
            m = e["coaching_event"].get("mistake", {})
            mtype = m.get("type", "")
            if not mtype:
                continue
            if mtype not in mistake_map:
                mistake_map[mtype] = {
                    "type": mtype,
                    "occurrences": 0,
                    "total_duration": 0.0,
                    "total_persist": 0.0,
                    "severity": m.get("severity", "low"),
                    "count": 0,
                }
            mistake_map[mtype]["occurrences"] += m.get("occurrences", 0)
            mistake_map[mtype]["total_duration"] += m.get("duration_seconds", 0)
            mistake_map[mtype]["total_persist"] += m.get("persistence_rate", 0)
            mistake_map[mtype]["count"] += 1

        mistakes = []
        for v in mistake_map.values():
            c = max(v["count"], 1)
            mistakes.append(
                {
                    "type": v["type"],
                    "occurrences": v["occurrences"],
                    "avg_duration_s": round(v["total_duration"] / c, 1),
                    "avg_persistence": round(v["total_persist"] / c, 2),
                    "severity": v["severity"],
                }
            )

        ok_aspects = infer_ok_aspects(self.events)
        quality_trend = infer_quality_trend(quality_scores)
        avg_quality = sum(quality_scores) / len(quality_scores)

        return {
            "exercise_name": self.exercise_name,
            "event_count": len(self.events),
            "quality_scores": [round(q, 2) for q in quality_scores],
            "avg_quality": round(avg_quality, 2),
            "quality_trend": quality_trend,
            "mistakes": mistakes,
            "ok_aspects": ok_aspects,
            "exercise_feedback": "",  # filled in after LLM call
        }

    def reset(self):
        self.events = []
        self.exercise_name = "Unknown"


# ── Session Manager ───────────────────────────────────────────────────────────


class SessionManager:
    """
    Core state machine. Feed coaching_event payloads via .ingest().
    Automatically triggers exercise summaries and phase reports.

    Usage:
        sm = SessionManager(patient_profile)
        sm.start()
        sm.ingest(payload)   # call every time a new event arrives
        # ... after 30s silence: exercise summary printed automatically
        # ... after 120s silence: phase report printed + JSON exported
    """

    EXERCISE_TIMEOUT_S = 30
    PHASE_TIMEOUT_S = 120

    def __init__(
        self,
        patient_profile: dict,
        ollama_model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434",
        verbose: bool = True,
    ):
        self.patient_profile = patient_profile
        self.verbose = verbose

        # LLM (shared for both exercise summary and phase report)
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.6,
            num_predict=400,
            num_ctx=2048,
        )
        self.parser = StrOutputParser()

        # State
        self.exercise_buffer = ExerciseBuffer()
        self.completed_exercises: list = []
        self.phase_start_ts: float = 0.0
        self.last_event_ts: float = 0.0
        self.last_ex_end_ts: float = 0.0
        self.phase_active: bool = False
        self.phase_ended: bool = False

        # RAG (optional — import from progress_tracker_agent if available)
        self._rag_retrieve = None
        self._load_rag()

        # Background watcher thread
        self._stop_event = threading.Event()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)

        self.exercise_feedbacks: list = []

    # ── RAG setup ─────────────────────────────────────────────────────────────

    def _load_rag(self):
        """Try to load the RAG knowledge base from progress_tracker_agent."""
        try:
            from progress_tracker_agent.rag_retriever import CoachingKnowledgeBase
            from progress_tracker_agent.prompts import build_rag_query
            from progress_tracker_agent.schemas import PatientContext, ConditionCategory, RehabPhase

            persist_dir = Path(__file__).parent / "chroma_coaching_db"
            if persist_dir.exists():
                kb = CoachingKnowledgeBase(persist_dir=str(persist_dir)).load_or_build()

                def _retrieve(query: str) -> str:
                    context, _ = kb.retrieve(query, k=3)
                    return context

                self._rag_retrieve = _retrieve
                if self.verbose:
                    print("RAG knowledge base loaded ✓")
            else:
                if self.verbose:
                    print("RAG DB not found — will generate without clinical context")
        except Exception as e:
            if self.verbose:
                print(f"RAG not available: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the session and the background watcher."""
        self.phase_start_ts = time.time()
        self.last_event_ts = time.time()
        self.last_ex_end_ts = time.time()
        self.phase_active = True
        self.phase_ended = False
        self._stop_event.clear()
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SESSION STARTED")
            print(f"  Patient : {self.patient_profile.get('condition')}")
            print(f"  Phase   : {self.patient_profile.get('rehab_phase')}")
            print(f"  Pain    : {self.patient_profile.get('pain_level')}/10")
            print(f"{'='*60}\n")

    def ingest(self, payload: dict):
        """Feed a new coaching_event payload into the session."""
        if not self.phase_active:
            print("⚠️  Session not started. Call .start() first.")
            return

        now = time.time()
        self.last_event_ts = now
        self.exercise_buffer.add(payload)

        event = payload.get("coaching_event", {})
        if self.verbose:
            ex = event.get("exercise", {}).get("name", "?")
            q = event.get("quality_score", 0)
            m = event.get("mistake", {}).get("type", "none")
            print(f"  ↳ event: {ex} | quality={q:.2f} | mistake={m}")

    def stop(self):
        """Manually stop the watcher thread."""
        self._stop_event.set()

    # ── Background watcher ────────────────────────────────────────────────────

    def _watch_loop(self):
        """Poll every second to check for exercise/phase timeouts."""
        while not self._stop_event.is_set():
            time.sleep(1)
            now = time.time()
            silent = now - self.last_event_ts
            ex_idle = now - self.last_ex_end_ts

            # ── Exercise timeout ──────────────────────────────────────────────
            if (
                silent >= self.EXERCISE_TIMEOUT_S
                and not self.exercise_buffer.is_empty()
            ):
                self._flush_exercise()

            # ── Phase timeout ─────────────────────────────────────────────────
            elif (
                ex_idle >= self.PHASE_TIMEOUT_S
                and self.completed_exercises
                and not self.phase_ended
            ):
                self._end_phase()
                self._stop_event.set()

    # ── Exercise flush ────────────────────────────────────────────────────────

    def _flush_exercise(self):
        summary = self.exercise_buffer.summarise()
        if not summary:
            return

        self.last_ex_end_ts = time.time()

        print(f"\n{'─'*60}")
        print(f"  🏋️  EXERCISE ENDED: {summary['exercise_name']}")
        print(
            f"  Events: {summary['event_count']} | "
            f"Avg quality: {summary['avg_quality']:.2f} | "
            f"Trend: {summary['quality_trend']}"
        )
        print(f"{'─'*60}")
        print("  Generating exercise feedback...\n")

        feedback = self._generate_exercise_summary(summary)
        summary["exercise_feedback"] = feedback
        self.completed_exercises.append(summary)
        self.exercise_feedbacks.append(
            {
                "exercise_name": summary["exercise_name"],
                "avg_quality": summary["avg_quality"],
                "quality_trend": summary["quality_trend"],
                "mistakes": summary["mistakes"],
                "ok_aspects": summary["ok_aspects"],
                "feedback": feedback,
            }
        )

        # ── 按段打印，替换原来的单块输出 ──────────────────────────
        for line in feedback.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            print(f"  {line}")
        print()
        # ──────────────────────────────────────────────────────────

        self.exercise_buffer.reset()

    # ── Phase end ─────────────────────────────────────────────────────────────
    def _end_phase(self):
        self.phase_ended = True
        phase_end_ts = time.time()

        print(f"\n{'='*60}")
        print(f"  📋 REHAB PHASE ENDED")
        print(f"  Exercises completed: {len(self.completed_exercises)}")
        print(f"{'='*60}")
        print("  Generating phase report...\n")

        report_text = self._generate_phase_report()

        # ── 按段打印 phase report ──────────────────────────────────
        print(f"  {'─'*56}")
        for line in report_text.strip().split("\n"):
            line = line.strip()
            if not line:
                print()
                continue
            print(f"  {line}")
        print(f"  {'─'*56}")
        # ──────────────────────────────────────────────────────────

        json_path = self._export_json(report_text, phase_end_ts)
        print(f"\n  💾 JSON saved → {json_path}")
        print(f"{'='*60}\n")

    # ── LLM: Exercise Summary ─────────────────────────────────────────────────

    def _generate_exercise_summary(self, summary: dict) -> str:
        p = self.patient_profile
        prompt = EXERCISE_SUMMARY_PROMPT.format(
            condition=p.get("condition", "musculoskeletal condition"),
            weeks_into_rehab=p.get("weeks_into_rehab", 1),
            rehab_phase=p.get("rehab_phase", "early"),
            pain_level=p.get("pain_level", 5),
            exercise_name=summary["exercise_name"],
            event_count=summary["event_count"],
            avg_quality=summary["avg_quality"],
            quality_trend=summary["quality_trend"],
            mistakes_text=format_mistakes_text(summary["mistakes"]),
            ok_aspects_text=format_ok_aspects_text(summary["ok_aspects"]),
        )
        try:
            result = self.llm.invoke(prompt)
            return result if isinstance(result, str) else self.parser.invoke(result)
        except Exception as e:
            return f"Exercise complete. Focus on form correction next set. (LLM error: {e})"

    # ── LLM: Phase Report ─────────────────────────────────────────────────────

    def _generate_phase_report(self) -> str:
        p = self.patient_profile

        # Pain trend: collect pain_level per exercise (static here, could vary)
        pain_trend = [p.get("pain_level", 5)] * len(self.completed_exercises)

        # Overall quality trend across all exercises
        all_scores = []
        for ex in self.completed_exercises:
            all_scores.extend(ex.get("quality_scores", []))
        overall_trend = infer_quality_trend(all_scores) if all_scores else "stable"

        # RAG clinical context
        condition = p.get("condition", "musculoskeletal condition")
        phase = p.get("rehab_phase", "early")
        if self._rag_retrieve:
            clinical_context = self._rag_retrieve(
                f"{condition} {phase} rehabilitation exercises progression"
            )[:1000]
        else:
            clinical_context = "No clinical context available."

        prompt = PHASE_REPORT_PROMPT.format(
            condition=condition,
            weeks_into_rehab=p.get("weeks_into_rehab", 1),
            rehab_phase=phase,
            pain_level=p.get("pain_level", 5),
            goals=p.get("goals", "improve function"),
            exercise_list=", ".join(
                ex["exercise_name"] for ex in self.completed_exercises
            ),
            overall_quality_trend=overall_trend,
            pain_trend=" → ".join(str(v) for v in pain_trend),
            exercise_breakdown=format_exercise_breakdown(self.completed_exercises),
            clinical_context=clinical_context,
        )
        try:
            result = self.llm.invoke(prompt)
            return result if isinstance(result, str) else self.parser.invoke(result)
        except Exception as e:
            return f"Session complete. Good work today. Please review your exercises for next session. (LLM error: {e})"

    # ── JSON Export ───────────────────────────────────────────────────────────

    def _export_json(self, report_text: str, phase_end_ts: float) -> Path:
        """Export structured JSON for Progress Tracker consumption."""
        p = self.patient_profile
        all_scores = []
        for ex in self.completed_exercises:
            all_scores.extend(ex.get("quality_scores", []))

        # Derive next phase focus from top mistakes across all exercises
        mistake_counts: dict = {}
        for ex in self.completed_exercises:
            for m in ex.get("mistakes", []):
                t = m["type"]
                mistake_counts[t] = mistake_counts.get(t, 0) + m.get("occurrences", 0)
        top_mistakes = sorted(mistake_counts, key=mistake_counts.get, reverse=True)[:2]
        next_focus = [f"Correct {m} in next phase" for m in top_mistakes]
        if not next_focus:
            next_focus = ["Maintain form quality", "Progress to next phase exercises"]

        payload = {
            "phase_summary": {
                "patient_id": p.get("patient_id", "unknown"),
                "condition": p.get("condition", ""),
                "rehab_phase": p.get("rehab_phase", ""),
                "pain_level": p.get("pain_level", 0),
                "weeks_into_rehab": p.get("weeks_into_rehab", 0),
                "goals": p.get("goals", ""),
                "phase_start_ts": round(self.phase_start_ts, 2),
                "phase_end_ts": round(phase_end_ts, 2),
                "phase_duration_s": round(phase_end_ts - self.phase_start_ts, 2),
            },
            "exercises": [
                {
                    "exercise_name": ex["exercise_name"],
                    "event_count": ex["event_count"],
                    "quality_scores": ex["quality_scores"],
                    "avg_quality": ex["avg_quality"],
                    "quality_trend": ex["quality_trend"],
                    "mistakes": ex["mistakes"],
                    "ok_aspects": ex["ok_aspects"],
                    "exercise_feedback": ex["exercise_feedback"],
                }
                for ex in self.completed_exercises
            ],
            "phase_pain_trend": [p.get("pain_level", 0)]
            * len(self.completed_exercises),
            "overall_quality_trend": (
                infer_quality_trend(all_scores) if all_scores else "stable"
            ),
            "next_phase_focus": next_focus,
            "phase_report": report_text,
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }

        # Write to phase_outputs/
        PHASE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        patient = p.get("patient_id", "unknown").replace(" ", "_")
        phase = p.get("rehab_phase", "phase")
        filename = f"{patient}_{phase}_{ts}.json"
        out_path = PHASE_OUTPUTS_DIR / filename

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return out_path
