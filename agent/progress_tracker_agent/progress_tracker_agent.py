"""
progress_tracker_agent.py — Progress Tracker Agent
Orchestrates: JSON Ingestion → Trend Analysis → LLM Report Generation
"""

import re
import time
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

from schemas import PatientContext, ProgressOutput, RehabPhase, make_sample_context
    PatientContext, ProgressOutput, RehabPhase, make_sample_context
)
from rag_retriever import ProgressKnowledgeBase
from prompts import (
    COACHING_PROMPT_FLAT, POLISH_PROMPT,
    build_rag_query, format_exercise_history
)


class ProgressTrackerAgent:
    """
    Progress Tracker Agent: analyzes phase JSON files and generates progress reports.
    
    Pipeline:
    phase JSON files → Trend Analysis → LLM Report
    """

    def __init__(
        self,
        knowledge_base: ProgressKnowledgeBase,
        ollama_model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434",
        retrieval_k: int = 3,           # Keep low to avoid memory pressure
        enable_polish: bool = True,
        verbose: bool = True,
    ):
        self.kb = knowledge_base
        self.retrieval_k = retrieval_k
        self.enable_polish = enable_polish
        self.verbose = verbose

        # ── LLM setup (gemma3:4b via Ollama) ────────────────────────────────
        # num_predict limits output tokens → prevents kernel crash from long outputs
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7,
            num_predict=512,        # Hard cap: prevents memory overflow
            num_ctx=2048,           # Context window: balanced for 4b model
        )
        self.parser = StrOutputParser()

        if self.verbose:
            print(f"ProgressTrackerAgent ready — model: {ollama_model}, retrieval_k: {retrieval_k}")

    # ── Main entry point ─────────────────────────────────────────────────────

    def generate_progress_report(self, context: PatientContext) -> ProgressOutput:
        """
        Full pipeline: receive context → retrieve → generate → polish → return.
        """
        start = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Generating coaching for patient: {context.patient_id}")
            print(f"Condition: {context.condition} | Phase: {context.rehab_phase.value}")
            print(f"Pain: {context.pain_level}/10 | Week: {context.weeks_into_rehab}")

        # Step 1: Build RAG query from context
        rag_query = build_rag_query(context)
        if self.verbose:
            print(f"\n[1/4] RAG query: '{rag_query}'")

        # Step 2: Retrieve clinical context
        clinical_context, sources = self.kb.retrieve(rag_query, k=self.retrieval_k)
        if self.verbose:
            print(f"[2/4] Retrieved {len(sources)} source(s): {sources}")

        # Step 3: Generate coaching with LLM
        if self.verbose:
            print(f"[3/4] Generating coaching (gemma3:4b)...")
        
        raw_response = self._generate(context, clinical_context)

        # Step 4: Polish response
        if self.enable_polish and len(raw_response) > 50:
            if self.verbose:
                print(f"[4/4] Polishing response...")
            polished = self._polish(raw_response)
        else:
            polished = raw_response
            if self.verbose:
                print(f"[4/4] Skipping polish (short response or disabled)")

        # Step 5: Parse output into structured ProgressOutput
        output = self._parse_output(polished, context, sources)
        
        elapsed = time.time() - start
        if self.verbose:
            print(f"\n✓ Done in {elapsed:.1f}s")
            print(f"{'='*60}")

        return output

    # ── Step 3: LLM Generation ───────────────────────────────────────────────

    def _generate(self, context: PatientContext, clinical_context: str) -> str:
        """Build prompt and call gemma3:4b."""

        # Format exercise history
        exercise_str = format_exercise_history(context.recent_exercises)

        # Optional fields
        age_info = f"Age: {context.age}\n" if context.age else ""
        goals_info = f"Patient's Goal: {context.goals}\n" if context.goals else ""

        # Fill prompt template
        prompt = COACHING_PROMPT_FLAT.format(
            condition=context.condition,
            rehab_phase=context.rehab_phase.value,
            pain_level=context.pain_level,
            weeks_into_rehab=context.weeks_into_rehab,
            age_info=age_info,
            goals_info=goals_info,
            recent_exercises=exercise_str,
            patient_message=context.patient_message,
            clinical_context=clinical_context[:1500],   # Hard trim: stay in context window
        )

        try:
            response = self.llm.invoke(prompt)
            return self.parser.invoke(response) if not isinstance(response, str) else response
        except Exception as e:
            print(f"  LLM generation error: {e}")
            return self._fallback_response(context)

    # ── Step 4: Polish ───────────────────────────────────────────────────────

    def _polish(self, raw_response: str) -> str:
        """Light polish pass for tone and formatting."""
        polish_prompt = POLISH_PROMPT.format(raw_response=raw_response)
        
        try:
            # Use lower temperature for polish to avoid creative drift
            polish_llm = OllamaLLM(
                model=self.llm.model,
                base_url=self.llm.base_url,
                temperature=0.3,
                num_predict=600,
                num_ctx=2048,
            )
            result = polish_llm.invoke(polish_prompt)
            polished = result if isinstance(result, str) else self.parser.invoke(result)
            
            # Safety check: if polish output is much shorter, use original
            if len(polished) < len(raw_response) * 0.5:
                print("  Polish result too short, using original")
                return raw_response
            return polished
        except Exception as e:
            print(f"  Polish error (using raw): {e}")
            return raw_response

    # ── Step 5: Parse into ProgressOutput ────────────────────────────────────

    def _parse_output(
        self, 
        text: str, 
        context: PatientContext, 
        sources: list
    ) -> ProgressOutput:
        """
        Parse the LLM text output into structured ProgressOutput.
        Extracts exercises, safety notes, and motivation where possible.
        """
        
        # Extract exercise suggestions (lines that look like exercise instructions)
        exercise_patterns = [
            r"(?:^|\n)\s*[-•*]\s*(.+(?:sets?|reps?|times?|seconds?|minutes?).+)",
            r"(?:^|\n)\s*\d+\.\s*(.+(?:sets?|reps?|exercise|stretch|squat|raise|rotation).+)",
        ]
        suggested_exercises = []
        for pattern in exercise_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            suggested_exercises.extend([m.strip() for m in matches if len(m.strip()) > 10])
        suggested_exercises = list(dict.fromkeys(suggested_exercises))[:5]  # dedupe, max 5

        # Extract safety notes (lines with warning keywords)
        safety_lines = []
        for line in text.split('\n'):
            if any(w in line.lower() for w in ['stop if', 'avoid', 'do not', 'warning', 'caution', '⚠️', 'consult']):
                safety_lines.append(line.strip())
        safety_notes = safety_lines[:3]

        # Extract motivational closing (last non-empty line or lines with 💪)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        motivational_note = ""
        for line in reversed(lines):
            if '💪' in line or any(w in line.lower() for w in ['goal', 'keep', 'great', 'well done', 'progress', 'believe']):
                motivational_note = line
                break
        if not motivational_note and lines:
            motivational_note = lines[-1]

        # Rough confidence: based on response length and source count
        confidence = min(0.9, 0.5 + (len(text) / 1000) * 0.2 + len(sources) * 0.1)

        return ProgressOutput(
            patient_id=context.patient_id,
            coaching_feedback=text,
            suggested_exercises=suggested_exercises,
            safety_notes=safety_notes,
            motivational_note=motivational_note,
            retrieved_sources=sources,
            confidence_score=round(confidence, 2),
        )

    # ── Fallback ─────────────────────────────────────────────────────────────

    def _fallback_response(self, context: PatientContext) -> str:
        """Return a safe fallback if LLM fails."""
        return (
            f"Thank you for completing your exercises this week. "
            f"Given your {context.condition} and current pain level of {context.pain_level}/10, "
            f"please continue with your prescribed programme at a comfortable pace. "
            f"If pain increases or you have concerns, please contact your physiotherapist directly."
        )


# ── Convenience runner ────────────────────────────────────────────────────────

def run_demo():
    """Quick demo — runs all three sample scenarios."""
    
    print("Initialising Coaching Agent Demo...\n")

    # Load knowledge base (will use existing DB if available)
    kb = ProgressKnowledgeBase(
        data_dir="dataset/clean",
        persist_dir="./chroma_coaching_db",
    ).load_or_build()

    # Create agent
    agent = ProgressTrackerAgent(knowledge_base=kb, verbose=True)

    # Run three scenarios
    for scenario in ["knee", "shoulder", "acl"]:
        context = make_sample_context(scenario)
        output = agent.generate_progress_report(context)

        print(f"\n{'='*60}")
        print(f"COACHING FEEDBACK — Patient {output.patient_id}")
        print(f"{'='*60}")
        print(output.coaching_feedback)
        
        if output.suggested_exercises:
            print(f"\n📋 Parsed Exercises:")
            for ex in output.suggested_exercises:
                print(f"  • {ex}")
        
        if output.safety_notes:
            print(f"\n⚠️ Safety Notes:")
            for note in output.safety_notes:
                print(f"  {note}")
        
        print(f"\n📊 Sources used: {output.retrieved_sources}")
        print(f"🎯 Confidence: {output.confidence_score:.0%}")
        print(f"\n{'-'*60}\n")
        
        # Pause between runs to avoid memory pressure
        input("Press Enter to continue to next scenario... (or Ctrl+C to stop)\n")


if __name__ == "__main__":
    run_demo()
