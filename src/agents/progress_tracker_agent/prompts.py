"""
prompts.py — Coaching Agent Prompt Templates
Designed for gemma3:4b via Ollama — concise, structured prompts that
stay within the model's context window without triggering memory pressure.
"""

from langchain_core.prompts import PromptTemplate


# ── System role definition ───────────────────────────────────────────────────

COACHING_SYSTEM = """You are a supportive physiotherapy coaching assistant.
Your role is to provide personalized exercise guidance and rehabilitation coaching based on:
- The patient's current condition and rehab phase
- Evidence-based guidance from clinical sources
- The patient's recent exercise performance and concerns

Rules:
- Be warm, encouraging, and clear
- Never diagnose or prescribe medication
- Always recommend consulting a physiotherapist for pain that worsens
- Keep responses practical and action-oriented
- Base advice on the provided clinical context"""


# ── Main coaching generation prompt ─────────────────────────────────────────

COACHING_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "condition", "rehab_phase", "pain_level", "weeks_into_rehab",
        "recent_exercises", "patient_message", "age_info", "goals_info",
        "clinical_context"
    ],
    template="""
{system}

=== PATIENT PROFILE ===
Condition: {condition}
Rehab Phase: {rehab_phase}
Pain Level (0-10): {pain_level}/10
Weeks into Rehab: {weeks_into_rehab}
{age_info}
{goals_info}

=== RECENT EXERCISE SESSION ===
{recent_exercises}

=== PATIENT'S MESSAGE ===
{patient_message}

=== CLINICAL REFERENCE (from evidence-based sources) ===
{clinical_context}

=== YOUR TASK ===
Write a personalized coaching response that includes:

1. ACKNOWLEDGEMENT: Briefly acknowledge the patient's effort and concerns (2-3 sentences)

2. EXERCISE FEEDBACK: Comment on their recent session performance - what went well, what to adjust (3-5 sentences)

3. NEXT SESSION RECOMMENDATIONS: List 3-5 specific exercises appropriate for their phase and condition, with sets/reps guidance

4. ADDRESSING THEIR CONCERN: Directly answer the patient's specific question or concern with practical advice

5. SAFETY REMINDER: One key safety note relevant to their condition/phase

6. MOTIVATION: End with a brief encouraging statement tied to their personal goal

Keep the total response under 400 words. Use simple, non-clinical language.
""".format(system=COACHING_SYSTEM, **{
        "condition": "{condition}",
        "rehab_phase": "{rehab_phase}",
        "pain_level": "{pain_level}",
        "weeks_into_rehab": "{weeks_into_rehab}",
        "age_info": "{age_info}",
        "goals_info": "{goals_info}",
        "recent_exercises": "{recent_exercises}",
        "patient_message": "{patient_message}",
        "clinical_context": "{clinical_context}",
    })
)


# ── Simpler flat template (better for gemma3:4b memory) ─────────────────────

COACHING_PROMPT_FLAT = """You are a supportive physiotherapy coaching assistant.

PATIENT: {condition}, week {weeks_into_rehab} of rehab, pain {pain_level}/10, phase: {rehab_phase}.
{age_info}{goals_info}

RECENT EXERCISES:
{recent_exercises}

PATIENT SAYS: "{patient_message}"

CLINICAL GUIDANCE (use this as your knowledge base):
{clinical_context}

Write a coaching response with these sections:
1. Acknowledge their session and concerns (2-3 sentences)
2. Specific feedback on what to continue and what to modify
3. Next session: 3-4 exercises with sets/reps (appropriate to their phase)
4. Answer their specific concern directly
5. One safety note
6. Brief encouragement linked to their goal

Keep it under 350 words. Be warm and practical. No medical diagnosis."""


# ── Polish/formatting prompt ──────────────────────────────────────────────────

POLISH_PROMPT = """Review and lightly polish this physiotherapy coaching message.

Original message:
{raw_response}

Requirements:
- Fix any grammar issues
- Ensure a warm, supportive tone throughout  
- Make sure exercises have clear sets/reps if mentioned
- Add "⚠️ Note:" prefix to any safety warnings
- End with "💪" emoji on the motivational closing line
- Do NOT add new medical advice or change exercise recommendations
- Keep under 400 words

Return only the polished message, no commentary."""


# ── RAG query construction ────────────────────────────────────────────────────

def build_rag_query(context) -> str:
    """
    Build an optimised retrieval query from PatientContext.
    Targets the specific exercise/condition info in your dataset.
    """
    base = f"{context.condition} exercises {context.rehab_phase.value} rehabilitation"
    
    # Add specific concern keywords if present
    concern_keywords = []
    msg = context.patient_message.lower()
    
    if any(w in msg for w in ["stair", "stairs", "steps"]):
        concern_keywords.append("stair climbing")
    if any(w in msg for w in ["pain", "hurt", "ache"]):
        concern_keywords.append("pain management")
    if any(w in msg for w in ["jog", "run", "sport", "return"]):
        concern_keywords.append("return to sport plyometric")
    if any(w in msg for w in ["sleep", "morning", "night"]):
        concern_keywords.append("activity modification")
    if any(w in msg for w in ["squat", "bend", "kneel"]):
        concern_keywords.append("squat progression")
    if any(w in msg for w in ["shoulder", "arm", "rotation"]):
        concern_keywords.append("shoulder exercises")

    if concern_keywords:
        base += " " + " ".join(concern_keywords)
    
    return base


def format_exercise_history(exercises) -> str:
    """Format exercise records for prompt injection."""
    if not exercises:
        return "No exercise records provided."
    
    lines = []
    for ex in exercises:
        status = "✓ Completed" if ex.completed else "✗ Not completed"
        parts = [f"- {ex.name}"]
        if ex.sets and ex.reps:
            parts[0] += f" ({ex.sets}×{ex.reps})"
        parts[0] += f" — {status}"
        if ex.difficulty_feedback:
            parts[0] += f" [felt: {ex.difficulty_feedback}]"
        lines.append(parts[0])
    
    return "\n".join(lines)
