# Progress Tracker Agent

Personalized physiotherapy coaching powered by **gemma3:4b** (local Ollama) + RAG.

## Architecture

```
PatientContext (upstream input)
        │
        ▼
┌─────────────────────────────────────────────────┐
│         Progress Tracker AGENT                  │
│                                                 │
│  ① Context Receiver                             │
│     PatientContext: condition, phase,           │
│     pain level, exercises, patient message      │
│                 │                               │
│  ② RAG Retriever ←── ChromaDB                  │
│     Build query from context                    │
│     Retrieve k=3 relevant clinical chunks       │
│                 │                               │
│  ③ Coaching Generator (gemma3:4b)               │
│     Structured prompt: profile + RAG context    │
│     num_predict=512, num_ctx=2048               │
│                 │                               │
│  ④ Response Polisher (gemma3:4b, temp=0.3)     │
│     Tone adjustment, safety prefixes,           │
│     emoji formatting                           │
│                 │                               │
└─────────────────┼───────────────────────────────┘
                  │
                  ▼
           CoachingOutput
           ├── coaching_feedback (main text)
           ├── suggested_exercises []
           ├── safety_notes []
           ├── motivational_note
           ├── retrieved_sources []
           └── confidence_score
```

## File Structure

```
src/agents/coaching_agent/
├── coaching_agent.py    # Main agent orchestration
├── rag_retriever.py     # ChromaDB knowledge base (reuses trial.ipynb logic)
├── prompts.py           # Prompt templates + RAG query builder
├── schemas.py           # PatientContext + CoachingOutput dataclasses
├── demo.ipynb           # Interactive demo notebook
└── README.md
```

## Setup

### Prerequisites
```bash
# 1. Ollama running with gemma3:4b
ollama serve
ollama pull gemma3:4b

# 2. Python dependencies (same as trial.ipynb)
pip install langchain langchain-ollama langchain-huggingface langchain-chroma
pip install chromadb sentence-transformers beautifulsoup4
```

### Data
Place your `.txt` and `.html` files in `dataset/clean/` (same as trial.ipynb).

### Run
```python
from coaching_agent import CoachingAgent
from rag_retriever import CoachingKnowledgeBase
from schemas import make_sample_context

kb = CoachingKnowledgeBase(data_dir='dataset/clean').load_or_build()
agent = CoachingAgent(knowledge_base=kb)
output = agent.generate_coaching(make_sample_context('knee'))
print(output.coaching_feedback)
```

Or open `demo.ipynb` and run cell by cell.

## Avoiding Kernel Crashes

The trial.ipynb crashed due to memory pressure when running evaluation loops.
This agent includes several mitigations:

| Setting | Value | Why |
|---------|-------|-----|
| `num_predict` | 512 | Caps output tokens — biggest crash cause |
| `num_ctx` | 2048 | Smaller context window |
| `retrieval_k` | 3 | Fewer docs = less prompt padding |
| Clinical context trim | 1500 chars | Prevents oversized prompts |
| `enable_polish` | toggleable | Skip 2nd LLM call if memory tight |

## PatientContext — Upstream Input Schema

```python
PatientContext(
    patient_id="P001",
    condition="knee osteoarthritis",
    condition_category=ConditionCategory.KNEE,
    rehab_phase=RehabPhase.MID,     # acute/early/mid/late/maintenance
    pain_level=4,                    # 0-10
    weeks_into_rehab=10,
    recent_exercises=[
        ExerciseRecord("Mini squats", sets=2, reps=8, completed=False, difficulty_feedback="too hard"),
    ],
    patient_message="The squats hurt my knee going down.",
    age=58,
    goals="Walk dog 30 mins daily",
)
```

## Extending the Agent

**Add memory/history**: Inject previous `CoachingOutput.coaching_feedback` into the prompt as conversation history.

**Add upstream agent**: Replace `make_sample_context()` with data from your upstream agent's output.

**Improve retrieval**: Add metadata filtering to ChromaDB (filter by `condition_category` tag).

**Evaluation**: Reuse the `CompleteRAGEvaluator` from `trial.ipynb` — pass the agent's `coaching_feedback` as `model_answer`.
