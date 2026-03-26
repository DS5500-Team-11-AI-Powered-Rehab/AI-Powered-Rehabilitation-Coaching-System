# Rehabilitation Coaching Agents

This folder contains the agent-level components of the rehabilitation coaching system. Agents are high-level orchestrators that leverage the RAG infrastructure (in `src/rag/`) to provide specialized coaching and progress tracking functionality.

## Agents Overview

### 1. **Coaching Agent** (`coaching_agent/`)

Personalized physiotherapy coaching powered by **gemma3:4b** (local Ollama) + RAG.

#### Architecture

```
PatientContext (upstream input)
        │
        ▼
┌─────────────────────────────────────────────────┐
│               COACHING AGENT                    │
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

#### File Structure

```
src/agents/coaching_agent/
├── session_manager.py       # Session state machine for exercise/phase lifecycle
├── session_prompts.py       # Prompts for exercise summaries and phase reports
├── demo_session.py          # Interactive demo script
├── demo_session.ipynb       # Jupyter notebook demo
├── chroma_coaching_db/      # ChromaDB vector store (Git LFS tracked)
└── __init__.py              # Package marker
```

#### Setup

##### Prerequisites
```bash
# 1. Ollama running with gemma3:4b
ollama serve
ollama pull gemma3:4b

# 2. Python dependencies
pip install langchain langchain-ollama langchain-huggingface langchain-chroma
pip install chromadb sentence-transformers beautifulsoup4
```

##### Data
Place your `.txt` and `.html` files in `data/pt_guidelines/` (see root README for details).

##### Run
```python
from coaching_agent.session_manager import SessionManager

manager = SessionManager(patient_id="P001", condition="knee")
manager.process_coaching_event({...})
```

Or open `demo_session.ipynb` and run cell by cell.

#### Avoiding Kernel Crashes

The agent includes memory-safe configurations:

| Setting | Value | Why |
|---------|-------|-----|
| `num_predict` | 512 | Caps output tokens — prevents kernel crash |
| `num_ctx` | 2048 | Smaller context window |
| `retrieval_k` | 3 | Fewer docs = less prompt padding |
| Clinical context trim | 1500 chars | Prevents oversized prompts |
| `enable_polish` | toggleable | Skip 2nd LLM call if memory tight |

#### PatientContext — Upstream Input Schema

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

---

### 2. **Progress Tracker Agent** (`progress_tracker_agent/`)

Longitudinal progress analysis and LLM-powered progress reports.

#### Pipeline

```
Phase JSON files → Trend Analysis → LLM Report Generation
```

#### File Structure

```
src/agents/progress_tracker_agent/
├── progress_tracker_agent.py        # Main orchestrator
├── progress_tracker.py              # Core progress analysis
├── upstream_adapter.py              # Merge coaching event + patient profile into PatientContext
├── rag_retriever.py                 # ChromaDB integration (shared schema with coaching agent)
├── schemas.py                       # PatientContext, ProgressOutput, RehabPhase enums
├── prompts.py                       # LLM prompt templates
├── demo_progress_tracker.py         # Demo script
├── demo_progress_tracker.ipynb      # Jupyter notebook demo
├── README.md                        # Detailed documentation
├── chroma_coaching_db/              # ChromaDB vector store (Git LFS tracked)
└── __init__.py                      # Package marker
```

#### Usage

```python
from progress_tracker_agent.progress_tracker_agent import ProgressTrackerAgent
from progress_tracker_agent.rag_retriever import ProgressKnowledgeBase

kb = ProgressKnowledgeBase(data_dir="data/pt_guidelines").load_or_build()
agent = ProgressTrackerAgent(knowledge_base=kb)
report = agent.generate_progress_report(patient_context_list)
print(report.progress_report)
```

---

## Architecture: Agents vs RAG

**Agents** (this folder):
- High-level orchestrators that coordinate multiple system components
- Handle session state, lifecycle management, user interactions
- Leverage RAG as a tool for retrieval and context enrichment

**RAG Infrastructure** (`src/rag/` in parent):
- Vector storage, indexing, retrieval mechanisms
- Embedding models, ChromaDB management
- Query building and ranking logic

**Relationship**: Agents consume RAG; RAG doesn't depend on agents.

---

## Extending the Agents

### Coaching Agent

- **Add memory/history**: Inject previous `CoachingOutput.coaching_feedback` into prompts
- **Add upstream integration**: Replace `make_sample_context()` with real coaching event data
- **Improve retrieval**: Add metadata filtering (e.g., by `condition_category`)
- **Evaluation**: Reuse `CompleteRAGEvaluator` from notebooks

### Progress Tracker Agent

- **Multi-phase analysis**: Chain multiple phase JSON files for longitudinal trends
- **Custom metrics**: Extend `ProgressOutput` with domain-specific scores
- **Export formats**: Add JSON/PDF report export logic
- **Real-time streaming**: Emit progress updates as they're computed

---

## Dependencies

Both agents require:
- `langchain` — LLM orchestration
- `langchain-ollama` — Local LLM integration (gemma3:4b)
- `chromadb` — Vector database
- `sentence-transformers` — Embedding model

See `rehab_ai_env.yml` (root) for full environment setup.

---

## License

See root LICENSE file.
