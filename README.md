# AI-Powered-Rehabilitation-Coaching-System

![Project Banner / Demo GIF Placeholder](https://via.placeholder.com/1200x400.png?text=Virtual+Physiotherapy+Assistant+Demo)  
*(Add a short demo GIF or screenshot here later — highly recommended!)*

## 🎯 The Problem

Recovering from an injury or surgery often requires patients to diligently perform prescribed rehabilitation exercises at home. However, two major challenges persist:

- **Incorrect form** — Without real-time professional guidance, many patients perform exercises improperly, which can slow recovery, worsen the injury, or lead to new complications.
- **Poor adherence** — Patient compliance (actually completing the full prescribed routine and frequency) remains one of the biggest barriers to successful at-home rehabilitation.

Traditional in-person physiotherapy is expensive, time-consuming, and not always accessible — especially in rural areas or during long-term recovery.

## 🚀 Our Solution

**Virtual Physiotherapy Assistant (VPA)** is an intelligent AI system that acts as your personal virtual physiotherapist — available anytime, anywhere, directly from your phone or webcam.

### Core Capabilities

- **Real-time pose estimation & movement analysis** — Uses your camera to track body keypoints and evaluate exercise execution.
- **Detailed, constructive feedback** — Tells you exactly what you're doing **correctly**, **moderately well**, or **poorly**, with specific, actionable suggestions to correct form (e.g. "Keep your knee aligned over your ankle — try shifting weight slightly forward").
- **Retrieval-Augmented Generation (RAG)** recommendation engine — Personalizes advice based on:
  - Your specific injury / condition
  - Doctor / physiotherapist recommendations
  - Evidence-based rehab protocols for common injuries
- **Patient-centric design** — Aims to increase adherence through clear, encouraging, human-like coaching.

The goal is simple: help people recover **faster**, **safer**, and **more consistently** from home — while reducing the burden on healthcare systems.

## ✨ Key Features (Initial Version)

- Video-based real-time exercise assessment
- Multi-level feedback (good / moderate / needs improvement)
- Personalized recommendations via RAG (injury-specific + protocol-aware)
- Chat interface for asking questions about exercises, pain, or progress
- (Planned) Progress tracking & adherence reports

## 🛠️ Technology Highlights

- **Computer Vision** → Human pose estimation (likely MediaPipe / OpenPose / RTMPose family)
- **AI Feedback Engine** → LLM-powered critique + natural language generation
- **Retrieval-Augmented Generation (RAG)** → For retrieving and grounding advice in trusted physiotherapy knowledge
- **Frontend** → (Web / mobile app — webcam access)
- **Backend** → Python-based inference pipeline

## 📦 Environment Setup

### Prerequisites
- **Conda** (Miniconda or Anaconda) installed on your system
- **Python 3.11** (specified in the environment file)

### Installation Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/DS5500-Team-11-AI-Powered-Rehab/AI-Powered-Rehabilitation-Coaching-System.git
   cd AI-Powered-Rehabilitation-Coaching-System
   ```

2. **Create the Conda environment** from the provided environment file:
   ```bash
   conda env create -f rehab_ai_env.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate rehab_ai_env
   ```

### What's Included

The environment includes:
- **Core scientific stack**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch (CPU-only), TorchVision, TorchAudio
- **RAG / Vector Database**: ChromaDB, Sentence Transformers
- **LLM Frameworks**: LangChain, LangGraph
- **LLM Clients**: OpenAI, Anthropic, Ollama
- **Data Processing**: PyPDF, python-docx, BeautifulSoup4
- **Jupyter**: Notebook environment for development and experimentation
- **Additional tools**: Transformers, Accelerate, Spacy, and more

### Deactivating the Environment

When you're done, deactivate the environment:
```bash
conda deactivate
```

## 📁 Project Structure

```
AI-Powered-Rehabilitation-Coaching-System/
│
├── README.md                        # This file — system overview
├── rehab_ai_env.yml                 # Conda environment specification
├── .env / .env.example              # Environment variables (API keys, model configs)
├── .gitignore                       # Ensure chroma_db & .env are ignored
│
├── data/
│   ├── pt_guideline_data/           # Physical therapy guidelines & protocols
│   ├── chroma_db/                   # Vector database (gitignored)
│   └── exercise_cache/              # Pre-computed Tier 1 response JSONs
│
├── notebooks/                       # Jupyter notebooks for exploration & evaluation
│   ├── llm_comprehensive_evaluation.ipynb
│   ├── validated_test_questions.json
│   └── evaluation_results/
│
├── src/                             # Production code
│   │
│   ├── cv/                          # Computer Vision pipeline
│   │   ├── __init__.py
│   │   ├── pose_estimator.py        # MediaPipe / OpenPose wrapper
│   │   ├── depth_estimator.py       # Depth Anything integration
│   │   ├── fusion.py                # 2D pose + depth → 3D
│   │   └── schemas.py               # CoachingEvent dataclass / Pydantic models
│   │
│   ├── integration/                 # Integration layer (CV → LLM bridge)
│   │   ├── __init__.py
│   │   ├── event_filter.py          # Temporal filtering, severity scoring
│   │   ├── deduplicator.py          # Prevents repetitive coaching cues
│   │   └── router.py                # Routes to Tier 1 / 2 / 3
│   │
│   ├── rag/                         # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── ingest.py                # Chunk & embed PT guidelines → ChromaDB
│   │   ├── retriever.py             # Query interface over ChromaDB
│   │   └── prompt_templates.py      # Tier 2 slot-based prompts
│   │
│   ├── agents/                      # LangGraph multi-agent system
│   │   ├── __init__.py
│   │   ├── state.py                 # Shared LangGraph state schema
│   │   ├── movement_analysis.py     # Movement Analysis Agent
│   │   ├── coaching.py              # Coaching Agent (conversational memory)
│   │   ├── progress.py              # Progress Tracking Agent
│   │   └── orchestrator.py          # LangGraph graph definition & routing
│   │
│   ├── feedback/                    # Feedback generation & delivery
│   │   ├── __init__.py
│   │   ├── tier1_cache.py           # Load/serve pre-computed audio cues
│   │   ├── tier2_generator.py       # RAG + GPT-4o-mini generation
│   │   ├── tier3_reasoner.py        # Full agent reasoning pass
│   │   └── delivery.py              # Timing logic (immediate / rep-end / rest)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Load .env, model names, thresholds
│       └── logging.py               # Logging utilities
│
├── tests/                           # Unit & integration tests
│   ├── test_event_filter.py
│   ├── test_retriever.py
│   ├── test_agents.py
│   └── test_tier_routing.py
│
├── scripts/                         # One-off runnable scripts
│   ├── ingest_pt_data.py            # Populate ChromaDB with PT guidelines
│   ├── build_tier1_cache.py         # Pre-compute top mistake responses
│   └── run_demo.py                  # End-to-end demo runner
│
└── docs/
    ├── architecture.html            # System architecture & design
    └── api_contracts.md             # CV ↔ Integration ↔ LLM interface specs
```

## Why This Matters

Incorrect exercise performance and low adherence are well-documented causes of prolonged recovery times and increased healthcare costs. By combining state-of-the-art **pose estimation**, **generative AI**, and **personalized retrieval**, VPA aims to bring high-quality, 24/7 physiotherapy guidance to anyone with a smartphone or laptop.

We're building this as an open-source project to encourage collaboration between AI researchers, physiotherapists, clinicians, and rehab tech enthusiasts.

## 🚧 Current Status

Early / proof-of-concept stage  
Actively developing core pose → feedback loop and RAG integration

Contributions, feedback, and domain expertise (especially from physiotherapists) are **very welcome**!

---

**Topics**: #pose-estimation #human-pose-estimation #computer-vision #rehabilitation #physiotherapy #healthcare-ai #exercise-feedback #rag #ai-healthcare #physical-therapy

Star ⭐ the repo if you're interested in AI for healthcare & rehabilitation!

Let's make high-quality rehab accessible to everyone.
