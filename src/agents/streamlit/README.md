# Rehab AI Agent (Streamlit Frontend)

This repository contains the **Streamlit frontend** for a Rehab AI system, including:

* Coaching Agent (real-time session simulation)
* Progress Tracker (longitudinal analysis across sessions)

---

# How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure **Ollama is running**:

```bash
ollama serve
```

---

# Project Structure

```
streamlit/
│
├── app.py                  # Entry point
├── page_coaching.py        # Coaching session UI
├── page_progress.py        # Progress tracker UI
│
../
├── coaching_agent/         # Core coaching logic (LLM + rules)
├── progress_tracker_agent/ # Long-term analysis
├── phase_outputs/          # Generated session JSONs (shared memory)
```

---

# Coaching Agent (page_coaching.py)

## What it does

Simulates a **live rehab session**:

* Events stream in every **5 seconds**
* Exercise feedback generated after **30s inactivity**
* Phase report generated after **120s inactivity**

Core engine:

```python
SessionManager(...)
```

---

## IMPORTANT: Mock Data Location

Currently, the session is driven by **hardcoded mock data**:

```python
EVENTS_EX1 = [...]
EVENTS_EX2 = [...]
```

Defined in:

```
page_coaching.py
```

These simulate:

* exercise name
* detected mistakes
* quality score
* severity
* biomechanical metrics

Example structure:

```python
{
  "coaching_event": {
    "exercise": {"name": "..."},
    "mistake": {"type": "..."},
    "quality_score": 0.4,
    ...
  }
}
```

---

## How to Replace with Real Data (VERY IMPORTANT)

To integrate with external systems (e.g. CV model / sensor pipeline):

### Step 1 — Remove mock generator

Replace:

```python
_run_session(...)
```

with your own event stream.

---

### Step 2 — Call SessionManager.ingest()

Your external system should send events like:

```python
sm.ingest(payload)
```

Where `payload` must match:

```json
{
  "coaching_event": {
    "exercise": {...},
    "mistake": {...},
    "metrics": {...},
    "quality_score": float,
    "severity": "low|medium|high"
  },
  "session_id": "...",
  "coaching_history": []
}
```

---

### Step 3 — Real-time integration options

You can plug in:

* WebSocket stream
* REST API polling
* Kafka / message queue
* Local model inference loop

---

## Output of Coaching Agent

During session:

* `sm.exercise_feedbacks` → per-exercise feedback cards
* `phase_outputs/*.json` → saved phase report

---

# Progress Tracker (page_progress.py)

## What it does

Reads **all historical phase outputs** and generates:

* pain trend 📉
* quality trend 📈
* mistake frequency ⚠️
* LLM-generated longitudinal report

---

## Data Source (VERY IMPORTANT)

All data comes from:

```
phase_outputs/
```

This folder acts as **long-term memory**.

Each file = one rehab phase:

```json
{
  "phase_summary": {...},
  "exercises": [...],
  "overall_quality_trend": "...",
  "phase_report": "...",
  "next_phase_focus": [...]
}
```

---

## How to Integrate External Data

If you already have your own backend:

### Option 1 — Direct JSON dump

Just write files into:

```
phase_outputs/
```

The UI will automatically pick them up.

---

### Option 2 — Replace loader

Modify:

```python
load_phase_jsons()
```

to read from:

* database (MongoDB / Postgres)
* API endpoint
* cloud storage (S3)

---

## LLM Usage

Progress report is generated via:

```python
ProgressTracker(...).run()
```

Using:

```
gemma3:4b (via Ollama)
```

---

# Data Flow Overview

```
[External Sensor / CV Model]
            ↓
      Coaching Events
            ↓
    SessionManager.ingest()
            ↓
  Exercise Feedback (real-time UI)
            ↓
    Phase JSON saved
            ↓
     phase_outputs/
            ↓
   Progress Tracker loads all
            ↓
   Trend analysis + LLM report
```

---

# Key Integration Points

### 1. Replace mock events

```
page_coaching.py → EVENTS_EX1 / EVENTS_EX2
```

---

### 2. Connect real-time stream

```
SessionManager.ingest(payload)
```

---

### 3. Persist phase outputs

```
phase_outputs/*.json
```

---

### 4. Progress Tracker reads ALL history

No API needed — just drop JSON files.

---

# Current Limitations

* Uses simulated data (not real-time yet)
* No authentication / multi-user separation
* Phase storage is file-based (not DB)

---

