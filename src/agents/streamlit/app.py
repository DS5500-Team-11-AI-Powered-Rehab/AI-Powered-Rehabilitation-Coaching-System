"""
app.py — Streamlit Frontend Entry Point
Run: streamlit run app.py

Structure:
  - sidebar navigation
  - page_coaching.py   → Coaching Agent live session
  - page_progress.py   → Progress Tracker report + charts
"""

import streamlit as st

st.set_page_config(
    page_title="Rehab AI Agent",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🏥 Rehab AI Agent")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏋️  Coaching Session", "📈  Progress Tracker"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Stack**\n"
    "- gemma3:4b via Ollama\n"
    "- ChromaDB + RAG\n"
    "- LangChain\n"
)
st.sidebar.markdown(
    "⚠️ Make sure Ollama is running:\n```\nollama serve\n```"
)

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "🏋️  Coaching Session":
    import page_coaching
    page_coaching.render()
else:
    import page_progress
    page_progress.render()
