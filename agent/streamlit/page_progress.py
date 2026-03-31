"""
page_progress.py — Progress Tracker Page
Reads phase JSON files, shows trend charts + LLM report.
"""

import sys
import json
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
TRACKER_DIR = Path(__file__).parent.parent / "progress_tracker_agent"
sys.path.insert(0, str(TRACKER_DIR))

from progress_tracker import (
    ProgressTracker, load_phase_jsons, PHASE_OUTPUTS_DIR,
    analyze_pain_trend, analyze_quality_trend, analyze_mistake_trend,
    analyze_exercise_progression,
)


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.title("📈  Progress Tracker")
    st.markdown(
        "Reads all phase JSON files from `phase_outputs/` and generates "
        "a longitudinal progress report."
    )

    # ── Load files ────────────────────────────────────────────────────────────
    json_files = sorted(PHASE_OUTPUTS_DIR.glob("*.json"))

    if not json_files:
        st.warning(
            "No phase JSON files found in `phase_outputs/`. "
            "Run a Coaching Session first to generate data."
        )
        return

    phases = load_phase_jsons()
    if not phases:
        st.error("Could not load phase data.")
        return

    # ── Phase file overview ───────────────────────────────────────────────────
    st.subheader("📂 Phase Files")
    cols = st.columns(len(phases))
    for i, (col, p) in enumerate(zip(cols, phases)):
        ps = p["phase_summary"]
        with col:
            with st.container(border=True):
                st.markdown(f"**Phase {i+1}: {ps.get('rehab_phase','?').upper()}**")
                st.metric("Pain",    f"{ps.get('pain_level','?')}/10")
                st.metric("Week",    ps.get("weeks_into_rehab", "?"))
                exs = [ex["exercise_name"] for ex in p.get("exercises", [])]
                st.caption(f"Exercises: {', '.join(exs)}")

    st.markdown("---")

    # ── Trend charts ──────────────────────────────────────────────────────────
    st.subheader("📊 Trend Charts")

    pain_data    = analyze_pain_trend(phases)
    quality_data = analyze_quality_trend(phases)
    mistake_data = analyze_mistake_trend(phases)

    phase_labels = [
        f"Phase {i+1} ({p['phase_summary'].get('rehab_phase','?')})"
        for i, p in enumerate(phases)
    ]

    chart_col1, chart_col2 = st.columns(2)

    # Pain chart
    with chart_col1:
        st.markdown("**📉 Pain Level per Phase**")
        import pandas as pd
        pain_df = pd.DataFrame({
            "Phase": phase_labels,
            "Pain Level": pain_data["values"],
        }).set_index("Phase")
        st.line_chart(pain_df, color="#ff4b4b")
        trend_color = (
            "🟢" if pain_data["trend"] == "improving"
            else "🔴" if pain_data["trend"] == "worsening"
            else "🟡"
        )
        st.caption(f"{trend_color} Trend: **{pain_data['trend']}** "
                   f"(change: {pain_data['total_change']:+d} pts)")

    # Quality chart
    with chart_col2:
        st.markdown("**📈 Avg Quality Score per Phase**")
        qual_values = [
            v if v is not None else 0
            for v in quality_data["values_per_phase"]
        ]
        qual_df = pd.DataFrame({
            "Phase": phase_labels,
            "Avg Quality": qual_values,
        }).set_index("Phase")
        st.line_chart(qual_df, color="#00cc88")
        trend_color = (
            "🟢" if quality_data["trend"] == "improving"
            else "🔴" if quality_data["trend"] == "declining"
            else "🟡"
        )
        st.caption(f"{trend_color} Trend: **{quality_data['trend']}**")

    # Mistake frequency bar chart
    st.markdown("**⚠️ Mistake Frequency Across All Phases**")
    if mistake_data["ranked"]:
        mistake_df = pd.DataFrame(
            mistake_data["ranked"][:8],
            columns=["Mistake Type", "Total Occurrences"]
        ).set_index("Mistake Type")
        st.bar_chart(mistake_df, color="#ffaa00")

        persistent = mistake_data["persistent"]
        if persistent:
            st.warning(
                f"**Persistent issues** (appear in 2+ phases): "
                + ", ".join(f"**{m}**" for m in persistent)
            )
    else:
        st.info("No mistakes recorded.")

    st.markdown("---")

    # ── Exercise progression ──────────────────────────────────────────────────
    st.subheader("🏋️ Exercise Progression")
    ex_lines = analyze_exercise_progression(phases)
    for line in ex_lines:
        st.markdown(f"- {line}")

    st.markdown("---")

    # ── Generate report ───────────────────────────────────────────────────────
    st.subheader("📝 Progress Report")

    if "progress_result" not in st.session_state:
        st.session_state.progress_result = None

    if st.button("🤖 Generate Progress Report", type="primary"):
        with st.spinner("Analysing phases and generating report with gemma3:4b..."):
            tracker = ProgressTracker(
                ollama_model="gemma3:4b",
                verbose=False,
            )
            result = tracker.run(patient_id=None)
            st.session_state.progress_result = result

    if st.session_state.progress_result:
        result = st.session_state.progress_result

        # Summary metrics
        pain    = result["pain_analysis"]
        qual    = result["quality_analysis"]
        mist    = result["mistake_analysis"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Phases Analysed", result["phases_analyzed"])
        m2.metric(
            "Pain Change",
            f"{pain['total_change']:+d} pts",
            delta_color="inverse",
        )
        m3.metric(
            "Quality Trend",
            qual["trend"].upper(),
        )
        m4.metric(
            "Persistent Issues",
            len(mist["persistent"]),
        )

        # Full report text
        with st.container(border=True):
            st.markdown("### Longitudinal Progress Report")
            st.markdown(result["progress_report"])

        # Next phase focus
        st.markdown("**🎯 Recommendations for Next Phase**")
        for f in result.get("next_phase_focus", []):
            # next_phase_focus not in result directly, get from latest phase JSON
            pass

        latest_json = sorted(PHASE_OUTPUTS_DIR.glob("*.json"))
        if latest_json:
            with open(latest_json[-1], encoding="utf-8") as f:
                latest_data = json.load(f)
            for focus in latest_data.get("next_phase_focus", []):
                st.markdown(f"- {focus}")

        # JSON preview
        st.markdown("---")
        st.subheader("🗂 JSON Preview")
        tabs = st.tabs([f.name for f in sorted(PHASE_OUTPUTS_DIR.glob("*.json"))])
        for tab, f in zip(tabs, sorted(PHASE_OUTPUTS_DIR.glob("*.json"))):
            with tab:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                st.json(data)
