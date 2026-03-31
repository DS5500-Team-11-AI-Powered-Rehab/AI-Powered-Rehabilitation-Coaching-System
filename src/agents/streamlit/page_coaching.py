"""
page_coaching.py — Coaching Session Page
Live session simulation: events flow in every 5s,
exercise feedback cards appear after 30s silence,
phase report appears after 120s silence.
"""

import sys
import time
import threading
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
COACHING_DIR  = Path(__file__).parent.parent / "coaching_agent"
TRACKER_DIR   = Path(__file__).parent.parent / "progress_tracker_agent"
sys.path.insert(0, str(COACHING_DIR))
sys.path.insert(0, str(TRACKER_DIR))

from session_manager import SessionManager


# ── Sample data ───────────────────────────────────────────────────────────────

PATIENT_PROFILE = {
    "patient_id":         "P001",
    "condition":          "knee osteoarthritis",
    "condition_category": "knee",
    "rehab_phase":        "mid",
    "pain_level":         4,
    "weeks_into_rehab":   10,
    "age":                58,
    "goals":              "Walk dog 30 minutes daily without pain",
}


def _make_event(exercise_name, mistake_type, quality, severity,
                occurrences, rom_level, session_time):
    return {
        "coaching_event": {
            "exercise": {"name": exercise_name, "confidence": 0.88},
            "mistake": {
                "type":             mistake_type,
                "confidence":       0.75,
                "duration_seconds": 3.5,
                "persistence_rate": 0.4,
                "occurrences":      occurrences,
            },
            "metrics": {
                "speed_rps":          0.9,
                "rom_level":          rom_level,
                "height_level":       3,
                "torso_rotation":     0,
                "direction":          "none",
                "no_obvious_issue_p": 0.1,
            },
            "quality_score":        quality,
            "severity":             severity,
            "is_recoaching":        False,
            "session_time_minutes": session_time,
            "tier":                 "tier_2",
            "cache_key":            None,
            "routing_reason":       "form issue detected",
        },
        "session_id":       "session_P001_mid",
        "coaching_history": [],
    }


EVENTS_EX1 = [
    _make_event("mini squat", "knee valgus", 0.28, "high",   31, 1, 0.0),
    _make_event("mini squat", "knee valgus", 0.31, "high",   28, 1, 0.1),
    _make_event("mini squat", "knee valgus", 0.35, "medium", 22, 2, 0.2),
    _make_event("mini squat", "knee valgus", 0.40, "medium", 18, 2, 0.3),
    _make_event("mini squat", "knee valgus", 0.44, "medium", 15, 2, 0.4),
    _make_event("mini squat", "knee valgus", 0.50, "low",    10, 3, 0.5),
]

EVENTS_EX2 = [
    _make_event("leg press", "forward lean", 0.55, "medium", 20, 2, 2.0),
    _make_event("leg press", "forward lean", 0.52, "medium", 18, 2, 2.1),
    _make_event("leg press", "forward lean", 0.58, "medium", 15, 3, 2.2),
    _make_event("leg press", "forward lean", 0.56, "low",    12, 3, 2.3),
    _make_event("leg press", "forward lean", 0.60, "low",    10, 3, 2.4),
    _make_event("leg press", "forward lean", 0.62, "low",     8, 3, 2.5),
]


# ── Session runner (background thread) ───────────────────────────────────────

def _run_session(sm: SessionManager, status_list: list):
    """
    Run the full session in a background thread.
    Appends status strings to status_list so the main thread can display them.
    """
    # Exercise 1
    status_list.append(("info", "--- Exercise 1: Mini Squat started ---"))
    for payload in EVENTS_EX1:
        sm.ingest(payload)
        e = payload["coaching_event"]
        status_list.append((
            "event",
            f"event: {e['exercise']['name']} | "
            f"quality={e['quality_score']:.2f} | "
            f"mistake={e['mistake']['type']}"
        ))
        time.sleep(5)

    status_list.append(("info", "Waiting for exercise 1 timeout (30s)..."))
    time.sleep(35)

    # Exercise 2
    status_list.append(("info", "--- Exercise 2: Leg Press started ---"))
    for payload in EVENTS_EX2:
        sm.ingest(payload)
        e = payload["coaching_event"]
        status_list.append((
            "event",
            f"event: {e['exercise']['name']} | "
            f"quality={e['quality_score']:.2f} | "
            f"mistake={e['mistake']['type']}"
        ))
        time.sleep(5)

    status_list.append(("info", "Waiting for exercise 2 timeout (30s)..."))
    time.sleep(35)

    status_list.append(("info", "Waiting for phase timeout (120s)..."))
    time.sleep(125)

    status_list.append(("done", "Session complete ✓"))


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.title("🏋️  Coaching Session")
    st.markdown(
        "Simulates a live rehab session. Events stream in every **5s**, "
        "exercise feedback appears after **30s** silence, "
        "phase report after **120s** silence."
    )

    # ── Patient info ──────────────────────────────────────────────────────────
    with st.expander("👤 Patient Profile", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient",   PATIENT_PROFILE["patient_id"])
        c2.metric("Phase",     PATIENT_PROFILE["rehab_phase"].upper())
        c3.metric("Pain",      f"{PATIENT_PROFILE['pain_level']}/10")
        c4.metric("Week",      PATIENT_PROFILE["weeks_into_rehab"])
        st.caption(f"Condition: {PATIENT_PROFILE['condition']} | "
                   f"Goal: {PATIENT_PROFILE['goals']}")

    st.markdown("---")

    # ── Session state init ────────────────────────────────────────────────────
    if "session_running"   not in st.session_state:
        st.session_state.session_running   = False
    if "session_done"      not in st.session_state:
        st.session_state.session_done      = False
    if "sm"                not in st.session_state:
        st.session_state.sm                = None
    if "status_list"       not in st.session_state:
        st.session_state.status_list       = []
    if "shown_feedbacks"   not in st.session_state:
        st.session_state.shown_feedbacks   = 0
    if "phase_report_shown" not in st.session_state:
        st.session_state.phase_report_shown = False

    # ── Start button ──────────────────────────────────────────────────────────
    col_btn, col_reset = st.columns([1, 5])
    with col_btn:
        start_clicked = st.button(
            "▶  Start Session",
            disabled=st.session_state.session_running,
            type="primary",
        )
    with col_reset:
        if st.button("↺  Reset", disabled=st.session_state.session_running):
            for key in ["session_running", "session_done", "sm",
                        "status_list", "shown_feedbacks", "phase_report_shown"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if start_clicked and not st.session_state.session_running:
        sm = SessionManager(
            patient_profile=PATIENT_PROFILE,
            ollama_model="gemma3:4b",
            verbose=False,
        )
        sm.start()
        st.session_state.sm             = sm
        st.session_state.status_list    = []
        st.session_state.session_running = True
        st.session_state.session_done   = False
        st.session_state.shown_feedbacks = 0
        st.session_state.phase_report_shown = False

        t = threading.Thread(
            target=_run_session,
            args=(sm, st.session_state.status_list),
            daemon=True,
        )
        t.start()
        st.rerun()

    # ── Live display ──────────────────────────────────────────────────────────
    if st.session_state.session_running or st.session_state.session_done:
        sm          = st.session_state.sm
        status_list = st.session_state.status_list

        # Event log
        st.subheader("📡 Event Stream")
        event_container = st.container()
        with event_container:
            for kind, msg in status_list:
                if kind == "event":
                    st.code(f"↳ {msg}", language=None)
                elif kind == "info":
                    st.info(msg)
                elif kind == "done":
                    st.success(msg)
                    st.session_state.session_running = False
                    st.session_state.session_done    = True

        # Exercise feedback cards
        if sm and len(sm.exercise_feedbacks) > st.session_state.shown_feedbacks:
            st.subheader("🏅 Exercise Feedback")
            for fb in sm.exercise_feedbacks[st.session_state.shown_feedbacks:]:
                with st.container(border=True):
                    st.markdown(f"### 🏋️ {fb['exercise_name']}")
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Avg Quality", f"{fb['avg_quality']:.2f}")
                    mc2.metric("Trend", fb["quality_trend"].upper())

                    st.markdown("**❌ What to Correct**")
                    for m in fb["mistakes"]:
                        sev_color = (
                            "🔴" if m["severity"] == "high"
                            else "🟡" if m["severity"] == "medium"
                            else "🟢"
                        )
                        st.markdown(
                            f"- {sev_color} **{m['type']}** — "
                            f"{m['occurrences']} occurrences ({m['severity']} severity)"
                        )

                    st.markdown("**✅ What You Did Well**")
                    for ok in fb["ok_aspects"]:
                        st.markdown(f"- {ok}")

                    st.markdown("**💡 Coaching Cue**")
                    st.info(fb["feedback"])

            st.session_state.shown_feedbacks = len(sm.exercise_feedbacks)

        # Phase report
        if (sm and sm.phase_ended
                and not st.session_state.phase_report_shown
                and sm.completed_exercises):

            st.subheader("📋 Phase Report")

            # Find latest JSON
            outputs_dir = Path(__file__).parent.parent / "phase_outputs"
            json_files  = sorted(outputs_dir.glob("*.json"))

            if json_files:
                import json as _json
                with open(json_files[-1], encoding="utf-8") as f:
                    data = _json.load(f)

                with st.container(border=True):
                    st.markdown(f"### Session Summary")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Exercises", len(data["exercises"]))
                    rc2.metric("Quality Trend", data["overall_quality_trend"].upper())
                    rc3.metric("Pain Level", f"{data['phase_summary']['pain_level']}/10")

                    st.markdown("**📝 Phase Report**")
                    st.markdown(data["phase_report"])

                    st.markdown("**🎯 Next Phase Focus**")
                    for focus in data["next_phase_focus"]:
                        st.markdown(f"- {focus}")

                    st.markdown("**🗂 JSON Preview**")
                    with st.expander("View full JSON"):
                        st.json(data)

            st.session_state.phase_report_shown = True

        # Auto-refresh while session is running
        if st.session_state.session_running:
            time.sleep(2)
            st.rerun()
