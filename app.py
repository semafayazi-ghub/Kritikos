"""
KRITIKOS - V5
=================================================================
Embedded, context-aware reflection assistant for critical AI use.

This is the V5 rebuild. Every change relative to V4 is driven by a
finding from the primary usability evaluation (heuristic E1/E2/E4,
SUS P1/P2/P3, reflection interview P1/P2). Each fix is tagged in the
code with  [V5-FIX #n]  so the report and the code documentation can
point straight at it.

V5 fixes
--------
 #1  First-run onboarding card (no-onboarding was the strongest finding;
     E1, E4, P1, P2). Explains what KRITIKOS is, that the AI answers and
     KRITIKOS only challenges, and that prompts are optional.
 #2  Clear AI-vs-KRITIKOS identity: the AI answer and the KRITIKOS
     reflection use distinct labels + avatars + colours (E1, P1).
 #3  Empty / invalid input handling: inline, plain-language hint instead
     of a silent no-op (E4, P2).
 #4  One focused question per reflection prompt (E1: multi-question
     prompts were overwhelming).
 #5  Visible question / conversation history (E4, P2).
 #6  Calmer accent colour (teal) instead of alarming red; transparent
     owl logo (E1).
 #7  Internal model signal is NEVER rendered to the user. In V4 a debug
     line ("this answer looks non-academic / weaker") leaked onto the
     screen; in V5 it lives only in the logs.
 #8  Starter-prompt suggestions for first-time users (E4).
 #9  Privacy note surfaced near the top, not only at the bottom (P2).
 #10 Even button spacing + mobile-friendly widths (E1, E2).

V5.1 fix (post-deployment, live-app user feedback)
--------------------------------------------------
 #11 Starter prompts now actually populate the question box. On the
     deployed app, tapping a starter and then "Ask" produced the empty-
     input hint: the un-keyed text_input lost its prefilled value on the
     next rerun. The input is now a keyed widget; starters write into
     its session state before instantiation, and the box is cleared via
     a flag handled at the top of the run. This closes the loop on
     [V5-FIX #3] + [V5-FIX #8], which the deployment surfaced as
     interacting badly.

Run:  streamlit run app.py
Author: Sema (Samaneh) Fayazi - Master Data-driven Design (D01), HU
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import streamlit as st

from reflection_engine import (
    REFLECTION_CUES,
    get_ai_answer,
    get_kritikos_reflection,
    score_answer_signal,
)

# ----------------------------------------------------------------------
# Logging - the internal model signal goes here, never to the screen.
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kritikos")

# ----------------------------------------------------------------------
# Logo - transparent PNG so it renders correctly on light AND dark themes.
# Falls back to the owl emoji if the file is missing (graceful degradation,
# same policy as the engine).
# ----------------------------------------------------------------------
_LOGO_PATH = Path(__file__).parent / "kritikos_logo.png"
try:
    _LOGO_B64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
except OSError:
    _LOGO_B64 = ""


def _logo_img(height_em: float, valign: str = "-0.18em") -> str:
    """Inline <img> tag for the transparent logo ('' if the file is absent)."""
    if not _LOGO_B64:
        return ""
    return (
        f'<img src="data:image/png;base64,{_LOGO_B64}" '
        f'style="height:{height_em}em;width:auto;vertical-align:{valign};" '
        f'alt="KRITIKOS owl logo"/>'
    )


_ICON_INLINE = _logo_img(1.05) or "🦉"   # small inline mark for block labels


def kritikos_header() -> None:
    """Page header: the real logo when available, emoji fallback otherwise."""
    if _LOGO_B64:
        st.markdown(
            f'<h1 style="display:flex;align-items:center;gap:.4em;">'
            f'{_logo_img(1.15, "0")}<span>KRITIKOS</span></h1>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("# 🦉 KRITIKOS")


try:
    from PIL import Image as _PILImage
    _PAGE_ICON = _PILImage.open(_LOGO_PATH)
except Exception:
    _PAGE_ICON = "🦉"

# ----------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="KRITIKOS",
    page_icon=_PAGE_ICON,
    layout="centered",          # [V5-FIX #10] centered reads better on mobile
    initial_sidebar_state="expanded",
)

# [V5-FIX #6] Calmer teal accent instead of the alarming red; clearer
# separation between the AI answer block and the KRITIKOS block.
ACCENT = "#0F8B8D"          # calm teal
AI_BG = "#EEF4FB"           # light blue   -> AI answer
KRITIKOS_BG = "#F0FBF7"     # light green  -> KRITIKOS reflection

st.markdown(
    f"""
    <style>
      .stApp {{ }}
      .kritikos-accent {{ color: {ACCENT}; }}
      .answer-block {{
          background: {AI_BG}; border-left: 4px solid #2E5E8C;
          padding: 1rem 1.2rem; border-radius: 8px; margin: .5rem 0 1rem;
      }}
      .kritikos-block {{
          background: {KRITIKOS_BG}; border-left: 4px solid {ACCENT};
          padding: 1rem 1.2rem; border-radius: 8px; margin: .5rem 0 1rem;
      }}
      .block-label {{
          font-size: .8rem; font-weight: 700; letter-spacing: .04em;
          text-transform: uppercase; margin-bottom: .3rem;
      }}
      .ai-label {{ color: #2E5E8C; }}
      .kritikos-label {{ color: {ACCENT}; }}
      /* [V5-FIX #10] even spacing across the reflection-cue buttons */
      div[data-testid="column"] {{ padding: 0 .35rem; }}
      .privacy-pill {{
          background: #F4F6F8; color: #475569; font-size: .82rem;
          padding: .5rem .8rem; border-radius: 999px; display: inline-block;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "show_prompts": True,
        "seen_onboarding": False,   # [V5-FIX #1]
        "history": [],              # [V5-FIX #5] list of {"q":..., "a":..., "signal":...}
        "active_reflection": None,  # currently open cue, or None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()

# ----------------------------------------------------------------------
# Sidebar - settings, model card, privacy
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")
    st.session_state.show_prompts = st.toggle(
        "Show reflection prompts",
        value=st.session_state.show_prompts,
        help="Reflection cues appear under each AI answer. They are always "
             "optional — turn them off any time.",
    )
    if st.button("Start over", use_container_width=True):
        st.session_state.history = []
        st.session_state.active_reflection = None
        st.rerun()

    st.divider()
    st.markdown("### About this prototype")
    st.write(
        "Ask a real question and get a real AI answer. Tap a reflection "
        "prompt to open a short critical dialogue with KRITIKOS. "
        "It won't give you the answer — it makes you think. A small ML "
        "model chooses which prompts to show."
    )
    st.markdown(
        "**Reflection model:** Logistic Regression  \n"
        "**5-fold CV accuracy:** 1.0  \n"
        "**Trained on:** 120 synthetic examples"
    )
    st.caption("Synthetic training data → proof of concept.")


# ----------------------------------------------------------------------
# [V5-FIX #1 + #9] First-run onboarding card
# ----------------------------------------------------------------------
def render_onboarding() -> None:
    kritikos_header()
    st.markdown(
        "A second look at every AI answer — reflection that challenges you, "
        "so you keep thinking for yourself."
    )
    with st.container(border=True):
        st.markdown("#### How it works")
        st.markdown(
            "- **You ask, the AI answers.** KRITIKOS does **not** write the "
            "answer — it only helps you look at it more critically.\n"
            "- **Reflection is optional.** After each answer you'll see a few "
            "small cues (like *Check the source*). Tap one to reflect, or just "
            "keep working.\n"
            "- **KRITIKOS never tells you what's true.** It asks one short "
            "question at a time and leaves the judgement to you."
        )
        # [V5-FIX #9] privacy surfaced up front, not only at the bottom
        st.markdown(
            '<span class="privacy-pill">🔒 No account, no tracking. '
            "KRITIKOS stores no chats or personal data.</span>",
            unsafe_allow_html=True,
        )
        if st.button("Got it — start asking", type="primary"):
            st.session_state.seen_onboarding = True
            st.rerun()


# ----------------------------------------------------------------------
# Answer + reflection rendering
# ----------------------------------------------------------------------
def render_answer(entry: dict, idx: int) -> None:
    """Render one AI answer with clear identity + optional reflection cues."""
    # [V5-FIX #2] explicit, distinct label for the AI answer
    st.markdown(
        f'<div class="answer-block"><div class="block-label ai-label">'
        f'🤖 AI answer</div>{entry["a"]}</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.show_prompts:
        return

    # [V5-FIX #7] The model signal decides WHICH cues to show. It is logged,
    # never printed to the UI.
    signal = entry["signal"]
    log.info("internal signal for answer #%d: %s (not shown to user)", idx, signal)
    cues = REFLECTION_CUES[signal]

    st.markdown("**Reflect with KRITIKOS** *(optional)*")
    # [V5-FIX #10] even columns -> even button spacing
    cols = st.columns(len(cues))
    for col, cue in zip(cols, cues):
        if col.button(cue, key=f"cue_{idx}_{cue}", use_container_width=True):
            st.session_state.active_reflection = (idx, cue)
            st.rerun()


def render_reflection(idx: int, cue: str) -> None:
    """Render the KRITIKOS reflective dialogue for one cue."""
    entry = st.session_state.history[idx]
    # [V5-FIX #2] distinct KRITIKOS identity (avatar + colour + label)
    # [V5-FIX #4] reflection_engine returns ONE focused question
    reflection = get_kritikos_reflection(cue, entry["q"], entry["a"])
    st.markdown(
        f'<div class="kritikos-block"><div class="block-label kritikos-label">'
        f'{_ICON_INLINE} KRITIKOS · reflecting on “{cue}”</div>{reflection}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Try another angle:")
    cues = REFLECTION_CUES[entry["signal"]]
    cols = st.columns(len(cues))
    for col, c in zip(cols, cues):
        if col.button(c, key=f"again_{idx}_{c}", use_container_width=True):
            st.session_state.active_reflection = (idx, c)
            st.rerun()
    if st.button("← Back to the answer", key=f"back_{idx}"):
        st.session_state.active_reflection = None
        st.rerun()


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
if not st.session_state.seen_onboarding:
    render_onboarding()
    st.stop()

kritikos_header()
st.caption(
    "A second look at every AI answer — reflection that challenges you, "
    "so you keep thinking for yourself."
)

# [V5-FIX #5] show the conversation history (most recent last)
for i, entry in enumerate(st.session_state.history):
    st.markdown(f"**You asked:** {entry['q']}")
    render_answer(entry, i)
    if (st.session_state.active_reflection
            and st.session_state.active_reflection[0] == i):
        render_reflection(i, st.session_state.active_reflection[1])
    st.divider()

# ----------------------------------------------------------------------
# Ask box  (+ [V5-FIX #3] validation, [V5-FIX #8] starter prompts)
# ----------------------------------------------------------------------
# [V5.1-FIX #11] Starter prompts now write straight into the *keyed* input
# widget's session state BEFORE the widget is instantiated. The previous
# pop-a-"prefill"-into-value pattern broke on the very next rerun: the
# default value flipped back to "", Streamlit treated that as a different
# widget and reset it, so pressing "Ask" after tapping a starter raised
# the "Please type a question first" hint even though a suggestion had
# just been chosen. With a stable key the typed/prefilled text survives
# every rerun, and the box is cleared through a flag that is handled
# before the widget exists (Streamlit forbids mutating a widget's state
# after instantiation within the same run).
st.session_state.setdefault("question_box", "")
if st.session_state.pop("clear_question", False):
    st.session_state.question_box = ""

with st.container(border=True):
    st.markdown("#### Ask the AI a question")

    # [V5-FIX #8] starter prompts for first-time users
    if not st.session_state.history:
        st.caption("New here? Try one of these:")
        starters = [
            "What does research say about social media and teen mental health?",
            "Is intermittent fasting actually effective?",
            "How can I find a job in the Netherlands?",
        ]
        scols = st.columns(len(starters))
        for col, s in zip(scols, starters):
            if col.button(s, key=f"starter_{s}", use_container_width=True):
                # [V5.1-FIX #11] set the widget state directly; the button's
                # own rerun re-creates the input with this text already in it.
                st.session_state.question_box = s
                st.rerun()

    question = st.text_input(
        "Your question",
        key="question_box",   # [V5.1-FIX #11] stable identity across reruns
        placeholder="e.g. What does research say about social media and teen mental health?",
        label_visibility="collapsed",
    )
    asked = st.button("Ask", type="primary")

if asked:
    # [V5-FIX #3] empty / invalid input -> plain-language hint, not a silent no-op
    if not question or not question.strip():
        st.warning("Please type a question first — KRITIKOS needs something to look at. 🙂")
    elif len(question.strip()) < 4:
        st.warning("That looks a little short. Try a full question so the AI has something to work with.")
    else:
        with st.spinner("Asking the AI…"):
            answer = get_ai_answer(question.strip())
            signal = score_answer_signal(question.strip(), answer)  # internal only
        st.session_state.history.append(
            {"q": question.strip(), "a": answer, "signal": signal}
        )
        st.session_state.active_reflection = None
        # [V5.1-FIX #11] request a clear of the input on the next run —
        # mutating st.session_state.question_box here would raise, because
        # the widget has already been instantiated in this run.
        st.session_state.clear_question = True
        st.rerun()

# [V5-FIX #9] privacy line still present at the bottom too, for persistence
st.divider()
st.caption(
    "🔒 KRITIKOS does not store your chats or personal data. Your messages "
    "are sent to the AI model only to generate replies."
)
