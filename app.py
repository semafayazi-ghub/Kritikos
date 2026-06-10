"""
app.py  —  KRITIKOS (redesigned, interactive version)
=====================================================
KRITIKOS is a reflective AI companion. The user asks a normal AI question and
gets an answer. Underneath, KRITIKOS shows context-aware reflection prompts.
When the user taps one (e.g. "Check the source"), KRITIKOS does NOT give a
verdict — it opens a short critical DIALOGUE: it asks probing questions,
challenges assumptions, and helps the user think about sources and bias.

Two roles, one model (Groq):
  - ANSWERER: a normal assistant that answers the user's question.
  - KRITIKOS COACH: a Socratic guide that never gives final answers; it asks
    questions, surfaces what to check, and pushes the user to reflect.

Responsible-AI rule: KRITIKOS supports the user's judgement, it does not
replace it. The reflection classifier only decides which prompts to show.

RUN
  pip install -r requirements.txt
  streamlit run app.py
Needs a GROQ_API_KEY in Streamlit secrets (or .streamlit/secrets.toml).
"""

import streamlit as st
from groq import Groq
from reflection_engine import train_model, classify_text, select_prompts
import os, base64

st.set_page_config(page_title="KRITIKOS", page_icon="🦉", layout="centered")

# load the owl logo if it is present in the repo (falls back to the emoji)
LOGO_PATH = "kritikos_logo.png"
HAS_LOGO = os.path.exists(LOGO_PATH)

def logo_data_uri():
    with open(LOGO_PATH, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

st.markdown("""
<style>
    .ai-answer {background:#EAF1F8; border-left:4px solid #2E6FB0;
                padding:14px 16px; border-radius:8px; margin:6px 0; color:#1F3A5F;}
    .reflect-label {color:#3F8E5C; font-size:0.85em; margin-top:8px;}
    .data-note {color:#888; font-size:0.8em; border-top:1px solid #eee;
                padding-top:8px; margin-top:18px;}
    .stButton>button {border-radius:18px; border:1px solid #2E6FB0; color:#2E6FB0;
                      background:white; font-size:0.85em; padding:2px 12px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return train_model()

model, metrics = get_model()


def get_client():
    key = st.secrets.get("GROQ_API_KEY", None)
    return Groq(api_key=key) if key else None

client = get_client()
MODEL = "llama-3.1-8b-instant"
OWL = LOGO_PATH if HAS_LOGO else "🦉"

# The personality that makes KRITIKOS a critical coach, not an answer machine.
COACH_SYSTEM = (
    "You are KRITIKOS, a critical-thinking coach for university students who use AI for "
    "research. You NEVER give the final answer or do the work for the student. Your job is "
    "to help them evaluate an AI answer critically. Be warm but challenging. Keep each reply "
    "SHORT (2-4 sentences). Always end with ONE pointed question that pushes the student to "
    "think about source quality, evidence, bias, missing perspectives, or how to verify the "
    "claim elsewhere. When relevant, suggest concrete ways to check (e.g. look for a "
    "peer-reviewed study, compare two sources, check the date/author), but make the student "
    "do the checking. Never claim something is simply true or false."
)

# Each prompt seeds the dialogue with a specific critical angle.
PROMPT_SEED = {
    "Check the source": "I want to check where this answer comes from. Push me to think about the source quality.",
    "This may be incomplete": "This answer may be incomplete. Help me notice what context or viewpoints might be missing.",
    "Compare with another source": "Help me think about how to compare this with another independent source.",
    "Verify references": "Help me think about how to verify any references or studies behind this claim.",
    "Compare with literature": "Help me think about how this compares with peer-reviewed literature.",
    "Based on academic sources": "This sounds academic. Push me to still check the method and limitations.",
}

if HAS_LOGO:
    c1, c2 = st.columns([1, 6])
    with c1:
        st.image(LOGO_PATH, width=64)
    with c2:
        st.markdown("## KRITIKOS")
        st.caption("A second look at every AI answer — reflection that challenges you, without doing the thinking for you.")
else:
    st.markdown("## 🦉 KRITIKOS")
    st.caption("A second look at every AI answer — reflection that challenges you, without doing the thinking for you.")

with st.sidebar:
    st.markdown("### Settings")
    show_reflection = st.toggle("Show reflection prompts", value=True,
                                help="You decide. Turn this off to use AI without reflection.")
    if st.button("Start over"):
        for k in ["answer", "question_text", "dialogue", "active_prompt"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("---")
    st.markdown("### About this prototype")
    st.write("Ask a real question and get a real AI answer. Tap a reflection prompt to open a "
             "short critical dialogue with KRITIKOS. It won't give you the answer — it makes "
             "you think. A small ML model chooses which prompts to show.")
    st.markdown(f"**Reflection model:** Logistic Regression  \n"
                f"**5-fold CV accuracy:** {metrics['cv_accuracy']}  \n"
                f"**Trained on:** {metrics['n_samples']} synthetic examples")
    st.caption("Synthetic training data → proof of concept.")

if client is None:
    st.error("No AI key found. Add GROQ_API_KEY in Streamlit secrets to enable live answers.")
    st.stop()

# session state
st.session_state.setdefault("answer", None)
st.session_state.setdefault("question_text", "")
st.session_state.setdefault("dialogue", [])       # list of (role, text) for the KRITIKOS chat
st.session_state.setdefault("active_prompt", None)


def ask_answerer(question):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=400,
    )
    return resp.choices[0].message.content


def ask_coach(history):
    """history = list of (role, text). Returns KRITIKOS's next critical reply."""
    msgs = [{"role": "system", "content": COACH_SYSTEM},
            {"role": "system", "content":
             f"The student asked an AI: '{st.session_state.question_text}'. "
             f"The AI answered: '{st.session_state.answer}'. "
             "Coach the student to evaluate THIS answer critically."}]
    for role, text in history:
        msgs.append({"role": "user" if role == "user" else "assistant", "content": text})
    resp = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=220)
    return resp.choices[0].message.content


# ---- 1) the AI question ----
q = st.text_input("Ask the AI a question:",
                  placeholder="e.g. What does research say about social media and teen mental health?")
if st.button("Ask") and q.strip():
    with st.spinner("Thinking..."):
        try:
            st.session_state.question_text = q
            st.session_state.answer = ask_answerer(q)
            st.session_state.dialogue = []
            st.session_state.active_prompt = None
        except Exception as e:
            st.error(f"Could not get an answer: {e}")

# ---- 2) the AI answer + reflection layer ----
if st.session_state.answer:
    st.markdown(f"<div class='ai-answer'>{st.session_state.answer}</div>", unsafe_allow_html=True)

    if show_reflection:
        prediction = classify_text(model, st.session_state.answer)
        prompts = select_prompts(prediction)
        st.markdown("<div class='reflect-label'>Reflect with KRITIKOS (optional):</div>",
                    unsafe_allow_html=True)
        cols = st.columns(len(prompts))
        for col, prompt in zip(cols, prompts):
            if col.button(prompt, key="p_" + prompt):
                # start a new critical dialogue seeded by this prompt
                st.session_state.active_prompt = prompt
                seed = PROMPT_SEED.get(prompt, "Help me reflect on this answer.")
                with st.spinner("KRITIKOS is thinking..."):
                    first = ask_coach([("user", seed)])
                st.session_state.dialogue = [("kritikos", first)]

        label = "academic / well-supported" if prediction == 1 else "non-academic / weaker"
        st.caption(f"(internal signal: this answer looks **{label}** — used only to choose "
                   "which prompts appear, never shown as a verdict)")

    # ---- 3) the ongoing critical dialogue ----
    if st.session_state.dialogue:
        st.markdown("---")
        st.markdown("#### Reflecting on: *" + (st.session_state.active_prompt or "") + "*")
        for role, text in st.session_state.dialogue:
            if role == "kritikos":
                with st.chat_message("assistant", avatar=OWL):
                    st.write(text)
            else:
                with st.chat_message("user"):
                    st.write(text)

        reply = st.chat_input("Reply to KRITIKOS...")
        if reply:
            st.session_state.dialogue.append(("user", reply))
            with st.spinner("KRITIKOS is thinking..."):
                nxt = ask_coach(st.session_state.dialogue)
            st.session_state.dialogue.append(("kritikos", nxt))
            st.rerun()

st.markdown("<div class='data-note'>KRITIKOS does not store your chats or personal data. "
            "Your messages are sent to the AI model only to generate replies.</div>",
            unsafe_allow_html=True)
