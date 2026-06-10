"""
app.py  —  KRITIKOS (redesigned, live version)
==============================================
A working prototype of the redesigned KRITIKOS reflective assistant, connected
to a real AI model (via Groq) so participants can ask their own questions.

WHAT IT DOES
  - The user types their OWN question and gets a real AI answer (Groq).
  - Under the answer, KRITIKOS shows lightweight reflection prompts that are
    OPTIONAL and CONTEXT-AWARE: which prompts appear is decided by a small
    machine-learning classifier (see reflection_engine.py) that reads the
    answer + its signals (citations? sources? hedging language?).
  - The user can ignore the prompts or tap one to reflect further.

RESPONSIBLE-AI RULE
  The model output and the classifier signal are never shown as a verdict.
  KRITIKOS supports the user's judgement; it does not replace it.

API KEY (kept secret, never in the code)
  Locally:  create a file  .streamlit/secrets.toml  with:
                GROQ_API_KEY = "your_key_here"
  Online (Streamlit Cloud):  add the same secret in the app's Settings.

RUN
  pip install -r requirements.txt
  streamlit run app.py
"""

import streamlit as st
from groq import Groq
from reflection_engine import train_model, classify_text, select_prompts

# ---------------------------------------------------------------------------
# Page setup + light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="KRITIKOS", page_icon="🦉", layout="centered")
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

# ---------------------------------------------------------------------------
# Train the small classifier once (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return train_model()       # returns (model, metrics)

model, metrics = get_model()

# ---------------------------------------------------------------------------
# Connect to Groq using the secret key
# ---------------------------------------------------------------------------
def get_client():
    key = st.secrets.get("GROQ_API_KEY", None)
    if not key:
        return None
    return Groq(api_key=key)

client = get_client()

# short reflective nudges shown when a prompt is tapped (questions, not answers)
PROMPT_HELP = {
    "Check the source": "Where does this claim come from? Is a source named, dated and trustworthy?",
    "This may be incomplete": "This answer may leave out context or other viewpoints. What might be missing?",
    "Compare with another source": "Try checking this against a second, independent source before relying on it.",
    "Verify references": "If references are mentioned, open them and confirm they say what the answer claims.",
    "Compare with literature": "How does this fit with other peer-reviewed work on the topic?",
    "Based on academic sources": "This sounds academic — still worth checking the method and limitations.",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 🦉 KRITIKOS")
st.caption("A second look at every AI answer — reflection that supports your work, never interrupts it.")

with st.sidebar:
    st.markdown("### Settings")
    show_reflection = st.toggle("Show reflection prompts", value=True,
                                help="You decide. Turn this off to use AI without reflection cues.")
    st.markdown("---")
    st.markdown("### About this prototype")
    st.write("You ask a real question and get a real AI answer. KRITIKOS then shows "
             "optional reflection prompts. A small machine-learning model decides which "
             "prompts to show — it never decides for you.")
    st.markdown(f"**Reflection model:** Logistic Regression  \n"
                f"**5-fold CV accuracy:** {metrics['cv_accuracy']}  \n"
                f"**Trained on:** {metrics['n_samples']} synthetic examples")
    st.caption("Synthetic training data → proof of concept, not a finished system.")

# ---------------------------------------------------------------------------
# Main interaction: user asks their own question
# ---------------------------------------------------------------------------
if client is None:
    st.error("No AI key found. Add GROQ_API_KEY in Streamlit secrets (or .streamlit/secrets.toml) "
             "to enable live answers.")
    st.stop()

# keep the last answer in memory so tapping a prompt doesn't re-ask the AI
if "answer" not in st.session_state:
    st.session_state.answer = None
    st.session_state.tapped = None

question = st.text_input("Ask the AI a question:",
                         placeholder="e.g. What does research say about social media and teen mental health?")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": question}],
                max_tokens=400,
            )
            st.session_state.answer = resp.choices[0].message.content
            st.session_state.tapped = None
        except Exception as e:
            st.error(f"Could not get an answer: {e}")
            st.session_state.answer = None

# show the answer + the embedded reflection layer
if st.session_state.answer:
    st.markdown(f"<div class='ai-answer'>{st.session_state.answer}</div>", unsafe_allow_html=True)

    if show_reflection:
        # the classifier reads the AI answer text and picks the prompts
        prediction = classify_text(model, st.session_state.answer)
        prompts = select_prompts(prediction)

        st.markdown("<div class='reflect-label'>Reflect on this answer (optional):</div>",
                    unsafe_allow_html=True)
        cols = st.columns(len(prompts))
        for col, prompt in zip(cols, prompts):
            if col.button(prompt, key=prompt):
                st.session_state.tapped = prompt

        if st.session_state.tapped in PROMPT_HELP:
            st.info("💡 " + PROMPT_HELP[st.session_state.tapped])

        label = "academic / well-supported" if prediction == 1 else "non-academic / weaker"
        st.caption(f"(internal signal: this answer looks **{label}** — used only to choose "
                   "which prompts appear, never shown as a verdict)")
    else:
        st.caption("Reflection prompts are off. Turn them on in the settings whenever you want them.")

st.markdown("<div class='data-note'>KRITIKOS does not store your chats or personal data. "
            "Your question is sent to the AI model only to generate an answer.</div>",
            unsafe_allow_html=True)
