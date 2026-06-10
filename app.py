"""
app.py  —  KRITIKOS (redesigned) prototype
===========================================
A working prototype of the redesigned KRITIKOS reflective assistant.

WHAT IT SHOWS
  - An AI answer (pre-written, offline, so the demo is reliable).
  - Lightweight reflection prompts that appear UNDER the answer (embedded),
    are OPTIONAL, and are CONTEXT-AWARE: which prompts appear is decided by a
    small machine-learning classifier (see reflection_engine.py).
  - The student can ignore the prompts, or tap one to reflect further.

HOW TO RUN
  1. pip install streamlit scikit-learn pandas
  2. streamlit run app.py
  3. A browser tab opens at http://localhost:8501

This file is the INTERFACE layer. The DATA-SCIENCE layer lives in
reflection_engine.py. Keeping them separate makes the relationship between
"what the model can do" and "what the interface shows" easy to explain.
"""

import streamlit as st
from reflection_engine import train_model, reflect

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="KRITIKOS", page_icon="🦉", layout="centered")

# a little styling to make the prompts look like lightweight cues
st.markdown("""
<style>
    .ai-answer {background:#EAF1F8; border-left:4px solid #2E6FB0;
                padding:14px 16px; border-radius:8px; margin-bottom:8px; color:#1F3A5F;}
    .source-box {background:#F7F9FB; border:1px solid #D7DEE7;
                 padding:10px 14px; border-radius:8px; font-size:0.9em; color:#444;}
    .reflect-label {color:#3F8E5C; font-size:0.85em; margin-top:6px;}
    .data-note {color:#888; font-size:0.8em; border-top:1px solid #eee;
                padding-top:8px; margin-top:18px;}
    .stButton>button {border-radius:18px; border:1px solid #2E6FB0; color:#2E6FB0;
                      background:white; font-size:0.85em; padding:2px 12px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Train the model once and cache it (so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return train_model()  # returns (model, metrics)

model, metrics = get_model()

# ---------------------------------------------------------------------------
# Pre-written demo answers (offline, reliable for the video).
# Each has the AI text + the "source" the answer leans on. The source
# metadata is what the classifier reads to pick the right prompts.
# ---------------------------------------------------------------------------
DEMO = {
    "What does research say about social media and teen mental health?": {
        "answer": ("Research suggests a link between heavy social media use and lower "
                   "wellbeing among teenagers, especially girls, though the effect size "
                   "is debated and many studies are correlational rather than causal."),
        "source": {
            "title": "Longitudinal study on adolescent social media use and wellbeing",
            "snippet": "Peer-reviewed; reports sample, method and limitations.",
            "source_type": "journal_article", "domain_type": "doi.org",
            "peer_reviewed": 1, "contains_citations": 1, "uses_promotional_language": 0,
        },
    },
    "How can I quickly improve my essay with AI?": {
        "answer": ("You can paste your essay into an AI tool and ask it to rewrite weak "
                   "paragraphs, fix grammar, and make the argument stronger in seconds."),
        "source": {
            "title": "10 AI hacks to finish your essay fast (sponsored)",
            "snippet": "No sources cited; promotional language; blog post.",
            "source_type": "ad", "domain_type": ".com",
            "peer_reviewed": 0, "contains_citations": 0, "uses_promotional_language": 1,
        },
    },
}

# short explanation shown when a prompt is tapped (reflection, not answers)
PROMPT_HELP = {
    "Check the source": "Where does this claim come from? Is the source named, dated and trustworthy?",
    "This may be incomplete": "This answer may leave out context or other viewpoints. What might be missing?",
    "Compare with another source": "Try checking this against a second, independent source before you rely on it.",
    "Verify references": "Open the cited references and confirm they actually say what the answer claims.",
    "Compare with literature": "How does this fit with other peer-reviewed work on the topic?",
    "Based on academic sources": "This leans on academic work — still worth checking the method and limitations.",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 🦉 KRITIKOS")
st.caption("A second look at every AI answer — reflection that supports your work, never interrupts it.")

# sidebar: autonomy + transparency controls
with st.sidebar:
    st.markdown("### Settings")
    show_reflection = st.toggle("Show reflection prompts", value=True,
                                help="You decide. Turn this off to use the tool as a one-off.")
    st.markdown("---")
    st.markdown("### About this prototype")
    st.write("Reflection prompts are chosen by a small machine-learning model "
             "that estimates how academic a source looks. The model never decides "
             "for you — it only picks which optional prompts to show.")
    st.markdown(f"**Model:** Logistic Regression  \n"
                f"**Test accuracy:** {metrics['test_accuracy']}  \n"
                f"**5-fold CV:** {metrics['cv_accuracy']}  \n"
                f"**Data:** {metrics['n_samples']} synthetic examples")
    st.caption("Synthetic data → proof of concept, not a finished system.")

# ---------------------------------------------------------------------------
# Main interaction
# ---------------------------------------------------------------------------
question = st.selectbox("Pick a question to ask the AI:", list(DEMO.keys()))
item = DEMO[question]

st.markdown(f"**You asked:** {question}")
st.markdown(f"<div class='ai-answer'>{item['answer']}</div>", unsafe_allow_html=True)

with st.expander("Source this answer leans on"):
    s = item["source"]
    st.markdown(f"<div class='source-box'><b>{s['title']}</b><br>{s['snippet']}<br>"
                f"<i>type: {s['source_type']} · domain: {s['domain_type']} · "
                f"peer-reviewed: {'yes' if s['peer_reviewed'] else 'no'}</i></div>",
                unsafe_allow_html=True)

# ---- the embedded, context-aware reflection layer ----
if show_reflection:
    result = reflect(model, item["source"])          # <-- model picks the prompts
    st.markdown("<div class='reflect-label'>Reflect on this answer "
                "(optional):</div>", unsafe_allow_html=True)
    cols = st.columns(len(result["prompts"]))
    for col, prompt in zip(cols, result["prompts"]):
        if col.button(prompt, key=prompt):
            st.session_state["tapped"] = prompt
    if "tapped" in st.session_state and st.session_state["tapped"] in PROMPT_HELP:
        st.info("💡 " + PROMPT_HELP[st.session_state["tapped"]])
    st.caption(f"(internal signal: this source looks **{result['label']}** — "
               "used only to choose which prompts appear)")
else:
    st.caption("Reflection prompts are off. Turn them on in the settings whenever you want them.")

st.markdown("<div class='data-note'>KRITIKOS does not store your chats or personal data. "
            "Everything runs locally in this prototype.</div>", unsafe_allow_html=True)
