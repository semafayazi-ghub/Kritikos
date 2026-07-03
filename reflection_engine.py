"""
reflection_engine.py
=================================================================
The "brain" behind KRITIKOS V5:

* get_ai_answer()         -> calls the hosted LLM (Groq) for the AI answer.
* score_answer_signal()   -> the INTERNAL model signal (academic vs weak)
                             that decides which reflection cues appear.
                             This value is never shown to the user.
* get_kritikos_reflection -> produces KRITIKOS's reflective reply for a
                             cue. [V5-FIX #4] enforces ONE focused
                             question per prompt.

The Groq calls degrade gracefully: if no API key is configured (e.g. in a
local demo or in CI), the functions fall back to deterministic stub text
so the UI still runs and can be screenshotted.

Author: Sema (Samaneh) Fayazi — Master Data-Driven Design (D01), HU
"""

from __future__ import annotations

import os
import re

# The classifier proof-of-concept lives next door; its pipeline is reused
# so the app and the appendix share exactly one model definition.
from kritikos_prompt_classifier import (
    FEATURE_COLUMNS,
    build_dataset,
    build_model,
    prediction_to_prompt,  # noqa: F401  (kept for parity with the appendix)
)
import pandas as pd

# ----------------------------------------------------------------------
# Reflection cues per internal signal.
# "weak"     -> answer looks non-academic / weakly supported
# "academic" -> answer looks academic / well-supported
# [V5-FIX #4] three short, single-idea cues; the heavy lifting (one
# question) happens in get_kritikos_reflection, not in the label.
# ----------------------------------------------------------------------
REFLECTION_CUES = {
    "weak": ["Check the source", "This may be incomplete", "Compare with another source"],
    "academic": ["Verify the references", "Compare with the literature", "What's the counter-view?"],
}

# ----------------------------------------------------------------------
# Internal model signal (NEVER shown to the user — [V5-FIX #7])
# ----------------------------------------------------------------------
_MODEL = None


def _model():
    """Lazily train the interpretable classifier once per process."""
    global _MODEL
    if _MODEL is None:
        df = build_dataset()
        _MODEL = build_model("logreg").fit(df[FEATURE_COLUMNS], df["label"])
    return _MODEL


def _heuristic_metadata(answer: str) -> dict:
    """Cheap metadata flags extracted from the answer text."""
    lower = answer.lower()
    cite = int(bool(re.search(r"\((19|20)\d{2}\)|doi|et al\.|journal", lower)))
    promo = int(bool(re.search(r"\b(best ever|buy now|miracle|guaranteed|amazing)\b", lower)))
    peer = int("peer-review" in lower or "peer reviewed" in lower)
    return {
        "source_type": "report" if cite else "blog",
        "domain_type": "academic" if cite else "commercial",
        "peer_reviewed": peer,
        "contains_citations": cite,
        "uses_promotional_language": promo,
    }


def score_answer_signal(question: str, answer: str) -> str:
    """
    Return 'academic' or 'weak' for an AI answer. INTERNAL USE ONLY:
    it picks which reflection cues to surface. It is logged by the app,
    never rendered as a verdict.
    """
    meta = _heuristic_metadata(answer)
    row = pd.DataFrame([{"text": f"{question} {answer[:400]}", **meta}])
    pred = int(_model().predict(row[FEATURE_COLUMNS])[0])
    return "academic" if pred == 1 else "weak"


# ----------------------------------------------------------------------
# LLM access (Groq) with a graceful offline fallback
# ----------------------------------------------------------------------
def _groq_chat(system: str, user: str, max_tokens: int = 600) -> str | None:
    """Call Groq if configured; return None to signal 'use fallback'."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq  # imported lazily so the app runs without it
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.environ.get("KRITIKOS_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception:  # noqa: BLE001  network/SDK issues -> graceful fallback
        return None


def get_ai_answer(question: str) -> str:
    """Get a normal AI answer to the user's question."""
    system = ("You are a helpful general assistant. Answer the user's "
              "question clearly and concisely.")
    out = _groq_chat(system, question, max_tokens=700)
    if out:
        return out
    # Offline fallback (keeps the prototype demoable without a key).
    return (
        f"Here is a general overview in response to: “{question}”. "
        "This is a demo answer generated without a live model key. In the "
        "deployed app this text comes from the hosted LLM. Notice that it "
        "does not cite a specific author, date, or publication — exactly the "
        "kind of answer KRITIKOS invites you to look at more carefully."
    )


# ----------------------------------------------------------------------
# Second internal signal — claim-type detection (V5.1)
# ----------------------------------------------------------------------
# Besides the source-quality signal, KRITIKOS looks at WHAT KIND of claim
# the answer makes, so the single reflective question can aim at it.
# Same rule as the source signal: logged, NEVER shown as a verdict.
# Marker lists are deliberately small, generic English discourse markers
# (hedging / causal / presupposition / normative), kept visible here so
# the criteria stay inspectable and adjustable.

CLAIM_MARKERS = {
    "causal": ["because", "leads to", "results in", "causes", "therefore",
               "due to", "consequently", "which means", "so that"],
    "assumption": ["obviously", "of course", "clearly", "naturally",
                   "everyone knows", "always", "never", "simply"],
    "value": ["should", "must", "the best", "ideally", "it is important",
              "need to", "ought to", "better to"],
    "surface": ["research suggests", "studies show", "often", "generally",
                "typically", "tends to", "commonly", "in many cases"],
}

# One targeted question per claim type — exactly ONE '?' each, so the
# single-question guarantee below is preserved.
CLAIM_QUESTIONS = {
    "surface": "The answer speaks in general terms — what evidence or "
               "mechanism would make this hold in your specific case?",
    "causal": "The answer draws a cause-and-effect link — what else could "
              "explain that connection?",
    "assumption": "What is this answer taking for granted — and would it "
                  "still stand if that assumption were dropped?",
    "value": "This answer contains a judgement about what is best — best "
             "for whom, under which priorities?",
}

# Cues whose question is aimed at the claim rather than at the source.
CLAIM_AWARE_CUES = {"This may be incomplete", "What's the counter-view?"}


def detect_claim_type(answer: str) -> str:
    """Classify the dominant claim type of an answer. INTERNAL USE ONLY."""
    lower = answer.lower()
    scores = {k: sum(lower.count(m) for m in v) for k, v in CLAIM_MARKERS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "surface"


# ----------------------------------------------------------------------
# KRITIKOS reflection — [V5-FIX #4] ONE focused question per prompt
# ----------------------------------------------------------------------
_ONE_QUESTION_RULE = (
    "You are KRITIKOS, a reflective study buddy. You NEVER give the answer "
    "and you NEVER say whether the AI is right or wrong. Your job is to make "
    "the student look once more, on their own terms.\n"
    "STRICT RULES:\n"
    "1. Ask exactly ONE short question. Never stack multiple questions.\n"
    "2. Keep it under 40 words.\n"
    "3. Be warm and non-judgemental — a hint, not a quiz.\n"
)

# Deterministic fallbacks: each is a single, focused question. [V5-FIX #4]
_FALLBACK = {
    "Check the source": "Where could you confirm this — is there an original "
        "author, date, or publication you could trace it back to?",
    "This may be incomplete": "What's one perspective or context this answer "
        "might be leaving out?",
    "Compare with another source": "Which independent source could you check "
        "to see if it says something different?",
    "Verify the references": "Could you open one cited reference and check it "
        "actually says what the answer claims?",
    "Compare with the literature": "Does this match what you've read elsewhere "
        "— or is there a study that disagrees?",
    "What's the counter-view?": "What would someone who disagrees with this "
        "answer point to first?",
}


def get_kritikos_reflection(cue: str, question: str, answer: str) -> str:
    """Return KRITIKOS's reflective reply for a cue — one question only.

    V5.1: the claim-type signal aims the question at the KIND of claim the
    answer makes. The signal itself is internal (never rendered).
    """
    claim = detect_claim_type(answer)  # internal only
    user = (
        f"The student asked: “{question}”.\n"
        f"The AI answered (excerpt): “{answer[:500]}”.\n"
        f"The student tapped the reflection cue: “{cue}”.\n"
        f"(Internal hint, never mention it: the answer reads as a {claim} claim; "
        "aim the question at that.)\n"
        "Respond with ONE short reflective question in that direction."
    )
    out = _groq_chat(_ONE_QUESTION_RULE, user, max_tokens=120)
    if out:
        return _enforce_single_question(out)
    if cue in CLAIM_AWARE_CUES:                     # claim-aware offline fallback
        return CLAIM_QUESTIONS[claim]
    return _FALLBACK.get(cue, "What makes you confident in this answer?")


def _enforce_single_question(text: str) -> str:
    """
    Safety net for [V5-FIX #4]: even if the model returns several
    questions, keep only the first. This guarantees the UI never shows a
    multi-question prompt (the exact issue E1 flagged in V4).
    """
    text = text.strip()
    # split on '?' and keep the first question
    if "?" in text:
        first = text.split("?")[0].strip()
        return first + "?"
    return text
