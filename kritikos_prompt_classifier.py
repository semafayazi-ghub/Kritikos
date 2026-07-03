"""
kritikos_prompt_classifier.py
=================================================================
KRITIKOS — context-aware reflection-prompt classifier (proof of concept)

Goal
----
Given a short description of the *source* an AI answer leans on (title +
snippet + a few metadata flags), estimate whether that source looks
"academic / well-supported" or "non-academic / weak". That estimate is
used ONLY to decide which reflection cue KRITIKOS surfaces. It is never
shown to the student as a verdict.

Design stance
-------------
* Interpretable baselines (Logistic Regression, Linear SVC) on purpose:
  KRITIKOS must be able to justify why a cue appeared, so a black box is
  the wrong tool here.
* The label is intentionally binary and the data is synthetic. This is a
  feasibility probe ("can a small signal drive prompt selection?"), not a
  production classifier. High accuracy reflects the simplicity of the
  synthetic task, not real-world robustness.

Author: Sema (Samaneh) Fayazi — Master Data-Driven Design (D01), HU
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Synthetic training data
# ----------------------------------------------------------------------
# Each row is a stylised "source description". label = 1 -> academic /
# well-supported; label = 0 -> non-academic / weak. The generator below
# expands a handful of templates with light variation so the model has
# ~120 examples to learn separable text + metadata patterns.

ACADEMIC_TEMPLATES = [
    ("A peer-reviewed study published in {journal} ({year}) examining {topic}.",
     "journal", "academic", 1, 1, 0),
    ("Systematic review in {journal} synthesising evidence on {topic}, with full citations.",
     "journal", "academic", 1, 1, 0),
    ("Randomised controlled trial on {topic}, methods and limitations reported, {journal}.",
     "journal", "academic", 1, 1, 0),
    ("University research report on {topic}, references included, {year}.",
     "report", "edu", 1, 1, 0),
    ("Meta-analysis of {topic} across multiple cohorts, published {journal}.",
     "journal", "academic", 1, 1, 0),
]

WEAK_TEMPLATES = [
    ("A blog post claiming {topic} is the 'best ever', no sources cited.",
     "blog", "commercial", 0, 0, 1),
    ("Sponsored article promoting a product related to {topic}. Buy now!",
     "news", "commercial", 0, 0, 1),
    ("Forum comment giving a personal opinion about {topic}.",
     "forum", "social", 0, 0, 0),
    ("Marketing page with bold claims about {topic} and limited-time offers.",
     "blog", "commercial", 0, 0, 1),
    ("Anonymous web page about {topic}, no author, no date, no references.",
     "blog", "unknown", 0, 0, 0),
]

JOURNALS = ["Nature", "The Lancet", "JAMA", "PLOS ONE", "Cognitive Science",
            "Human Factors", "Societies"]
TOPICS = ["social media and teen mental health", "remote work productivity",
          "the Mediterranean diet", "AI in education", "sleep and memory",
          "urban air quality", "online learning outcomes", "vaccine uptake"]
YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]


def build_dataset(n_per_template: int = 12, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Expand templates into a small synthetic dataset (~120 rows)."""
    rng = np.random.default_rng(seed)
    rows = []
    for templates in (ACADEMIC_TEMPLATES, WEAK_TEMPLATES):
        for tmpl, src, dom, peer, cite, promo in templates:
            for _ in range(n_per_template):
                text = tmpl.format(
                    journal=rng.choice(JOURNALS),
                    topic=rng.choice(TOPICS),
                    year=rng.choice(YEARS),
                )
                label = 1 if templates is ACADEMIC_TEMPLATES else 0
                rows.append({
                    "text": text,
                    "source_type": src,
                    "domain_type": dom,
                    "peer_reviewed": peer,
                    "contains_citations": cite,
                    "uses_promotional_language": promo,
                    "label": label,
                })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 2. Feature pipeline  (text + metadata in one ColumnTransformer)
# ----------------------------------------------------------------------
FEATURE_COLUMNS = ["text", "source_type", "domain_type",
                   "peer_reviewed", "contains_citations",
                   "uses_promotional_language"]


def build_preprocessor() -> ColumnTransformer:
    """Combine TF-IDF text features with one-hot + passthrough metadata."""
    return ColumnTransformer(transformers=[
        ("text", TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
                ["source_type", "domain_type"]),
        ("num", "passthrough", ["peer_reviewed",
                "contains_citations", "uses_promotional_language"]),
    ])


def build_model(kind: str = "logreg") -> Pipeline:
    """Return a full pipeline: preprocessing + an interpretable classifier."""
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif kind == "linsvc":
        clf = LinearSVC(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown model kind: {kind!r}")
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


# ----------------------------------------------------------------------
# 3. Translate a model prediction into a reflection prompt
# ----------------------------------------------------------------------
# IMPORTANT: this mapping is the only place the model touches the UI.
# It chooses WHICH cue to surface — it never produces a verdict string.

def prediction_to_prompt(pred: int) -> str:
    if pred == 1:        # looks academic / structured
        return "Verify references | Compare with literature"
    return "Check source | This may be incomplete"   # looks non-academic / weak


# ----------------------------------------------------------------------
# 4. Train, cross-validate, and demo
# ----------------------------------------------------------------------
def main() -> None:
    df = build_dataset()
    X, y = df[FEATURE_COLUMNS], df["label"]
    print(f"Dataset: {len(df)} synthetic examples "
          f"({int(y.sum())} academic / {int((1 - y).sum())} weak)\n")

    for name, kind in [("Logistic Regression", "logreg"),
                       ("Linear SVC", "linsvc")]:
        model = build_model(kind)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"{name:22s}  5-fold CV accuracy = {scores.mean():.2f} "
              f"(+/- {scores.std():.2f})")

    # Fit the LogReg on all data and show the cue it would surface.
    model = build_model("logreg").fit(X, y)
    examples = pd.DataFrame([
        {"text": "Peer-reviewed cohort study in The Lancet on sleep and memory.",
         "source_type": "journal", "domain_type": "academic",
         "peer_reviewed": 1, "contains_citations": 1,
         "uses_promotional_language": 0},
        {"text": "Sponsored blog: this miracle method fixes everything, buy today!",
         "source_type": "blog", "domain_type": "commercial",
         "peer_reviewed": 0, "contains_citations": 0,
         "uses_promotional_language": 1},
    ])
    print("\nDemo — which cue KRITIKOS would surface (internal use only):")
    for text, pred in zip(examples["text"], model.predict(examples)):
        print(f"  [{pred}] {prediction_to_prompt(pred):44s} <- {text[:48]}...")


if __name__ == "__main__":
    main()
