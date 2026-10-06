"""
complaint_pipeline.py

Standalone inference module for the Smart Complaint Management System.

This is NOT a rewrite of the ML model -- it loads the artifacts already
trained and saved by complaint_pipeline.ipynb (Sections 7-9e) and reproduces
the exact same feature computation and gate logic used there, so a separate
app (Streamlit or otherwise) can call predict_single_complaint() / 
analyze_complaint() without re-running training.

REQUIRED before this module will work:
  1. Run complaint_pipeline.ipynb top to bottom at least once, so that
     ./processed/best_transformer_model.pkl and
     ./processed/category_classifier.pkl exist.
  2. Have complaints.csv-derived config match (HAZARD_LEXICON, engineered
     feature list) -- these are duplicated here rather than imported from the
     notebook, since notebooks aren't directly importable modules. If you
     ever change the hazard lexicon in the notebook, update it here too.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = os.environ.get("COMPLAINT_MODEL_DIR", "processed")
CATEGORY_COL = "category"

LOW_STAKES_THRESHOLD = 0.35
STRICT_THRESHOLD = 0.60
HAZARD_OVERRIDE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Hazard lexicon -- kept in sync with Section 4 of the notebook
# ---------------------------------------------------------------------------
HAZARD_LEXICON = {
    "fire": {
        "keywords": ["fire", "smoke", "burning", "flames", "smoldering"],
        "support": ["evacuate", "alarm", "extinguisher", "panel", "electrical", "sparks", "wire"],
    },
    "gas": {
        "keywords": ["gas leak", "gas smell", "natural gas", "smells like gas"],
        "support": ["evacuate", "ventilation", "odor", "basement", "lab"],
    },
    "electrical": {
        "keywords": ["sparking", "exposed wiring", "electrical fire", "shock", "sparks"],
        "support": ["outlet", "panel", "wire", "burning smell", "smoke"],
    },
    "structural": {
        "keywords": ["ceiling collapse", "structural", "collapsed", "crack in foundation", "caved in"],
        "support": ["unsafe", "evacuate", "inspection", "falling"],
    },
    "medical": {
        "keywords": ["collapsed", "unconscious", "overdose", "seizure", "not breathing", "medical emergency"],
        "support": ["ambulance", "help", "immediately", "unresponsive"],
    },
    "assault": {
        "keywords": [
            "assault", "attacked", "physically", "weapon", "threatening", "threatened me",
            "knife", "gun", "armed", "stalking", "stalker",
            "following me", "chasing me", "harassing me", "harassed me",
        ],
        "support": ["security", "police", "unsafe", "afraid", "violence"],
    },
    "entrapment": {
        # Added after a real complaint ("lift struck between floors, people
        # inside, need backup immediately") scored 0.0 hazard because none of
        # the original 6 categories covered someone physically trapped
        # somewhere (stuck elevator, locked room, etc.). Tested against
        # benign "stuck" usages (stuck printer, stuck registration, feeling
        # stuck about a decision) to confirm those still correctly score 0.
        "keywords": [
        "stuck between", "struck between", "stuck in the elevator", "stuck in elevator",
        "elevator stuck", "lift stuck", "stuck lift", "stuck elevator",
        "trapped inside", "stuck inside", "people inside", "people in it",
        "locked inside", "locked in",
    ],
        "support": ["help", "immediately", "backup", "rescue", "hours", "floor", "stuck for"],
    },
}

UNLIKELY_CATEGORY_FOR_HAZARD = {
    "fire": ["IT Support", "Financial Aid", "Academic"],
    "gas": ["IT Support", "Financial Aid", "Academic", "Campus Life"],
    "electrical": ["Financial Aid", "Academic"],
    "structural": ["Financial Aid", "IT Support"],
    "medical": ["Financial Aid", "IT Support"],
    "assault": ["Financial Aid", "IT Support", "Facilities"],
    "entrapment": ["Financial Aid", "Academic"],
}


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text


def _score_hazard(text_lower: str, hazard_type: str, category: str) -> float:
    lex = HAZARD_LEXICON[hazard_type]
    if not any(kw in text_lower for kw in lex["keywords"]):
        return 0.0
    support_hits = sum(1 for sw in lex["support"] if sw in text_lower)
    base = 0.5 + min(support_hits, 3) * 0.15
    if category in UNLIKELY_CATEGORY_FOR_HAZARD.get(hazard_type, []):
        base *= 0.5
    return round(min(base, 1.0), 3)


def add_hazard_features(df: pd.DataFrame, text_col: str = "clean_text",
                         category_col: str = "category") -> pd.DataFrame:
    df = df.copy()
    text_lower = df[text_col].astype(str).str.lower()
    for hazard_type in HAZARD_LEXICON:
        df[f"hazard_{hazard_type}"] = [
            _score_hazard(t, hazard_type, cat) for t, cat in zip(text_lower, df[category_col])
        ]
    hazard_cols = [f"hazard_{h}" for h in HAZARD_LEXICON]
    df["hazard_max_score"] = df[hazard_cols].max(axis=1)
    df["hazard_count"] = (df[hazard_cols] > 0).sum(axis=1)
    return df


def add_sentiment_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    from textblob import TextBlob
    df = df.copy()
    df["sentiment_polarity"] = df[text_col].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity)
    df["sentiment_subjectivity"] = df[text_col].astype(str).apply(lambda t: TextBlob(t).sentiment.subjectivity)
    return df


def build_category_dummies(df: pd.DataFrame, all_categories: list) -> pd.DataFrame:
    dummies = pd.get_dummies(df[CATEGORY_COL], prefix="cat")
    for cat in all_categories:
        col = f"cat_{cat}"
        if col not in dummies.columns:
            dummies[col] = 0
    return dummies[[f"cat_{c}" for c in all_categories]].astype(int)


# ---------------------------------------------------------------------------
# Load trained artifacts once, at import time (this is the expensive part --
# a Streamlit app should only pay this cost once per session, via st.cache_resource)
# ---------------------------------------------------------------------------
class PipelineNotReadyError(RuntimeError):
    """Raised when the notebook hasn't been run yet / models aren't saved."""
    pass


def _load_artifacts():
    priority_path = os.path.join(PROCESSED_DIR, "best_transformer_model.pkl")
    category_path = os.path.join(PROCESSED_DIR, "category_classifier.pkl")

    if not os.path.exists(priority_path) or not os.path.exists(category_path):
        raise PipelineNotReadyError(
            f"Could not find trained model files in '{PROCESSED_DIR}/'. "
            f"Run complaint_pipeline.ipynb top to bottom first (Section 9e must "
            f"complete) so that best_transformer_model.pkl and "
            f"category_classifier.pkl are saved there."
        )

    with open(priority_path, "rb") as f:
        priority_bundle = pickle.load(f)
    with open(category_path, "rb") as f:
        category_bundle = pickle.load(f)

    encoder = SentenceTransformer(priority_bundle["embedding_model_name"])

    return {
        "priority_model": priority_bundle["model"],
        "label_encoder": priority_bundle.get("label_encoder"),
        "engineered_cols": priority_bundle["engineered_cols"],
        "all_categories": priority_bundle["categories"],
        "category_model": category_bundle["model"],
        "encoder": encoder,
    }


_artifacts = None


def _get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_complaint(complaint_text: str) -> dict:
    """
    Full analysis of a single complaint: predicted category, priority,
    per-class probabilities, hazard breakdown, and gate decision.
    """
    art = _get_artifacts()
    encoder = art["encoder"]
    priority_model = art["priority_model"]
    label_encoder = art["label_encoder"]
    engineered_cols = art["engineered_cols"]
    all_categories = art["all_categories"]
    category_model = art["category_model"]

    cleaned = clean_text(complaint_text)
    embedding = encoder.encode([cleaned])

    predicted_category = category_model.predict(embedding)[0]

    single_df = pd.DataFrame({"clean_text": [cleaned], CATEGORY_COL: [predicted_category]})
    single_df = add_hazard_features(single_df)
    single_df = add_sentiment_features(single_df)
    single_cat = build_category_dummies(single_df, all_categories)

    numeric_features = single_df[engineered_cols].to_numpy(dtype=np.float32)
    category_features = single_cat.to_numpy(dtype=np.float32)
    X_sample = np.hstack([embedding.astype(np.float32), numeric_features, category_features])

    proba_row = priority_model.predict_proba(X_sample)[0]
    if label_encoder is not None:
        class_labels = list(label_encoder.classes_)
    else:
        class_labels = list(priority_model.classes_)
    probabilities = {label: float(p) for label, p in zip(class_labels, proba_row)}

    prediction = max(probabilities, key=probabilities.get)
    confidence = probabilities[prediction]
    hazard_score = float(single_df["hazard_max_score"].iloc[0])
    hazard_breakdown = {h: float(single_df[f"hazard_{h}"].iloc[0]) for h in HAZARD_LEXICON}

    is_low_stakes = (prediction in ("Low", "Medium")) and (hazard_score == 0)
    effective_threshold = LOW_STAKES_THRESHOLD if is_low_stakes else STRICT_THRESHOLD

    if hazard_score >= HAZARD_OVERRIDE_THRESHOLD:
        gated_decision, gate_reason = "human_review", "hazard_override"
    elif confidence < effective_threshold:
        gated_decision, gate_reason = "human_review", "low_confidence"
    else:
        gated_decision, gate_reason = "auto_assign", ""

    return {
        "complaint_text": complaint_text,
        "cleaned_text": cleaned,
        "predicted_category": predicted_category,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "hazard_score": hazard_score,
        "hazard_breakdown": hazard_breakdown,
        "sentiment_polarity": float(single_df["sentiment_polarity"].iloc[0]),
        "sentiment_subjectivity": float(single_df["sentiment_subjectivity"].iloc[0]),
        "threshold_used": effective_threshold,
        "tier": "low_stakes" if is_low_stakes else "strict",
        "gated_decision": gated_decision,
        "gate_reason": gate_reason,
    }


def predict_single_complaint(complaint_text: str):
    """
    Matches the exact signature used in the notebook:
    returns (prediction, confidence, predicted_category, gated_decision)
    """
    result = analyze_complaint(complaint_text)
    return (
        result["prediction"],
        result["confidence"],
        result["predicted_category"],
        result["gated_decision"],
    )


if __name__ == "__main__":
    # Quick self-test
    r = analyze_complaint(
        "Electrical panel near room 204 is sparking and smoke is coming out, please help immediately."
    )
    print(r)
