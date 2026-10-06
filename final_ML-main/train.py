"""
train.py -- one-shot training script for the Smart Complaint Management System

Run this ONCE, locally, with internet access (it needs to download the
Sentence-BERT embedding model from Hugging Face the first time). It reproduces
the exact model-saving steps from complaint_pipeline.ipynb (Sections 7, 9, 9e)
and writes the two files complaint_pipeline.py / app.py actually need:

    processed/best_transformer_model.pkl
    processed/category_classifier.pkl

Usage:
    python train.py
    python train.py --input complaints_updated.csv --output processed

Requires: pandas, numpy, scikit-learn, scipy, textblob, sentence-transformers,
xgboost (see requirements.txt). Run `python -m textblob.download_corpora`
once if TextBlob complains about missing corpora.
"""

import argparse
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, recall_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

TEXT_COL = "complaint_text"
CATEGORY_COL = "category"
LABEL_COL = "priority_level"
RANDOM_STATE = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Text cleaning -- identical to Section 3 of the notebook
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text


# ---------------------------------------------------------------------------
# Hazard-entity scoring -- identical to Section 4 of the notebook
# (includes the expanded assault keyword list: knife/gun/armed/stalking/etc.)
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
        # somewhere. Verified against benign "stuck" usages (stuck printer,
        # stuck registration, feeling stuck about a decision) -- those still
        # correctly score 0.
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


# ---------------------------------------------------------------------------
# Sentiment features -- identical to Section 5 of the notebook
# ---------------------------------------------------------------------------
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


def evaluate(model, X_eval, y_eval, name):
    preds = model.predict(X_eval)
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_eval, preds, digits=3))
    em_recall = recall_score(y_eval, preds, labels=["Emergency"], average="micro")
    print(f"Emergency recall: {em_recall:.3f}")
    return em_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="complaints_updated.csv", help="Path to the complaints CSV")
    parser.add_argument("--output", default="processed", help="Folder to save trained model files into")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Load + clean + split (Sections 2, 3, 7) ---
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df["clean_text"] = df[TEXT_COL].apply(clean_text)
    df = df.drop_duplicates(subset=TEXT_COL).reset_index(drop=True)
    print(f"Loaded {len(df)} rows after de-duplication.")

    all_categories = sorted(df[CATEGORY_COL].unique().tolist())
    print(f"Categories found: {all_categories}")

    train, temp = train_test_split(df, test_size=0.3, stratify=df[LABEL_COL], random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[LABEL_COL], random_state=RANDOM_STATE)
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")

    train = add_hazard_features(train)
    val = add_hazard_features(val)
    test = add_hazard_features(test)

    train = add_sentiment_features(train)
    val = add_sentiment_features(val)
    test = add_sentiment_features(test)

    engineered_cols = (
        [f"hazard_{h}" for h in HAZARD_LEXICON]
        + ["hazard_max_score", "hazard_count", "sentiment_polarity", "sentiment_subjectivity"]
    )

    train_cat = build_category_dummies(train, all_categories)
    val_cat = build_category_dummies(val, all_categories)
    test_cat = build_category_dummies(test, all_categories)

    y_train, y_val, y_test = train[LABEL_COL], val[LABEL_COL], test[LABEL_COL]

    # --- Sentence-BERT embeddings (Section 9) ---
    from sentence_transformers import SentenceTransformer
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} (downloads from Hugging Face on first run) ...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding complaint text ...")
    emb_train = np.array(encoder.encode(train["clean_text"].tolist(), show_progress_bar=True, convert_to_numpy=True))
    emb_val = np.array(encoder.encode(val["clean_text"].tolist(), show_progress_bar=True, convert_to_numpy=True))
    emb_test = np.array(encoder.encode(test["clean_text"].tolist(), show_progress_bar=True, convert_to_numpy=True))

    def combine_embeddings(embeddings, split_df, cat_df):
        numeric_features = split_df[engineered_cols].to_numpy(dtype=np.float32)
        category_features = cat_df.to_numpy(dtype=np.float32)
        return np.hstack([embeddings.astype(np.float32), numeric_features, category_features])

    X_train_emb = combine_embeddings(emb_train, train, train_cat)
    X_val_emb = combine_embeddings(emb_val, val, val_cat)
    X_test_emb = combine_embeddings(emb_test, test, test_cat)

    # --- Priority model: Logistic Regression vs XGBoost, pick the winner (Section 9a-9c) ---
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train_emb, y_train)
    evaluate(clf, X_val_emb, y_val, "Logistic Regression -- Validation")
    logreg_recall = evaluate(clf, X_test_emb, y_test, "Logistic Regression -- Test")
    logreg_f1 = f1_score(y_test, clf.predict(X_test_emb), average="macro")

    from xgboost import XGBClassifier  # NOTE: this import was missing in the original notebook

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    sample_weights = compute_sample_weight("balanced", y_train_enc)

    xgb_model = XGBClassifier(
        objective="multi:softprob", num_class=len(le.classes_),
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
    )
    print("\nTraining XGBoost ...")
    xgb_model.fit(X_train_emb, y_train_enc, sample_weight=sample_weights)
    xgb_preds = le.inverse_transform(xgb_model.predict(X_test_emb))
    print("\n" + "=" * 60 + "\nXGBoost -- Test\n" + "=" * 60)
    print(classification_report(y_test, xgb_preds, digits=3))
    xgb_recall = recall_score(y_test, xgb_preds, labels=["Emergency"], average="micro")
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro")
    print(f"Emergency recall: {xgb_recall:.3f}  Macro F1: {xgb_f1:.3f}")

    print("\n" + "=" * 60 + "\nMODEL COMPARISON\n" + "=" * 60)
    print(f"Logistic Regression -- Emergency Recall: {logreg_recall:.3f}  Macro F1: {logreg_f1:.3f}")
    print(f"XGBoost             -- Emergency Recall: {xgb_recall:.3f}  Macro F1: {xgb_f1:.3f}")

    if xgb_recall > logreg_recall or (xgb_recall == logreg_recall and xgb_f1 > logreg_f1):
        best_model, best_name, uses_label_encoder = xgb_model, "XGBoost", True
    else:
        best_model, best_name, uses_label_encoder = clf, "Logistic Regression", False
    print(f"\nBest priority model: {best_name}")

    with open(os.path.join(args.output, "best_transformer_model.pkl"), "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "label_encoder": le if uses_label_encoder else None,
            "embedding_model_name": EMBEDDING_MODEL,
            "engineered_cols": engineered_cols,
            "categories": all_categories,
        }, f)
    print(f"Saved {args.output}/best_transformer_model.pkl")

    # --- Category classifier (Section 9e) ---
    category_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    category_clf.fit(emb_train, train[CATEGORY_COL])
    cat_preds = category_clf.predict(emb_test)
    print(f"\nCategory classifier accuracy: {accuracy_score(test[CATEGORY_COL], cat_preds):.3f}")
    print(classification_report(test[CATEGORY_COL], cat_preds, digits=3))

    with open(os.path.join(args.output, "category_classifier.pkl"), "wb") as f:
        pickle.dump({
            "model": category_clf,
            "categories": all_categories,
            "embedding_model_name": EMBEDDING_MODEL,
        }, f)
    print(f"Saved {args.output}/category_classifier.pkl")

    print("\nDone. Copy/keep the whole '" + args.output + "' folder next to app.py, then run:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
