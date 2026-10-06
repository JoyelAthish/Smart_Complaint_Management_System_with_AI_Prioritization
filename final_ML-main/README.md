# 🎓 Smart Complaint Management System

An AI-powered university complaint prioritization and intelligent routing dashboard.
Complaints are automatically categorized, scored for priority (Low / Medium / High /
Emergency), checked for genuine safety hazards (not just alarming keywords), and
routed either straight through or to a human reviewer via a risk-tiered confidence gate.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard UI |
| `complaint_pipeline.py` | Inference module -- loads the trained models and exposes `analyze_complaint()` / `predict_single_complaint()` |
| `complaint_pipeline.ipynb` | The notebook that trains and saves the models this app loads |
| `requirements.txt` | Python dependencies |

## ⚠️ Before running the app: train the models once

This app does **not** train anything itself -- it loads models already trained and
saved by the notebook. The app runs in demo/mock mode with clearly-labeled fake
predictions until you've done this once:

1. Open `complaint_pipeline.ipynb` and run every cell top to bottom, with
   `complaints.csv` in the same folder.
2. Confirm this produces a `processed/` folder containing, at minimum:
   - `processed/best_transformer_model.pkl`
   - `processed/category_classifier.pkl`
3. Keep the `processed/` folder next to `app.py` (or set the `COMPLAINT_MODEL_DIR`
   environment variable to point at it).

If these files are missing, the app still runs but shows a warning banner and uses
mock predictions so the UI can be reviewed/demoed independently of model training.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Push this folder (including the `processed/` folder with your trained model
   files, or a way to regenerate them) to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   **New app**.
3. Point it at your repository, branch, and `app.py` as the entry point.
4. Streamlit Cloud installs `requirements.txt` automatically and launches the app.

**Note on model size:** the sentence-embedding model (`all-MiniLM-L6-v2`) and
`torch` are sizeable dependencies. If you hit resource limits on the free tier,
consider committing only the small `.pkl` files and re-downloading the embedding
model at startup (already handled automatically by `sentence-transformers`), rather
than trying to commit model weights directly to the repo.

## How the pipeline works (what the app is visualizing)

1. **Text cleaning** -- whitespace/URL normalization, punctuation kept (urgency signal).
2. **Category prediction** -- a classifier trained on the same sentence embeddings
   predicts which department the complaint belongs to, so no manual category input
   is needed.
3. **Hazard-entity scoring** -- checks for genuine safety-hazard language (fire, gas
   leak, electrical, structural, medical, assault) with corroborating context, not
   just keyword matching -- so "this wifi issue is on fire" scores low while "smoke
   coming from an electrical panel" scores high.
4. **Sentiment features** -- polarity/subjectivity via TextBlob, as one additional signal.
5. **Priority classification** -- Sentence-BERT embeddings + engineered features feed
   a trained classifier producing Low / Medium / High / Emergency probabilities.
6. **Risk-tiered confidence gate** -- Low/Medium predictions with zero hazard signal
   only need 35% confidence to auto-assign; everything else needs 60%; any hazard
   score of 0.70+ forces human review regardless of confidence.

## Screenshots

_Add screenshots of the running app here once deployed._

- Hero + complaint input
- Prediction summary cards
- Probability charts (bar / radar / donut)
- Hazard analytics section
- Historical dashboard
