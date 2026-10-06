"""
app.py -- Smart Complaint Management System dashboard

Wraps complaint_pipeline.py (the notebook's trained models) in a Streamlit UI.
Does NOT reimplement or retrain the ML model -- it only calls
complaint_pipeline.analyze_complaint() and visualizes the result.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ==============================
# IMPORT PROJECT PIPELINE
# ==============================
try:
    from complaint_pipeline import analyze_complaint, PipelineNotReadyError
    PIPELINE_AVAILABLE = True
    PIPELINE_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - exercised only when the real pipeline can't load
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(e)

    class PipelineNotReadyError(RuntimeError):
        pass

    def analyze_complaint(complaint_text: str) -> dict:
        """
        Temporary mock used ONLY when complaint_pipeline.py can't be imported
        (e.g. models not trained yet, or sentence-transformers not installed
        in this environment). Lets the dashboard UI be demoed/developed without
        the real model. Never used if the real pipeline is available.
        """
        text_lower = complaint_text.lower()
        if any(w in text_lower for w in ["fire", "gas leak", "assault", "weapon", "unconscious"]):
            probs = {"Low": 0.03, "Medium": 0.07, "High": 0.15, "Emergency": 0.75}
        elif any(w in text_lower for w in ["urgent", "asap", "immediately", "broken"]):
            probs = {"Low": 0.10, "Medium": 0.25, "High": 0.55, "Emergency": 0.10}
        else:
            probs = {"Low": 0.55, "Medium": 0.30, "High": 0.10, "Emergency": 0.05}
        prediction = max(probs, key=probs.get)
        confidence = probs[prediction]
        hazard_score = 0.75 if prediction == "Emergency" else 0.05
        gated = "human_review" if (confidence < 0.60 or hazard_score >= 0.70) else "auto_assign"
        return {
            "complaint_text": complaint_text,
            "cleaned_text": complaint_text.strip(),
            "predicted_category": "IT Support",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs,
            "hazard_score": hazard_score,
            "hazard_breakdown": {"fire": hazard_score, "gas": 0.0, "electrical": 0.0,
                                  "structural": 0.0, "medical": 0.0, "assault": 0.0},
            "sentiment_polarity": -0.2,
            "sentiment_subjectivity": 0.5,
            "threshold_used": 0.60,
            "tier": "strict",
            "gated_decision": gated,
            "gate_reason": "mock_data" ,
        }

PRIORITY_ORDER = ["Low", "Medium", "High", "Emergency"]
PRIORITY_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Emergency": "#7b0000",
}

# ==============================
# 1. PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Smart Complaint Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# 14. CUSTOM CSS / STYLING
# ==============================
st.markdown("""
<style>
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #4f46e5, #9333ea, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}
.card {
    background: var(--background-color, #ffffff);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 14px rgba(0,0,0,0.08);
    border: 1px solid rgba(120,120,120,0.12);
    margin-bottom: 1rem;
}
.metric-card {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    color: white;
}
.metric-label {
    font-size: 0.8rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 0.2rem;
}
.char-count {
    text-align: right;
    color: #9ca3af;
    font-size: 0.85rem;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120,120,120,0.15);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 2. HERO SECTION
# ==============================
st.markdown('<div class="hero-title">🎓 Smart Complaint Management System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered university complaint prioritization and intelligent routing system</div>',
    unsafe_allow_html=True,
)
st.divider()

# ==============================
# 3. SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("### 🎓 Smart Complaint Management System")
    st.caption("University complaint triage, powered by a hazard-aware AI pipeline.")

    st.markdown("---")
    st.markdown("**Model pipeline**")
    st.markdown(
        "- Text cleaning\n"
        "- Hazard-entity corroboration scoring\n"
        "- Sentiment features (TextBlob)\n"
        "- Sentence-BERT embeddings\n"
        "- Category classifier (auto-detected)\n"
        "- Priority classifier (Low / Medium / High / Emergency)\n"
        "- Risk-tiered confidence gate"
    )

    st.markdown("---")
    st.markdown("**Confidence gate settings**")
    st.markdown(
        f"- Low-stakes threshold: **0.35**\n"
        f"- Strict threshold: **0.60**\n"
        f"- Hazard override threshold: **0.70**"
    )

    st.markdown("---")
    st.markdown("**Supported classes**")
    for p in PRIORITY_ORDER:
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{PRIORITY_COLORS[p]};margin-right:6px"></span>{p}',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.write(
            "This dashboard wraps a trained complaint-prioritization pipeline. "
            "It detects genuine safety hazards (fire, gas leak, assault, medical "
            "emergencies) rather than just matching keywords, and routes "
            "low-confidence or high-hazard complaints to a human reviewer instead "
            "of guessing."
        )
        if not PIPELINE_AVAILABLE:
            st.warning(
                "Running in DEMO MODE with mock predictions -- the trained model "
                "files were not found. See the error details in the main panel."
            )

# ==============================
# Pipeline availability banner
# ==============================
if not PIPELINE_AVAILABLE:
    st.error(
        "⚠️ Could not load the trained pipeline (`complaint_pipeline.py`). "
        "The app is running with MOCK predictions so the UI can still be used/demoed.\n\n"
        f"Import error: `{PIPELINE_IMPORT_ERROR}`\n\n"
        "To fix: run `complaint_pipeline.ipynb` fully at least once so that "
        "`processed/best_transformer_model.pkl` and `processed/category_classifier.pkl` "
        "exist, then restart this app."
    )

# ==============================
# 4. COMPLAINT INPUT CARD
# ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Submit a Complaint")

complaint_text = st.text_area(
    "Complaint text",
    height=140,
    placeholder=(
        "Example: Electrical panel near room 216 is sparking and smoke is coming out, "
        "please help immediately."
    ),
    label_visibility="collapsed",
    key="complaint_input",
)
st.markdown(f'<div class="char-count">{len(complaint_text)} characters</div>', unsafe_allow_html=True)

analyze_clicked = st.button("🔍 Analyze Complaint", type="primary", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# 13. ERROR HANDLING around the prediction call
# ==============================
result = None
if analyze_clicked:
    if not complaint_text.strip():
        st.warning("Please enter a complaint before analyzing.")
    else:
        try:
            with st.spinner("Analyzing complaint..."):
                result = analyze_complaint(complaint_text)
            st.session_state["last_result"] = result
        except PipelineNotReadyError as e:
            st.error(f"Model not ready: {e}")
        except Exception as e:
            st.error(f"Something went wrong while analyzing this complaint: {e}")

if result is None and "last_result" in st.session_state:
    result = st.session_state["last_result"]

# ==============================
# 5. PREDICTION SUMMARY
# ==============================
if result:
    st.markdown("### 📊 Prediction Summary")
    prediction = result["prediction"]
    priority_color = PRIORITY_COLORS.get(prediction, "#6b7280")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card" style="background:{priority_color};">'
            f'<div class="metric-label">Predicted Priority</div>'
            f'<div class="metric-value">{prediction}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card" style="background:#4f46e5;">'
            f'<div class="metric-label">Confidence</div>'
            f'<div class="metric-value">{result["confidence"]*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card" style="background:#0ea5e9;">'
            f'<div class="metric-label">Predicted Category</div>'
            f'<div class="metric-value" style="font-size:1.1rem">{result["predicted_category"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        gate_color = "#e74c3c" if result["gated_decision"] == "human_review" else "#2ecc71"
        gate_label = "Human Review" if result["gated_decision"] == "human_review" else "Auto-Assign"
        st.markdown(
            f'<div class="metric-card" style="background:{gate_color};">'
            f'<div class="metric-label">Routing Decision</div>'
            f'<div class="metric-value" style="font-size:1.2rem">{gate_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ==============================
    # 6. PROBABILITY BAR CHART
    # ==============================
    st.markdown("### 📈 Class Probability Distribution")
    probs = result["probabilities"]
    prob_df = pd.DataFrame({
        "Priority": PRIORITY_ORDER,
        "Probability": [probs.get(p, 0) * 100 for p in PRIORITY_ORDER],
    })
    fig_bar = go.Figure(go.Bar(
        x=prob_df["Probability"],
        y=prob_df["Priority"],
        orientation="h",
        marker_color=[PRIORITY_COLORS[p] for p in prob_df["Priority"]],
        text=[f"{v:.1f}%" for v in prob_df["Probability"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Probability (%)", yaxis_title="", height=320,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 105],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ==============================
    # 6b. RADAR + DONUT
    # ==============================
    st.markdown("### 🎯 Model Prediction Confidence Analysis")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_label, top_val = sorted_probs[0]
    second_label, second_val = sorted_probs[1] if len(sorted_probs) > 1 else ("-", 0)
    gap_pts = (top_val - second_val) * 100

    if gap_pts >= 50:
        certainty_label = "Highly confident"
    elif gap_pts >= 20:
        certainty_label = "Moderately confident"
    else:
        certainty_label = "Uncertain"

    col_radar, col_donut = st.columns(2)

    with col_radar:
        st.markdown("**Prediction Confidence Radar**")
        radar_vals = [probs.get(p, 0) * 100 for p in PRIORITY_ORDER] + [probs.get(PRIORITY_ORDER[0], 0) * 100]
        radar_labels = PRIORITY_ORDER + [PRIORITY_ORDER[0]]
        marker_colors = ["#111827" if p == prediction else "#9ca3af" for p in PRIORITY_ORDER] + ["#111827"]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals, theta=radar_labels, fill="toself",
            line=dict(color="#4f46e5", width=2),
            marker=dict(size=[14 if p == prediction else 6 for p in radar_labels], color=marker_colors),
            name="Confidence",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=360, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_donut:
        st.markdown("**Predicted Class vs. Everything Else**")
        other_prob = 1 - top_val
        fig_donut = go.Figure(go.Pie(
            labels=[top_label, "Other classes"],
            values=[top_val * 100, other_prob * 100],
            hole=0.6,
            marker_colors=[PRIORITY_COLORS.get(top_label, "#4f46e5"), "#e5e7eb"],
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            height=360, margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f"{top_label}<br>{top_val*100:.0f}%", x=0.5, y=0.5,
                font_size=16, showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.caption(
        f"**{top_label}** is the dominant prediction at **{top_val*100:.1f}%**, "
        f"{gap_pts:.1f} percentage points ahead of **{second_label}** ({second_val*100:.1f}%). "
        f"Model is **{certainty_label.lower()}** about this prediction."
    )

    # ==============================
    # 7. CONFIDENCE GAUGE
    # ==============================
    st.markdown("### 🧭 Confidence Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["confidence"] * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": priority_color},
            "steps": [
                {"range": [0, 35], "color": "#fee2e2"},
                {"range": [35, 60], "color": "#fef3c7"},
                {"range": [60, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": result["threshold_used"] * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ==============================
    # 8. HAZARD ANALYTICS
    # ==============================
    st.markdown("### 🚨 Hazard Analytics")
    hazard_score = result["hazard_score"]
    st.progress(min(hazard_score, 1.0), text=f"Hazard score: {hazard_score:.2f}")

    if hazard_score < 0.3:
        st.success(f"✅ Low hazard signal ({hazard_score:.2f}) -- no strong safety-critical language detected.")
    elif hazard_score < 0.7:
        st.warning(f"⚠️ Moderate hazard signal ({hazard_score:.2f}) -- some hazard-related language present.")
    else:
        st.error(f"🔴 High hazard signal ({hazard_score:.2f}) -- forces human review regardless of confidence.")

    hc1, hc2 = st.columns(2)
    with hc1:
        st.metric("Threshold used", f"{result['threshold_used']:.2f}")
    with hc2:
        tier_label = "Low-stakes tier (35%)" if result["tier"] == "low_stakes" else "Strict tier (60%)"
        st.metric("Gate tier applied", tier_label)

    with st.expander("Hazard type breakdown"):
        hb_df = pd.DataFrame({
            "Hazard type": list(result["hazard_breakdown"].keys()),
            "Score": list(result["hazard_breakdown"].values()),
        })
        st.bar_chart(hb_df.set_index("Hazard type"))

    # ==============================
    # 9. EXPLAINABILITY SECTION
    # ==============================
    with st.expander("🔬 Model Explanation"):
        st.write(f"**Top predicted class:** {top_label} ({top_val*100:.1f}%)")
        st.write(f"**Second most probable class:** {second_label} ({second_val*100:.1f}%)")
        st.write(f"**Probability gap:** {gap_pts:.1f} percentage points")
        if gap_pts >= 50:
            interpretation = "Highly certain -- the model strongly favors one class over all others."
        elif gap_pts >= 20:
            interpretation = "Moderately certain -- the top class leads, but not overwhelmingly."
        else:
            interpretation = "Ambiguous prediction -- the top two classes are close, worth a second look."
        st.write(f"**Interpretation:** {interpretation}")

    # ==============================
    # 11. TECHNICAL DETAILS
    # ==============================
    with st.expander("⚙️ Technical Details"):
        st.write("**Cleaned complaint text:**", result["cleaned_text"])
        st.write("**Predicted category:**", result["predicted_category"])
        st.write("**Hazard score:**", f"{result['hazard_score']:.3f}")
        st.write("**Confidence score:**", f"{result['confidence']:.3f}")
        st.write("**Routing decision:**", result["gated_decision"],
                  f"({result['gate_reason']})" if result["gate_reason"] else "")
        st.write("**Sentiment polarity / subjectivity:**",
                  f"{result['sentiment_polarity']:.2f} / {result['sentiment_subjectivity']:.2f}")
        st.write("**Pipeline steps executed:**")
        st.code(
            "1. clean_text()\n"
            "2. SentenceTransformer.encode() -> embedding\n"
            "3. category_classifier.predict() -> predicted_category\n"
            "4. add_hazard_features() -> hazard scores\n"
            "5. add_sentiment_features() -> sentiment scores\n"
            "6. build_category_dummies()\n"
            "7. priority_model.predict_proba() -> class probabilities\n"
            "8. risk-tiered confidence gate -> routing decision",
            language="text",
        )

    # ==============================
    # 12. DOWNLOAD FEATURE
    # ==============================
    download_payload = {k: v for k, v in result.items()}
    st.download_button(
        "⬇️ Download prediction as JSON",
        data=json.dumps(download_payload, indent=2),
        file_name="complaint_prediction.json",
        mime="application/json",
    )

st.divider()

# ==============================
# 10. HISTORICAL DASHBOARD (sample data)
# ==============================
st.markdown("## 📚 Historical Complaint Analytics (sample data)")
st.caption("Illustrative data for dashboard purposes -- not connected to a live complaint log.")


@st.cache_data
def _generate_sample_history(n=40):
    rng = np.random.default_rng(42)
    priorities = rng.choice(PRIORITY_ORDER, size=n, p=[0.40, 0.33, 0.17, 0.10])
    dates = [datetime.now() - timedelta(days=int(d)) for d in rng.integers(0, 30, size=n)]
    decisions = [
        "human_review" if (p in ("High", "Emergency") or rng.random() < 0.25) else "auto_assign"
        for p in priorities
    ]
    return pd.DataFrame({"date": dates, "priority": priorities, "decision": decisions}).sort_values("date")


history_df = _generate_sample_history()

hcol1, hcol2, hcol3 = st.columns(3)

with hcol1:
    st.markdown("**Complaint distribution by priority**")
    dist = history_df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    fig_pie = go.Figure(go.Pie(
        labels=dist.index, values=dist.values,
        marker_colors=[PRIORITY_COLORS[p] for p in dist.index],
    ))
    fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with hcol2:
    st.markdown("**Model comparison: accuracy vs. Emergency recall**")
    # Real numbers from an actual run of the notebook against the current dataset --
    # not sample/illustrative data like the other two charts. Shown as a GROUPED
    # bar (accuracy + Emergency recall) rather than accuracy alone, because
    # accuracy alone is misleading here: XGBoost has the highest raw accuracy
    # (67.3%) but was correctly NOT selected as the best model, since its
    # Emergency recall (63.2%) is weaker than Logistic Regression's (77.9%) --
    # and Emergency recall is the metric this project prioritizes on purpose.
    algo_comparison = pd.DataFrame({
        "Algorithm": [
            "Random Forest\n(TF-IDF)", "SVM\n(TF-IDF)", "SVM + Threshold\nTuning",
            "Transformer +\nLogReg (selected)", "Transformer +\nXGBoost",
        ],
        "Accuracy": [63.5, 63.3, 62.5, 60.4, 67.3],
        "Emergency Recall": [57.4, 61.8, 66.2, 77.9, 63.2],
    })
    fig_algo = go.Figure()
    fig_algo.add_trace(go.Bar(
        name="Accuracy", x=algo_comparison["Algorithm"], y=algo_comparison["Accuracy"],
        marker_color="#4f46e5",
        text=[f"{v:.1f}%" for v in algo_comparison["Accuracy"]], textposition="outside",
    ))
    fig_algo.add_trace(go.Bar(
        name="Emergency Recall", x=algo_comparison["Algorithm"], y=algo_comparison["Emergency Recall"],
        marker_color="#16a34a",
        text=[f"{v:.1f}%" for v in algo_comparison["Emergency Recall"]], textposition="outside",
    ))
    fig_algo.update_layout(
        barmode="group", yaxis_title="Score (%)", yaxis_range=[0, 90],
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_algo, use_container_width=True)
    st.caption(
        "Transformer + Logistic Regression (highlighted) was selected as the best "
        "model -- not the highest-accuracy option (XGBoost, 67.3%), but the one "
        "with the strongest Emergency recall (77.9%), since missing a real "
        "emergency is the more costly mistake in this project."
    )

with hcol3:
    st.markdown("**Auto-assigned vs. human review**")
    dec_counts = history_df["decision"].value_counts()
    fig_dec = go.Figure(go.Bar(
        x=dec_counts.index, y=dec_counts.values,
        marker_color=["#2ecc71" if d == "auto_assign" else "#e74c3c" for d in dec_counts.index],
    ))
    fig_dec.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_dec, use_container_width=True)

st.caption("Built on the Smart Complaint Management System pipeline -- hazard-aware, risk-tiered, confidence-gated.")
