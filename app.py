import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
SAMPLE_PATH = BASE_DIR / "sample_customers.csv"

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PAYMENTS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]

# ---------------------------------------------------------------------
# Visual theme (frontend only)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
      --ink: #152238;
      --muted: #68768a;
      --line: #e4eaf2;
      --panel: #ffffff;
      --canvas: #f4f7fb;
      --teal: #087f8c;
      --teal-soft: #e3f6f5;
      --blue: #4263eb;
      --blue-soft: #edf1ff;
      --amber: #b96b09;
      --amber-soft: #fff3db;
      --red: #c43d4b;
      --red-soft: #ffeaed;
    }
    html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,251,.88); }
    [data-testid="stSidebar"] {
      background: #101c2e;
      border-right: 1px solid #223149;
    }
    [data-testid="stSidebar"] * { color: #e8eef8; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: #aebbd0;
    }
    [data-testid="stSidebar"] label { color: #e8eef8 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
      background: #1a2940;
    }
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: -0.035em; }
    h1 { font-weight: 800; }
    [data-testid="stMetric"] {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: 0 4px 18px rgba(28,45,75,.035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-weight: 800; }
    div[data-testid="stTabs"] button { font-weight: 700; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
      color: var(--teal);
      border-bottom-color: var(--teal);
    }
    .hero {
      background: linear-gradient(115deg, #14243a 0%, #173c55 58%, #087f8c 100%);
      border-radius: 22px;
      padding: 30px 34px;
      color: white;
      margin-bottom: 22px;
      box-shadow: 0 12px 35px rgba(20,36,58,.12);
    }
    .hero-eyebrow { text-transform: uppercase; letter-spacing: .15em; font-size: .73rem; font-weight: 800; color: #91e3dc; }
    .hero-title { color: white; font-family: 'Manrope', sans-serif; font-size: clamp(1.8rem, 4vw, 2.65rem); font-weight: 800; line-height: 1.12; margin: 10px 0; }
    .hero-copy { color: #d8e5ef; max-width: 720px; font-size: 1rem; line-height: 1.65; }
    .section-kicker { color: var(--teal); text-transform: uppercase; font-size: .72rem; letter-spacing: .13em; font-weight: 800; margin-bottom: 4px; }
    .section-title { font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 800; letter-spacing: -.03em; margin-bottom: 4px; }
    .section-copy { color: var(--muted); font-size: .92rem; margin-bottom: 15px; }
    .surface {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 4px 18px rgba(28,45,75,.03);
    }
    .risk-pill {
      display: inline-block; padding: 5px 11px; border-radius: 999px;
      font-weight: 800; font-size: .78rem; letter-spacing: .04em;
    }
    .risk-high { background: var(--red-soft); color: var(--red); }
    .risk-watch { background: var(--amber-soft); color: var(--amber); }
    .risk-low { background: var(--teal-soft); color: var(--teal); }
    .small-muted { color: var(--muted); font-size: .84rem; }
    .footer {
      border-top: 1px solid var(--line); margin-top: 30px; padding-top: 16px;
      color: var(--muted); font-size: .82rem;
    }
    .stButton > button, .stDownloadButton > button {
      border-radius: 10px; font-weight: 700; min-height: 2.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Model artifacts and original training-serving feature encoding
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH.name}")
    model = joblib.load(MODEL_PATH)
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw: pd.DataFrame, columns) -> pd.DataFrame:
    """Preserve the serving-time encoder used by the original project."""
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = raw[RAW_COLS].copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0)
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0)
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0)
    encoded = pd.get_dummies(frame)
    return encoded.reindex(columns=columns, fill_value=0).astype(float)


def score(raw: pd.DataFrame, model, columns) -> np.ndarray:
    features = prepare_input(raw, columns)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not expose predict_proba().")
    probabilities = model.predict_proba(features)[:, 1]
    return np.clip(probabilities, 0, 1)


def make_customer(
    tenure, contract, monthly, internet, services, phone, multi, payment,
    paperless, gender, senior, partner, dependents,
):
    row = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(tenure) * float(monthly),
    }
    for service in SERVICES:
        row[service] = (
            "No internet service" if internet == "No"
            else "Yes" if service in services else "No"
        )
    return pd.DataFrame([row], columns=RAW_COLS)


def risk_band(probability, threshold):
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold / 2:
        return "WATCH"
    return "LOW"


def render_risk_badge(band):
    css_class = {"HIGH": "risk-high", "WATCH": "risk-watch", "LOW": "risk-low"}[band]
    st.markdown(f'<span class="risk-pill {css_class}">{band} RISK</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Load model; keep the original model and feature columns unchanged
# ---------------------------------------------------------------------
try:
    model, meta = load_artifacts()
    COLS = meta["feature_columns"]
except Exception as exc:
    st.error("The dashboard could not load the model artifacts.")
    st.exception(exc)
    st.stop()

threshold_default = float(meta.get("threshold", 0.30))
model_name = str(meta.get("model_name", "Saved churn model"))
cv_auc = meta.get("cv_auc")
cv_auc_std = meta.get("cv_auc_std")

# ---------------------------------------------------------------------
# Sidebar: profile controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ◈ ChurnIQ")
    st.markdown("CUSTOMER INTELLIGENCE")
    st.divider()
    st.markdown("### Customer profile")
    st.caption("Adjust the profile to see the model's estimated churn probability.")

    tenure = st.slider("Tenure (months)", 0, 72, 4)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly charges (USD)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

    services = []
    if internet != "No":
        services = st.multiselect(
            "Add-on services",
            SERVICES,
            default=["StreamingTV"],
            format_func=lambda x: x.replace("Online", "Online ").replace("Tech", "Tech "),
        )

    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    multi = "No phone service"
    if phone == "Yes":
        multi = st.radio("Multiple lines", ["No", "Yes"], horizontal=True)

    payment = st.selectbox("Payment method", PAYMENTS)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    with st.expander("Demographics", expanded=False):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    st.divider()
    threshold = st.slider(
        "Contact threshold",
        min_value=0.05,
        max_value=0.95,
        value=min(max(threshold_default, 0.05), 0.95),
        step=0.05,
        help="Used to label a customer as HIGH risk. Changing this does not change the model probability.",
    )
    st.caption("Model predictions are decision support, not certainty.")

customer = make_customer(
    tenure, contract, monthly, internet, services, phone, multi, payment,
    paperless, gender, senior, partner, dependents,
)
probability = float(score(customer, model, COLS)[0])
band = risk_band(probability, threshold)

# ---------------------------------------------------------------------
# Header and dashboard navigation
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-eyebrow">CUSTOMER RETENTION • MACHINE LEARNING</div>
      <div class="hero-title">Understand churn.<br>Act with confidence.</div>
      <div class="hero-copy">Explore customer risk, compare retention scenarios, and review batch predictions in one clear workspace.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2, top3, top4 = st.columns(4)
top1.metric("Current churn probability", f"{probability:.1%}")
top2.metric("Risk classification", band)
top3.metric("Customer tenure", f"{tenure} mo")
top4.metric("Monthly charges", f"${monthly:,.2f}")

st.progress(probability, text=f"Estimated churn probability: {probability:.1%}")
st.markdown(
    '<div class="small-muted">Probability is produced by the existing saved model. '
    'The threshold only changes the risk label, not the probability.</div>',
    unsafe_allow_html=True,
)
st.write("")

tab_overview, tab_scenarios, tab_batch, tab_model = st.tabs(
    ["Overview", "What-if scenarios", "Batch analysis", "Model & limitations"]
)

with tab_overview:
    st.markdown('<div class="section-kicker">CUSTOMER SNAPSHOT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A quick interpretation of the selected customer profile.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Risk assessment")
            render_risk_badge(band)
            st.write("")
            if band == "HIGH":
                st.markdown("**Suggested priority: review soon**")
                st.write("This profile meets or exceeds the selected contact threshold. Consider reviewing the customer's needs before deciding on a retention action.")
            elif band == "WATCH":
                st.markdown("**Suggested priority: monitor**")
                st.write("This profile is below the contact threshold but above half of it. Consider monitoring it alongside other customer signals.")
            else:
                st.markdown("**Suggested priority: routine service**")
                st.write("This profile is below half of the selected threshold. Continue normal customer support and monitoring.")
            st.caption("These are workflow suggestions, not guaranteed interventions or causal explanations.")
    with right:
        with st.container(border=True):
            st.markdown("#### Profile details")
            profile = pd.DataFrame(
                [
                    ("Contract", contract),
                    ("Internet service", internet),
                    ("Tenure", f"{tenure} months"),
                    ("Monthly charges", f"${monthly:,.2f}"),
                    ("Payment method", payment),
                    ("Paperless billing", paperless),
                    ("Phone service", phone),
                ],
                columns=["Attribute", "Selected value"],
            )
            st.dataframe(profile, hide_index=True, use_container_width=True)

    st.markdown("#### Probability guide")
    guide1, guide2, guide3 = st.columns(3)
    with guide1:
        st.markdown('<div class="surface"><span class="risk-pill risk-low">LOW RISK</span><h3>Below half</h3><p class="small-muted">Probability is below half of the current threshold.</p></div>', unsafe_allow_html=True)
    with guide2:
        st.markdown('<div class="surface"><span class="risk-pill risk-watch">WATCH</span><h3>Monitor</h3><p class="small-muted">Probability is at least half the threshold but below it.</p></div>', unsafe_allow_html=True)
    with guide3:
        st.markdown('<div class="surface"><span class="risk-pill risk-high">HIGH RISK</span><h3>Review</h3><p class="small-muted">Probability meets or exceeds the current threshold.</p></div>', unsafe_allow_html=True)

with tab_scenarios:
    st.markdown('<div class="section-kicker">SCENARIO EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What could the model score under other profiles?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">These are model comparisons, not proof that changing a feature will cause churn to decrease.</div>', unsafe_allow_html=True)

    alternatives = []
    for alternative_contract in ["One year", "Two year"]:
        if alternative_contract != contract:
            alternatives.append(("Contract", alternative_contract))
    if payment == "Electronic check":
        alternatives.append(("PaymentMethod", "Credit card (automatic)"))
    if internet != "No" and "TechSupport" not in services:
        alternatives.append(("TechSupport", "Yes"))

    scenario_rows = []
    for feature, value in alternatives:
        alt = customer.copy()
        if feature == "Contract":
            alt.loc[:, feature] = value
        elif feature == "PaymentMethod":
            alt.loc[:, feature] = value
        elif feature == "TechSupport":
            alt.loc[:, feature] = value
        new_probability = float(score(alt, model, COLS)[0])
        scenario_rows.append(
            {
                "Scenario": f"{feature}: {value}",
                "New probability": new_probability,
                "Change vs. current": new_probability - probability,
            }
        )

    if scenario_rows:
        scenarios = pd.DataFrame(scenario_rows).sort_values("Change vs. current")
        display = scenarios.copy()
        display["New probability"] = display["New probability"].map(lambda x: f"{x:.1%}")
        display["Change vs. current"] = display["Change vs. current"].map(lambda x: f"{x:+.1%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("A negative difference means the saved model assigns a lower probability to that alternate profile.")
    else:
        st.info("No alternate scenarios are available for the current profile.")

    final_estimator = model[-1] if hasattr(model, "steps") and len(model.steps) else model
    if hasattr(model, "steps") and hasattr(final_estimator, "coef_"):
        with st.expander("Inspect model coefficients for this customer"):
            try:
                transformed = model[:-1].transform(prepare_input(customer, COLS))[0]
                contributions = pd.Series(final_estimator.coef_[0] * transformed, index=COLS)
                top = contributions.reindex(
                    contributions.abs().sort_values(ascending=False).index
                ).head(8).sort_values()
                st.bar_chart(top)
                st.caption("For this linear model, positive contributions push the linear score upward and negative contributions push it downward. This is not a causal explanation.")
            except Exception as exc:
                st.warning(f"Coefficient view is unavailable for this model: {exc}")
    else:
        st.caption("The saved model does not expose the Logistic Regression coefficient view used by this optional panel.")

with tab_batch:
    st.markdown('<div class="section-kicker">BATCH WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Score a customer file</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Upload a CSV with the original Telco customer columns. The model and serving-time encoding remain unchanged.</div>', unsafe_allow_html=True)

    if SAMPLE_PATH.exists():
        with open(SAMPLE_PATH, "rb") as sample_file:
            st.download_button(
                "Download sample CSV",
                data=sample_file.read(),
                file_name="sample_customers.csv",
                mime="text/csv",
            )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [column for column in RAW_COLS if column not in data.columns]
            if missing:
                st.error("Your file is missing required columns: " + ", ".join(missing))
            elif data.empty:
                st.warning("The uploaded CSV has no customer rows.")
            else:
                batch_probabilities = score(data, model, COLS)
                scored = data.copy()
                scored["churn_probability"] = np.round(batch_probabilities, 4)
                scored["risk_band"] = [risk_band(float(p), threshold) for p in batch_probabilities]
                scored["review_recommended"] = np.where(batch_probabilities >= threshold, "Yes", "No")

                high_count = int((batch_probabilities >= threshold).sum())
                avg_probability = float(np.mean(batch_probabilities))
                m1, m2, m3 = st.columns(3)
                m1.metric("Customers scored", f"{len(scored):,}")
                m2.metric("At/above threshold", f"{high_count:,}")
                m3.metric("Average predicted probability", f"{avg_probability:.1%}")

                st.markdown("#### Highest predicted risk")
                st.dataframe(
                    scored.sort_values("churn_probability", ascending=False).head(50),
                    hide_index=True,
                    use_container_width=True,
                )
                csv_bytes = scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download scored results",
                    data=csv_bytes,
                    file_name="scored_customers.csv",
                    mime="text/csv",
                    type="primary",
                )
                st.caption("Only the first 50 rows are shown in the preview; the download contains all scored rows.")
        except Exception as exc:
            st.error("The CSV could not be processed. Check the file format and required columns.")
            st.exception(exc)
    else:
        st.info("Upload a CSV to begin, or download the sample file above.")

with tab_model:
    st.markdown('<div class="section-kicker">MODEL INFORMATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model card & limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Transparency about the model is part of the product.</div>', unsafe_allow_html=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("Model", str(meta.get("model_name", "Not recorded")))
    info2.metric("Cross-validation AUC", f"{float(cv_auc):.3f}" if cv_auc is not None else "Not recorded")
    info3.metric("CV AUC standard deviation", f"{float(cv_auc_std):.3f}" if cv_auc_std is not None else "Not recorded")

    safe_meta = {k: v for k, v in meta.items() if k != "feature_columns"}
    with st.expander("View saved metadata"):
        st.json(safe_meta)

    st.markdown("#### Intended use")
    st.write("Rank existing telecom customers by estimated churn risk so a retention team can prioritize review. Predictions should support human decisions, not automate them.")
    st.markdown("#### Important limitations")
    st.markdown(
        """
        - The model was trained on a historical US telecom dataset; performance may not transfer to other markets or providers.
        - Associations learned by a model do not establish that changing a customer attribute will change their behavior.
        - Risk labels depend on the selected threshold. The underlying probability is unchanged when the threshold moves.
        - Review current validation metrics and subgroup performance before using predictions operationally.
        - Do not use this model to deny service, set prices, or make credit decisions.
        """
    )

st.markdown(
    f'<div class="footer">ChurnIQ · ML-Powered Customer Analytics &nbsp; • &nbsp; {model_name} &nbsp; • &nbsp; Decision support only</div>',
    unsafe_allow_html=True,
)
