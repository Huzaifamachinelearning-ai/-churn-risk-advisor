import io
import json
import math
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Advisor",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
SAMPLE_PATH = BASE_DIR / "sample_customers.csv"

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PAYMENTS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]

# Design tokens
INK, MUTED, LINE = "#0f1b3d", "#6b7494", "#e6e8f2"
VIOLET, MINT, AMBER, ROSE = "#5b4bdb", "#12b886", "#f59f00", "#e5484d"
BAND_COLOR = {"HIGH": ROSE, "WATCH": AMBER, "LOW": MINT}
BAND_TITLE = {"HIGH": "Likely to leave", "WATCH": "Worth watching", "LOW": "Likely to stay"}
BAND_NOTE = {
    "HIGH": "Contact this customer for a retention review.",
    "WATCH": "Keep an eye on this account.",
    "LOW": "Routine retention is enough for now.",
}
BAND_ACTION = {"HIGH": "Contact for review", "WATCH": "Monitor", "LOW": "Routine retention"}

LEVER_TITLE = {
    ("Contract", "One year"): "Move to a one-year contract",
    ("Contract", "Two year"): "Move to a two-year contract",
    ("PaymentMethod", "Credit card (automatic)"): "Switch to automatic card payment",
    ("TechSupport", "Yes"): "Add tech support",
}


def html(markup: str) -> None:
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Public+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Public Sans', sans-serif; }
.stApp { background: #f5f6fb; color: #0f1b3d; }
.block-container { max-width: 1320px; padding-top: 1.2rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif; color: #0f1b3d; letter-spacing: -0.02em; }
[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e8f2; }
.brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 6px; }
.brand-mark { width: 40px; height: 40px; border-radius: 12px; background: #0f1b3d;
  display: flex; align-items: flex-end; justify-content: center; gap: 3px; padding: 9px 0; }
.brand-mark i { display: block; width: 4px; border-radius: 2px; background: #8f84ff; }
.brand-name { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; line-height: 1.1; }
.brand-sub { color: #6b7494; font-size: .78rem; }
.side-h { font-family: 'Sora', sans-serif; font-weight: 600; font-size: .86rem;
  color: #0f1b3d; margin: 18px 0 2px; padding-top: 14px; border-top: 1px solid #e6e8f2; }

/* Top bar */
.top { display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.top h1 { font-size: 1.55rem; margin: 0; font-weight: 700; }
.chip-model { background: #fff; border: 1px solid #e6e8f2; border-radius: 999px;
  padding: 6px 14px; font-size: .82rem; color: #6b7494; }
.chip-model b { color: #0f1b3d; font-weight: 600; }

/* Tabs as pills */
div[data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom: none; }
div[data-testid="stTabs"] button[role="tab"] { background: #fff; border: 1px solid #e6e8f2;
  border-radius: 999px; padding: 6px 18px; height: auto; }
div[data-testid="stTabs"] button[role="tab"] p { font-weight: 600; font-size: .9rem; }
div[data-testid="stTabs"] button[aria-selected="true"] { background: #0f1b3d; border-color: #0f1b3d; }
div[data-testid="stTabs"] button[aria-selected="true"] p { color: #fff; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

/* Verdict hero */
.hero { background: linear-gradient(125deg, #0f1b3d 0%, #1c2a6e 100%);
  border-radius: 24px; padding: 28px 32px; color: #fff; display: flex;
  flex-wrap: wrap; gap: 34px; align-items: center; margin: 14px 0 26px; }
.ring { position: relative; width: 200px; height: 200px; flex: none; }
.ring svg { width: 200px; height: 200px; }
.ring-mid { position: absolute; inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center; }
.ring-num { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.6rem; line-height: 1; }
.ring-lbl { color: #b8c0e8; font-size: .78rem; margin-top: 4px; }
.hero-body { flex: 1; min-width: 260px; }
.hero-body h2 { color: #fff; font-size: 2rem; margin: 0 0 6px; font-weight: 700; }
.hero-body p { color: #c9d0f2; margin: 0 0 16px; max-width: 460px; }
.signal { display: flex; align-items: flex-end; gap: 5px; height: 46px; margin-bottom: 4px; }
.signal i { display: block; width: 11px; border-radius: 3px; }
.signal-lbl { color: #b8c0e8; font-size: .78rem; }
.stats { display: grid; grid-template-columns: repeat(2, minmax(130px, 1fr)); gap: 10px; flex: none; }
.stat { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12);
  border-radius: 14px; padding: 12px 16px; }
.stat b { display: block; font-family: 'Sora', sans-serif; font-size: 1.25rem; font-weight: 700; }
.stat span { color: #b8c0e8; font-size: .76rem; }

/* Sections */
.sec h3 { font-size: 1.15rem; margin: 0 0 2px; font-weight: 700; }
.sec p { color: #6b7494; font-size: .88rem; margin: 0 0 14px; }

/* Levers */
.levers { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
.lever { background: #fff; border: 1px solid #e6e8f2; border-radius: 16px; padding: 16px; }
.lever.best { border: 2px solid #5b4bdb; }
.lever-t { font-weight: 600; font-size: .92rem; margin-bottom: 10px; min-height: 2.4em; }
.lever-n { font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; }
.lever-n span { color: #6b7494; padding: 0 4px; }
.delta { display: inline-block; border-radius: 999px; padding: 3px 10px;
  font-weight: 700; font-size: .78rem; }
.delta.down { background: #e3f8f0; color: #0b8a63; }
.delta.up { background: #ffe8ea; color: #c9343a; }
.flag { color: #5b4bdb; font-size: .74rem; font-weight: 700; margin-left: 8px; }

/* KPI cards */
.kpi { background: #fff; border: 1px solid #e6e8f2; border-radius: 16px; padding: 16px 20px; }
.kpi span { color: #6b7494; font-size: .82rem; font-weight: 500; }
.kpi b { display: block; font-family: 'Sora', sans-serif; font-size: 1.7rem;
  font-weight: 700; margin: 2px 0; font-variant-numeric: tabular-nums; }
.kpi small { color: #6b7494; font-size: .78rem; }

/* Info cards */
.info { background: #fff; border: 1px solid #e6e8f2; border-radius: 16px; padding: 18px 20px; height: 100%; }
.info h4 { margin: 0 0 8px; font-size: 1rem; }
.info ul { margin: 0; padding-left: 18px; color: #3a4466; font-size: .9rem; }
.info li { margin-bottom: 4px; }
.empty { background: #fff; border: 2px dashed #cfd3ea; border-radius: 20px;
  padding: 34px; text-align: center; color: #6b7494; margin-top: 10px; }
.empty b { color: #0f1b3d; font-family: 'Sora', sans-serif; font-size: 1.1rem; }

.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH.name}")
    loaded_model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "feature_columns" not in metadata:
        raise ValueError("model_meta.json must contain feature_columns.")
    return loaded_model, metadata


try:
    model, meta = load_artifacts()
    COLS = meta["feature_columns"]
except Exception as exc:
    st.error(f"Could not load model artifacts: {exc}")
    st.stop()


# ----------------------------------------------------------------------
# Feature preparation, scoring and explanation
# ----------------------------------------------------------------------
def prepare_input(raw, columns):
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = raw[RAW_COLS].copy()
    for col in NUMERIC_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[NUMERIC_COLS].isna().any().any():
        raise ValueError("Numeric customer fields contain invalid or missing values.")

    # Do not use drop_first=True when serving.
    encoded = pd.get_dummies(data)
    return encoded.reindex(columns=columns, fill_value=0).astype(float)


def score(raw):
    features = prepare_input(raw, COLS)
    if not hasattr(model, "predict_proba"):
        raise ValueError("This app requires a model with predict_proba().")

    probabilities = model.predict_proba(features)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)

    if classes is not None:
        idx = [i for i, v in enumerate(classes)
               if str(v).strip().lower() in {"yes", "1", "true", "churn"}]
        if idx:
            return probabilities[:, idx[0]]
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    raise ValueError("Unable to identify the churn class in model.classes_.")


def explain(customer_row):
    """Per-customer factor contributions (log-odds) for linear and XGBoost models."""
    features = prepare_input(customer_row, COLS)
    is_pipeline = hasattr(model, "steps")
    final = model.steps[-1][1] if is_pipeline else model

    if is_pipeline and len(model.steps) > 1:
        x = np.asarray(model[:-1].transform(features), dtype=float)
    else:
        x = features.to_numpy(dtype=float)

    if hasattr(final, "coef_"):
        values = final.coef_[0] * x[0]
    elif hasattr(final, "get_booster"):
        import xgboost as xgb

        contribs = final.get_booster().predict(
            xgb.DMatrix(x), pred_contribs=True, validate_features=False
        )
        values = contribs[0][:-1]  # last column is the bias term
    else:
        return None

    if len(values) != len(COLS):
        return None
    return pd.Series(values, index=COLS)


def band_of(p, threshold):
    return "HIGH" if p >= threshold else "WATCH" if p >= threshold / 2 else "LOW"


@st.cache_data(show_spinner="Scoring customers...")
def score_file(file_bytes: bytes) -> pd.DataFrame:
    data = pd.read_csv(io.BytesIO(file_bytes))
    missing = [c for c in RAW_COLS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("The uploaded CSV contains no customers.")
    # Blank TotalCharges (brand-new customers) count as zero.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["p_churn"] = np.round(score(data), 4)
    return data


def pretty(name: str) -> str:
    return name.replace("_", ": ", 1).replace("_", " ")


def tidy(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#eceef6",
                        domainColor="#d9dcec", tickColor="#d9dcec",
                        labelFont="Public Sans", titleFont="Public Sans")
        .configure(background="transparent")
    )


def kpi(label, value, note=""):
    html(f'<div class="kpi"><span>{label}</span><b>{value}</b><small>{note}</small></div>')


def section(title, text=""):
    html(f'<div class="sec"><h3>{title}</h3><p>{text}</p></div>')


# ----------------------------------------------------------------------
# Sidebar: customer profile
# ----------------------------------------------------------------------
with st.sidebar:
    html("""
    <div class="brand">
      <div class="brand-mark"><i style="height:7px"></i><i style="height:11px"></i>
        <i style="height:15px"></i><i style="height:19px"></i></div>
      <div><div class="brand-name">Churn Risk Advisor</div>
      <div class="brand-sub">Retention desk</div></div>
    </div>
    """)

    html('<div class="side-h">Plan and billing</div>')
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 4)
    monthly = st.number_input("Monthly charges (USD)", min_value=18.0, max_value=120.0,
                              value=85.0, step=0.5)
    payment = st.selectbox("Payment method", PAYMENTS)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    html('<div class="side-h">Services</div>')
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    services = []
    if internet != "No":
        services = st.multiselect("Add-on services", SERVICES, default=["StreamingTV"])
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    multi = "No phone service"
    if phone == "Yes":
        multi = st.radio("Multiple lines", ["No", "Yes"], horizontal=True)

    html('<div class="side-h">Household</div>')
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    senior = st.checkbox("Senior citizen")
    partner = st.checkbox("Has partner")
    dependents = st.checkbox("Has dependents")

    html('<div class="side-h">Decision rule</div>')
    threshold = st.slider(
        "Contact threshold",
        min_value=0.05, max_value=0.95,
        value=float(min(max(meta.get("threshold", 0.30), 0.05), 0.95)),
        step=0.05,
        help="Customers at or above this probability are marked HIGH. "
             "It changes the label, not the model's probability.",
    )

row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No",
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multi,
    "InternetService": internet,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,
}
for service in SERVICES:
    row[service] = ("No internet service" if internet == "No"
                    else "Yes" if service in services else "No")
customer = pd.DataFrame([row], columns=RAW_COLS)


# ----------------------------------------------------------------------
# Top bar
# ----------------------------------------------------------------------
cv_auc, cv_std = meta.get("cv_auc"), meta.get("cv_auc_std")
auc_text = f" &nbsp;·&nbsp; AUC <b>{float(cv_auc):.3f}</b>" if cv_auc is not None else ""
html(f"""
<div class="top">
  <h1>Customer churn risk</h1>
  <div class="chip-model">Model <b>{meta.get('model_name', 'Not recorded')}</b>{auc_text}</div>
</div>
""")

tab1, tab2, tab3 = st.tabs(["One customer", "Batch analysis", "About the model"])


# ----------------------------------------------------------------------
# Tab 1: single customer
# ----------------------------------------------------------------------
with tab1:
    try:
        p = float(score(customer)[0])
        band = band_of(p, threshold)
        color = BAND_COLOR[band]
        annual = monthly * 12
        at_risk = p * annual

        # Ring gauge geometry
        r = 76
        circ = 2 * math.pi * r
        ang = math.radians(-90 + threshold * 360)
        x1, y1 = 100 + (r - 9) * math.cos(ang), 100 + (r - 9) * math.sin(ang)
        x2, y2 = 100 + (r + 9) * math.cos(ang), 100 + (r + 9) * math.sin(ang)

        lit = int(np.clip(round((1 - p) * 5), 1, 5))
        bars = "".join(
            f'<i style="height:{h}px;background:{color if i < lit else "rgba(255,255,255,.18)"}"></i>'
            for i, h in enumerate([12, 20, 28, 36, 46])
        )

        html(f"""
        <div class="hero">
          <div class="ring">
            <svg viewBox="0 0 200 200">
              <circle cx="100" cy="100" r="{r}" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="14"/>
              <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="14"
                stroke-linecap="round" stroke-dasharray="{circ * p:.1f} {circ:.1f}"
                transform="rotate(-90 100 100)"/>
              <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#fff" stroke-width="3"/>
            </svg>
            <div class="ring-mid"><div class="ring-num">{p:.0%}</div>
              <div class="ring-lbl">chance of churn</div></div>
          </div>
          <div class="hero-body">
            <h2>{BAND_TITLE[band]}</h2>
            <p>{BAND_NOTE[band]} The white tick on the ring marks your {threshold:.0%} contact threshold.</p>
            <div class="signal">{bars}</div>
            <div class="signal-lbl">Retention signal: {lit} of 5 bars</div>
          </div>
          <div class="stats">
            <div class="stat"><b>${annual:,.0f}</b><span>Annual bill</span></div>
            <div class="stat"><b>${at_risk:,.0f}</b><span>Revenue at risk</span></div>
            <div class="stat"><b>{tenure} mo</b><span>Tenure</span></div>
            <div class="stat"><b>{BAND_ACTION[band]}</b><span>Suggested action</span></div>
          </div>
        </div>
        """)

        left, right = st.columns([1.15, 1], gap="large")

        # ---- Retention levers --------------------------------------------
        with left:
            section("What could lower the risk?",
                    "Model comparisons, not guaranteed effects. Each card changes one thing.")

            changes = [("Contract", v) for v in ["One year", "Two year"] if v != contract]
            if payment == "Electronic check":
                changes.append(("PaymentMethod", "Credit card (automatic)"))
            if internet != "No" and "TechSupport" not in services:
                changes.append(("TechSupport", "Yes"))

            results = []
            for feature, value in changes:
                alt_customer = customer.copy()
                alt_customer.loc[:, feature] = value
                new_p = float(score(alt_customer)[0])
                results.append((LEVER_TITLE.get((feature, value), f"{feature}: {value}"),
                                new_p, new_p - p))

            if results:
                results.sort(key=lambda t: t[2])
                cards = ""
                for i, (title, new_p, diff) in enumerate(results):
                    best = i == 0 and diff < 0
                    cls = "down" if diff < 0 else "up"
                    flag = '<span class="flag">Biggest drop</span>' if best else ""
                    cards += (
                        f'<div class="lever{" best" if best else ""}">'
                        f'<div class="lever-t">{title}</div>'
                        f'<div class="lever-n">{p:.1%}<span>→</span>{new_p:.1%}</div>'
                        f'<span class="delta {cls}">{diff * 100:+.1f} pts</span>{flag}</div>'
                    )
                html(f'<div class="levers">{cards}</div>')
            else:
                st.info("No alternative scenarios for this profile.")

        # ---- Drivers -----------------------------------------------------
        with right:
            section("What drives this score",
                    "Right pushes toward churn, left pushes away. Not a causal explanation.")
            try:
                contrib = explain(customer)
                if contrib is None:
                    st.info("Factor breakdown is not available for this model type.")
                else:
                    top = (contrib.reindex(contrib.abs().sort_values(ascending=False).index)
                           .head(8).rename("Contribution").reset_index()
                           .rename(columns={"index": "Factor"}))
                    top["Factor"] = top["Factor"].map(pretty)
                    chart = (
                        alt.Chart(top)
                        .mark_bar(cornerRadiusEnd=4, height=18)
                        .encode(
                            y=alt.Y("Factor:N", sort=alt.SortField("Contribution", order="descending"),
                                    title=None, axis=alt.Axis(labelLimit=220)),
                            x=alt.X("Contribution:Q", title="Effect on churn score"),
                            color=alt.condition(alt.datum.Contribution > 0,
                                                alt.value(ROSE), alt.value(MINT)),
                            tooltip=["Factor", alt.Tooltip("Contribution:Q", format="+.3f")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(tidy(chart), theme=None)
            except Exception as exc:
                st.info(f"Factor breakdown is unavailable: {exc}")

    except Exception as exc:
        st.error(f"Could not score this customer: {exc}")


# ----------------------------------------------------------------------
# Tab 2: batch analysis
# ----------------------------------------------------------------------
with tab2:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        section("Score a customer file",
                "Upload a CSV with the original Telco columns to see where risk is concentrated.")
    with head_r:
        if SAMPLE_PATH.exists():
            st.download_button("Download sample CSV", SAMPLE_PATH.read_bytes(),
                               file_name="sample_customers.csv", mime="text/csv")

    uploaded = st.file_uploader("Customer CSV file", type=["csv"], label_visibility="collapsed")
    use_sample = False
    if uploaded is None and SAMPLE_PATH.exists():
        use_sample = st.checkbox("Explore the built-in sample data instead")

    file_bytes = (uploaded.getvalue() if uploaded is not None
                  else SAMPLE_PATH.read_bytes() if use_sample else None)

    if file_bytes is None:
        html("""
        <div class="empty"><b>No file yet</b><br>
        Drop a CSV above, or tick the sample option to preview the analysis.</div>
        """)
    else:
        try:
            df = score_file(file_bytes)
            df["risk_band"] = [band_of(v, threshold) for v in df["p_churn"]]
            df["annual_value"] = df["MonthlyCharges"] * 12
            df["revenue_at_risk"] = df["p_churn"] * df["annual_value"]

            with st.expander("Filters"):
                f1, f2, f3 = st.columns(3)
                contracts = sorted(df["Contract"].dropna().unique())
                nets = sorted(df["InternetService"].dropna().unique())
                sel_contract = f1.multiselect("Contract", contracts, default=contracts)
                sel_internet = f2.multiselect("Internet service", nets, default=nets)
                sel_band = f3.multiselect("Risk band", ["HIGH", "WATCH", "LOW"],
                                          default=["HIGH", "WATCH", "LOW"])

            view = df[df["Contract"].isin(sel_contract)
                      & df["InternetService"].isin(sel_internet)
                      & df["risk_band"].isin(sel_band)].copy()

            if view.empty:
                st.warning("No customers match these filters.")
            else:
                high = view[view["p_churn"] >= threshold]
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    kpi("Customers", f"{len(view):,}", "In the current filter")
                with k2:
                    kpi("Above threshold", f"{len(high):,}", f"{len(high) / len(view):.1%} of customers")
                with k3:
                    kpi("Average risk", f"{view['p_churn'].mean():.1%}", "Mean churn probability")
                with k4:
                    kpi("Revenue at risk", f"${view['revenue_at_risk'].sum():,.0f}",
                        "Probability × monthly × 12")

                st.write("")
                c1, c2 = st.columns(2, gap="large")

                with c1:
                    section("Risk distribution", "How many customers fall at each probability.")
                    hist = alt.Chart(view).mark_bar(color=VIOLET, cornerRadiusEnd=3).encode(
                        x=alt.X("p_churn:Q", bin=alt.Bin(maxbins=25), title="Churn probability",
                                axis=alt.Axis(format="%")),
                        y=alt.Y("count()", title="Customers"),
                    )
                    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
                        color=ROSE, strokeDash=[6, 4], size=2).encode(x="t:Q")
                    st.altair_chart(tidy((hist + rule).properties(height=290)), theme=None)

                with c2:
                    section("Customers by risk band", "Hover a bar to see revenue at risk.")
                    bands = (view.groupby("risk_band")
                             .agg(Customers=("p_churn", "size"), Revenue=("revenue_at_risk", "sum"))
                             .reindex(["HIGH", "WATCH", "LOW"]).fillna(0).reset_index())
                    bar = alt.Chart(bands).mark_bar(cornerRadiusEnd=4, height=34).encode(
                        y=alt.Y("risk_band:N", sort=["HIGH", "WATCH", "LOW"], title=None),
                        x=alt.X("Customers:Q"),
                        color=alt.Color("risk_band:N", legend=None,
                                        scale=alt.Scale(domain=list(BAND_COLOR),
                                                        range=list(BAND_COLOR.values()))),
                        tooltip=["risk_band", "Customers", alt.Tooltip("Revenue:Q", format="$,.0f")],
                    ).properties(height=290)
                    st.altair_chart(tidy(bar), theme=None)

                st.write("")
                section("Where is the risk concentrated?",
                        "Smaller groups give less reliable averages, so check the customer count.")
                view["tenure_group"] = pd.cut(
                    view["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
                    labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"])
                segments = {"Contract": "Contract", "Internet": "InternetService",
                            "Payment": "PaymentMethod", "Tenure": "tenure_group"}
                for seg_tab, (label, col) in zip(st.tabs(list(segments)), segments.items()):
                    with seg_tab:
                        g = (view.groupby(col, observed=True)
                             .agg(Customers=("p_churn", "size"), Average_risk=("p_churn", "mean"),
                                  High_risk=("p_churn", lambda s: int((s >= threshold).sum())),
                                  Revenue_at_risk=("revenue_at_risk", "sum"))
                             .reset_index())
                        g[col] = g[col].astype(str)
                        seg = alt.Chart(g).mark_bar(color=VIOLET, cornerRadiusEnd=4, height=26).encode(
                            y=alt.Y(f"{col}:N", title=None,
                                    sort=None if col == "tenure_group" else "-x"),
                            x=alt.X("Average_risk:Q", title="Average churn probability",
                                    axis=alt.Axis(format="%")),
                            tooltip=[col, "Customers", alt.Tooltip("Average_risk:Q", format=".1%"),
                                     "High_risk", alt.Tooltip("Revenue_at_risk:Q", format="$,.0f")],
                        ).properties(height=max(140, 52 * len(g)))
                        st.altair_chart(tidy(seg), theme=None)

                st.write("")
                section("Contact list", "Ordered by revenue at risk, so the biggest accounts come first.")
                ranked = view.sort_values("revenue_at_risk", ascending=False)
                cols = ([c for c in ["customerID"] if c in ranked.columns]
                        + ["p_churn", "risk_band", "revenue_at_risk", "Contract", "tenure",
                           "MonthlyCharges", "InternetService", "PaymentMethod"])
                st.dataframe(
                    ranked[cols].head(50), hide_index=True,
                    column_config={
                        "p_churn": st.column_config.ProgressColumn(
                            "Churn probability", format="percent", min_value=0, max_value=1),
                        "risk_band": "Risk band",
                        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.0f"),
                        "tenure": "Tenure (mo)",
                        "MonthlyCharges": st.column_config.NumberColumn("Monthly", format="$%.2f"),
                    },
                )
                st.caption("Showing the top 50. The download includes every customer in the current filter.")

                export = ranked.drop(columns=["tenure_group"], errors="ignore")
                export["contact"] = np.where(export["p_churn"] >= threshold, "Yes", "")
                st.download_button("Download scored CSV", export.to_csv(index=False).encode("utf-8"),
                                   file_name="scored_customers.csv", mime="text/csv", type="primary")

        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")


# ----------------------------------------------------------------------
# Tab 3: about
# ----------------------------------------------------------------------
with tab3:
    section("About the model", "What it is, how well it ranks customers, and where it should not be used.")
    m1, m2, m3 = st.columns(3)
    with m1:
        kpi("Model", str(meta.get("model_name", "Not recorded")), "Saved with the app")
    with m2:
        kpi("Cross-validation AUC", f"{float(cv_auc):.3f}" if cv_auc is not None else "Not recorded",
            "Higher ranks churners better")
    with m3:
        kpi("AUC spread", f"± {float(cv_std):.3f}" if cv_std is not None else "Not recorded",
            "Variation across folds")

    st.write("")
    a, b, c = st.columns(3, gap="medium")
    with a:
        html("""<div class="info"><h4>Intended use</h4><ul>
        <li>Help a telecom retention team decide who to review first.</li>
        <li>Support human judgement. Do not replace it.</li></ul></div>""")
    with b:
        html("""<div class="info"><h4>Limitations</h4><ul>
        <li>Trained on a US telecom dataset. Not validated on Pakistani telecom customers.</li>
        <li>What-if results show associations, not proof that a change prevents churn.</li>
        <li>Revenue at risk ignores discounts, retention cost, and customer lifetime.</li></ul></div>""")
    with c:
        html("""<div class="info"><h4>Not for</h4><ul>
        <li>Credit decisions.</li><li>Service denial.</li>
        <li>Other high-impact automated decisions.</li></ul></div>""")

    st.write("")
    with st.expander("Saved metadata"):
        st.json({k: v for k, v in meta.items() if k != "feature_columns"})
