import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Page + simple styling
# -------------------------
st.set_page_config(page_title="Credit Risk Demo", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.kpi-card {
  padding: 14px 16px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.badge {
  display:inline-block; padding: 7px 12px; border-radius: 999px;
  font-weight: 700; font-size: 0.9rem;
}
.badge-ok  {background: rgba(0, 200, 140, .15); border: 1px solid rgba(0,200,140,.35);}
.badge-bad {background: rgba(255, 80, 80, .15); border: 1px solid rgba(255,80,80,.35);}
.small {opacity: .85; font-size: 0.95rem;}
hr {border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Feature engineering (same as notebook)
# -------------------------
def apply_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ordinal mapping
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    default_map = {'Y': 1, 'N': 0}
    df["loan_grade"] = df["loan_grade"].map(grade_map)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(default_map)

    # Logs
    df["person_income_log"] = np.log1p(df["person_income"])
    df["loan_amnt_log"] = np.log1p(df["loan_amnt"])

    # Binning
    inc_bins  = [0, 0.20, 0.35, 0.50, np.inf]
    rate_bins = [0, 8, 12, 16, np.inf]
    emp_bins  = [0, 1, 3, 7, 15, np.inf]
    age_bins  = [0, 25, 35, 50, np.inf]

    df["loan_percent_group"] = pd.cut(
        df["loan_percent_income"], bins=inc_bins, labels=[1, 2, 3, 4], include_lowest=True
    ).astype(float)

    df["int_rate_group"] = pd.cut(
        df["loan_int_rate"], bins=rate_bins, labels=[1, 2, 3, 4], include_lowest=True
    ).astype(float)

    df["emp_length_group"] = pd.cut(
        df["person_emp_length"], bins=emp_bins, labels=[1, 2, 3, 4, 5], include_lowest=True
    ).astype(float)

    df["age_group"] = pd.cut(
        df["person_age"], bins=age_bins, labels=[1, 2, 3, 4], include_lowest=True
    ).astype(float)

    # Ensure exists
    if "cb_person_cred_hist_length" not in df.columns:
        df["cb_person_cred_hist_length"] = np.nan

    return df

# -------------------------
# Load model assets
# -------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("model.joblib")
    thr = joblib.load("threshold.joblib")
    return model, float(thr)

model, saved_thr = load_assets()

# -------------------------
# Header
# -------------------------
st.markdown("## 📊 Credit Risk Prediction — Demo")
st.markdown('<div class="small">LightGBM Pipeline + CV threshold • Portfolio showcase</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Decision Threshold")
thr = st.sidebar.slider("Threshold", 0.05, 0.95, saved_thr, 0.01)

st.sidebar.markdown("**How to read**")
st.sidebar.write("- Probability = risk score (class=1)\n- Threshold controls decision")

show_debug = st.sidebar.checkbox("Show debug (engineered features)", value=False)

# -------------------------
# Main layout
# -------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Input Features")
    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            person_age = st.number_input("person_age", min_value=18, max_value=80, value=30)
            person_income = st.number_input("person_income", min_value=0, value=50000)
            person_emp_length = st.number_input("person_emp_length", min_value=0.0, value=5.0)
            loan_amnt = st.number_input("loan_amnt", min_value=0, value=10000)
            loan_int_rate = st.number_input("loan_int_rate", min_value=0.0, value=12.0)
            cb_person_cred_hist_length = st.number_input("cb_person_cred_hist_length", min_value=0, value=10)

        with col2:
            loan_percent_income = st.number_input("loan_percent_income", min_value=0.0, value=0.2)
            loan_grade = st.selectbox("loan_grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
            cb_person_default_on_file = st.selectbox("cb_person_default_on_file", ["Y", "N"], index=1)
            person_home_ownership = st.selectbox("person_home_ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=0)
            loan_intent = st.selectbox(
                "loan_intent",
                ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
                index=3
            )

        submitted = st.form_submit_button("Predict")

with right:
    st.subheader("Decision Panel")

    if not submitted:
        st.info("Fill the form and click **Predict** to see results.")
    else:
        # Raw input (as trained)
        X = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file,
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
        }])

        # Apply FE required by model
        X = apply_fe(X)

        # Predict
        proba = float(model.predict_proba(X)[:, 1][0])
        pred = int(proba >= thr)

        # Status badge
        label = "RISKY (Reject / Review)" if pred == 1 else "NOT RISKY (Approve)"
        badge_cls = "badge-bad" if pred == 1 else "badge-ok"
        st.markdown(f'<span class="badge {badge_cls}">{label}</span>', unsafe_allow_html=True)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Risk Probability", f"{proba:.3f}")
        k2.metric("Threshold", f"{thr:.2f}")
        k3.metric("Class", str(pred))

        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.write("**Quick explanation**")
        st.write(
            "Probability is the estimated chance of being class=1 (risky). "
            "If probability ≥ threshold, the case is flagged as risky."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Optional debug
        if show_debug:
            st.markdown("### Debug — Engineered Features")
            st.dataframe(X, use_container_width=True)

        # Bonus: tiny business impact toy
        st.markdown("### 💰 Business Impact (toy)")
        lgd = st.slider("LGD (loss given default)", 0.10, 0.90, 0.60, 0.05)
        fp_cost = st.slider("Opportunity cost (FP)", 0.00, 0.50, 0.10, 0.01)

        expected_loss_if_approve = proba * lgd
        expected_loss_if_reject = (1 - proba) * fp_cost

        b1, b2 = st.columns(2)
        b1.metric("Expected loss if Approve", f"{expected_loss_if_approve:.3f}")
        b2.metric("Expected loss if Reject", f"{expected_loss_if_reject:.3f}")

        st.caption("Toy model for storytelling. Real business impact requires calibrated costs and policy constraints.")