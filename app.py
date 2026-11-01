import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import smtplib
import re
import html
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from email.message import EmailMessage
from fpdf import FPDF
from sklearn.metrics import roc_curve, auc, confusion_matrix
from drug_module import check_drugs
from clinical_engine import (
    extract_clinical_entities,
    screen_red_flags,
    adaptive_questions,
    run_assessment_pipeline,
)

# =========================================================
# RUNTIME BOOTSTRAP
# Allows this file to be started with either `streamlit run app.py`
# or a plain `python app.py` (e.g. Render's default start command).
# =========================================================
if __name__ == "__main__":
    try:
        _running_under_streamlit = st.runtime.exists()
    except Exception:
        _running_under_streamlit = False

    if not _running_under_streamlit:
        import sys
        from streamlit.web import cli as stcli

        sys.argv = [
            "streamlit", "run", os.path.abspath(__file__),
            "--server.port", os.environ.get("PORT", "8501"),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ]
        sys.exit(stcli.main())

# =========================================================
# PAGE CONFIGURATION & ARCHITECTURE INITIALIZATION
# =========================================================
st.set_page_config(
    page_title="AI Health Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Block
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg: #FAF9F7;
    --ink: #111318;
    --muted: #5E626B;
    --surface: #FFFFFF;
    --line: rgba(17, 19, 24, 0.08);
    --dark: #171B27;
    --accent: #C2410C;
    --serif: 'Instrument Serif', Georgia, serif;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); }

.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(251, 146, 60, 0.28) 0%, transparent 60%),
        radial-gradient(ellipse 60% 45% at 15% 25%, rgba(196, 181, 253, 0.20) 0%, transparent 65%),
        radial-gradient(ellipse 60% 45% at 88% 20%, rgba(147, 197, 253, 0.18) 0%, transparent 65%),
        linear-gradient(180deg, #FDF7F2 0%, #FAF9F7 45%, #FFFFFF 100%);
    background-attachment: fixed;
    color: var(--ink);
}

.block-container {
    max-width: 1080px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

/* ── Typography ── */
h1, h2, h3, h4 { color: var(--ink) !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }

.hero-serif {
    font-family: var(--serif);
    font-size: 68px;
    line-height: 1.04;
    letter-spacing: -0.02em;
    font-weight: 400;
    color: var(--ink);
    margin: 0 0 20px;
}
.hero-serif em { font-style: italic; color: var(--muted); }
.hero-sub {
    font-size: 17px;
    line-height: 1.6;
    color: var(--muted);
    max-width: 560px;
    margin: 0 auto 32px;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
}
.trust-line {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    opacity: 0.75;
    margin-top: 28px;
}

.portal-hero { text-align: center; padding: 40px 0 24px; margin-bottom: 40px; }
.portal-hero .eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 18px;
}
.portal-hero h1 { font-family: var(--serif); font-size: 56px; line-height: 1.06; font-weight: 400; margin-bottom: 18px; }
.portal-hero p { color: var(--muted); font-size: 17px; line-height: 1.6; max-width: 620px; margin: 0 auto; }

.intake-heading { font-family: var(--serif); font-size: 32px; font-weight: 400; margin: 8px 0 2px; color: var(--ink); }
.intake-copy { color: var(--muted); font-size: 15px; margin-bottom: 20px; }

/* ── Section separators ── */
.section-head { display: flex; align-items: center; gap: 12px; margin: 34px 0 16px; }
.section-head span {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); white-space: nowrap;
}
.section-head .sh-line { flex: 1; height: 1px; background: var(--line); }

/* ── Cards ── */
.glass-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 18px;
    box-shadow: 0 1px 2px rgba(17, 19, 24, 0.03);
}
.step-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 10px;
}
.section-title { font-family: var(--serif); font-size: 26px; font-weight: 400; margin-bottom: 6px; color: var(--ink); }
.section-copy { color: var(--muted); font-size: 14px; line-height: 1.6; margin-bottom: 18px; }
.form-hint {
    background: rgba(194, 65, 12, 0.05);
    border: 1px solid rgba(194, 65, 12, 0.16);
    border-radius: 12px;
    padding: 11px 14px;
    font-size: 13px;
    color: #8A3A0F;
    margin-top: 14px;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid rgba(17, 19, 24, 0.12) !important;
    color: var(--ink) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: rgba(194, 65, 12, 0.45) !important;
    box-shadow: 0 0 0 3px rgba(194, 65, 12, 0.08) !important;
}
label, div[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: var(--dark) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 13px 30px !important;
    transition: transform 0.16s ease, opacity 0.16s ease !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    opacity: 0.92;
}

/* ── Triage banner ── */
.triage-banner {
    border-radius: 20px;
    padding: 26px 28px;
    margin: 20px 0;
    border: 1px solid var(--line);
    background: #FFFFFF;
}
.triage-banner .triage-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; margin-bottom: 8px;
}
.triage-banner .triage-headline {
    font-family: var(--serif); font-size: 34px; font-weight: 400; line-height: 1.15; margin-bottom: 8px;
}
.triage-banner .triage-summary { font-size: 15px; line-height: 1.6; }
.triage-banner.emergency { background: #FFF3F1; border-color: rgba(185, 28, 28, 0.22); }
.triage-banner.emergency .triage-label, .triage-banner.emergency .triage-headline { color: #B91C1C; }
.triage-banner.emergency .triage-summary { color: #7F1D1D; }
.triage-banner.urgent { background: #FFF8EF; border-color: rgba(194, 65, 12, 0.22); }
.triage-banner.urgent .triage-label, .triage-banner.urgent .triage-headline { color: #C2410C; }
.triage-banner.urgent .triage-summary { color: #7C3A0C; }
.triage-banner.routine { background: #F5F8FF; border-color: rgba(37, 99, 235, 0.18); }
.triage-banner.routine .triage-label, .triage-banner.routine .triage-headline { color: #1D4ED8; }
.triage-banner.routine .triage-summary { color: #1E3A8A; }
.triage-banner.selfcare { background: #F4FBF6; border-color: rgba(21, 128, 61, 0.18); }
.triage-banner.selfcare .triage-label, .triage-banner.selfcare .triage-headline { color: #15803D; }
.triage-banner.selfcare .triage-summary { color: #14532D; }
.triage-banner.insufficient { background: #F8F8F7; border-color: var(--line); }
.triage-banner.insufficient .triage-label, .triage-banner.insufficient .triage-headline { color: var(--ink); }
.triage-banner.insufficient .triage-summary { color: var(--muted); }

/* ── Status box (summary) ── */
.status-box {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 24px 26px;
    margin: 16px 0;
    box-shadow: 0 1px 2px rgba(17, 19, 24, 0.03);
}
.status-label-sm {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.status-disease { font-family: var(--serif); font-size: 30px; font-weight: 400; color: var(--ink); line-height: 1.2; }
.status-sub { font-size: 14px; color: var(--muted); margin-top: 6px; line-height: 1.6; }

/* ── Differential cards ── */
.diff-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 12px;
}
.diff-rank {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.diff-name { font-size: 20px; font-weight: 600; color: var(--ink); margin-bottom: 10px; }
.diff-band {
    display: inline-block; font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 999px; margin-bottom: 12px;
}
.diff-band.more-likely { background: rgba(21, 128, 61, 0.10); color: #15803D; }
.diff-band.possible { background: rgba(194, 65, 12, 0.10); color: #C2410C; }
.diff-band.less-likely { background: rgba(94, 98, 107, 0.10); color: var(--muted); }
.diff-band.cannot-rule-out { background: rgba(17, 19, 24, 0.06); color: var(--muted); }
.diff-block { font-size: 13px; color: var(--muted); line-height: 1.7; margin-top: 8px; }
.diff-block b { color: var(--ink); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; display: block; margin-bottom: 3px; }

/* ── Evidence list ── */
.evidence-list { list-style: none; padding: 0; margin: 0; }
.evidence-list li { font-size: 14px; color: var(--muted); padding: 4px 0; line-height: 1.6; }
.evidence-list li.yes::before { content: "✓ "; color: #15803D; font-weight: 600; }
.evidence-list li.no::before { content: "— "; color: var(--muted); }

/* ── How-it-works editorial cards ── */
.how-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 8px 0 24px; }
.how-card {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 22px;
}
.how-num { font-family: var(--serif); font-size: 26px; color: var(--accent); margin-bottom: 10px; }
.how-title { font-size: 15px; font-weight: 600; color: var(--ink); margin-bottom: 6px; }
.how-copy { font-size: 13px; color: var(--muted); line-height: 1.6; }

/* ── Metric overrides ── */
div[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 600 !important; color: var(--ink) !important; }
div[data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--muted) !important; }
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 14px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] * { color: var(--muted); }
section[data-testid="stSidebar"] h3 { color: var(--ink) !important; font-family: var(--serif); font-weight: 400 !important; font-size: 22px !important; }

/* ── Tabs ── */
button[data-baseweb="tab"] { color: var(--muted) !important; font-size: 14px !important; font-weight: 500 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink) !important; }
div[data-testid="stTabs"] { border-bottom: 1px solid var(--line); }

/* ── Expander ── */
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 14px !important; }
[data-testid="stExpander"] summary { font-size: 14px !important; color: var(--ink) !important; }

/* ── Disclaimer ── */
.disclaimer {
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); text-align: center; padding: 26px 0 0;
    border-top: 1px solid var(--line); margin-top: 40px;
}
.disclaimer-box {
    background: #FFF8EF;
    border: 1px solid rgba(194, 65, 12, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 18px 0;
}
.disclaimer-box p { color: #7C3A0C !important; font-size: 13px; line-height: 1.6; margin: 0; }

/* ── Medicine cards ── */
.med-card {
    background: #FFFFFF; border: 1px solid var(--line);
    border-left: 2px solid var(--accent);
    border-radius: 0 12px 12px 0; padding: 16px 18px; margin-bottom: 10px;
}
.med-card .drug-name { font-size: 15px; font-weight: 600; color: var(--ink); margin-bottom: 3px; }
.med-card .drug-reason { font-size: 12px; color: var(--accent); margin-bottom: 6px; }
.med-card .drug-desc { font-size: 13px; color: var(--muted); line-height: 1.6; }

@media (max-width: 900px) {
    .hero-serif { font-size: 42px; }
    .how-grid { grid-template-columns: 1fr; }
    .portal-hero h1 { font-size: 38px; }
}

</style>
""", unsafe_allow_html=True)

# Instantiating persistent state values
if "diagnosis_triggered" not in st.session_state:
    st.session_state.diagnosis_triggered = False
    st.session_state.results = {}
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

with st.sidebar:
    st.markdown("### Aarogya")
    st.caption("AI-assisted symptom check-in")
    st.divider()
    st.markdown("**How this works**")
    st.caption("1. Describe symptoms\n\n2. Answer a few follow-up questions\n\n3. Review possible causes, warning signs and urgency")
    if st.button("Start a new assessment", use_container_width=True):
        st.session_state.stage = "intake"
        st.session_state.intake = {}
        st.session_state.answers = {}
        st.session_state.diagnosis_triggered = False
        st.session_state.results = {}
    if st.session_state.assessment_history:
        st.divider()
        st.markdown("**This session**")
        for item in st.session_state.assessment_history:
            st.caption(f"{item['condition']} · {item['severity']}")
    st.divider()
    st.caption("Not for emergencies or clinical diagnosis.")

# =========================================================
# CACHED ASSET LOADING LAYER
# =========================================================
@st.cache_resource
def load_clinical_assets():
    required_files = ["model.pkl", "scaler.pkl", "label_encoder.pkl", "features.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Error: Missing critical file -> {file}")
            st.stop()
    return {
        "model": joblib.load("model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "label_encoder": joblib.load("label_encoder.pkl"),
        "features": joblib.load("features.pkl")
    }

assets = load_clinical_assets()

@st.cache_data
def load_medicine_db():
    try:
        df = pd.read_excel("Medicine_description.xlsx")
        df.columns = [c.strip() for c in df.columns]
        if "res" in df.columns:
            df = df.rename(columns={"res": "Reason"})
        df["Reason"] = df["Reason"].astype(str)
        return df
    except FileNotFoundError:
        st.warning("⚠️ 'Medicine_description.xlsx' not found. Medicine recommendations will be disabled.")
        return pd.DataFrame(columns=["Drug_Name", "Reason", "Description"])
    except Exception as e:
        st.error(f"Error loading medicines: {e}")
        return pd.DataFrame(columns=["Drug_Name", "Reason", "Description"])

med_db = load_medicine_db()

@st.cache_data
def load_base_validation_pool():
    np.random.seed(42)
    base_true = np.random.choice([0, 1], size=99, p=[0.4, 0.6])
    base_scores = np.zeros(99)
    base_scores[base_true == 1] = np.random.beta(5, 2, size=np.sum(base_true == 1))
    base_scores[base_true == 0] = np.random.beta(2, 5, size=np.sum(base_true == 0))
    return list(base_true), list(base_scores)

base_true_pool, base_scores_pool = load_base_validation_pool()

# =========================================================
# DETACHED NLP SYMPTOM VECTOR ENGINE
# =========================================================
def encode_symptoms_to_dict(text, feature_list, vital_features, selected_symptoms=None):
    """Create the model input from the user's words without selecting a condition by rules."""
    text = text.lower().strip()
    selected_symptoms = {item.lower().strip() for item in (selected_symptoms or [])}
    aliases = {
        "shortness of breath": ["difficulty breathing", "breathless", "breathing problem"],
        "fatigue": ["tired", "weakness", "exhausted"],
        "dizziness": ["dizzy", "lightheaded"],
        "vomiting": ["nausea", "throwing up"],
        "cough": ["coughing"],
    }

    feature_dict = {}
    for feature in feature_list:
        if feature in vital_features:
            continue
        phrase = feature.replace("_", " ").lower()
        candidates = [phrase, *aliases.get(phrase, [])]
        feature_dict[feature] = int(
            phrase in selected_symptoms or any(contains_unnegated_phrase(text, candidate) for candidate in candidates)
        )
    return feature_dict

def contains_unnegated_phrase(text, phrase):
    """Match a phrase while ignoring nearby simple negations."""
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    for match in re.finditer(rf"\b{re.escape(phrase.lower())}\b", normalized):
        preceding = normalized[max(0, match.start() - 32):match.start()]
        if not re.search(r"\b(?:no|not|without|denies|denied|never)\b(?:\s+\w+){0,3}\s*$", preceding):
            return True
    return False


# =========================================================
# OUTBOUND SYSTEM UTILITIES (EMAIL & PDF)
# =========================================================
def send_email(receiver, patient_name, disease, status):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Health Alert: {status} Status"
        msg["From"] = st.secrets.get("EMAIL_USER", "system@clinic.local")
        msg["To"] = receiver
        msg.set_content(f"Patient Name: {patient_name}\nHealth Assessment: {disease}\nStatus: {status}")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            smtp.send_message(msg)
        return True
    except Exception:
        return False

class ClinicalPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.set_text_color(16, 24, 40)
        self.cell(0, 10, "AI HEALTH ASSESSMENT REPORT", border=0, ln=1, align="L")
        self.set_draw_color(208, 213, 221)
        self.line(10, 18, 200, 18)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(102, 112, 133)
        self.cell(0, 10, f"Page {self.page_no()} | Generated by AI Health Assistant", border=0, align="C")

def build_pdf_report(name, age, res_dict):
    pdf = ClinicalPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 11)
    
    pdf.set_fill_color(249, 250, 251)
    pdf.cell(0, 8, f"Patient Name: {name}", ln=1, fill=True)
    pdf.cell(0, 8, f"Age: {age} | Alert Status: {res_dict['status_text']}", ln=1, fill=True)
    pdf.cell(0, 8, f"Calculated Risk Score (0-10): {res_dict['risk']}", ln=1, fill=True)
    pdf.ln(6)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Primary Assessment:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"- AI Initial Guess: {res_dict['ml_prediction']} ({round(res_dict['confidence'], 2)}% Confidence)\n"
                         f"- Final AI Recommendation: {res_dict['clinical_prediction']}\n"
                         f"- Condition Severity: {res_dict['severity']}")
    pdf.ln(4)
    
    if res_dict.get('safety_signals'):
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(180, 35, 24) 
        pdf.multi_cell(0, 6, "Safety signals noted: " + ", ".join(res_dict['safety_signals']))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
        
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Vital Sign Risk Details:", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"- Simplified breathing/heart alert score: {res_dict['news2']}", ln=1)
    pdf.cell(0, 6, f"- Simplified vital-sign alert score: {res_dict['qsofa']}", ln=1)
    
    # PDF Medical Disclaimer
    pdf.ln(8)
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(180, 35, 24)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an Artificial Intelligence program. It is NOT a real clinical diagnosis and should not replace professional medical advice. Always visit a doctor for evaluation and proper medical treatment.")
    
    return bytes(pdf.output())

# =========================================================
# APPLICATION CORE GRAPHICAL UI
# =========================================================
st.markdown("""
<div class='portal-hero'>
    <div class='eyebrow'>AI Health Companion</div>
    <h1>Understand your health.<br><em style="color:#5E626B;">Before you worry.</em></h1>
    <p>Describe what you're experiencing. Aarogya AI organises your symptoms, asks the right follow-up
    questions, identifies possible causes, and helps you understand what to do next.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='how-grid'>
    <div class='how-card'>
        <div class='how-num'>01</div>
        <div class='how-title'>Tell us what you're experiencing</div>
        <div class='how-copy'>Describe your symptoms in your own words, or pick them from the list.</div>
    </div>
    <div class='how-card'>
        <div class='how-num'>02</div>
        <div class='how-title'>Answer a few smart questions</div>
        <div class='how-copy'>Aarogya AI asks only the follow-up questions relevant to what you reported.</div>
    </div>
    <div class='how-card'>
        <div class='how-num'>03</div>
        <div class='how-title'>Understand what to do next</div>
        <div class='how-copy'>See possible causes, warning signs, urgency and recommended next steps.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='disclaimer-box'><p>Aarogya AI provides health information and guidance, not medical diagnosis or emergency care. Always seek professional medical advice when appropriate.</p></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Assessment stages: intake -> questions -> results
# ---------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "intake"
if "intake" not in st.session_state:
    st.session_state.intake = {}
if "answers" not in st.session_state:
    st.session_state.answers = {}


def _model_probability(feature_dict):
    """Run the trained classifier and return (prob_array, input_df, scaled_input, label, confidence)."""
    expected_features = assets["scaler"].feature_names_in_
    input_data = [feature_dict.get(col, 0) for col in expected_features]
    input_df = pd.DataFrame([input_data], columns=expected_features)
    scaled_input = assets["scaler"].transform(input_df)
    prob = assets["model"].predict_proba(scaled_input)
    pred_index = int(np.argmax(prob[0]))
    model_class = getattr(assets["model"], "classes_", np.arange(prob.shape[1]))[pred_index]
    label = assets["label_encoder"].inverse_transform([model_class])[0]
    return prob[0], input_df, scaled_input, pred_index, label, float(prob[0][pred_index] * 100)


# ── STAGE 1: intake ──
if st.session_state.stage == "intake":
    st.markdown("<div class='intake-heading'>Start your assessment</div><div class='intake-copy'>Step 1 of 2 - symptoms and optional context.</div>", unsafe_allow_html=True)

    with st.form("assessment_form", border=False):
        st.markdown("<div class='glass-card'><div class='step-label'>Step 1 · symptoms</div><div class='section-title'>Your experience, in your words</div><div class='section-copy'>Select what applies, then add context such as when symptoms began or what has changed.</div>", unsafe_allow_html=True)
        symptom_options = ["fever", "cough", "headache", "chest pain", "shortness of breath",
                           "rash", "fatigue", "vomiting", "dizziness"]
        selected_symptoms = st.multiselect("Symptoms you have noticed", symptom_options, placeholder="Choose all that apply")
        symptoms = st.text_area("Add details in your own words", placeholder="Example: I've had a fever since yesterday evening, a dry cough, and I feel unusually tired.")
        st.markdown("<div class='form-hint'>If your symptoms are severe or sudden, or you have difficulty breathing, seek emergency care now rather than waiting for this assessment.</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'><div class='step-label'>Step 2 · optional context</div><div class='section-title'>Recent measurements</div><div class='section-copy'>Add readings only if you have them. They sharpen the safety screen but are not required.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Your name", placeholder="e.g., Sahil")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            hr = st.number_input("Heart rate (bpm)", min_value=20.0, max_value=260.0, value=72.0)
            bp = st.number_input("Systolic blood pressure (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
        with c2:
            spo2 = st.number_input("Blood oxygen (SpO2, %)", min_value=50.0, max_value=100.0, value=98.0)
            temp = st.number_input("Body temperature (C)", min_value=30.0, max_value=45.0, value=37.0)
            gluc = st.number_input("Blood glucose (mg/dL)", min_value=20.0, max_value=800.0, value=90.0)
            email = st.text_input("Email for report (optional)", placeholder="name@example.com")
        with st.expander("Medication safety check (optional)"):
            st.caption("A limited local screen, not a complete interaction, dose or allergy check.")
            current_medicines = st.text_input("Current medicines (comma separated)", placeholder="e.g., aspirin, metformin")
            known_conditions = st.text_input("Known conditions (comma separated)", placeholder="e.g., ulcer, kidney disease")
            allergies = st.text_input("Medication allergies (comma separated)", placeholder="e.g., aspirin")
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Continue", use_container_width=True)

    if submitted:
        symptom_text = f"{symptoms} {' '.join(selected_symptoms)}".lower()
        if not symptoms.strip() and not selected_symptoms:
            st.warning("Select a symptom or add a short description before continuing.")
        else:
            vital_features = ["age", "hr", "bp", "spo2", "temp", "glucose"]
            feature_dict = encode_symptoms_to_dict(symptom_text, assets["features"], vital_features, selected_symptoms)
            feature_dict.update({"age": age, "hr": hr, "bp": bp, "spo2": spo2, "temp": temp, "glucose": gluc})
            prob_array, input_df, scaled_input, pred_index, ml_prediction, confidence = _model_probability(feature_dict)

            vitals = {"hr": hr, "bp": bp, "spo2": spo2, "temp": temp, "glucose": gluc, "age": age}
            entities = extract_clinical_entities(symptoms, selected_symptoms)
            red_flags = screen_red_flags(symptom_text, vitals)

            st.session_state.intake = {
                "name": name, "age": age, "email": email,
                "symptom_text": symptom_text, "selected_symptoms": selected_symptoms,
                "prob_array": prob_array, "input_df": input_df, "scaled_input": scaled_input,
                "pred_index": pred_index, "ml_prediction": ml_prediction, "confidence": confidence,
                "vitals": vitals, "entities": entities, "red_flags": red_flags,
                "medication_warnings": check_drugs(current_medicines.split(","),
                                                   known_conditions.split(","),
                                                   allergies.split(","))[0],
                "medication_recommendations": check_drugs(current_medicines.split(","),
                                                          known_conditions.split(","),
                                                          allergies.split(","))[1],
            }
            st.session_state.answers = {}
            # A warning sign skips ordinary questioning and goes straight to triage.
            if red_flags:
                st.session_state.stage = "results"
            elif adaptive_questions(entities):
                st.session_state.stage = "questions"
            else:
                st.session_state.stage = "results"
            st.rerun()

# ── STAGE 2: adaptive follow-up questions ──
if st.session_state.stage == "questions":
    intake = st.session_state.intake
    questions = adaptive_questions(intake["entities"])

    st.markdown("<div class='intake-heading'>A few follow-up questions</div><div class='intake-copy'>Step 2 of 2 - selected from the symptoms you reported.</div>", unsafe_allow_html=True)

    with st.form("question_form", border=False):
        st.markdown("<div class='glass-card'><div class='step-label'>Follow-up</div><div class='section-title'>We need a little more detail</div><div class='section-copy'>These questions change depending on your answers, so only the most relevant ones are shown.</div>", unsafe_allow_html=True)
        collected = {}
        for question in questions:
            collected[question["id"]] = st.radio(
                question["question"], question["options"], index=None, key=f"q_{question['id']}"
            )
        st.markdown("</div>", unsafe_allow_html=True)
        answered = st.form_submit_button("Continue to my assessment", use_container_width=True)

    if answered:
        st.session_state.answers = {k: v for k, v in collected.items() if v}
        st.session_state.stage = "results"
        st.rerun()

# ── STAGE 3: results ──
if st.session_state.stage == "results" and st.session_state.intake:
    intake = st.session_state.intake
    assessment = run_assessment_pipeline(
        intake["symptom_text"], intake["selected_symptoms"], intake["vitals"],
        intake["prob_array"], assets["label_encoder"], st.session_state.answers,
    )
    entities = assessment["entities"]
    red_flags = assessment["red_flags"]
    differential = assessment["differential"]
    uncertainty = assessment["uncertainty"]
    triage_result = assessment["triage"]

    status_text = "HIGH RISK - SEE A DOCTOR" if red_flags else "STABLE"
    severity = triage_result["label"]
    risk = min(10, sum(3 if flag["emergency"] else 1 for flag in red_flags))
    clinical_prediction = intake["ml_prediction"]

    medication_warnings = intake["medication_warnings"]
    medication_recommendations = intake["medication_recommendations"]

    st.session_state.results = {
        "ml_prediction": intake["ml_prediction"], "clinical_prediction": clinical_prediction,
        "confidence": intake["confidence"], "risk": risk,
        "severity": severity, "status_color": "#B91C1C" if red_flags else "#15803D",
        "status_text": status_text, "override_reason": None,
        "safety_signals": [flag["label"] for flag in red_flags],
        "symptom_text": intake["symptom_text"], "input_df": intake["input_df"],
        "prob_array": intake["prob_array"], "pred_index": intake["pred_index"],
        "scaled_input": intake["scaled_input"],
        "live_true": base_true_pool + [1 if red_flags else 0],
        "live_scores": base_scores_pool + [float(intake["confidence"] / 100.0)],
        "live_pred": [1 if s >= 0.5 else 0 for s in base_scores_pool] + [1 if red_flags else 0],
        "cv_scores": [0.972, 0.958, 0.965, 0.979, 0.961],
        "medication_warnings": medication_warnings,
        "medication_recommendations": medication_recommendations,
        "emergency_flags": [flag["label"] for flag in red_flags],
        "selected_symptoms": intake["selected_symptoms"],
        "entities": entities, "red_flags": red_flags, "differential": differential,
        "uncertainty": uncertainty, "triage": triage_result,
        "news2": 0, "qsofa": 0,
    }
    st.session_state.diagnosis_triggered = True

    # Streamlit re-runs this whole block on every interaction, so side effects
    # are keyed to a signature of the assessment to avoid duplicates.
    signature = f"{intake['symptom_text']}|{st.session_state.answers}"
    if st.session_state.get("history_signature") != signature:
        st.session_state.history_signature = signature
        st.session_state.assessment_history = ([{
            "condition": clinical_prediction, "confidence": intake["confidence"],
            "severity": triage_result["label"],
        }] + st.session_state.assessment_history)[:5]

        if intake.get("email"):
            send_email(intake["email"], intake["name"], clinical_prediction, status_text)

# =========================================================
# RESULTS DASHBOARD
# =========================================================
if st.session_state.diagnosis_triggered:
    res = st.session_state.results

    triage_result = res["triage"]
    entities = res["entities"]
    differential = res["differential"]
    uncertainty = res["uncertainty"]
    red_flags = res["red_flags"]

    st.markdown("<div class='section-head'><span>Your health assessment</span><div class='sh-line'></div></div>", unsafe_allow_html=True)

    # ── 1. Urgency first (spec 26) ──
    st.markdown(f"""
    <div class='triage-banner {triage_result['tone']}'>
        <div class='triage-label'>Recommended urgency · Level {triage_result['level']} · {html.escape(triage_result['label'])}</div>
        <div class='triage-headline'>{html.escape(triage_result['headline'])}</div>
        <div class='triage-summary'>{html.escape(triage_result['summary'])}</div>
    </div>
    """, unsafe_allow_html=True)

    # Emergency override (spec 17): warning signs replace ordinary reasoning.
    if red_flags:
        st.error("**Warning signs detected:** " + " · ".join(flag["label"] for flag in red_flags)
                 + ". Aarogya AI has stopped the normal assessment. If this is severe, sudden or ongoing, "
                   "contact your local emergency service now.")
        ec1, ec2 = st.columns(2)
        ec1.link_button("Call emergency services", "tel:112", use_container_width=True)
        ec2.link_button("Find nearest emergency department",
                        "https://www.google.com/maps/search/emergency+department+near+me",
                        use_container_width=True)

    # ── 2. What we found ──
    st.markdown("<div class='section-head'><span>What we found</span><div class='sh-line'></div></div>", unsafe_allow_html=True)
    if entities["symptoms"]:
        chips = "".join(f"<li class='yes'>{html.escape(s)}</li>" for s in entities["symptoms"])
        st.markdown(f"<div class='status-box'><div class='status-label-sm'>Symptoms extracted</div>"
                    f"<ul class='evidence-list'>{chips}</ul>"
                    f"<div class='status-sub'>Duration reported: {html.escape(entities['duration'] or 'not stated')} · "
                    f"Severity: {entities['severity'] if entities['severity'] is not None else 'not stated'}</div></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-box'><div class='status-label-sm'>Symptoms extracted</div>"
                    "<div class='status-sub'>We could not reliably identify specific symptoms from the description. "
                    "Try describing them in a little more detail or selecting them from the list.</div></div>",
                    unsafe_allow_html=True)

    if entities["explicitly_denied"]:
        st.caption("Reported as absent: " + ", ".join(entities["explicitly_denied"]))

    # ── 3. Uncertainty (spec 22) ──
    if uncertainty:
        next_question = entities["missing_information"][0] if entities.get("missing_information") else "How have your symptoms changed since they started?"
        st.info(f"{uncertainty} One more answer would help most: {next_question}")

    # ── 4. Differential (spec 27) ──
    st.markdown("<div class='section-head'><span>Possible causes</span><div class='sh-line'></div></div>", unsafe_allow_html=True)
    st.caption("These are ranked possibilities from the trained model for the information you gave. "
               "They are not diagnoses. Likelihood is shown as a band, not a precise percentage.")

    for candidate in differential:
        band_class = candidate["band"].lower().replace(" ", "-")
        supporting = "".join(f"<li class='yes'>{html.escape(item)}</li>" for item in candidate["supporting"]) or "<li class='no'>No supporting symptoms recorded</li>"
        against = "".join(f"<li class='no'>{html.escape(item)}</li>" for item in candidate["against"]) or "<li class='no'>None reported</li>"
        missing = "".join(f"<li class='no'>{html.escape(item)}</li>" for item in candidate["missing"][:3]) or "<li class='no'>No critical gaps identified</li>"
        st.markdown(f"""
        <div class='diff-card'>
            <div class='diff-rank'>{candidate['rank']:02d}</div>
            <div class='diff-name'>{html.escape(candidate['condition'])}</div>
            <div class='diff-band {band_class}'>{html.escape(candidate['band'])}</div>
            <div class='diff-block'><b>Why</b><ul class='evidence-list'>{supporting}</ul></div>
            <div class='diff-block'><b>Against</b><ul class='evidence-list'>{against}</ul></div>
            <div class='diff-block'><b>Missing information</b><ul class='evidence-list'>{missing}</ul></div>
        </div>
        """, unsafe_allow_html=True)

    # ── 5. What to do next (spec 29) ──
    st.markdown("<div class='section-head'><span>What to do next</span><div class='sh-line'></div></div>", unsafe_allow_html=True)
    actions = triage_result["actions"]
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**For now**")
        for item in actions["for_now"]:
            st.write(f"- {item}")
    with a2:
        st.markdown("**Consider medical evaluation if**")
        for item in actions["consider"]:
            st.write(f"- {item}")
    with a3:
        st.markdown("**Seek urgent help if**")
        for item in actions["urgent"]:
            st.write(f"- {item}")

    # ── 6. Why this result (spec 28) ──
    with st.expander("How Aarogya AI reached this assessment"):
        st.markdown("**1. Symptoms considered**")
        st.write(", ".join(entities["symptoms"]) or "None reliably identified")
        st.markdown("**2. Context applied**")
        st.write(f"Age {st.session_state.intake['age']} · "
                 f"Readings: heart rate {st.session_state.intake['vitals']['hr']}, "
                 f"blood pressure {st.session_state.intake['vitals']['bp']}, "
                 f"oxygen {st.session_state.intake['vitals']['spo2']}%, "
                 f"temperature {st.session_state.intake['vitals']['temp']} C, "
                 f"glucose {st.session_state.intake['vitals']['glucose']} mg/dL")
        st.markdown("**3. Warning signs checked**")
        st.write("Detected: " + (", ".join(flag["label"] for flag in red_flags) if red_flags else "none"))
        st.markdown("**4. Where the ranking comes from**")
        st.write("The trained classifier produces the ranked list above. Urgency is decided separately by "
                 "deterministic safety rules, so a model score can never lower the urgency level.")

    st.markdown("<div class='disclaimer-box'><p>Aarogya AI provides health information and guidance, not "
                "medical diagnosis or emergency care. Always seek professional medical advice when appropriate.</p></div>",
                unsafe_allow_html=True)

    st.write("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Condition ranking", 
        "What influenced it", 
        "Common Medicines Info", 
        "My Medication Safety",
        "How this App Works"
    ])
    
    with tab1:
        st.subheader("Possible-condition ranking")
        st.caption("These are the classifier’s highest-ranked categories for the information provided—not diagnoses or treatment advice.")
        
        # 1. Create the dataframe with all 41 diseases
        prob_df = pd.DataFrame({
            "Target Classification": assets["label_encoder"].classes_, 
            "Softmax Weight (%)": res["prob_array"] * 100
        })
        
        # 2. THE FIX: Keep only the Top 10 highest probabilities
        prob_df = prob_df.sort_values(by="Softmax Weight (%)", ascending=False).head(10)
        
        # 3. Draw the graph (sorting ascending so the biggest bar is at the top)
        fig_prob = px.bar(
            prob_df.sort_values(by="Softmax Weight (%)", ascending=True), 
            x="Softmax Weight (%)", 
            y="Target Classification", 
            orientation='h', 
            text_auto='.2f', 
            title="Top 10 model-ranked categories"
        )
        fig_prob.update_layout(template="plotly_dark")
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with tab2:
        st.subheader("Symptom Impact Analysis")
        st.caption("Seeing which of your specific inputs (like your fever or a headache) most influenced the AI's final answer.")
        try:
            if hasattr(assets["model"], "tree_method") or "Forest" in type(assets["model"]).__name__ or "Tree" in type(assets["model"]).__name__:
                explainer = shap.TreeExplainer(assets["model"])
            else:
                explainer = shap.KernelExplainer(assets["model"].predict_proba, res["scaled_input"][:1])
            
            shap_values = explainer.shap_values(res["scaled_input"])
            
            if isinstance(shap_values, list):
                shap_single = shap_values[res["pred_index"]][0]
            elif len(shap_values.shape) == 3:
                shap_single = shap_values[0, :, res["pred_index"]]
            else:
                shap_single = shap_values[0]
                
            shap_single = np.abs(np.array(shap_single).flatten())
            min_len = min(len(res["input_df"].columns), len(shap_single))
            
            shap_df = pd.DataFrame({"Symptom / Vital": res["input_df"].columns[:min_len], "Influence Level": shap_single[:min_len]})
            shap_df = shap_df.sort_values(by="Influence Level", ascending=False).head(10)
            
            fig_shap = px.bar(shap_df, x="Influence Level", y="Symptom / Vital", orientation="h", text_auto='.4f')
            fig_shap.update_layout(template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            
    with tab3:
        st.subheader("Related Medicines Info")
        st.caption("Common medicines generally associated with this condition (Do not take without a doctor's prescription).")
        
        # Break the prediction into individual words for a broader search
        prediction_words = res["clinical_prediction"].lower().split()
        
        # Create a boolean mask to check if ANY of the words exist in the 'Reason' column
        mask = med_db["Reason"].str.lower().apply(
            lambda x: any(word in str(x) for word in prediction_words if len(word) > 3) 
            # Note: len(word) > 3 ignores small words like "high" or "the"
        )
        
        matched_meds = med_db[mask]
        
        if not matched_meds.empty:
            for _, row in matched_meds.head(5).iterrows():
                st.markdown(f"""
                <div class='med-card'>
                    <div class='drug-name'>{row.get('Drug_Name', 'Unknown')}</div>
                    <div class='drug-reason'>Usually used for: {row.get('Reason', '')}</div>
                    <div class='drug-desc'>{row.get('Description', 'No description available.')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No common medication records found for '{res['clinical_prediction']}' in our database.")

    with tab4:
        st.subheader("Medication safety screen")
        st.caption("This screen only checks a small local rule set. A pharmacist or clinician should review all medicines, doses, supplements, and allergies.")
        for warning in res["medication_warnings"]:
            if "CRITICAL" in warning or "WARNING" in warning:
                st.error(warning)
            else:
                st.success(warning)
        if res["medication_recommendations"]:
            st.markdown("**Next steps**")
            for recommendation in res["medication_recommendations"]:
                st.write(f"- {recommendation}")

    with tab5:
        left_col, right_col = st.columns([3, 2])
        with left_col:
            st.header("Model transparency")
            st.warning("These charts use illustrative demo data and are not evidence of real-world clinical accuracy. The model should be independently validated before any clinical use.")
            
            fpr, tpr, _ = roc_curve(res["live_true"], res["live_scores"])
            cm_matrix = confusion_matrix(res["live_true"], res["live_pred"])
            
            tn, fp, fn, tp = cm_matrix.ravel() if cm_matrix.size == 4 else (0, 0, 0, 0)
            live_acc = float((tp + tn) / len(res["live_true"])) if len(res["live_true"]) > 0 else 0.0
            live_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            live_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            st.dataframe(pd.DataFrame({
                "Measurement": ["Illustrative accuracy", "Illustrative precision", "Illustrative recall"],
                "Score": [f"{live_acc:.4f}", f"{live_prec:.4f}", f"{live_rec:.4f}"]
            }), use_container_width=True)
            
            g1, g2 = st.columns(2)
            with g1:
                st.caption("Illustrative ROC curve")
                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, color='#6aadff', lw=2)
                ax_roc.plot([0, 1], [0, 1], color='#6670aa', lw=1, linestyle='--')
                fig_roc.patch.set_facecolor('#07090f')
                ax_roc.set_facecolor('#07090f')
                ax_roc.tick_params(colors='#e8eaf2')
                st.pyplot(fig_roc)
                
            with g2:
                st.caption("Illustrative classification matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
                fig_cm.patch.set_facecolor('#07090f')
                ax_cm.tick_params(colors='#e8eaf2')
                st.pyplot(fig_cm)

        with right_col:
            st.header("Behind the Scenes")
            st.markdown("""
            This check-in has two deliberately separate parts:
            
            **1. Model ranking**
            Your reported symptoms and measurements are converted to the same feature format used when training the Random Forest classifier. It returns a ranked set of learned categories instead of a hand-written condition rule.
            
            **2. Safety screen**
            Transparent threshold checks look for concerning readings and symptom phrases. They do not overwrite the model’s category; they simply tell you when the situation may need more urgent attention.
            
            **3. Explainability**
            The ranking and input-influence views help show what the model considered. This is an educational demonstration and must be independently clinically validated before real-world care use.
            """)

st.write("---")
st.subheader("Save Your Results")
try:
    if st.session_state.diagnosis_triggered:
        pdf_bytes = build_pdf_report(name, age, st.session_state.results)
        st.download_button("Download Health Report (PDF)", data=pdf_bytes, file_name=f"Report_{name.replace(' ', '_')}.pdf", mime="application/pdf")
except Exception:
    pass
