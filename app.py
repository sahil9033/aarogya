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
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: #000000;
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.6);
    --accent-orange: #FF6B00;
    --accent-blue: #007AFF;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --mesh-gradient: 
        radial-gradient(at 0% 0%, rgba(255, 107, 0, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(0, 122, 255, 0.15) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(255, 255, 255, 0.05) 0px, transparent 50%);
}

.stApp {
    background-color: var(--bg-dark);
    background-image: var(--mesh-gradient);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    max-width: 1000px;
    padding-top: 4rem;
    padding-bottom: 6rem;
}

/* ── Typography ── */
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
}

.main-title {
    font-size: 64px;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -3px;
    text-align: center;
    margin-bottom: 24px;
    background: linear-gradient(to bottom, #fff 0%, rgba(255,255,255,0.7) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    font-size: 20px;
    color: var(--text-secondary);
    text-align: center;
    max-width: 600px;
    margin: 0 auto 48px;
    line-height: 1.5;
}

/* ── Hero Section ── */
.portal-hero {
    text-align: center;
    margin-bottom: 64px;
}

.portal-hero .eyebrow {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--accent-blue);
    margin-bottom: 16px;
}

/* ── Cards & Panels ── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 32px;
    padding: 32px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}

.step-label {
    color: var(--accent-blue);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-copy {
    color: var(--text-secondary);
    font-size: 15px;
    margin-bottom: 24px;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

label, [data-testid="stWidgetLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    margin-bottom: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: white !important;
    color: black !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 600 !important;
    padding: 16px 48px !important;
    font-size: 18px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 32px auto !important;
    display: flex !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 15px 30px rgba(0,0,0,0.3) !important;
}

/* ── Results Dashboard ── */
.status-box {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 32px;
    padding: 48px;
    text-align: center;
    margin-bottom: 32px;
    backdrop-filter: blur(20px);
}

.status-disease {
    font-size: 48px;
    font-weight: 800;
    margin: 16px 0;
    letter-spacing: -0.04em;
    background: linear-gradient(to right, #fff, #aaa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.8) !important;
    border-right: 1px solid var(--glass-border);
    backdrop-filter: blur(20px);
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 16px !important;
    border-bottom-width: 2px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom-color: var(--text-primary) !important;
}

/* ── Disclaimer ── */
.disclaimer-box {
    background: rgba(255, 59, 48, 0.05);
    border: 1px solid rgba(255, 59, 48, 0.2);
    border-radius: 24px;
    padding: 24px;
    margin: 24px 0;
}

.disclaimer-box p {
    color: #ff9b9b !important;
    font-size: 15px !important;
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
    st.caption("1. Describe symptoms\n\n2. Add optional readings\n\n3. Review the model ranking and separate safety signals")
    if st.session_state.assessment_history:
        st.divider()
        st.markdown("**This session**")
        for item in st.session_state.assessment_history:
            st.caption(f"{item['condition']} · {item['confidence']:.0f}% confidence · {item['severity']} safety signal")
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

def identify_emergency_flags(text):
    emergency_phrases = [
        "chest pain", "tight chest", "severe difficulty breathing", "gasping",
        "choking", "passed out", "unconscious", "not able to get words out"
    ]
    return [phrase for phrase in emergency_phrases if contains_unnegated_phrase(text, phrase)]

def evaluate_safety_signals(hr, bp, spo2, temp, gluc, symptom_text):
    """Safety signals are deliberately separate from the trained condition model."""
    checks = [
        ("Very low oxygen reading", spo2 < 90),
        ("Very high heart rate", hr >= 145),
        ("Very high temperature", temp >= 39),
        ("High blood glucose reading", gluc > 200),
        ("Low blood pressure", bp < 90),
        ("Chest pain mentioned", contains_unnegated_phrase(symptom_text, "chest pain")),
    ]
    return [label for label, is_present in checks if is_present]

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
    <div class='eyebrow'>Aarogya · AI-assisted symptom check-in</div>
    <h1>A clearer first step for your health.</h1>
    <p>Describe what is happening in your own words. The trained model ranks possible conditions, while a separate safety screen highlights readings that may need prompt care.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='intake-heading'>Tell us what you’re experiencing</div><div class='intake-copy'>Start with symptoms. Add measurements only if you have recent readings.</div>", unsafe_allow_html=True)

with st.form("assessment_form", border=False):
    st.markdown("<div class='glass-card'><div class='step-label'>Step 1 · symptoms</div><div class='section-title'>Your experience, in your words</div><div class='section-copy'>Select what applies, then add context such as when symptoms began or what has changed.</div>", unsafe_allow_html=True)
    symptom_options = [
        "fever", "cough", "headache", "chest pain", "shortness of breath",
        "rash", "fatigue", "vomiting", "dizziness"
    ]
    selected_symptoms = st.multiselect("Symptoms you have noticed", symptom_options, placeholder="Choose all that apply")
    symptoms = st.text_area("Add details in your own words", placeholder="Example: Fever since yesterday evening, dry cough, and feeling unusually tired.")
    st.markdown("<div class='form-hint'>For emergencies, severe or sudden symptoms, or difficulty breathing: seek urgent care now. Do not wait for this check-in.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='margin-top:16px'><div class='step-label'>Step 2 · optional context</div><div class='section-title'>Measurements & report delivery</div><div class='section-copy'>Leave anything blank only when a reading is unavailable; values are not required for the model to use your symptoms.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Your name", placeholder="e.g., Alex")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        hr = st.number_input("Heart rate (beats per minute)", min_value=20.0, max_value=260.0, value=72.0, help="Use a recent resting reading if available.")
        bp = st.number_input("Systolic blood pressure (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
    with col2:
        spo2 = st.number_input("Blood oxygen (SpO₂, %)", min_value=50.0, max_value=100.0, value=98.0)
        temp = st.number_input("Body temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)
        gluc = st.number_input("Blood glucose (mg/dL)", min_value=20.0, max_value=800.0, value=90.0)
        email = st.text_input("Email for report (optional)", placeholder="name@example.com")
    with st.expander("Medication safety check (optional)"):
        st.caption("This is a limited local screen, not a complete interaction, dose, or allergy check.")
        current_medicines = st.text_input("Current medicines (separate with commas)", placeholder="e.g., aspirin, metformin")
        known_conditions = st.text_input("Known conditions (separate with commas)", placeholder="e.g., ulcer, kidney disease")
        allergies = st.text_input("Medication allergies (separate with commas)", placeholder="e.g., aspirin")
    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Generate my AI summary", use_container_width=True)

# =========================================================
# COMPUTATION PIPELINE
# =========================================================
if submitted:
    symptom_text = f"{symptoms} {' '.join(selected_symptoms)}".lower()
    if not symptoms.strip() and not selected_symptoms:
        st.warning("Select a symptom or add a short description before checking your results.")
        st.stop()
    vital_features = ["age", "hr", "bp", "spo2", "temp", "glucose"]
    
    feature_dict = encode_symptoms_to_dict(symptom_text, assets["features"], vital_features, selected_symptoms)
    feature_dict["age"] = age
    feature_dict["hr"] = hr
    feature_dict["bp"] = bp
    feature_dict["spo2"] = spo2
    feature_dict["temp"] = temp
    feature_dict["glucose"] = gluc
    
    expected_features = assets["scaler"].feature_names_in_
    input_data = [feature_dict.get(col, 0) for col in expected_features]
    input_df = pd.DataFrame([input_data], columns=expected_features)
    
    # ML Stage
    scaled_input = assets["scaler"].transform(input_df)
    prob = assets["model"].predict_proba(scaled_input)
    pred_index = np.argmax(prob[0])
    model_class = getattr(assets["model"], "classes_", np.arange(prob.shape[1]))[pred_index]
    ml_prediction = assets["label_encoder"].inverse_transform([model_class])[0]
    confidence = float(prob[0][pred_index] * 100)
    
    # Keep the model's ranked condition separate from transparent safety triage.
    clinical_prediction = ml_prediction
    safety_signals = evaluate_safety_signals(hr, bp, spo2, temp, gluc, symptom_text)
    override_reason = None

    risk_weights = {
        "Very low oxygen reading": 3, "Very high heart rate": 3,
        "Very high temperature": 2, "High blood glucose reading": 2,
        "Low blood pressure": 2, "Chest pain mentioned": 3,
    }
    risk = min(10, sum(risk_weights[signal] for signal in safety_signals))
    
    news2 = sum([3 if spo2 < 91 else (2 if spo2 < 94 else 0),
                 3 if temp > 39 else (1 if temp > 38 else 0),
                 3 if hr > 130 else (2 if hr > 110 else 0)])
    
    qsofa = sum([1 if bp < 100 else 0, 1 if hr > 120 else 0, 1 if spo2 < 90 else 0])
    severity = "Critical" if risk >= 6 else ("Severe" if risk >= 4 else ("Moderate" if risk >= 2 else "Mild"))
    
    if severity in ["Severe", "Critical"]:
        status_color = "#e24b4a" # Red
        status_text = "HIGH RISK - SEE A DOCTOR"
        live_label = 1
    else:
        status_color = "#3ec97a" # Green
        status_text = "STABLE"
        live_label = 0

    medication_warnings, medication_recommendations = check_drugs(
        current_medicines.split(","), known_conditions.split(","), allergies.split(",")
    )
    emergency_flags = identify_emergency_flags(symptom_text)
    
    live_true = base_true_pool + [live_label]
    live_scores = base_scores_pool + [float(confidence / 100.0)]
    live_pred = [1 if score >= 0.5 else 0 for score in live_scores]
    cv_scores = [0.972, 0.958, 0.965, 0.979, 0.961]
    
    st.session_state.results = {
        "ml_prediction": ml_prediction, "clinical_prediction": clinical_prediction,
        "confidence": confidence, "risk": risk, "news2": news2, "qsofa": qsofa,
        "severity": severity, "status_color": status_color, "status_text": status_text, 
        "override_reason": override_reason, "safety_signals": safety_signals, "symptom_text": symptom_text, 
        "input_df": input_df, "prob_array": prob[0], "pred_index": pred_index,
        "scaled_input": scaled_input, "live_true": live_true, "live_scores": live_scores,
        "live_pred": live_pred, "cv_scores": cv_scores,
        "medication_warnings": medication_warnings,
        "medication_recommendations": medication_recommendations,
        "emergency_flags": emergency_flags,
        "selected_symptoms": selected_symptoms
    }
    st.session_state.diagnosis_triggered = True
    st.session_state.assessment_history = ([{
        "condition": clinical_prediction, "confidence": confidence, "severity": severity
    }] + st.session_state.assessment_history)[:5]
    
    if email:
        send_email(email, name, clinical_prediction, status_text)

# =========================================================
# RESULTS DASHBOARD
# =========================================================
if st.session_state.diagnosis_triggered:
    res = st.session_state.results

    if res["emergency_flags"]:
        st.error("**Emergency warning:** You mentioned " + ", ".join(res["emergency_flags"]) + ". If this is severe, sudden, or ongoing, contact your local emergency service now. Do not wait for this tool's result.")
    
    box_style = f"border-left: 6px solid {res['status_color']};" if "HIGH RISK" in res["status_text"] else ""
    condition_label = html.escape(str(res["clinical_prediction"]))
    
    st.markdown(f"""
    <div class='status-box' style='{box_style}'>
        <div style="font-size: 12px; color: #b7cee9; text-transform: uppercase; letter-spacing: .1em;">Model’s leading possibility</div>
        <div style="font-size: 28px; font-weight: 750; color: #f3f8ff;">{condition_label}</div>
        <div style="font-size: 14px; color: #c0d3e9;">Confidence for this model output: {round(res['confidence'], 2)}% · A confidence score is not a diagnosis or a measure of medical urgency.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class='disclaimer-box'>
        <p><b>⚠️ Important Medical Notice:</b> This result is generated by an Artificial Intelligence program for informational purposes only. It is NOT a real medical diagnosis. You must consult a doctor or visit a hospital immediately for proper medical advice and treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Safety signal score", f"{res['risk']} / 10")
    mc2.metric("Urgency signal", res["severity"])
    mc3.metric("Model confidence", f"{round(res['confidence'], 2)}%")

    if res["safety_signals"]:
        st.warning("**Safety signals noted:** " + " · ".join(res["safety_signals"]) + ". These are separate from the model’s condition ranking.")
    else:
        st.success("No immediate safety signals were found from the values entered. This does not rule out a health problem.")

    if res["severity"] in ["Severe", "Critical"]:
        st.warning("**Recommended next step:** Seek prompt in-person medical assessment. If symptoms rapidly worsen or feel life-threatening, use your local emergency service.")
    else:
        st.info("**Recommended next step:** Monitor your symptoms and discuss persistent, worsening, or worrying symptoms with a qualified healthcare professional.")
    
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
