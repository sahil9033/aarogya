import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from fpdf import FPDF
import datetime
from nlp_utils import encode_symptoms_to_dict, evaluate_safety_signals

app = FastAPI(title="Clinic AI API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
# We assume the models are in the parent directory (clinical-ai-system-main)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # Initialize dummies for testing if files are missing
    model = None
    scaler = None
    label_encoder = None
    features = []

# Pydantic schema for request
class SymptomRequest(BaseModel):
    symptoms: str
    selected_symptoms: List[str] = []
    age: Optional[int] = 30
    hr: Optional[float] = 72.0
    bp: Optional[float] = 120.0
    spo2: Optional[float] = 98.0
    temp: Optional[float] = 37.0
    glucose: Optional[float] = 90.0

class AnalysisResponse(BaseModel):
    primary_condition: str
    possible_causes: str
    next_steps: str
    when_to_seek_care: str
    safety_signals: List[str]
    confidence_score: float

def generate_rich_text(condition: str, symptoms: str, safety_signals: List[str], file_context: str = "") -> dict:
    """Uses Gemini to generate rich text if API key is present, otherwise falls back to templates."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            A patient presents with these symptoms: "{symptoms}".
            A machine learning model has classified their likely primary condition as: "{condition}".
            Safety signals identified: {', '.join(safety_signals) if safety_signals else 'None'}.
            
            Additional context from attached medical document: "{file_context}"
            
            Based on this, please provide three concise, empathetic, and medically sound paragraphs for a patient-facing app:
            1. possible_causes: Briefly explain what this condition is and why the symptoms align with it. Do NOT say 'The ML model says'. Speak directly to the patient (e.g. "Based on your symptoms...").
            2. next_steps: 3-4 bullet points of immediate actionable advice (e.g. rest, hydration).
            3. when_to_seek_care: 2-3 bullet points of warning signs that indicate they need immediate medical attention.
            
            Return ONLY a JSON object with keys: "possible_causes", "next_steps", "when_to_seek_care".
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            import json
            return json.loads(response.text)
            
        except Exception as e:
            print(f"Gemini API error: {e}. Falling back to templates.")
            
    # Fallback template
    return {
        "possible_causes": f"Based on the symptoms described, this resembles a case of <strong>{condition}</strong>. This can cause the symptoms you are experiencing.",
        "next_steps": "<ul><li>Get plenty of rest.</li><li>Stay hydrated by drinking water.</li><li>Monitor your symptoms closely.</li></ul>",
        "when_to_seek_care": "<ul><li>If symptoms rapidly worsen.</li><li>If you experience difficulty breathing or severe pain.</li><li>If symptoms persist for more than a few days without improvement.</li></ul>"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_symptoms(
    symptoms: str = Form(""),
    selected_symptoms: str = Form(""),
    age: int = Form(30),
    hr: float = Form(72.0),
    bp: float = Form(120.0),
    spo2: float = Form(98.0),
    temp: float = Form(37.0),
    glucose: float = Form(90.0),
    file: Optional[UploadFile] = File(None)
):
    # Parse selected symptoms
    symptom_list = [s.strip() for s in selected_symptoms.split(",")] if selected_symptoms else []
    symptom_text = f"{symptoms} {' '.join(symptom_list)}".lower()
    
    file_context = ""
    # Process file if attached
    if file and file.filename:
        try:
            content = await file.read()
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                client = genai.Client(api_key=api_key)
                # Quick parse of text using Gemini
                prompt = "Extract the key medical findings, diagnoses, or lab results from this document concisely."
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        types.Part.from_bytes(data=content, mime_type=file.content_type),
                        prompt
                    ]
                )
                file_context = response.text
                symptom_text += f" [Attached Document Summary: {file_context}]"
        except Exception as e:
            print(f"File processing error: {e}")

    if not model:
        # Mock response if model wasn't loaded
        rich_text = generate_rich_text("General Viral Infection", symptom_text, [], file_context)
        return AnalysisResponse(
            primary_condition="General Viral Infection",
            possible_causes=rich_text["possible_causes"],
            next_steps=rich_text["next_steps"],
            when_to_seek_care=rich_text["when_to_seek_care"],
            safety_signals=[],
            confidence_score=85.0
        )
    
    vital_features = ["Age", "HeartRate", "BP_Systolic", "SpO2", "Temperature", "Glucose"]
    feature_dict = encode_symptoms_to_dict(symptom_text, features, vital_features, symptom_list)
    
    # Add vitals with exact casing expected by the scaler
    feature_dict["Age"] = age
    feature_dict["HeartRate"] = hr
    feature_dict["BP_Systolic"] = bp
    feature_dict["SpO2"] = spo2
    feature_dict["Temperature"] = temp
    feature_dict["Glucose"] = glucose
    
    expected_features = scaler.feature_names_in_
    input_data = [feature_dict.get(col, 0) for col in expected_features]
    input_df = pd.DataFrame([input_data], columns=expected_features)
    
    # ML Inference
    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)
    pred_index = np.argmax(prob[0])
    ml_prediction = label_encoder.inverse_transform([pred_index])[0]
    confidence_score = float(prob[0][pred_index] * 100)
    
    safety_signals = evaluate_safety_signals(symptom_text)
    
    # Generative AI Enrichment
    rich_text = generate_rich_text(ml_prediction, symptom_text, safety_signals, file_context)
    
    # Check if safety signals exist and override
    if safety_signals:
        rich_text["when_to_seek_care"] = "<strong>CRITICAL WARNING:</strong> You mentioned symptoms that require immediate care. Please contact emergency services or visit a doctor right away.<br><br>" + rich_text["when_to_seek_care"]
        
    return AnalysisResponse(
        primary_condition=ml_prediction,
        possible_causes=rich_text["possible_causes"],
        next_steps=rich_text["next_steps"],
        when_to_seek_care=rich_text["when_to_seek_care"],
        safety_signals=safety_signals,
        confidence_score=confidence_score
    )

class ReportRequest(BaseModel):
    condition: str
    confidence: float
    causes: str
    steps: str
    care: str
    safety_signals: List[str]

@app.post("/api/report/download")
async def download_report(req: ReportRequest):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Clinic AI Assessment Report", ln=1, align="C")
    
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Primary Assessment", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Condition: {req.condition} (Model Confidence: {req.confidence:.1f}%)", ln=1)
    pdf.ln(5)
    
    if req.safety_signals:
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(226, 75, 74) # Red
        pdf.cell(0, 10, "Safety Signals Noted:", ln=1)
        pdf.set_font("Arial", "", 12)
        for sig in req.safety_signals:
            pdf.cell(0, 8, f"- {sig}", ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
    def add_section(title, text):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=1)
        pdf.set_font("Arial", "", 11)
        # Clean HTML tags very roughly
        clean_text = text.replace("<strong>", "").replace("</strong>", "").replace("<ul>", "").replace("</ul>", "").replace("<li>", "- ").replace("</li>", "\n").replace("<br>", "\n")
        pdf.multi_cell(0, 6, clean_text)
        pdf.ln(5)
        
    add_section("Possible Causes", req.causes)
    add_section("Next Steps", req.steps)
    add_section("When to Seek Care", req.care)
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an Artificial Intelligence program. It is NOT a real clinical diagnosis and should not replace professional medical advice. Always visit a doctor for evaluation and proper medical treatment.")
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=ClinicAI_Report.pdf"})

# Serve the frontend statically
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
