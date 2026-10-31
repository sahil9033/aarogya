"""FastAPI application for the Clinic AI browser experience."""
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

try:
    from .nlp_utils import encode_symptoms_to_dict, evaluate_safety_signals
except ImportError:  # Supports `python backend/main.py` during local development.
    from nlp_utils import encode_symptoms_to_dict, evaluate_safety_signals

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
VITAL_FEATURES = ("age", "hr", "bp", "spo2", "temp", "glucose")
app = FastAPI(title="Clinic AI", version="2.0.0")


def load_assets():
    try:
        return {
            "model": joblib.load(BASE_DIR / "model.pkl"), "scaler": joblib.load(BASE_DIR / "scaler.pkl"),
            "labels": joblib.load(BASE_DIR / "label_encoder.pkl"), "features": joblib.load(BASE_DIR / "features.pkl"),
        }
    except Exception as exc:
        raise RuntimeError("Clinical model files could not be loaded.") from exc


try:
    ASSETS, ASSET_ERROR = load_assets(), None
except RuntimeError as exc:
    ASSETS, ASSET_ERROR = None, str(exc)


class SymptomRequest(BaseModel):
    symptoms: str = Field(..., min_length=1, max_length=2000)
    selected_symptoms: List[str] = Field(default_factory=list, max_items=15)
    age: int = Field(30, ge=1, le=120)
    hr: float = Field(72, ge=20, le=260)
    bp: float = Field(120, ge=50, le=250)
    spo2: float = Field(98, ge=50, le=100)
    temp: float = Field(37, ge=30, le=45)
    glucose: float = Field(90, ge=20, le=800)

    @validator("selected_symptoms", each_item=True)
    def clean_selected_symptoms(cls, value):
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("Each selected symptom must be between 1 and 80 characters.")
        return value


class AnalysisResponse(BaseModel):
    primary_condition: str
    confidence: float
    alternatives: List[str]
    possible_causes: str
    next_steps: List[str]
    when_to_seek_care: List[str]
    safety_signals: List[str]
    urgency: str
    model_notice: str


def model_ranking(request):
    if ASSETS is None:
        raise HTTPException(status_code=503, detail=ASSET_ERROR or "The clinical model is unavailable.")
    values = encode_symptoms_to_dict(request.symptoms, ASSETS["features"], VITAL_FEATURES, request.selected_symptoms)
    values.update({key: getattr(request, key) for key in VITAL_FEATURES})
    expected = list(ASSETS["scaler"].feature_names_in_)
    frame = pd.DataFrame([[values.get(key, 0) for key in expected]], columns=expected)
    probabilities = ASSETS["model"].predict_proba(ASSETS["scaler"].transform(frame))[0]
    order = np.argsort(probabilities)[::-1]
    classes = np.asarray(getattr(ASSETS["model"], "classes_", np.arange(len(probabilities))))
    labels = ASSETS["labels"].inverse_transform(classes[order].astype(int))
    return [(str(label), float(probabilities[index])) for label, index in zip(labels, order)]


def guidance(condition, safety_signals):
    """Provides tailored medical guidance based on the identified condition and safety signals."""
    condition_lower = condition.lower()
    
    # Default guidance
    cause = (f"Your reported symptoms align with patterns the AI model associates with {condition}. "
             "This is a probabilistic assessment and not a clinical diagnosis.")
    steps = [
        "Monitor your symptoms closely, noting any changes or new developments.",
        "Ensure you are well-hydrated and getting adequate rest.",
        "Keep a log of your vital signs if possible (temperature, heart rate, etc.)."
    ]
    care = [
        "Consult with a healthcare professional for a formal evaluation.",
        "Seek immediate medical attention if you experience sudden worsening of symptoms."
    ]

    # Tailored guidance for specific conditions
    if "heart attack" in condition_lower or "hypertension" in condition_lower:
        cause = f"The model indicates a high correlation with cardiovascular patterns ({condition}). This requires professional validation."
        steps.insert(0, "Rest immediately and avoid any strenuous physical activity.")
        care.insert(0, "If you experience chest pain, shortness of breath, or numbness, call emergency services immediately.")
    elif "diabetes" in condition_lower or "hypoglycemia" in condition_lower:
        cause = f"Symptoms suggest a potential metabolic or blood sugar related issue ({condition})."
        steps.append("Check your blood glucose levels if you have the means to do so.")
        care.append("Schedule an appointment with an endocrinologist or your primary care doctor.")
    elif "pneumonia" in condition_lower or "bronchial asthma" in condition_lower or "respiratory" in condition_lower:
        cause = f"The AI model identified respiratory patterns consistent with {condition}."
        steps.append("Monitor your breathing rate and oxygen saturation (SpO2).")
        care.insert(0, "Seek urgent care if you experience significant difficulty breathing or blue-tinted lips/fingernails.")
    elif "allergy" in condition_lower or "drug reaction" in condition_lower:
        cause = f"The model suggests a potential allergic response or reaction ({condition})."
        steps.append("Identify and avoid any new foods, medications, or environmental triggers.")
        care.insert(0, "Seek emergency care if you experience swelling of the face, tongue, or throat.")

    if safety_signals:
        care.insert(0, "CRITICAL: One or more safety signals were detected. You should seek an in-person medical assessment as soon as possible.")
        
    return cause, steps, care


@app.get("/api/health")
def health():
    return {"status": "ok" if ASSETS else "degraded", "model_available": ASSETS is not None}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_symptoms(request: SymptomRequest):
    rankings = model_ranking(request)
    condition, score = rankings[0]
    joined = f"{request.symptoms} {' '.join(request.selected_symptoms)}"
    signals = evaluate_safety_signals(joined, request.hr, request.bp, request.spo2, request.temp, request.glucose)
    cause, steps, care = guidance(condition, signals)
    return AnalysisResponse(primary_condition=condition, confidence=round(score * 100, 1),
        alternatives=[label for label, _ in rankings[1:3]], possible_causes=cause, next_steps=steps,
        when_to_seek_care=care, safety_signals=signals, urgency="urgent" if signals else "routine",
        model_notice="This screening tool cannot diagnose illness or rule out emergencies.")


# Mount last so API routes retain priority and one command serves the whole app.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
