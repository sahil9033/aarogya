"""Deterministic helpers for symptom feature extraction and safety screening."""
import re

ALIASES = {
    "shortness of breath": ("difficulty breathing", "breathless", "cannot breathe", "breathing problem", "dyspnea", "gasping"),
    "fatigue": ("tired", "exhausted", "low energy", "weakness", "lethargy", "malaise"),
    "dizziness": ("dizzy", "lightheaded", "faint", "vertigo", "unsteady"),
    "vomiting": ("nausea", "throwing up", "throw up", "emesis", "sick to stomach"),
    "cough": ("coughing", "hacking", "phlegm"),
    "fever": ("high temperature", "feverish", "chills", "pyrexia"),
    "chest pain": ("tight chest", "chest pressure", "angina", "heart pain"),
    "headache": ("migraine", "throbbing head", "head pain"),
    "rash": ("skin irritation", "hives", "dermatitis", "itchy skin"),
    "stomach pain": ("abdominal pain", "belly ache", "cramps"),
}
NEGATION_WORDS = r"no|not|without|denies|denied|never|negative for|free of"

def contains_unnegated_phrase(text: str, phrase: str) -> bool:
    """Match a phrase while respecting a nearby simple negation."""
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    # Simple check for the phrase first
    if phrase.lower() not in normalized:
        return False
        
    for match in re.finditer(rf"\b{re.escape(phrase.lower())}\b", normalized):
        # Look back up to 40 characters for negations
        preceding = normalized[max(0, match.start() - 40):match.start()]
        if not re.search(rf"\b(?:{NEGATION_WORDS})\b(?:\s+\w+){{0,4}}\s*$", preceding):
            return True
    return False

def encode_symptoms_to_dict(text, feature_list, vital_features, selected_symptoms=None):
    """Map input to the model's exact feature names without condition rules."""
    selected = {str(item).lower().strip() for item in (selected_symptoms or [])}
    values = {}
    for feature in feature_list:
        if feature in vital_features:
            continue
        phrase = str(feature).replace("_", " ").lower()
        candidates = (phrase, *ALIASES.get(phrase, ()))
        values[feature] = int(phrase in selected or any(contains_unnegated_phrase(text, item) for item in candidates))
    return values

def evaluate_safety_signals(symptom_text, hr=72, bp=120, spo2=98, temp=37, glucose=90):
    """Return explainable safety signals; this is intentionally not a diagnosis."""
    checks = (
        ("Critical oxygen level (SpO2 < 90%)", spo2 < 90),
        ("Low oxygen level (SpO2 90-94%)", 90 <= spo2 < 94),
        ("Tachycardia (Heart rate >= 140 bpm)", hr >= 140),
        ("Bradycardia (Heart rate < 50 bpm)", hr < 50),
        ("Hypotension (Blood pressure < 90 mmHg)", bp < 90),
        ("Hypertension (Blood pressure > 180 mmHg)", bp > 180),
        ("High fever (Temp >= 39.4°C / 103°F)", temp >= 39.4),
        ("Hyperglycemia (Glucose >= 300 mg/dL)", glucose >= 300),
        ("Hypoglycemia (Glucose < 70 mg/dL)", glucose < 70),
        ("Chest pain / Pressure mentioned", contains_unnegated_phrase(symptom_text, "chest pain")),
        ("Severe breathing difficulty", any(contains_unnegated_phrase(symptom_text, value) for value in ("shortness of breath", "difficulty breathing", "gasping", "cannot breathe"))),
        ("Loss of consciousness", any(contains_unnegated_phrase(symptom_text, value) for value in ("passed out", "unconscious", "fainted", "syncope"))),
        ("Possible stroke signs (FAST)", any(contains_unnegated_phrase(symptom_text, value) for value in ("face drooping", "slurred speech", "one sided weakness", "arm drift"))),
        ("Severe allergic reaction signs", any(contains_unnegated_phrase(symptom_text, value) for value in ("throat closing", "swelling of tongue", "hives everywhere"))),
    )
    return [label for label, present in checks if present]
