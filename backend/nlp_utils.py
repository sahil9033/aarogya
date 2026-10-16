"""Deterministic symptom extraction and separate safety screening."""
import re

ALIASES = {
    "shortness of breath": ("difficulty breathing", "breathless", "cannot breathe", "breathing problem"),
    "fatigue": ("tired", "exhausted", "low energy", "weakness"),
    "dizziness": ("dizzy", "lightheaded", "faint"),
    "vomiting": ("nausea", "throwing up", "throw up"),
    "cough": ("coughing",), "fever": ("high temperature", "feverish"),
    "chest pain": ("tight chest", "chest pressure"),
}


def contains_unnegated_phrase(text, phrase):
    """Match a phrase unless it is preceded by a nearby simple negation."""
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    for match in re.finditer(rf"\b{re.escape(phrase.lower())}\b", text):
        preceding = text[max(0, match.start() - 32):match.start()]
        if not re.search(r"\b(?:no|not|without|denies|denied|never)\b(?:\s+\w+){0,3}\s*$", preceding):
            return True
    return False


def encode_symptoms_to_dict(text, feature_list, vital_features, selected_symptoms=None):
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
    """Return explainable triage signals, never a diagnosis or risk score."""
    checks = (
        ("Very low oxygen reading", spo2 < 90), ("Low oxygen reading", 90 <= spo2 < 94),
        ("Very high heart rate", hr >= 140), ("Very low blood pressure", bp < 90),
        ("High temperature", temp >= 39.4), ("Very high blood glucose reading", glucose >= 300),
        ("Chest pain mentioned", contains_unnegated_phrase(symptom_text, "chest pain")),
        ("Breathing difficulty", any(contains_unnegated_phrase(symptom_text, item) for item in ("shortness of breath", "difficulty breathing", "gasping"))),
        ("Loss of consciousness mentioned", any(contains_unnegated_phrase(symptom_text, item) for item in ("passed out", "unconscious", "fainted"))),
        ("Stroke warning symptom mentioned", any(contains_unnegated_phrase(symptom_text, item) for item in ("face drooping", "slurred speech", "one sided weakness"))),
    )
    return [label for label, present in checks if present]
