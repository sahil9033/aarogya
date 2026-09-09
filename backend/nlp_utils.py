import re

def encode_symptoms_to_dict(text, feature_list, vital_features, selected_symptoms=None):
    """Create the model input from the user's words."""
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
    """Match a phrase while ignoring simple statements such as 'no chest pain'."""
    phrase_pattern = re.escape(phrase)
    return bool(re.search(rf"(?<!no )(?<!denies ){phrase_pattern}", text.lower()))

def evaluate_safety_signals(symptom_text):
    """Basic safety checks based on text."""
    checks = [
        ("Chest pain mentioned", contains_unnegated_phrase(symptom_text, "chest pain")),
        ("Breathing difficulty", contains_unnegated_phrase(symptom_text, "difficulty breathing") or contains_unnegated_phrase(symptom_text, "shortness of breath")),
        ("Unconsciousness", contains_unnegated_phrase(symptom_text, "passed out") or contains_unnegated_phrase(symptom_text, "unconscious"))
    ]
    return [label for label, is_present in checks if is_present]
