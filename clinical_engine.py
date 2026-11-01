"""
Aarogya AI - Clinical reasoning layer.

This module implements the deterministic parts of the assessment pipeline
described in the product specification:

    entity extraction -> red-flag screening -> differential -> triage

Design rules enforced here:
  * Safety screening is deterministic and never depends on the ML model
    (spec 46: safety must be separable from generation).
  * No fabricated probabilities are produced. The model's own output is
    only ever shown as a ranked candidate; the user-facing likelihood is a
    coarse band (spec 21).
  * Nothing in this file invents clinical facts. Thresholds and warning
    signs are drawn from the specification's own red-flag list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# 1. SYMPTOM LEXICON
# ---------------------------------------------------------------------------

SYMPTOM_LEXICON = {
    "fever": ("fever", "feverish", "high temperature", "chills", "pyrexia", "temperature"),
    "cough": ("cough", "coughing", "hacking cough", "phlegm"),
    "productive cough": ("productive cough", "cough with phlegm", "coughing up mucus", "green sputum"),
    "sore throat": ("sore throat", "throat pain", "painful swallow", "scratchy throat"),
    "headache": ("headache", "head pain", "migraine", "throbbing head"),
    "shortness of breath": ("shortness of breath", "breathless", "difficulty breathing", "cannot breathe", "dyspnea", "gasping", "winded"),
    "chest pain": ("chest pain", "tight chest", "chest pressure", "chest tightness", "angina"),
    "fatigue": ("fatigue", "tired", "exhausted", "low energy", "weakness", "lethargy", "malaise"),
    "body ache": ("body ache", "body aches", "muscle ache", "aching all over", "myalgia"),
    "nausea": ("nausea", "nauseous", "queasy", "sick to my stomach"),
    "vomiting": ("vomiting", "throwing up", "vomited", "emesis"),
    "diarrhea": ("diarrhea", "diarrhoea", "loose stools", "watery stool"),
    "abdominal pain": ("abdominal pain", "stomach pain", "stomach ache", "belly pain", "cramps"),
    "dizziness": ("dizzy", "dizziness", "lightheaded", "vertigo", "unsteady"),
    "rash": ("rash", "hives", "skin irritation", "itchy skin", "dermatitis"),
    "joint pain": ("joint pain", "achy joints", "swollen joints", "arthralgia"),
    "loss of appetite": ("loss of appetite", "not hungry", "no appetite", "appetite loss"),
    "weight loss": ("weight loss", "losing weight"),
    "night sweats": ("night sweats", "sweating at night"),
    "blurred vision": ("blurred vision", "vision changes", "double vision"),
    "light sensitivity": ("light sensitivity", "photophobia", "bothered by light"),
    "numbness": ("numbness", "numb", "tingling", "pins and needles"),
    "confusion": ("confusion", "confused", "disoriented", "cannot think clearly"),
    "palpitations": ("palpitations", "racing heart", "heart pounding"),
}

NEGATION_PATTERN = r"no|not|without|denies|denied|never|negative for|free of|haven't had|didn't have"

DURATION_PATTERN = re.compile(
    r"\b(?:for|since|past|last)\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(hour|hours|day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

SEVERITY_PATTERN = re.compile(r"\b(\d{1,2})\s*(?:/|out of)\s*10\b", re.IGNORECASE)


def _is_negated(text: str, start: int, window: int = 44) -> bool:
    """True when the matched phrase sits behind a nearby negation."""
    preceding = text[max(0, start - window):start]
    return bool(re.search(rf"\b(?:{NEGATION_PATTERN})\b(?:\s+\w+){{0,4}}\s*$", preceding))


def find_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive, negation-aware phrase match."""
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    for match in re.finditer(rf"\b{re.escape(phrase.lower())}\b", normalized):
        if not _is_negated(normalized, match.start()):
            return True
    return False


def extract_clinical_entities(text: str, selected_symptoms=None) -> dict:
    """Turn free text into structured symptoms, duration, severity and context."""
    raw = text or ""
    selected = {str(item).lower().strip() for item in (selected_symptoms or [])}

    detected = []
    for canonical, synonyms in SYMPTOM_LEXICON.items():
        if canonical in selected or any(find_phrase(raw, synonym) for synonym in synonyms):
            detected.append(canonical)

    duration_match = DURATION_PATTERN.search(raw.lower())
    duration = None
    if duration_match:
        duration = f"{duration_match.group(1)} {duration_match.group(2)}"

    severity_match = SEVERITY_PATTERN.search(raw)
    severity = int(severity_match.group(1)) if severity_match else None

    negated = []
    for canonical, synonyms in SYMPTOM_LEXICON.items():
        if canonical in detected:
            continue
        for synonym in synonyms:
            normalized = re.sub(r"\s+", " ", raw.lower())
            for match in re.finditer(rf"\b{re.escape(synonym)}\b", normalized):
                if _is_negated(normalized, match.start()):
                    negated.append(canonical)
                    break
            else:
                continue
            break

    return {
        "symptoms": detected,
        "duration": duration,
        "severity": severity,
        "explicitly_denied": sorted(set(negated)),
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# 2. RED FLAG ENGINE  (spec 17 - runs before any reasoning)
# ---------------------------------------------------------------------------

@dataclass
class RedFlag:
    id: str
    label: str
    message: str
    emergency: bool = True


# Text-based warning signs. Deliberately conservative: any hit escalates.
TEXT_RED_FLAGS = (
    RedFlag("breathing", "Severe breathing difficulty",
            "Difficulty breathing can indicate a problem that needs emergency assessment.", True),
    RedFlag("chest_pain", "Chest pain or pressure",
            "Chest pain or pressure should be assessed in person without delay.", True),
    RedFlag("consciousness", "Loss of consciousness",
            "Fainting or loss of consciousness needs urgent medical evaluation.", True),
    RedFlag("stroke", "Possible stroke signs",
            "Sudden weakness, facial drooping or slurred speech are time-critical.", True),
    RedFlag("seizure", "Seizure activity",
            "A new seizure requires emergency assessment.", True),
    RedFlag("confusion", "Severe confusion",
            "New confusion or disorientation needs urgent assessment.", True),
    RedFlag("sudden_headache", "Sudden severe headache",
            "A sudden, severe or 'worst ever' headache needs urgent assessment.", True),
    RedFlag("allergy", "Severe allergic reaction",
            "Swelling of the face, lips or throat can become an emergency.", True),
    RedFlag("bleeding", "Major bleeding",
            "Heavy or uncontrolled bleeding requires emergency care.", True),
    RedFlag("self_harm", "Self-harm or safety risk",
            "If you are at risk of harming yourself, please contact emergency services or a crisis line now.", True),
)

TEXT_RED_FLAG_PATTERNS = {
    "breathing": ("severe difficulty breathing", "cannot breathe", "can't breathe", "gasping for air",
                  "struggling to breathe", "shortness of breath", "breathless", "choking"),
    "chest_pain": ("chest pain", "tight chest", "chest pressure", "chest tightness", "crushing chest",
                   "pain in my chest"),
    "consciousness": ("passed out", "unconscious", "fainted", "blacked out", "lost consciousness"),
    "stroke": ("face drooping", "facial drooping", "slurred speech", "one sided weakness",
               "one-sided weakness", "cannot lift my arm", "arm drift"),
    "seizure": ("seizure", "convulsion", "fitting", "fits"),
    "confusion": ("severe confusion", "confused and disoriented", "cannot recognise", "can't recognise"),
    "sudden_headache": ("worst headache of my life", "sudden severe headache", "thunderclap headache",
                        "worst ever headache"),
    "allergy": ("throat closing", "swollen tongue", "swelling of my tongue", "face swelling",
                "lips swelling", "anaphylaxis"),
    "bleeding": ("severe bleeding", "heavy bleeding", "bleeding heavily", "coughing up blood",
                 "vomiting blood", "blood in stool"),
    "self_harm": ("kill myself", "end my life", "suicidal", "harm myself", "self harm"),
}

# Reading-based warning signs (spec 17 + vitals thresholds).
VITAL_RED_FLAGS = (
    ("spo2_critical", "Very low blood oxygen", lambda v: v.get("spo2", 98) < 90,
     "An oxygen saturation below 90% needs urgent assessment."),
    ("spo2_low", "Low blood oxygen", lambda v: 90 <= v.get("spo2", 98) < 94,
     "An oxygen saturation between 90% and 94% should be reviewed promptly."),
    ("hr_high", "Very high heart rate", lambda v: v.get("hr", 72) >= 140,
     "A resting heart rate at or above 140 bpm needs prompt review."),
    ("hr_low", "Very low heart rate", lambda v: v.get("hr", 72) < 50,
     "A resting heart rate below 50 bpm should be reviewed."),
    ("bp_low", "Low blood pressure", lambda v: v.get("bp", 120) < 90,
     "A systolic reading below 90 mmHg can indicate a circulation problem."),
    ("bp_high", "Very high blood pressure", lambda v: v.get("bp", 120) > 180,
     "A systolic reading above 180 mmHg needs prompt review."),
    ("temp_high", "High fever", lambda v: v.get("temp", 37) >= 39.4,
     "A temperature at or above 39.4 C needs prompt review."),
    ("temp_low", "Low body temperature", lambda v: v.get("temp", 37) < 35,
     "A temperature below 35 C needs urgent assessment."),
    ("glucose_high", "Very high blood glucose", lambda v: v.get("glucose", 90) >= 300,
     "A glucose reading at or above 300 mg/dL needs prompt review."),
    ("glucose_low", "Low blood glucose", lambda v: v.get("glucose", 90) < 70,
     "A glucose reading below 70 mg/dL needs prompt treatment."),
)


def screen_red_flags(text: str, vitals: dict | None = None) -> list[dict]:
    """Return every warning sign that applies. Empty list means 'none detected'."""
    vitals = vitals or {}
    found: list[RedFlag] = []

    for flag in TEXT_RED_FLAGS:
        patterns = TEXT_RED_FLAG_PATTERNS.get(flag.id, ())
        if any(find_phrase(text, pattern) for pattern in patterns):
            found.append(flag)

    by_id = {flag.id: flag for flag in found}
    for flag_id, label, test, message in VITAL_RED_FLAGS:
        try:
            applies = bool(test(vitals))
        except Exception:
            applies = False
        if applies and flag_id not in by_id:
            found.append(RedFlag(flag_id, label, message, True))
            by_id[flag_id] = found[-1]

    return [asdict(flag) for flag in found]


# ---------------------------------------------------------------------------
# 3. DIFFERENTIAL ENGINE  (spec 19-21)
# ---------------------------------------------------------------------------

# Coarse, user-facing likelihood bands. The spec explicitly forbids showing
# false precision such as "87.42% chance".
BANDS = (
    (0.55, "More likely"),
    (0.20, "Possible"),
    (0.08, "Less likely"),
    (0.0,  "Cannot rule out"),
)


def _band_for(score: float) -> str:
    for threshold, label in BANDS:
        if score >= threshold:
            return label
    return "Cannot rule out"


def _condition_label(condition: str) -> str:
    """Turn a model class such as 'viral_upper_respiratory_infection' into a title."""
    text = str(condition).replace("_", " ").strip()
    return text.title() if text else str(condition)


def build_differential(prob_array, label_encoder, entities: dict, top_k: int = 4) -> list[dict]:
    """
    Rank candidate explanations from the model's own output.

    The model supplies the ranking. Supporting evidence is drawn only from
    symptoms the user actually reported, so nothing is invented.
    """
    import numpy as np

    probabilities = np.asarray(prob_array, dtype=float).flatten()
    classes = list(getattr(label_encoder, "classes_", range(len(probabilities))))
    order = np.argsort(probabilities)[::-1][:top_k]

    reported = entities.get("symptoms", [])
    missing = entities.get("missing_information", [])

    candidates = []
    for position, index in enumerate(order, start=1):
        raw = str(classes[index]) if index < len(classes) else f"Category {index}"
        score = float(probabilities[index])
        candidates.append({
            "rank": position,
            "raw_condition": raw,
            "condition": _condition_label(raw),
            "band": _band_for(score),
            "supporting": reported,
            "against": entities.get("explicitly_denied", []),
            "missing": missing,
            "score": score,
        })
    return candidates


# ---------------------------------------------------------------------------
# 4. UNCERTAINTY ENGINE  (spec 22)
# ---------------------------------------------------------------------------

def uncertainty_message(entities: dict, candidates: list[dict]) -> str | None:
    """Warn instead of pretending to be certain when the signal is weak."""
    if not candidates:
        return "We do not have enough information to suggest any possible cause."
    if candidates[0]["score"] < 0.35:
        return ("We don't have enough information to confidently distinguish between "
                "these possibilities.")
    return None


# ---------------------------------------------------------------------------
# 5. ADAPTIVE QUESTIONS  (spec 16)
# ---------------------------------------------------------------------------

QUESTION_BANK = {
    "headache": (
        ("sudden_onset", "Did the headache start suddenly, or build up gradually?",
         ["Suddenly", "Gradually", "Not sure"]),
        ("vision", "Any vision changes alongside the headache?", ["Yes", "No", "Not sure"]),
        ("fever_with_headache", "Do you also have a fever?", ["Yes", "No", "Not sure"]),
    ),
    "shortness of breath": (
        ("breath_severity", "Is the breathing difficulty severe, or mild?", ["Severe", "Mild", "Not sure"]),
        ("breath_rest", "Is it present at rest, or only on exertion?", ["At rest", "On exertion", "Not sure"]),
    ),
    "chest pain": (
        ("chest_exertion", "Does the chest pain come on with exertion?", ["Yes", "No", "Not sure"]),
        ("chest_radiate", "Does it spread to the arm, neck or jaw?", ["Yes", "No", "Not sure"]),
    ),
    "abdominal pain": (
        ("abdomen_location", "Where is the pain worst?", ["Upper", "Lower", "Right side", "Left side", "All over"]),
        ("abdomen_blood", "Any blood in vomit or stool?", ["Yes", "No", "Not sure"]),
    ),
    "fever": (
        ("fever_duration", "How many days has the fever lasted?", ["Under 3 days", "3-7 days", "More than a week"]),
        ("fever_breathing", "Any difficulty breathing with the fever?", ["Yes", "No", "Not sure"]),
    ),
    "cough": (
        ("cough_productive", "Is the cough dry, or producing phlegm?", ["Dry", "With phlegm", "Not sure"]),
        ("cough_duration", "How long have you had the cough?", ["Under 1 week", "1-3 weeks", "More than 3 weeks"]),
    ),
    "vomiting": (
        ("vomit_blood", "Any blood in the vomit?", ["Yes", "No", "Not sure"]),
        ("vomit_keep_fluids", "Can you keep fluids down?", ["Yes", "No", "Not sure"]),
    ),
}

DEFAULT_QUESTIONS = (
    ("duration_change", "Are your symptoms getting better, staying the same, or getting worse?",
     ["Better", "Same", "Worse"]),
    ("overall_severity", "Overall, how much are these symptoms affecting you?",
     ["Mildly", "Moderately", "Severely"]),
)


def adaptive_questions(entities: dict, limit: int = 4) -> list[dict]:
    """Pick only the questions relevant to the symptoms that were detected."""
    detected = entities.get("symptoms", [])
    chosen: list[dict] = []
    seen = set()

    for symptom in detected:
        for question_id, prompt, options in QUESTION_BANK.get(symptom, ()):
            if question_id in seen:
                continue
            seen.add(question_id)
            chosen.append({"id": question_id, "question": prompt, "options": options,
                           "triggered_by": symptom})
            if len(chosen) >= limit:
                return chosen

    if not chosen:
        for question_id, prompt, options in DEFAULT_QUESTIONS:
            chosen.append({"id": question_id, "question": prompt, "options": options,
                           "triggered_by": "general"})
    return chosen[:limit]


def answer_is_concerning(answers: dict) -> list[str]:
    """Interpret follow-up answers into additional warning signs."""
    concerning_pattern = {
        "sudden_onset": "Suddenly",
        "vision": "Yes",
        "breath_severity": "Severe",
        "breath_rest": "At rest",
        "chest_exertion": "Yes",
        "chest_radiate": "Yes",
        "abdomen_blood": "Yes",
        "fever_breathing": "Yes",
        "vomit_blood": "Yes",
        "vomit_keep_fluids": "No",
        "fever_duration": "More than a week",
        "cough_duration": "More than 3 weeks",
        "duration_change": "Worse",
        "overall_severity": "Severely",
    }
    hits = []
    for key, expected in concerning_pattern.items():
        if str(answers.get(key, "")).strip() == expected:
            hits.append(key)
    return hits


# ---------------------------------------------------------------------------
# 6. TRIAGE ENGINE  (spec 25)
# ---------------------------------------------------------------------------

TRIAGE_LEVELS = {
    1: {
        "level": 1, "label": "Emergency",
        "headline": "Seek urgent medical care",
        "summary": "Your symptoms include a warning sign that should not be assessed through an AI tool alone.",
        "tone": "emergency",
    },
    2: {
        "level": 2, "label": "Urgent",
        "headline": "Get medical evaluation soon",
        "summary": "What you have described should be reviewed by a healthcare professional soon.",
        "tone": "urgent",
    },
    3: {
        "level": 3, "label": "Routine appointment",
        "headline": "Consider seeing a healthcare professional",
        "summary": "There is no emergency warning sign, but a routine appointment would be sensible.",
        "tone": "routine",
    },
    4: {
        "level": 4, "label": "Self-care and monitoring",
        "headline": "Monitor your symptoms",
        "summary": "No warning signs were detected. Monitor how things change and seek care if they worsen.",
        "tone": "selfcare",
    },
    5: {
        "level": 5, "label": "Insufficient information",
        "headline": "We need a little more information",
        "summary": "There is not yet enough detail to assess urgency reliably.",
        "tone": "insufficient",
    },
}


NEXT_STEPS = {
    1: {
        "for_now": ["Call your local emergency number or go to the nearest emergency department.",
                    "Do not drive yourself if you feel faint, breathless or confused.",
                    "Share this summary with whoever treats you."],
        "consider": ["Any worsening of breathing, chest pain or alertness."],
        "urgent": ["Severe breathing difficulty", "Chest pain", "Fainting or confusion"],
    },
    2: {
        "for_now": ["Arrange an in-person medical review today or as soon as possible.",
                    "Rest and keep fluids up if you can safely swallow.",
                    "Note down when symptoms started and what makes them worse."],
        "consider": ["Symptoms that persist beyond a day or two."],
        "urgent": ["Severe breathing difficulty", "Chest pain", "Confusion", "Fainting"],
    },
    3: {
        "for_now": ["Book a routine appointment with a healthcare professional.",
                    "Keep a simple symptom log with dates and severity.",
                    "Mention any new or changed medication at your visit."],
        "consider": ["Symptoms that do not settle within a week."],
        "urgent": ["Severe breathing difficulty", "Chest pain", "Confusion", "Fainting"],
    },
    4: {
        "for_now": ["Rest, stay hydrated, and monitor your temperature if relevant.",
                    "Track how symptoms change over the next few days.",
                    "Re-assess with Aarogya if anything changes."],
        "consider": ["Symptoms that worsen or fail to improve."],
        "urgent": ["Severe breathing difficulty", "Chest pain", "Confusion", "Fainting"],
    },
    5: {
        "for_now": ["Answer the follow-up questions so the assessment can continue.",
                    "Describe how long symptoms have lasted and how severe they feel."],
        "consider": ["Anything that is changing quickly."],
        "urgent": ["Severe breathing difficulty", "Chest pain", "Confusion", "Fainting"],
    },
}


def triage(red_flags: list[dict], entities: dict, vitals: dict | None = None,
           answers: dict | None = None) -> dict:
    """Produce exactly one urgency level with its next actions."""
    vitals = vitals or {}
    answers = answers or {}

    if red_flags:
        level = 1
    else:
        concerning = answer_is_concerning(answers)
        moderate_vitals = (
            94 <= vitals.get("spo2", 98) < 96
            or vitals.get("temp", 37) >= 38.5
            or vitals.get("hr", 72) >= 120
            or vitals.get("glucose", 90) >= 250
        )
        if concerning or moderate_vitals:
            level = 2
        elif entities.get("symptoms") and entities.get("duration"):
            level = 3
        elif entities.get("symptoms"):
            level = 3
        else:
            level = 5

    if not entities.get("symptoms") and not red_flags:
        level = 5

    result = dict(TRIAGE_LEVELS[level])
    result["actions"] = NEXT_STEPS[level]
    result["red_flags"] = red_flags
    return result


# ---------------------------------------------------------------------------
# 7. CONVENIENCE ENTRY POINT
# ---------------------------------------------------------------------------

def run_assessment_pipeline(text: str, selected_symptoms, vitals: dict,
                            prob_array, label_encoder, answers: dict | None = None) -> dict:
    """Run every deterministic stage and return one assessment object."""
    entities = extract_clinical_entities(text, selected_symptoms)
    red_flags = screen_red_flags(text, vitals)
    entities["missing_information"] = [
        question["question"] for question in adaptive_questions(entities)
    ]
    candidates = build_differential(prob_array, label_encoder, entities)
    uncertainty = uncertainty_message(entities, candidates)
    triage_result = triage(red_flags, entities, vitals, answers)

    return {
        "entities": entities,
        "red_flags": red_flags,
        "differential": candidates,
        "uncertainty": uncertainty,
        "triage": triage_result,
        "questions": adaptive_questions(entities),
    }
