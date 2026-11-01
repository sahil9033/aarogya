"""End-to-end smoke tests for the paths that must never regress.

These assert behaviour, not implementation: what the patient is shown, and the
guarantees the safety layer makes. Run from the ``backend`` directory:

    pip install pytest
    pytest
"""

from __future__ import annotations

FEVER_TEXT = "Fever since yesterday evening with a dry cough, no chest pain."
EMERGENCY_TEXT = "Crushing chest pain and I cannot breathe."


def _assess(client, **overrides):
    payload = {
        "symptom_text": FEVER_TEXT,
        "selected_symptoms": ["fever", "cough"],
        "session_id": "test-session",
    }
    payload.update(overrides)
    return client.post("/api/v1/assessments", json=payload)


def test_health_reports_a_working_engine(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["engine_loaded"] is True
    assert body["database"] == "sqlite"


def test_questionnaire_asks_follow_ups_when_nothing_is_flagged(client):
    body = client.post(
        "/api/v1/assessments/questionnaire",
        json={"symptom_text": FEVER_TEXT, "selected_symptoms": ["fever", "cough"]},
    ).json()
    assert body["red_flags"] == []
    assert body["follow_up_questions"]


def test_questionnaire_skips_questions_when_a_warning_sign_is_present(client):
    body = client.post(
        "/api/v1/assessments/questionnaire",
        json={"symptom_text": EMERGENCY_TEXT},
    ).json()
    assert body["red_flags"]
    # Walking a patient with a warning sign through ordinary questions is the
    # failure mode this guard exists to prevent.
    assert body["follow_up_questions"] == []


def test_assessment_reports_likelihood_as_bands_never_percentages(client):
    response = _assess(client)
    assert response.status_code == 201
    body = response.json()

    assert body["differential"], "expected at least one candidate"
    for candidate in body["differential"]:
        assert candidate["band"] in {
            "More likely",
            "Possible",
            "Less likely",
            "Cannot rule out",
        }
        assert "%" not in candidate["band"]


def test_negated_symptoms_are_recorded_as_absent(client):
    body = _assess(client).json()
    assert "chest pain" in body["entities"]["explicitly_denied"]
    assert "chest pain" not in body["entities"]["symptoms"]


def test_red_flags_force_the_emergency_level(client):
    body = _assess(
        client,
        symptom_text=EMERGENCY_TEXT,
        vitals={"age": 61, "heart_rate": 128, "systolic_bp": 88, "spo2": 88},
    ).json()

    assert body["triage"]["level"] == 1
    assert body["red_flags"]
    # A narrative is always produced, with or without an LLM provider.
    assert body["narrative"]


def test_chat_admits_uncertainty_instead_of_inventing_an_answer(client):
    _assess(client, session_id="plain-session")
    chat = client.post(
        "/api/v1/chat",
        json={"message": "Do I need to see a doctor?", "session_id": "plain-session"},
    ).json()

    assert chat["grounded"] is False, "nothing has been ingested to quote from"
    assert chat["reply"]
    assert chat["safety_notice"] is None
    assert "not serious" not in chat["reply"].lower()


def test_chat_cannot_be_used_to_step_around_an_assessment_red_flag(client):
    created = _assess(client, symptom_text=EMERGENCY_TEXT, session_id="escalate-session").json()

    chat = client.post(
        "/api/v1/chat",
        json={
            "message": "What should I eat tonight?",
            "session_id": "escalate-session",
            "assessment_id": created["id"],
        },
    ).json()

    # A mild question must not be a route past the warning sign.
    assert chat["safety_notice"] is not None
    assert "emergency" in chat["reply"].lower()


def test_knowledge_document_round_trips_and_is_retrievable(client):
    created = client.post(
        "/api/v1/knowledge",
        json={
            "title": "Sepsis warning signs",
            "content": "Seek emergency care for confusion, rapid breathing, mottled skin "
                       "or a high temperature with a racing heart.",
            "source": "test fixture",
        },
    )
    assert created.status_code == 201
    document = created.json()
    assert document["has_embedding"] is True

    listing = client.get("/api/v1/knowledge").json()
    assert any(item["id"] == document["id"] for item in listing)

    results = client.post(
        "/api/v1/knowledge/search",
        json={"query": "confusion rapid breathing emergency", "min_score": 0.0},
    ).json()
    assert results
