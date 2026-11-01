"""Narrative summary generation.

The narrative is a *presentation* of findings the deterministic engine already
produced. It is never allowed to introduce a new clinical claim.

When an LLM provider is configured it is given only the structured facts and is
instructed to rephrase them. When no provider is configured -- which is the
default, and the state of every test run -- a deterministic template produces
the same summary from the same facts. Callers always receive a narrative, so
the API never degrades to an empty field.
"""

from __future__ import annotations

import json

from app.services.llm import chat_completion, provider_ready

SYSTEM_PROMPT = (
    "You are the explanation layer of a symptom-checking tool. You are given "
    "structured findings that were already computed by deterministic rules. "
    "Rewrite them as two or three short, calm sentences for the patient.\n"
    "Rules you must not break:\n"
    "- Do not add any symptom, condition, medicine, dose or diagnosis that is "
    "not present in the findings.\n"
    "- Do not state or imply how likely a condition is; the findings use bands "
    "for exactly that reason.\n"
    "- Do not soften or contradict the recommended urgency.\n"
    "- If the findings contain a warning sign, say plainly that this must be "
    "assessed by a clinician and do not offer reassurance.\n"
    "- Use plain language and no markdown."
)


def _facts(
    entities: dict,
    red_flags: list[dict],
    candidates: list[dict],
    triage_result: dict,
    uncertainty: str | None,
) -> dict:
    return {
        "reported_symptoms": entities.get("symptoms", []),
        "reported_as_absent": entities.get("explicitly_denied", []),
        "duration": entities.get("duration"),
        "severity_0_to_10": entities.get("severity"),
        "warning_signs": [flag["label"] for flag in red_flags],
        "possibilities": [
            {"condition": item["condition"], "likelihood_band": item["band"]}
            for item in candidates
        ],
        "urgency_level_1_to_5": triage_result.get("level"),
        "urgency_label": triage_result.get("label"),
        "uncertainty": uncertainty,
    }


def deterministic_narrative(
    entities: dict,
    red_flags: list[dict],
    candidates: list[dict],
    triage_result: dict,
    uncertainty: str | None = None,
) -> str:
    """Compose the summary from the findings alone, with no model involved."""
    sentences: list[str] = []

    symptoms = entities.get("symptoms", [])
    if symptoms:
        listed = ", ".join(symptoms[:4])
        if len(symptoms) > 4:
            listed += f" and {len(symptoms) - 4} more"
        sentences.append(f"You described {listed}.")
    else:
        sentences.append("We could not reliably identify specific symptoms from the description you gave.")

    if red_flags:
        sentences.append(
            "You also reported "
            + " and ".join(flag["label"].lower() for flag in red_flags)
            + ", which needs to be assessed by a clinician rather than through this tool."
        )

    if candidates:
        top = candidates[0]
        others = len(candidates) - 1
        if others:
            sentences.append(
                f"The pattern is most consistent with {top['condition']} "
                f"({top['band'].lower()}), with {others} other possibility"
                f"{'ies' if others > 1 else ''} also considered."
            )
        else:
            sentences.append(f"The pattern is most consistent with {top['condition']} ({top['band'].lower()}).")
    elif not red_flags:
        sentences.append("Nothing in what you described points to a specific common condition.")

    if uncertainty:
        sentences.append(uncertainty)

    label = triage_result.get("label")
    if label:
        sentences.append(f"Recommended urgency: {label}. {triage_result.get('headline', '')}".strip())

    sentences.append("This is health information, not a diagnosis.")
    return " ".join(sentences)


async def generate_narrative(
    entities: dict,
    red_flags: list[dict],
    candidates: list[dict],
    triage_result: dict,
    uncertainty: str | None = None,
    enabled: bool = True,
) -> tuple[str, str]:
    """Return ``(narrative, source)`` where source is a provider name or 'deterministic'."""
    fallback = deterministic_narrative(entities, red_flags, candidates, triage_result, uncertainty)

    if not enabled or provider_ready() is None:
        return fallback, "deterministic"

    user_prompt = (
        "Findings (JSON):\n"
        + json.dumps(_facts(entities, red_flags, candidates, triage_result, uncertainty), indent=2, default=str)
        + "\n\nWrite the patient-facing summary now."
    )

    # A narration failure must never fail the assessment: the deterministic text
    # already contains every required fact.
    result = await chat_completion(SYSTEM_PROMPT, user_prompt, max_tokens=400)
    if result is None:
        return fallback, "deterministic"

    text, provider = result
    return text, provider
