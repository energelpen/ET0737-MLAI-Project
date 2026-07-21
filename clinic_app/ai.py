"""Optional AI assistant: Claude-generated guidance for patients and clinicians.

This is a thin, defensive wrapper around the Anthropic Messages API using the smallest
Claude model (Haiku). It is entirely optional:

  * If the `anthropic` package isn't installed, or no API key is set, or the admin has
    switched the assistant off, every helper returns None and the UI simply shows the
    existing rule-based text. Nothing here can break a screening.

Both prompts are grounded in the patient's *structured* fields (the same ones the model
scored and the rule engine surfaced), and the system prompt forbids diagnosis and forces
a screening disclaimer — the model summarises what's already known, it does not invent
new clinical claims.
"""
from __future__ import annotations

import os

from . import config, settings_store
from .schema import humanize

# --- lazy, cached client --------------------------------------------------------
_client = "unset"  # sentinel: not yet initialised


def _get_client():
    """Return an Anthropic client, or None if the SDK/key isn't available."""
    global _client
    if _client == "unset":
        api_key = os.environ.get(config.AI_API_KEY_ENV)
        if not api_key:
            _client = None
        else:
            try:
                import anthropic
                _client = anthropic.Anthropic(api_key=api_key)
            except Exception:  # SDK missing or failed to construct
                _client = None
    return _client


def is_configured() -> bool:
    """True when an API key + SDK are present (independent of the admin on/off switch)."""
    return _get_client() is not None


def is_enabled() -> bool:
    """True when the assistant is both configured and switched on in settings."""
    return is_configured() and bool(settings_store.get_settings().get("ai_enabled", True))


def status() -> dict:
    """Small dict for the admin panel to render the assistant's state."""
    return {
        "sdk_and_key": is_configured(),
        "toggled_on": bool(settings_store.get_settings().get("ai_enabled", True)),
        "model": config.AI_MODEL,
        "key_env": config.AI_API_KEY_ENV,
    }


# --- prompt building ------------------------------------------------------------
def _patient_context(record: dict, factors: list[dict]) -> str:
    """A compact, readable digest of a screening for the model to reason over."""
    f = record.get("features", {})
    lines = [
        f"Screening risk score: {record.get('percent')}% (model triage tier: {record.get('tier')}).",
        f"Age: {f.get('AgeYears', 'n/a')}. Sex: {humanize('Sex', f['Sex']) if 'Sex' in f else 'n/a'}. "
        f"BMI: {f.get('BMI', 'n/a')}.",
    ]
    present = [rf["label"] for rf in factors]
    lines.append("Known risk factors present: " + (", ".join(present) if present else "none flagged") + ".")
    return "\n".join(lines)


def _generate(system: str, user: str, max_tokens: int = 500) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.messages.create(
            model=config.AI_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception:
        return None
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    text = "".join(parts).strip()
    return text or None


# --- public helpers -------------------------------------------------------------
_PATIENT_SYSTEM = (
    "You are a friendly health-screening assistant at a community clinic. You are given "
    "the structured results of a diabetes-RISK questionnaire (a screening tool, not a "
    "diagnostic test). Write a short, warm, plain-language note for the patient in 2–3 "
    "short paragraphs. Explain what their result means in everyday terms, gently highlight "
    "the modifiable risk factors present, and give encouraging, practical next steps. "
    "Rules: never state or imply a diagnosis; never invent facts beyond what you're given; "
    "do not give specific medication or dosage advice; always remind them this is a "
    "screening result to discuss with a clinician. Do not use markdown headings."
)

_CLINICIAN_SYSTEM = (
    "You are a clinical decision-support assistant summarising a diabetes-risk screening "
    "for a busy clinician. Produce a concise briefing (a one-line impression, then 2–4 "
    "short bullet points) covering the risk score/tier, the most salient modifiable and "
    "non-modifiable factors, and suggested triage actions consistent with a screening "
    "(e.g. fasting glucose / HbA1c follow-up, lifestyle counselling). Rules: this is "
    "decision support, not a diagnosis or an order; be factual and grounded only in the "
    "data provided; flag if information is limited. Keep it under ~120 words."
)


def patient_guidance(record: dict, factors: list[dict]) -> str | None:
    if not is_enabled():
        return None
    return _generate(_PATIENT_SYSTEM, _patient_context(record, factors), max_tokens=500)


def clinician_briefing(record: dict, factors: list[dict]) -> str | None:
    if not is_enabled():
        return None
    return _generate(_CLINICIAN_SYSTEM, _patient_context(record, factors), max_tokens=350)
