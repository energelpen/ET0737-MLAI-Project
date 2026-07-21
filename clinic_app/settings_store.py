"""Runtime-editable clinic settings, persisted in TinyDB.

These are the "details inside the model" an admin can safely tune from the web UI
*without* retraining or changing the model's decision line:

  - the monitor / "elevated" band cut-off (a soft triage aid, not the model's decision);
  - the plain-language copy shown for each triage tier on the patient result page;
  - the lifestyle tip attached to each known risk factor;
  - clinic branding (name + tagline) and whether the AI assistant is switched on.

The model's own recall-first *decision threshold* is deliberately NOT here — it is the
operating point chosen in notebook 02 §6.6 and is loaded read-only from the .meta.json,
so a clinic admin can't silently move the referral line and change who gets flagged.

Everything falls back to a hard-coded default, so a fresh database (or a settings key
added in a later version) always resolves to something sensible.
"""
from __future__ import annotations

import copy

from tinydb import Query

from . import config

# The single row in the `settings` table holds this shape. Defaults mirror the values
# the app shipped with, so behaviour is unchanged until an admin edits them.
DEFAULTS: dict = {
    "clinic_name": "ClearCheck",
    "clinic_tagline": "Diabetes Risk Screening",
    # Soft "monitor" band: p in [elevated_cutoff, decision_threshold) -> tier "elevated".
    "elevated_cutoff": config.ELEVATED_CUTOFF,
    "ai_enabled": True,
    # Headline / subtext / call-to-action per triage tier (patient result page).
    "tier_copy": {
        "high": {
            "headline": "Your answers suggest a higher risk of diabetes",
            "subtext": ("Based on your responses, we recommend a follow-up blood-glucose "
                        "test. This is a screening result, not a diagnosis — please speak "
                        "with the clinic team today."),
            "cta": "A clinician has been notified to review your result.",
        },
        "elevated": {
            "headline": "Your answers suggest a moderately elevated risk",
            "subtext": ("You are below the referral threshold, but some risk factors are "
                        "present. Small lifestyle changes now can make a real difference."),
            "cta": "Consider a routine glucose check at your next visit.",
        },
        "low": {
            "headline": "Your answers suggest a lower risk of diabetes",
            "subtext": ("Your responses don't show a strong pattern of diabetes risk right now. "
                        "Keep up healthy habits and screen periodically."),
            "cta": "No follow-up glucose test is indicated at this time.",
        },
    },
    # Lifestyle tip per risk-factor rule id (see insights._RULES). Editing these lets a
    # clinic reword the advice patients see without touching code.
    "factor_tips": {
        "HighBP": "Work with your clinician on blood-pressure control (diet, activity, medication).",
        "HighChol": "A diet lower in saturated fat and regular activity can improve cholesterol.",
        "BMI_obese": "Even a 5–7% weight loss meaningfully lowers diabetes risk.",
        "BMI_over": "Gradual weight loss through diet and activity reduces risk.",
        "Smoker": "Quitting smoking lowers your risk of diabetes and its complications.",
        "PhysInactive": "Aim for 150 minutes of moderate activity a week.",
        "LowFruit": "Add a daily serving of fruit for fibre and micronutrients.",
        "LowVeg": "Fill half your plate with vegetables at main meals.",
        "HvyAlcohol": "Reducing alcohol helps weight, blood pressure and liver health.",
        "Age60": "Risk rises with age — regular screening becomes more important.",
        "PoorGenHlth": "Discuss your overall health with your clinician.",
        "DiffWalk": "Mobility support and tailored activity can still help.",
        "HeartDisease": "Cardiovascular and metabolic risk are closely linked — stay in regular care.",
    },
}


def _table():
    # Imported lazily to keep db.py -> settings_store import direction one-way.
    from .db import get_db
    return get_db().table("settings")


def _deep_merge(base: dict, override: dict) -> dict:
    """Return `base` overlaid with `override`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def get_settings() -> dict:
    """Current settings, with any stored overrides layered onto the defaults."""
    row = _table().get(Query().key == "clinic")
    stored = row.get("value", {}) if row else {}
    return _deep_merge(DEFAULTS, stored)


def update_settings(changes: dict) -> dict:
    """Merge `changes` into the stored settings and return the full new settings."""
    table = _table()
    row = table.get(Query().key == "clinic")
    stored = row.get("value", {}) if row else {}
    merged = _deep_merge(stored, changes)
    if row:
        table.update({"value": merged}, Query().key == "clinic")
    else:
        table.insert({"key": "clinic", "value": merged})
    return _deep_merge(DEFAULTS, merged)


def reset_settings() -> None:
    """Drop all overrides, reverting to the shipped defaults."""
    _table().remove(Query().key == "clinic")
