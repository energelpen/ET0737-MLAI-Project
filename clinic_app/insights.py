"""Transparent, rule-based risk factors and lifestyle guidance for a patient result.

IMPORTANT: these are *not* the model's internal reasons (no SHAP here). They are
well-established diabetes risk factors that we check against the patient's own answers,
so the result page can explain — honestly and simply — which known factors are present
and which are modifiable. The numeric risk score itself comes from the XGBoost model.

The wording of each factor's tip and of the per-tier headline/subtext/CTA is read from
the editable settings (settings_store.py), so a clinic admin can reword the guidance
without changing the underlying rules.
"""
from __future__ import annotations

from . import settings_store

# (rule_id, predicate on features, label, is_modifiable). The tip text is looked up by
# rule_id from settings.factor_tips, so it can be edited in the admin panel.
_RULES = [
    ("HighBP", lambda f: f.get("HighBP") == 1, "High blood pressure", True),
    ("HighChol", lambda f: f.get("HighChol") == 1, "High cholesterol", True),
    ("BMI_obese", lambda f: f.get("BMI", 0) >= 30, "Obesity (BMI 30+)", True),
    ("BMI_over", lambda f: 25 <= f.get("BMI", 0) < 30, "Overweight (BMI 25–29.9)", True),
    ("Smoker", lambda f: f.get("Smoker") == 1, "History of smoking", True),
    ("PhysInactive", lambda f: f.get("PhysActivity") == 0, "Physically inactive", True),
    ("LowFruit", lambda f: f.get("Fruits") == 0, "Low fruit intake", True),
    ("LowVeg", lambda f: f.get("Veggies") == 0, "Low vegetable intake", True),
    ("HvyAlcohol", lambda f: f.get("HvyAlcoholConsump") == 1, "Heavy alcohol consumption", True),
    ("Age60", lambda f: f.get("AgeYears", 0) >= 60, "Age 60 or over", False),
    ("PoorGenHlth", lambda f: f.get("GenHlth", 0) >= 4, "Fair or poor general health", False),
    ("DiffWalk", lambda f: f.get("DiffWalk") == 1, "Difficulty walking", False),
    ("HeartDisease", lambda f: f.get("HeartDiseaseorAttack") == 1, "Existing heart disease", False),
    ("Stroke", lambda f: f.get("Stroke") == 1, "History of stroke", False),
    ("Kidney", lambda f: f.get("KidneyDisease") == 1, "Kidney disease", False),
]


def present_risk_factors(features: dict) -> list[dict]:
    """Known diabetes risk factors present in this patient's answers."""
    tips = settings_store.get_settings()["factor_tips"]
    found = []
    for rule_id, pred, label, modifiable in _RULES:
        if pred(features):
            found.append({
                "id": rule_id,
                "label": label,
                "modifiable": modifiable,
                "tip": tips.get(rule_id),  # None if this factor has no configured tip
            })
    return found


def lifestyle_tips(features: dict) -> list[str]:
    """Actionable tips for the *modifiable* factors this patient has."""
    return [rf["tip"] for rf in present_risk_factors(features)
            if rf["modifiable"] and rf["tip"]]


def result_copy(assessment: dict) -> dict:
    """Headline + guidance text keyed to the triage tier (editable in settings)."""
    tier = assessment["tier"]
    tier_copy = settings_store.get_settings()["tier_copy"]
    return tier_copy.get(tier, tier_copy["low"])
