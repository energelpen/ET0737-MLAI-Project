"""Aggregate screening records into the figures the doctor dashboard renders.

Everything is computed in Python and handed to the template as plain numbers; the
charts are server-rendered CSS/SVG bars (no client-side charting library), so the
dashboard is fully self-contained and works offline.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from .schema import YESNO_FEATURES

# Risk factors we track prevalence for on the dashboard (feature -> label).
TRACKED_FACTORS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "BMI30": "Obesity (BMI 30+)",
    "Smoker": "Smoker",
    "PhysInactive": "Physically inactive",
    "DiffWalk": "Difficulty walking",
    "Age60": "Age 60+",
    "PoorGenHlth": "Fair/poor health",
    "HeartDiseaseorAttack": "Heart disease",
}


def _has_factor(features: dict, key: str) -> bool:
    if key == "BMI30":
        return features.get("BMI", 0) >= 30
    if key == "PhysInactive":
        return features.get("PhysActivity") == 0
    if key == "Age60":
        return features.get("AgeYears", 0) >= 60
    if key == "PoorGenHlth":
        return features.get("GenHlth", 0) >= 4
    return features.get(key) == 1


def _age_band(age: int) -> str:
    if age < 40:
        return "18–39"
    if age < 50:
        return "40–49"
    if age < 60:
        return "50–59"
    if age < 70:
        return "60–69"
    return "70+"


def _bmi_band(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def summarize(records: list[dict]) -> dict:
    """Build the full dashboard payload from all patient records."""
    total = len(records)
    tiers = Counter(r.get("tier", "low") for r in records)
    high = tiers.get("high", 0)
    elevated = tiers.get("elevated", 0)
    low = tiers.get("low", 0)

    probs = [r.get("probability", 0) for r in records]
    avg_prob = round(sum(probs) / total, 3) if total else 0.0

    now = datetime.now(timezone.utc)
    today = now.date()
    week_ago = now - timedelta(days=7)

    def _dt(rec):
        try:
            return datetime.fromisoformat(rec["created_at"])
        except (KeyError, ValueError):
            return None

    screened_today = sum(1 for r in records if (d := _dt(r)) and d.date() == today)
    screened_week = sum(1 for r in records if (d := _dt(r)) and d >= week_ago)
    pending_review = sum(1 for r in records if r.get("tier") == "high" and not r.get("reviewed"))

    # Risk-factor prevalence: overall vs among the high-risk cohort.
    high_recs = [r for r in records if r.get("tier") == "high"]
    factor_rows = []
    for key, label in TRACKED_FACTORS.items():
        overall_n = sum(1 for r in records if _has_factor(r.get("features", {}), key))
        high_n = sum(1 for r in high_recs if _has_factor(r.get("features", {}), key))
        factor_rows.append({
            "label": label,
            "overall_pct": _pct(overall_n, total),
            "high_pct": _pct(high_n, len(high_recs)),
            "high_n": high_n,
        })
    factor_rows.sort(key=lambda x: x["high_pct"], reverse=True)

    # Distributions for the demographic charts.
    age_counter: Counter = Counter()
    bmi_counter: Counter = Counter()
    gen_counter: Counter = Counter()
    for r in records:
        f = r.get("features", {})
        if "AgeYears" in f:
            age_counter[_age_band(f["AgeYears"])] += 1
        if "BMI" in f:
            bmi_counter[_bmi_band(f["BMI"])] += 1
        if "GenHlth" in f:
            gen_counter[f["GenHlth"]] += 1

    age_order = ["18–39", "40–49", "50–59", "60–69", "70+"]
    bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
    gen_labels = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

    age_dist = [{"label": b, "count": age_counter.get(b, 0),
                 "pct": _pct(age_counter.get(b, 0), total)} for b in age_order]
    bmi_dist = [{"label": b, "count": bmi_counter.get(b, 0),
                 "pct": _pct(bmi_counter.get(b, 0), total)} for b in bmi_order]
    gen_dist = [{"label": gen_labels[k], "count": gen_counter.get(k, 0),
                 "pct": _pct(gen_counter.get(k, 0), total)} for k in range(1, 6)]

    return {
        "total": total,
        "high": high, "elevated": elevated, "low": low,
        "high_pct": _pct(high, total),
        "elevated_pct": _pct(elevated, total),
        "low_pct": _pct(low, total),
        "avg_prob": avg_prob,
        "avg_prob_pct": round(avg_prob * 100, 1),
        "screened_today": screened_today,
        "screened_week": screened_week,
        "pending_review": pending_review,
        "factor_rows": factor_rows,
        "age_dist": age_dist,
        "bmi_dist": bmi_dist,
        "gen_dist": gen_dist,
    }
