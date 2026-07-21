"""Populate the clinic with a spread of realistic demo patients.

    python -m clinic_app.seed_demo          # add demo patients
    python -m clinic_app.seed_demo --reset  # wipe existing patients first

Handy for showing the dashboard with data before any real screenings exist. It samples
real rows from the cleaned BRFSS dataset so the demo cohort reflects genuine patterns.
"""
from __future__ import annotations

import random
import sys

import pandas as pd

from . import config, db
from .model_service import get_model

_FIRST = ["Alex", "Priya", "Wei", "Mira", "Jordan", "Sam", "Noor", "Diego", "Aisha",
          "Ken", "Lena", "Omar", "Rosa", "Tariq", "Ivy", "Marcus", "Hana", "Leo"]
_LAST = ["Tan", "Kumar", "Lee", "Silva", "Ng", "Osei", "Patel", "Garcia", "Wong",
         "Ahmed", "Rossi", "Chen", "Adams", "Khan", "Lim", "Diaz"]


def seed(n: int = 40, reset: bool = False) -> None:
    if reset:
        db.clear_patients()

    model = get_model()
    df = pd.read_csv(config.BASE_DIR / "data" / "processed" / "diabetes_2015_clean.csv")
    sample = df.sample(n=min(n, len(df)), random_state=7)

    rng = random.Random(7)
    added = 0
    for _, row in sample.iterrows():
        features = {c: (float(row[c]) if c == "BMI" else int(row[c])) for c in model.feature_order}
        assessment = model.assess(features)
        h = rng.randint(150, 190)
        # back out a plausible weight from BMI so the record looks like a real intake
        w = round(features["BMI"] * (h / 100) ** 2, 1)
        db.add_patient({
            "patient_name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "height_cm": h, "weight_kg": w,
            "features": features,
            "probability": assessment["probability"],
            "percent": assessment["percent"],
            "tier": assessment["tier"],
            "refer": assessment["refer"],
        })
        added += 1

    print(f"Seeded {added} demo patients into {config.DB_PATH}")


if __name__ == "__main__":
    seed(reset="--reset" in sys.argv)
