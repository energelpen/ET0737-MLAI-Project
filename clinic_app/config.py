"""Central configuration for the diabetes-screening clinic app.

Everything the rest of the package needs to find on disk lives here, so paths are
defined once. The project layout is:

    <project root>/
        models/   diabetes_stack_recall_first.joblib (+ .meta.json)  <- exported by notebook 02 §12
        data/     clinic_db.json                  (TinyDB, created at runtime)
        clinic_app/  <- this package
"""
from __future__ import annotations

import os
from pathlib import Path

# clinic_app/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "clinic_db.json"


def _load_dotenv() -> None:
    """Load KEY=value lines from a project-root `.env` into os.environ.

    A tiny, dependency-free reader so secrets (e.g. ANTHROPIC_API_KEY) can live in an
    untracked `.env` file instead of being typed into the shell each time. Existing
    environment variables always win, so a value set in the real environment is never
    overridden by the file. Runs at import, before any os.environ.get() below.
    """
    env_path = BASE_DIR / ".env"
    try:
        text = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip surrounding quotes and inline whitespace from the value.
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Model artifacts written by notebook 02 §12. The whole fitted pipeline (preprocessor +
# SVM+XGBoost stacking ensemble + Logistic Regression meta-learner) is a single joblib file,
# paired with a JSON sidecar carrying the recall-first threshold and feature order.
PIPE_PATH = MODELS_DIR / "diabetes_stack_recall_first.joblib"
META_PATH = MODELS_DIR / "diabetes_stack_recall_first.meta.json"

# Flask session signing key. Override in real deployments via the environment.
SECRET_KEY = os.environ.get("CLINIC_SECRET", "dev-clinic-secret-change-me-in-production")

# Triage banding. The model's own recall-first cut-off (loaded from the .meta.json,
# ~0.485) is the REFER line — at or above it the patient is flagged high-risk. Below
# that we still surface an "elevated / monitor" band as a soft clinical triage aid;
# it is a UX convenience, not part of the model's decision. This is only the *default*:
# an admin can adjust the monitor-band cut-off at runtime (see settings_store.py). The
# decision threshold itself is fixed to the notebook's operating point and is NOT editable.
ELEVATED_CUTOFF = 0.30

# Roles used across the app. `admin` can do everything a `clinician` can, plus manage
# settings and accounts; `patient` sees only their own screening history.
ROLES = ("admin", "clinician", "patient")

# Seed staff accounts created on first run (demo credentials — change for real use).
DEFAULT_ADMIN = {
    "username": "admin",
    "password": "admin123",
    "name": "Clinic Administrator",
    "role": "admin",
}
DEFAULT_DOCTOR = {
    "username": "doctor",
    "password": "clinic123",
    "name": "Dr. Alex Tan",
    "role": "clinician",
}

# --- AI (Claude) assistant -------------------------------------------------------
# Agentic plain-language guidance for patients and briefings for clinicians. Uses the
# smallest Claude model; the feature degrades gracefully to the rule-based text when no
# API key is configured. Set ANTHROPIC_API_KEY in the environment to enable it.
AI_MODEL = os.environ.get("CLINIC_AI_MODEL", "claude-haiku-4-5")
AI_API_KEY_ENV = "ANTHROPIC_API_KEY"
