"""Central configuration for the diabetes-screening clinic app.

Everything the rest of the package needs to find on disk lives here, so paths are
defined once. The project layout is:

    <project root>/
        models/   diabetes_risk_model.joblib (+ .meta.json)  <- exported by notebook 02 §12
        data/     clinic_db.json                  (TinyDB, created at runtime)
        clinic_app/  <- this package
"""
from __future__ import annotations

import os
from pathlib import Path

# clinic_app/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Where mutable runtime state (the TinyDB file) lives.
#
# Locally this is <project>/data and nothing needs configuring. On a hosted platform the
# application directory is usually EPHEMERAL — Render rebuilds the container on every deploy,
# so anything written next to the code is lost. Pointing CLINIC_DATA_DIR at a mounted disk
# (e.g. /var/data) is what makes patient records survive a redeploy. See render.yaml.
DATA_DIR = Path(os.environ.get("CLINIC_DATA_DIR") or (BASE_DIR / "data"))
DB_PATH = DATA_DIR / "clinic_db.json"

# Render sets this automatically; it is the most reliable "am I in production?" signal we get.
# It only ever tightens behaviour — every guard below is a no-op on a local machine.
ON_RENDER = os.environ.get("RENDER", "").lower() == "true"


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

# Model artifacts written by notebook 02 §12. The whole fitted estimator -- the preprocessor and
# whichever classifier §6.4b selected, wrapped in the isotonic calibrator from §6.7 -- is a single
# joblib file, paired with a JSON sidecar carrying the recall-first threshold and feature order.
#
# The filenames are deliberately named for the artifact's ROLE, not for the winning algorithm. The
# notebook compares nine models and picks one by rule, so hard-coding an algorithm name here
# would mean editing this file every time that comparison is re-run. Read the actual model name
# from the sidecar (`model_service.RiskModel.model_name`) instead -- the admin model card does.
#
# The model is CALIBRATED, which is what makes `probability * 100` safe to show a patient: the raw
# model was over-confident by ~2.5x (it read "85%" where the true rate was ~50%). Because
# calibration only rescales the probability, the threshold moved but the referral decisions did
# not. Do NOT swap in an uncalibrated artifact without revisiting the displayed percentages.
PIPE_PATH = MODELS_DIR / "diabetes_risk_model.joblib"
META_PATH = MODELS_DIR / "diabetes_risk_model.meta.json"

# Flask session signing key. Override in real deployments via the environment.
_DEV_SECRET = "dev-clinic-secret-change-me-in-production"
SECRET_KEY = os.environ.get("CLINIC_SECRET", _DEV_SECRET)

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

# Seed staff accounts created on first run.
#
# The demo credentials below are PUBLIC — they are in the README, and letting a visitor sign in
# as `admin` / `admin123` is the point of a showcase deployment. This app is a simulation: the
# patient records are synthetic, there is no real PII, and the whole thing exists to demonstrate
# the model. So the defaults are allowed to stand anywhere, including on a public URL.
#
# They are still overridable, which is what a real deployment would use:
#     CLINIC_ADMIN_USER / CLINIC_ADMIN_PASSWORD
#     CLINIC_DOCTOR_USER / CLINIC_DOCTOR_PASSWORD
#
# `db._seed_staff()` only creates an account that does not already exist, so changing these
# later does NOT rotate an existing password — delete the user (or the DB file) to re-seed.
_DEV_ADMIN_PW = "admin123"
_DEV_DOCTOR_PW = "clinic123"

DEFAULT_ADMIN = {
    "username": os.environ.get("CLINIC_ADMIN_USER", "admin"),
    "password": os.environ.get("CLINIC_ADMIN_PASSWORD", _DEV_ADMIN_PW),
    "name": "Clinic Administrator",
    "role": "admin",
}
DEFAULT_DOCTOR = {
    "username": os.environ.get("CLINIC_DOCTOR_USER", "doctor"),
    "password": os.environ.get("CLINIC_DOCTOR_PASSWORD", _DEV_DOCTOR_PW),
    "name": "Dr. Alex Tan",
    "role": "clinician",
}


def _warn_public_defaults() -> None:
    """Record which public defaults are live on a hosted deploy — log only, never fatal.

    Deliberately NOT an error. This is a demo app and its credentials are meant to be shared;
    blocking startup over that would break the thing it is for. The warning exists so the fact
    is visible in the Render logs rather than forgotten, and so that if this app is ever reused
    with real data the reader has already been told exactly which knobs to set.
    """
    import sys

    public = []
    if SECRET_KEY == _DEV_SECRET:
        public.append("CLINIC_SECRET (session cookies are signed with the repo's public key, "
                      "so they can be forged)")
    if DEFAULT_ADMIN["password"] == _DEV_ADMIN_PW:
        public.append("CLINIC_ADMIN_PASSWORD (admin/admin123)")
    if DEFAULT_DOCTOR["password"] == _DEV_DOCTOR_PW:
        public.append("CLINIC_DOCTOR_PASSWORD (doctor/clinic123)")
    if not public:
        return
    print("[clinic_app] NOTE - running on a hosted deployment with public demo credentials:",
          file=sys.stderr)
    for item in public:
        print(f"[clinic_app]   * {item}", file=sys.stderr)
    print("[clinic_app] Intended for this screening demo (synthetic data, no real PII). "
          "Set the variables above before reusing this app with anything real.", file=sys.stderr)


if ON_RENDER:
    _warn_public_defaults()

# --- AI (Claude) assistant -------------------------------------------------------
# Agentic plain-language guidance for patients and briefings for clinicians. Uses the
# smallest Claude model; the feature degrades gracefully to the rule-based text when no
# API key is configured. Set ANTHROPIC_API_KEY in the environment to enable it.
AI_MODEL = os.environ.get("CLINIC_AI_MODEL", "claude-haiku-4-5")
AI_API_KEY_ENV = "ANTHROPIC_API_KEY"
