# ClearCheck — Diabetes Screening Clinic (web app)

An interactive Flask web application that turns the exported stacking-ensemble model
(SVM + XGBoost → Logistic Regression meta-learner, notebook 02 §12) into a working clinic
tool with three roles:

- **Patients** register, complete a diabetes-risk questionnaire, get a clear friendly
  result, and see their own screening history at `/portal`. Walk-in (anonymous)
  screening still works without an account.
- **Clinicians** sign in to an analytics dashboard, see which patients were flagged
  **high-risk (refer)**, drill into any record, and mark it reviewed.
- **Admins** get everything a clinician has, plus an **admin panel** (`/admin`) to manage
  accounts and edit the clinic-facing details of the model — see below.

An optional **AI assistant** (Claude Haiku) adds a plain-language "what this means for
you" note on the patient result and a concise clinical briefing on the clinician's
patient page. It's grounded in the structured fields, carries a screening disclaimer, and
**degrades gracefully** to the rule-based text when no API key is set.

Everything is Python + server-rendered HTML (Jinja templates) styled with **Tailwind
CSS**; storage is **TinyDB** (a single JSON file — no database server to run).

---

## Quick start

```bash
# 1. install deps (into the environment that has the notebook's packages)
pip install -r requirements.txt

# 2. (optional) populate the dashboard with 40 realistic demo patients
python -m clinic_app.seed_demo --reset

# 3. run
python run_clinic.py
# open http://127.0.0.1:5000/
```

**Demo logins (seeded on first run):**

| Role | Username | Password | Lands on |
|---|---|---|---|
| Admin | `admin` | `admin123` | `/admin` |
| Clinician | `doctor` | `clinic123` | `/staff` |
| Patient | *register at `/register`* | — | `/portal` |

**Optional AI assistant.** Set the `ANTHROPIC_API_KEY` environment variable and
`pip install anthropic` to switch it on (uses the smallest Claude model, `claude-haiku-4-5`).
The admin panel shows whether the key/SDK are detected and lets an admin toggle it off.

The app loads the model artifacts from `../models/` at startup, so run notebook 02 §12
first if `models/diabetes_stack_recall_first.joblib` is missing.

---

## What the admin can edit

The admin panel tunes the **clinic-facing details** of the model without retraining:

- clinic name + tagline;
- the **monitor / "elevated" band cutoff** (a soft triage aid);
- the plain-language headline / explanation / call-to-action for each triage tier;
- the lifestyle tip attached to each known risk factor;
- whether the AI assistant is switched on.

The model's own **decision (referral) threshold is deliberately locked** — it's the
recall-first operating point chosen in the notebook, shown read-only so the clinic can't
silently move the referral line. Editable settings live in the `settings` TinyDB table
(`settings_store.py`) and fall back to shipped defaults.

---

## How it fits the model

The app scores patients with the **whole fitted pipeline** from §12 — one joblib file
holding the preprocessor, the SVM + XGBoost stacking ensemble and its Logistic Regression
meta-learner — the exact, bit-for-bit pipeline evaluated in the notebook (no retraining).
The `.meta.json` supplies the 29-feature order and the **recall-first decision threshold**;
at or above it a patient is flagged *high-risk / refer*.

| Triage band | Rule | Meaning |
|---|---|---|
| **High** | `p ≥ threshold` | Refer for a glucose test (model's operating point) |
| **Elevated** | `0.30 ≤ p < threshold` | Monitor — a soft UX triage aid, not the model's line |
| **Low** | `p < 0.30` | No follow-up indicated |

BMI is not asked directly — the questionnaire takes height + weight and computes it, the
way a clinic actually measures.

---

## Architecture

```
clinic_app/
  config.py         paths, secret key, triage defaults, seed accounts, AI config
  model_service.py  loads the .joblib pipeline + .meta.json; scores + bands a patient
  settings_store.py runtime-editable clinic settings (admin panel) with defaults
  db.py             TinyDB: user accounts (3 roles) + patient records + settings
  schema.py         the 29-feature questionnaire + BRFSS code→label maps + parsing/validation
  insights.py       transparent rule-based risk factors + patient guidance (editable copy)
  ai.py             optional Claude assistant (patient + clinician briefings), graceful fallback
  analytics.py      aggregates records into the dashboard figures
  routes.py         patient / clinician / admin flows with role-based auth
  seed_demo.py      sample real dataset rows to fill the demo dashboard
  templates/        Tailwind + Jinja pages (dark-mode aware)
run_clinic.py       entry point
```

Design notes: the dashboard charts are **server-rendered CSS bars** (no client charting
library), so it works fully offline; the risk colours come from a validated,
colourblind-safe status palette (red/amber/green). Only two tiny bits of JS exist (the
dark-mode toggle and smooth section scrolling).

---

## Ideas to make it better (next steps)

**Model & explanation**
- Show **per-patient SHAP contributions** instead of only rule-based factors, so a
  clinician sees *why the model* scored someone high.
- Let the clinician record the **actual glucose-test outcome** and periodically compare
  it against the flag — a live precision/recall monitor that would catch model drift.

**Clinical usefulness**
- Multi-step questionnaire with a progress bar + inline validation for a smoother
  tablet experience; save-and-resume for interrupted intakes.
- Configurable threshold slider on the dashboard so the clinic lead can trade recall vs.
  referral volume and see the effect on the cohort immediately.
- Export a patient's result to **PDF** for the physical file; CSV export of the cohort.

**Data & fairness**
- A **fairness panel**: flag rate broken down by sex / age / income to check the model
  isn't systematically over- or under-referring a group.
- Track **calibration** (predicted vs. observed risk) as outcomes accumulate.

**Engineering & security** *(this is a teaching demo, not production)*
- Move off the Flask dev server to a WSGI server (waitress/gunicorn); pin Tailwind as a
  built stylesheet instead of the Play CDN.
- Real accounts with roles, password policy, rate-limited login, CSRF tokens on POST
  forms (Flask-WTF), and audit logging of who viewed which record.
- Encrypt the datastore / move to a real DB once records contain PII; add consent capture
  and a retention policy.

**Wow-factor**
- A live **"clinic today"** ticker and trend-over-time chart of daily flag rate.
- Optional GenAI summary: turn a flagged record into a plain-language briefing for the
  clinician (validated against the structured fields).
