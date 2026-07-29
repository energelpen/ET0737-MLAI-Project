# ClearCheck — Diabetes Screening Clinic (web app)

An interactive Flask web application that turns the exported, calibrated risk model
(notebook 02 §12) into a working clinic tool with three roles:

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
first if `models/diabetes_risk_model.joblib` is missing.

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

> **First run: you need the model artifact.** The app cannot start without
> `models/diabetes_risk_model.joblib`. If it is absent, **run
> `notebooks/02_Diabetes_2015_Modelling.ipynb` end-to-end** — §12a writes it, and §12c verifies the
> app can load it before the notebook finishes. Starting the app without it raises an explicit
> error naming that fix rather than a `FileNotFoundError` traceback.

The app scores patients with the **whole fitted estimator** from §12 — one joblib file
holding the preprocessor and the selected classifier inside the isotonic calibrator from
§6.7 — the exact, bit-for-bit model evaluated in the notebook (no retraining).
The `.meta.json` supplies the 29-feature order and the **recall-first decision threshold**;
at or above it a patient is flagged *high-risk / refer*.

**Nothing in this app names an algorithm.** Notebook 02 compares nine models across eight
families and picks one by an explicit rule (§6.4b), so the winner can change between runs.
The artifact is therefore named for its role (`diabetes_risk_model.joblib`) and the model's
actual name is read from the sidecar and shown on the admin model card. Everything here needs
only `predict_proba`, which every candidate provides.

The model is **calibrated**, which is what makes the risk percentage on the result page
honest. The raw model was over-confident by ~2.5×: a band it scored at 85% had a true
at-risk rate of ~50%. Calibration is a monotone rescaling, so it changed no referral
decision and no metric in §7 — it only made the displayed number trustworthy.

| Triage band | Rule | Meaning |
|---|---|---|
| **High** | `p ≥ threshold` | Refer for a glucose test (model's operating point) |
| **Elevated** | `0.30 ≤ p < threshold` | Monitor — a soft UX triage aid, not the model's line |
| **Low** | `p < 0.30` | No follow-up indicated |

BMI is not asked directly — the questionnaire takes height + weight and computes it, the
way a clinic actually measures.

---

## Deploying to Render

The app runs locally and hosted from the same codebase — the only difference is the entry point.

| | local | Render |
|---|---|---|
| entry point | `run_clinic.py` (Flask dev server, `debug=True`) | `wsgi.py` via gunicorn |
| dependencies | `requirements.txt` (the full notebook environment) | `requirements-app.txt` (runtime only) |
| config | none needed | `render.yaml` provisions it |

Push to GitHub, then in Render pick **New → Blueprint** and point it at the repo; `render.yaml`
supplies the build command, start command and secrets. **Run notebook 02 first** — Render builds
from the repo, so `models/diabetes_risk_model.joblib` has to be committed. §12a prints whether the
artifact is inside GitHub's 100 MB limit and §12c fails the notebook if it isn't.

Three things are deliberate rather than accidental:

* **`wsgi.py` exists so a deploy can never start the debug server.** `run_clinic.py` sets
  `debug=True`, and the Werkzeug debugger is an interactive Python console — fine on `127.0.0.1`,
  remote code execution on a public URL. Separate files, no chance of mixing them up.
* **One gunicorn worker, scaled with threads.** TinyDB is a single JSON file with no cross-process
  locking, and `db._db` / `model_service._MODEL` are per-process singletons — a second worker would
  silently clobber writes. `RiskModel` holds a lock, so threads are safe. Postgres is what would
  lift this.
* **`requirements-app.txt` drops jupyter, matplotlib and seaborn.** The server never imports them,
  and on a 512 MB free instance the memory matters. lightgbm is excluded too: the sidecar records
  its version as a display string, but nothing imports it unless LightGBM is the deployed model.

**Known limitation — storage is ephemeral.** Render wipes the filesystem on every deploy and
restart, so `clinic_db.json` resets: staff accounts are re-seeded automatically, but registered
patients and their screening history are lost. Fine for a demo link. To fix it, uncomment the disk
block in `render.yaml` (needs a paid instance) — `CLINIC_DATA_DIR` moves the database onto the
mounted volume and `get_db()` creates the directory on first use.

**Credentials are public on purpose.** The demo logins (`admin`/`admin123`, `doctor`/`clinic123`)
are left live on a hosted deploy so anyone opening the link can try both roles — this is a
simulation with synthetic patients and no real PII. `config.py` logs a note listing exactly which
public defaults are in play, and every one is overridable (`CLINIC_ADMIN_PASSWORD`,
`CLINIC_DOCTOR_PASSWORD`, `CLINIC_SECRET`) if this is ever reused with real data.

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
