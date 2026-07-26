"""All HTTP routes.

Three audiences, separated by role:
  * patient   — registers, screens, and sees their own history (/portal)
  * clinician — reviews flagged patients + analytics (/staff …)
  * admin     — everything a clinician can do, plus settings + accounts (/admin …)

Anonymous walk-in screening is still supported (no login required to fill the form).
"""
from __future__ import annotations

import functools

from flask import (Blueprint, flash, redirect, render_template, request,
                   session, url_for)

from . import ai, analytics, config, db, insights, settings_store
from .model_service import get_model
from .schema import SECTIONS, ValidationError, parse_submission

bp = Blueprint("clinic", __name__)


# ------------------------------------------------------------------ auth helpers
def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("clinic.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def role_required(*roles):
    """Restrict a view to the given roles (admin implicitly passes clinician gates)."""
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            user = session.get("user")
            if not user:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("clinic.login", next=request.path))
            allowed = set(roles)
            if "clinician" in allowed:
                allowed.add("admin")  # admins can do everything clinicians can
            if user["role"] not in allowed:
                flash("You don't have access to that page.", "error")
                return redirect(url_for("clinic.home"))
            return view(*args, **kwargs)
        return wrapped
    return decorator


def _home_for(user: dict) -> str:
    """Where a signed-in user should land, by role."""
    if user["role"] == "admin":
        return url_for("clinic.admin")
    if user["role"] == "clinician":
        return url_for("clinic.dashboard")
    return url_for("clinic.portal")


@bp.app_context_processor
def inject_globals():
    return {"current_user": session.get("user"),
            "settings": settings_store.get_settings(),
            "ai_enabled": ai.is_enabled()}


# ---------------------------------------------------------------- patient-facing
@bp.route("/")
def landing():
    return render_template("landing.html")


@bp.route("/home")
def home():
    """Send a signed-in user to their role's home; otherwise the landing page."""
    user = session.get("user")
    return redirect(_home_for(user) if user else url_for("clinic.landing"))


@bp.route("/screening", methods=["GET", "POST"])
def screening():
    user = session.get("user")
    if request.method == "POST":
        try:
            features, meta = parse_submission(request.form)
        except ValidationError as exc:
            flash("Please review the highlighted questions.", "error")
            return render_template("screening.html", sections=SECTIONS,
                                   errors=exc.errors, values=request.form), 400

        assessment = get_model().assess(features)
        record = {
            "patient_name": meta.get("patient_name", "").strip(),
            "height_cm": meta.get("height_cm"),
            "weight_kg": meta.get("weight_kg"),
            "features": features,
            "probability": assessment["probability"],
            "percent": assessment["percent"],
            "tier": assessment["tier"],
            "refer": assessment["refer"],
            # A logged-in patient owns their screening; staff/walk-ins are anonymous.
            "patient_username": user["username"] if user and user["role"] == "patient" else None,
        }
        doc_id = db.add_patient(record)
        session["last_result"] = doc_id  # lets an anonymous patient view the result they just made
        return redirect(url_for("clinic.result", doc_id=doc_id))

    # Pre-fill the name for a logged-in patient.
    values = {}
    if user and user["role"] == "patient":
        values = {"patient_name": user["name"]}
    return render_template("screening.html", sections=SECTIONS, errors={}, values=values)


def _may_view_result(record: dict) -> bool:
    user = session.get("user")
    if user and user["role"] in ("admin", "clinician"):
        return True
    if user and record.get("patient_username") == user["username"]:
        return True
    return session.get("last_result") == record["id"]


@bp.route("/screening/result/<int:doc_id>")
def result(doc_id: int):
    record = db.get_patient(doc_id)
    if record is None:
        flash("That screening record could not be found.", "error")
        return redirect(url_for("clinic.landing"))
    if not _may_view_result(record):
        flash("Please sign in to view that result.", "warning")
        return redirect(url_for("clinic.login", next=request.path))
    model = get_model()
    assessment = {"tier": record["tier"], "probability": record["probability"],
                  "percent": record["percent"], "refer": record["refer"],
                  "threshold": model.threshold}
    copy = insights.result_copy(assessment)
    factors = insights.present_risk_factors(record["features"])
    tips = insights.lifestyle_tips(record["features"])
    return render_template("result.html", record=record, assessment=assessment,
                           copy=copy, factors=factors, tips=tips,
                           ai_text=record.get("ai", {}).get("patient"),
                           threshold_pct=round(model.threshold * 100, 1))


@bp.route("/screening/result/<int:doc_id>/ai", methods=["POST"])
def result_ai(doc_id: int):
    """Generate (and cache) the patient-facing AI note for a screening."""
    record = db.get_patient(doc_id)
    if record is None or not _may_view_result(record):
        flash("That screening record could not be found.", "error")
        return redirect(url_for("clinic.landing"))
    factors = insights.present_risk_factors(record["features"])
    text = ai.patient_guidance(record, factors)
    if text:
        db.set_ai_text(doc_id, "patient", text)
    else:
        flash("The AI assistant is unavailable right now.", "warning")
    return redirect(url_for("clinic.result", doc_id=doc_id))


# ------------------------------------------------------------ accounts / sessions
@bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user"):
        return redirect(url_for("clinic.home"))
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        name = (request.form.get("name") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""
        errors = {}
        if len(username) < 3:
            errors["username"] = "Choose a username of at least 3 characters."
        if len(password) < 6:
            errors["password"] = "Use a password of at least 6 characters."
        elif password != confirm:
            errors["confirm"] = "The passwords don't match."
        if not errors:
            try:
                user = db.create_user(username, password, name or username, role="patient")
            except ValueError as exc:
                errors["username"] = str(exc)
            else:
                session["user"] = user
                flash(f"Welcome, {user['name']}! Your account is ready.", "success")
                return redirect(url_for("clinic.portal"))
        flash("Please fix the highlighted fields.", "error")
        return render_template("register.html", errors=errors, values=request.form), 400
    return render_template("register.html", errors={}, values={})


@bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("clinic.home"))
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = db.verify_user(username, password)
        if user:
            session["user"] = user
            flash(f"Welcome back, {user['name']}.", "success")
            nxt = request.args.get("next")
            return redirect(nxt or _home_for(user))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("last_result", None)
    flash("You have been signed out.", "success")
    return redirect(url_for("clinic.landing"))


# Backwards-compatible aliases for the old clinician-only entry points.
@bp.route("/staff/login")
def staff_login():
    return redirect(url_for("clinic.login", next=request.args.get("next")))


@bp.route("/staff/logout")
def staff_logout():
    return redirect(url_for("clinic.logout"))


# --------------------------------------------------------------- patient portal
@bp.route("/portal")
@role_required("patient")
def portal():
    user = session["user"]
    records = db.patients_for_user(user["username"])
    return render_template("portal.html", records=records)


# ------------------------------------------------------------------ clinician side
@bp.route("/staff")
@role_required("clinician")
def dashboard():
    records = db.all_patients()
    summary = analytics.summarize(records)
    recent = sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)[:6]
    return render_template("dashboard.html", s=summary, recent=recent)


@bp.route("/staff/patients")
@role_required("clinician")
def patients():
    records = db.all_patients()
    tier_filter = request.args.get("tier", "all")
    sort = request.args.get("sort", "recent")
    if tier_filter in {"high", "elevated", "low"}:
        records = [r for r in records if r.get("tier") == tier_filter]
    if sort == "risk":
        records.sort(key=lambda r: r.get("probability", 0), reverse=True)
    elif sort == "name":
        records.sort(key=lambda r: (r.get("patient_name") or "").lower())
    else:  # recent
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return render_template("patients.html", records=records,
                           tier_filter=tier_filter, sort=sort, total=len(records))


@bp.route("/staff/patients/<int:doc_id>")
@role_required("clinician")
def patient_detail(doc_id: int):
    record = db.get_patient(doc_id)
    if record is None:
        flash("Patient record not found.", "error")
        return redirect(url_for("clinic.patients"))
    factors = insights.present_risk_factors(record["features"])
    model = get_model()
    return render_template("patient_detail.html", record=record, factors=factors,
                           sections=SECTIONS,
                           ai_text=record.get("ai", {}).get("clinician"),
                           threshold_pct=round(model.threshold * 100, 1),
                           feature_order=model.feature_order)


@bp.route("/staff/patients/<int:doc_id>/review", methods=["POST"])
@role_required("clinician")
def review_patient(doc_id: int):
    record = db.get_patient(doc_id)
    if record is None:
        flash("Patient record not found.", "error")
        return redirect(url_for("clinic.patients"))
    reviewed = request.form.get("reviewed") == "on"
    note = (request.form.get("clinician_note") or "").strip()
    db.update_review(doc_id, reviewed, note)
    flash("Review saved.", "success")
    return redirect(url_for("clinic.patient_detail", doc_id=doc_id))


@bp.route("/staff/patients/<int:doc_id>/ai", methods=["POST"])
@role_required("clinician")
def patient_ai(doc_id: int):
    """Generate (and cache) the clinician briefing for a patient."""
    record = db.get_patient(doc_id)
    if record is None:
        flash("Patient record not found.", "error")
        return redirect(url_for("clinic.patients"))
    factors = insights.present_risk_factors(record["features"])
    text = ai.clinician_briefing(record, factors)
    if text:
        db.set_ai_text(doc_id, "clinician", text)
    else:
        flash("The AI assistant is unavailable right now.", "warning")
    return redirect(url_for("clinic.patient_detail", doc_id=doc_id))


# ---------------------------------------------------------------------- admin side
@bp.route("/admin")
@role_required("admin")
def admin():
    model = get_model()
    return render_template(
        "admin.html",
        settings=settings_store.get_settings(),
        users=db.list_users(),
        ai_status=ai.status(),
        roles=config.ROLES,
        model_info={
            "name": model.model_name,
            "threshold_pct": round(model.threshold * 100, 2),
            "libraries": f"scikit-learn {model.sklearn_version} · XGBoost {model.xgboost_version}",
            "n_features": len(model.feature_order),
            "positive_class": model.positive_class,
        },
    )


@bp.route("/admin/settings", methods=["POST"])
@role_required("admin")
def admin_settings():
    form = request.form
    changes: dict = {
        "clinic_name": (form.get("clinic_name") or "").strip() or settings_store.DEFAULTS["clinic_name"],
        "clinic_tagline": (form.get("clinic_tagline") or "").strip(),
        "ai_enabled": form.get("ai_enabled") == "on",
    }
    # Monitor-band cutoff: must be a fraction below the (locked) decision threshold.
    try:
        cutoff = float(form.get("elevated_cutoff", ""))
        threshold = get_model().threshold
        if 0.0 <= cutoff < threshold:
            changes["elevated_cutoff"] = round(cutoff, 3)
        else:
            flash(f"Monitor-band cutoff must be between 0 and the referral line "
                  f"({round(threshold*100,1)}%). Left unchanged.", "warning")
    except (TypeError, ValueError):
        flash("Monitor-band cutoff must be a number. Left unchanged.", "warning")

    # Editable tier copy.
    tier_copy = {}
    for tier in ("high", "elevated", "low"):
        tier_copy[tier] = {
            "headline": (form.get(f"{tier}_headline") or "").strip(),
            "subtext": (form.get(f"{tier}_subtext") or "").strip(),
            "cta": (form.get(f"{tier}_cta") or "").strip(),
        }
    changes["tier_copy"] = tier_copy

    # Editable factor tips (only keys that were rendered on the form).
    tips = {}
    for key in settings_store.DEFAULTS["factor_tips"]:
        val = form.get(f"tip_{key}")
        if val is not None:
            tips[key] = val.strip()
    changes["factor_tips"] = tips

    settings_store.update_settings(changes)
    flash("Settings saved.", "success")
    return redirect(url_for("clinic.admin"))


@bp.route("/admin/settings/reset", methods=["POST"])
@role_required("admin")
def admin_settings_reset():
    settings_store.reset_settings()
    flash("Settings reset to defaults.", "success")
    return redirect(url_for("clinic.admin"))


@bp.route("/admin/users", methods=["POST"])
@role_required("admin")
def admin_create_user():
    username = (request.form.get("username") or "").strip()
    name = (request.form.get("name") or "").strip()
    password = request.form.get("password") or ""
    role = request.form.get("role") or "clinician"
    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("clinic.admin"))
    try:
        db.create_user(username, password, name or username, role)
        flash(f"Account '{username}' created.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    return redirect(url_for("clinic.admin"))


@bp.route("/admin/users/<username>/role", methods=["POST"])
@role_required("admin")
def admin_set_role(username: str):
    role = request.form.get("role") or ""
    if username == session["user"]["username"]:
        flash("You can't change your own role.", "error")
        return redirect(url_for("clinic.admin"))
    try:
        db.set_user_role(username, role)
        flash(f"Updated role for '{username}'.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    return redirect(url_for("clinic.admin"))


@bp.route("/admin/users/<username>/delete", methods=["POST"])
@role_required("admin")
def admin_delete_user(username: str):
    if username == session["user"]["username"]:
        flash("You can't delete your own account.", "error")
        return redirect(url_for("clinic.admin"))
    db.delete_user(username)
    flash(f"Deleted account '{username}'.", "success")
    return redirect(url_for("clinic.admin"))
