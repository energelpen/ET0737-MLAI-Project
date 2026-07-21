"""TinyDB persistence: user accounts, patient screening records, and clinic settings.

TinyDB is a pure-Python, single-file document store — perfect for a simulated clinic:
no server to run, the whole database is a human-readable JSON file under `data/`.

Tables
------
users     : admin / clinician / patient accounts (passwords hashed with Werkzeug)
patients  : one screening each; `patient_username` links it to the account that owns it
settings  : a single editable-settings row (see settings_store.py)
"""
from __future__ import annotations

from datetime import datetime, timezone

from tinydb import Query, TinyDB
from werkzeug.security import check_password_hash, generate_password_hash

from . import config

_db: TinyDB | None = None


def get_db() -> TinyDB:
    """Open (once) the single-file TinyDB and make sure the seed staff accounts exist."""
    global _db
    if _db is None:
        config.DATA_DIR.mkdir(exist_ok=True)
        _db = TinyDB(config.DB_PATH, indent=2, ensure_ascii=False)
        _seed_staff(_db)
    return _db


def _seed_staff(db: TinyDB) -> None:
    users = db.table("users")
    for seed in (config.DEFAULT_ADMIN, config.DEFAULT_DOCTOR):
        if not users.get(Query().username == seed["username"]):
            users.insert({
                "username": seed["username"],
                "password_hash": generate_password_hash(seed["password"]),
                "name": seed["name"],
                "role": seed["role"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })


# --------------------------------------------------------------------------- users
def _public_user(user: dict) -> dict:
    """A user dict safe to store in the session (no password hash)."""
    return {"username": user["username"], "name": user["name"], "role": user["role"]}


def verify_user(username: str, password: str) -> dict | None:
    user = get_db().table("users").get(Query().username == username)
    if user and check_password_hash(user["password_hash"], password):
        return _public_user(user)
    return None


def get_user(username: str) -> dict | None:
    user = get_db().table("users").get(Query().username == username)
    return _public_user(user) if user else None


def create_user(username: str, password: str, name: str, role: str) -> dict:
    """Create an account. Raises ValueError if the username is taken or the role is bad."""
    username = (username or "").strip()
    name = (name or "").strip() or username
    if role not in config.ROLES:
        raise ValueError("Unknown role.")
    users = get_db().table("users")
    if users.get(Query().username == username):
        raise ValueError("That username is already taken.")
    users.insert({
        "username": username,
        "password_hash": generate_password_hash(password),
        "name": name,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"username": username, "name": name, "role": role}


def list_users() -> list[dict]:
    out = []
    for u in get_db().table("users").all():
        out.append({"username": u["username"], "name": u["name"], "role": u["role"],
                    "created_at": u.get("created_at", "")})
    return sorted(out, key=lambda u: (u["role"], u["username"]))


def set_user_role(username: str, role: str) -> None:
    if role not in config.ROLES:
        raise ValueError("Unknown role.")
    get_db().table("users").update({"role": role}, Query().username == username)


def delete_user(username: str) -> None:
    get_db().table("users").remove(Query().username == username)


# ------------------------------------------------------------------------ patients
def add_patient(record: dict) -> int:
    """Store one screening. Returns the TinyDB doc id (used as the record's public id)."""
    record = dict(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    record["reviewed"] = False
    record["clinician_note"] = ""
    record.setdefault("patient_username", None)  # None => anonymous / walk-in screening
    record.setdefault("ai", {})                  # cache of generated AI text, keyed by audience
    return get_db().table("patients").insert(record)


def get_patient(doc_id: int) -> dict | None:
    rec = get_db().table("patients").get(doc_id=doc_id)
    if rec is not None:
        rec = dict(rec)
        rec["id"] = doc_id
    return rec


def all_patients() -> list[dict]:
    """Every screening record, each carrying its TinyDB doc id as `id`."""
    out = []
    for doc in get_db().table("patients").all():
        d = dict(doc)
        d["id"] = doc.doc_id  # TinyDB Documents expose their id here
        out.append(d)
    return out


def patients_for_user(username: str) -> list[dict]:
    """Screening records owned by one patient account, newest first."""
    recs = [r for r in all_patients() if r.get("patient_username") == username]
    recs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return recs


def update_review(doc_id: int, reviewed: bool, note: str) -> None:
    get_db().table("patients").update(
        {"reviewed": reviewed, "clinician_note": note}, doc_ids=[doc_id]
    )


def set_ai_text(doc_id: int, audience: str, text: str) -> None:
    """Cache generated AI text on a record so we don't re-call the API on every view."""
    rec = get_db().table("patients").get(doc_id=doc_id)
    ai = dict(rec.get("ai", {})) if rec else {}
    ai[audience] = text
    get_db().table("patients").update({"ai": ai}, doc_ids=[doc_id])


def clear_patients() -> None:
    """Wipe screening records (used by the demo-seeding helper)."""
    get_db().table("patients").truncate()
