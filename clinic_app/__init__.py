"""Application factory for the ClearCheck diabetes-screening clinic app.

Run it with the sibling `run_clinic.py`, or:  flask --app clinic_app run
"""
from __future__ import annotations

from flask import Flask

from . import config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY

    from .routes import bp
    app.register_blueprint(bp)

    # Expose the code->label helper to templates for the clinician review pages.
    from .schema import humanize
    app.jinja_env.globals["humanize"] = humanize

    # Fail fast at startup if the model artifacts are missing, rather than on the
    # first patient submission.
    from .model_service import get_model
    get_model()

    # Ensure the TinyDB file + seed clinician exist from the start.
    from .db import get_db
    get_db()

    return app
