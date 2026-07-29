"""Production WSGI entry point — this is what gunicorn imports on Render.

    gunicorn wsgi:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT

Use `run_clinic.py` for local development instead. The two are deliberately separate:
`run_clinic.py` starts Flask's development server with `debug=True`, and the Werkzeug
debugger exposes an interactive console that executes arbitrary Python — harmless on
127.0.0.1, remote code execution on a public URL. Keeping the production entry point in its
own file means a hosted deployment can never accidentally start the debug server.

The app object is built at import time so gunicorn fails fast on a bad deploy: a missing
model artifact or an unset CLINIC_SECRET raises here, during startup, rather than on the
first patient request.

WHY ONE WORKER. Storage is TinyDB — a single JSON file with no cross-process locking — and
`db._db` / `model_service._MODEL` are per-process singletons. Two worker processes would each
hold their own handle to the same file and silently clobber each other's writes. Threads are
fine (`RiskModel` guards its estimator with a lock), so scale with `--threads`, not
`--workers`. Moving to Postgres is what would lift that restriction.
"""
from clinic_app import create_app

app = create_app()
