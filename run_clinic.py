"""Launch the ClearCheck diabetes-screening clinic web app.

    python run_clinic.py

Then open http://127.0.0.1:5001/ in a browser.

Roles (demo accounts, seeded on first run):
    admin  / admin123    -> settings, accounts, model info  (/admin)
    doctor / clinic123   -> analytics dashboard + patients   (/staff)
    patients register at /register and see their own history (/portal)

Optional AI assistant: set ANTHROPIC_API_KEY (and `pip install anthropic`) to enable
Claude-generated guidance for patients and briefings for clinicians. Without it, the app
falls back to the built-in rule-based text.
"""
from clinic_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
