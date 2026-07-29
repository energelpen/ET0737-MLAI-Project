"""Loads the exported model artifact and turns a patient's answers into a score.

This deliberately reuses the deployment artifact from notebook 02 §12 — a single joblib
file holding the *whole* fitted estimator (preprocessor → the classifier §6.4b selected,
wrapped in the isotonic calibrator from §6.7) — so the clinic app scores patients with the
exact model that was evaluated in the notebook (no retraining, no drift). The deployment
recipe printed at the end of §12 is implemented verbatim here.

Nothing in this file names an algorithm. The notebook compares nine models across eight
families and picks one by an explicit rule, so the winner can change between runs; the model
name is read from the sidecar and surfaced on the admin model card. Everything here needs
only `predict_proba`, which every candidate provides.

Why the calibrator matters to this file: `assess()` returns `percent`, which the templates
render straight to the patient. The raw model was over-confident by ~2.5x — it scored a
band of patients at 85% whose true at-risk rate was ~50% — so showing its probability would
have misinformed people. Calibration is monotone, so it left every referral decision (and
every metric in §7) untouched while making the displayed number mean what it says.
"""
from __future__ import annotations

import json
import threading

import joblib
import pandas as pd

from . import config


class RiskModel:
    """Thin wrapper around the fitted, calibrated pipeline with the recall-first cut-off."""

    def __init__(self) -> None:
        # The artifact is produced by notebook 02, not by this package, so "it hasn't been
        # generated yet" is the single most likely first-run failure. A bare FileNotFoundError
        # from deep inside pathlib tells a new deployer nothing, so name the cause and the fix.
        missing = [p.name for p in (config.META_PATH, config.PIPE_PATH) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Model artifact(s) not found in {config.MODELS_DIR}: {', '.join(missing)}.\n"
                "The clinic app scores patients with the exact estimator exported by the "
                "modelling notebook, so it cannot start without it.\n"
                "Fix: run notebooks/02_Diabetes_2015_Modelling.ipynb end-to-end; its §12 cell "
                "writes both files. Do not hand-build a replacement — the app's displayed "
                "percentages assume the calibrated model §6.7 produced."
            )

        meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
        self.feature_order: list[str] = list(meta["feature_order"])
        self.threshold: float = float(meta["decision_threshold"])
        self.model_name: str = meta.get("model", "calibrated risk model")
        self.positive_class: str = meta.get("positive_class", "at_risk")
        # Library versions the artifact was trained with — shown on the admin model card so a
        # deployer can spot a version mismatch that could break unpickling. Which of these
        # actually matters depends on the winning model, so all three are recorded.
        self.sklearn_version: str = meta.get("sklearn_version", "unknown")
        self.xgboost_version: str = meta.get("xgboost_version", "unknown")
        self.lightgbm_version: str = meta.get("lightgbm_version", "unknown")

        # One joblib file restores the entire fitted object graph (preprocessor + model +
        # calibration map) — nothing to reassemble.
        self.pipeline = joblib.load(config.PIPE_PATH)

        # predict is read-only, but guard the shared object so a threaded dev server
        # can't interleave transforms mid-call.
        self._lock = threading.Lock()

    def predict_proba(self, features: dict) -> float:
        """P(at risk) for one patient. `features` must contain every model feature."""
        missing = [c for c in self.feature_order if c not in features]
        if missing:
            raise KeyError(f"missing model features: {missing}")
        row = {c: features[c] for c in self.feature_order}
        X = pd.DataFrame([row], columns=self.feature_order)
        with self._lock:
            proba = float(self.pipeline.predict_proba(X)[0, 1])
        return proba

    def assess(self, features: dict) -> dict:
        """Score a patient and band the result for triage.

        The referral line (`self.threshold`) is the model's fixed operating point. The
        softer "elevated / monitor" cut-off is read live from the editable settings, so
        an admin can widen or narrow the monitor band without touching the model.
        """
        from . import settings_store
        elevated_cutoff = settings_store.get_settings()["elevated_cutoff"]

        p = self.predict_proba(features)
        refer = p >= self.threshold
        if refer:
            tier = "high"
        elif p >= elevated_cutoff:
            tier = "elevated"
        else:
            tier = "low"
        return {
            "probability": round(p, 4),
            "percent": round(p * 100, 1),
            "refer": refer,
            "tier": tier,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


_MODEL: RiskModel | None = None


def get_model() -> RiskModel:
    """Process-wide singleton — the artifact is loaded once at startup."""
    global _MODEL
    if _MODEL is None:
        _MODEL = RiskModel()
    return _MODEL
