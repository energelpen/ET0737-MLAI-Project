"""Loads the exported stacking-ensemble pipeline and turns a patient's answers into a score.

This deliberately reuses the deployment artifact from notebook 02 §12 — a single joblib
file holding the *whole* fitted pipeline (preprocessor → SVM + XGBoost stacking ensemble →
Logistic Regression meta-learner) — so the clinic app scores patients with the exact
pipeline that was evaluated in the notebook (no retraining, no drift). The deployment
recipe printed at the end of §12 is implemented verbatim here.
"""
from __future__ import annotations

import json
import threading

import joblib
import pandas as pd

from . import config


class RiskModel:
    """Thin wrapper around the fitted stacking pipeline with the recall-first cut-off."""

    def __init__(self) -> None:
        meta = json.loads(config.META_PATH.read_text())
        self.feature_order: list[str] = list(meta["feature_order"])
        self.threshold: float = float(meta["decision_threshold"])
        self.model_name: str = meta.get("model", "Stacking (SVM+XGB)")
        self.positive_class: str = meta.get("positive_class", "at_risk")
        # Library versions the artifact was trained with — shown on the admin model card so a
        # deployer can spot a scikit-learn / XGBoost mismatch that could break unpickling.
        self.sklearn_version: str = meta.get("sklearn_version", "unknown")
        self.xgboost_version: str = meta.get("xgboost_version", "unknown")

        # One joblib file restores the entire fitted object graph (preprocessor + both base
        # models + meta-learner) — nothing to reassemble.
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
