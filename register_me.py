"""Register yourself in the User_Registry experiment."""

import os
import sys

# Project root on PATH so `rationai` resolves without install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rationai.mlkit.provenance.user import register_new_user

# ── Config ───────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://mlflow-jiribuchta.dyn.cloud.trusted.e-infra.cz",
)

USERNAME = "jiribuchta"
REAL_NAME = "Jiri Buchta"
EMAIL = "jiri.buchta@rationai.com"
ORGANIZATION = "RationAI"
LEAD_NAME = "Jiri Buchta"
LEAD_EMAIL = "jiri.buchta@rationai.com"
# ─────────────────────────────────────────────────────────────

import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if __name__ == "__main__":
    run_id = register_new_user(
        username=USERNAME,
        real_name=REAL_NAME,
        email=EMAIL,
        organization=ORGANIZATION,
        lead_name=LEAD_NAME,
        lead_email=LEAD_EMAIL,
    )
    print(f"Done. User run_id: {run_id}")
