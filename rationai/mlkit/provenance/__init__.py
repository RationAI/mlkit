"""Provenance tracking — PROV-O-aware logging to MLflow.

Submodules:
    provenance        – @autolog decorator, internal helpers
    register_dataset  – register_dataset (hash-based) and
                        register_dataset_as_provenance (legacy CSV-based)
    register_user     – register_new_user

MLflow tracking URI defaults to ``http://localhost:5000``.  Override with
the ``MLFLOW_TRACKING_URI`` environment variable.
"""

from __future__ import annotations

import os

# ── Set default tracking URI before any mlflow import runs ───────────
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Now safe to import – all child modules will pick up the env var
from .provenance import autolog  # noqa: E402
from .register_dataset import (  # noqa: E402
    register_dataset,
    register_dataset_as_provenance,
)
from .register_user import register_new_user  # noqa: E402

__all__ = [
    # Core provenance
    "autolog",
    # Dataset registration
    "register_dataset",
    "register_dataset_as_provenance",
    # User registration
    "register_new_user",
]
