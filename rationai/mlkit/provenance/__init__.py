"""Provenance tracking — PROV-O-aware logging to MLflow.

Submodules:
    provenance        – internal helpers (lookup, verification)
    register_dataset  – register_dataset (hash-based)
    register_user     – register_new_user

For automatic provenance capture with Lightning, use
:class:`~rationai.mlkit.lightning.callbacks.provenance.ProvenanceCallback`.

MLflow tracking URI defaults to ``http://localhost:5000``.  Override with
the ``MLFLOW_TRACKING_URI`` environment variable.
"""

from __future__ import annotations

import os

# ── Set default tracking URI before any mlflow import runs ───────────
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# Now safe to import – all child modules will pick up the env var
from .register_dataset import (  # noqa: E402
    register_dataset,
    verify_dataset,
)
from .register_user import register_new_user  # noqa: E402

__all__ = [
    # Dataset
    "register_dataset",
    "verify_dataset",
    # User registration
    "register_new_user",
]


def __getattr__(name: str):
    """Raise helpful error for removed ``autolog``."""
    if name == "autolog":
        raise ImportError(
            "provenance.autolog has been removed. "
            "Use ProvenanceCallback instead:\n\n"
            "  from rationai.mlkit.lightning.callbacks import ProvenanceCallback\n"
            "  trainer = Trainer(callbacks=[ProvenanceCallback(model_name='...')], ...)\n"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
