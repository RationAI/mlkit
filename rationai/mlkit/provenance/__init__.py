"""Provenance tracking - PROV-O-aware logging to MLflow.

Submodules:
    common          - shared helpers (prefixes, IDs, timestamps)
    user            - build_user_prov + register_new_user
    dataset         - build_dataset_prov + register_dataset + verify_dataset
    run             - build_training_run_prov

For automatic provenance capture with Lightning, use
:class:`~rationai.mlkit.lightning.callbacks.provenance.ProvenanceCallback`.

MLflow tracking URI defaults to ``http://localhost:5000``.  Override with
the ``MLFLOW_TRACKING_URI`` environment variable.
"""

from __future__ import annotations

import os
from typing import Any


# ── Set default tracking URI before any mlflow import runs ───────────
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

from rationai.mlkit.provenance.dataset import (
    build_dataset_prov,
    register_dataset,
    verify_dataset,
)
from rationai.mlkit.provenance.run import build_training_run_prov
from rationai.mlkit.provenance.user import (
    build_user_prov,
    register_new_user,
)


__all__ = [
    "build_dataset_prov",
    "build_training_run_prov",
    "build_user_prov",
    "register_dataset",
    "register_new_user",
    "verify_dataset",
]


def __getattr__(name: str) -> Any:
    """Raise helpful error for removed ``autolog``."""
    if name == "autolog":
        raise ImportError(
            "provenance.autolog has been removed. "
            "Use ProvenanceCallback instead:\n\n"
            "  from rationai.mlkit.lightning.callbacks import ProvenanceCallback\n"
            "  trainer = Trainer(callbacks=[ProvenanceCallback(model_name='...')], ...)\n"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
