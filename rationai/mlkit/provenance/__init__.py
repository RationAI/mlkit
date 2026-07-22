"""Provenance tracking - PROV-O-aware logging to MLflow.

Submodules:
    prov              - PROV-O document builders (W3C PROV compatible)
    register_dataset  - register_dataset (hash-based, emits prov.json)
    register_user     - register_new_user (emits prov.json)

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

# Now safe to import - all child modules will pick up the env var
from rationai.mlkit.provenance.prov import (
    build_dataset_prov,
    build_user_prov,
)
from rationai.mlkit.provenance.register_dataset import (
    register_dataset,
    verify_dataset,
)
from rationai.mlkit.provenance.register_user import register_new_user


__all__ = [
    "build_dataset_prov",
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
