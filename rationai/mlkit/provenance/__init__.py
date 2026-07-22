"""Provenance tracking - PROV-O-aware logging to MLflow.

Submodules:
    common          - shared helpers (prefixes, IDs, timestamps)
    user            - build_user_prov + register_new_user
    dataset         - build_dataset_prov + register_dataset + verify_dataset
    run             - build_training_run_prov

For automatic provenance capture with Lightning, use
:class:`~rationai.mlkit.lightning.callbacks.provenance.ProvenanceCallback`.
"""

from __future__ import annotations

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
