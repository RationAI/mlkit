"""Provenance tracking - PROV-O-aware logging to MLflow.

Submodules:
    common          - shared helpers (prefixes, IDs, timestamps)
    user            - build_user_prov + register_new_user
    dataset         - build_dataset_prov + register_dataset + verify_dataset
    run             - build_training_run_prov
    log_provenance  - universal run provenance (any pipeline step,
                      incl. splits via dict[split_name, DataFrame])

For automatic provenance capture with Lightning, use
:class:`~rationai.mlkit.lightning.callbacks.provenance.ProvenanceCallback`.
"""

from __future__ import annotations

from rationai.mlkit.provenance.dataset import (
    build_dataset_prov,
    register_dataset,
    verify_dataset,
)
from rationai.mlkit.provenance.environment import (
    capture_environment,
    log_environment,
)
from rationai.mlkit.provenance.log_provenance import log_provenance
from rationai.mlkit.provenance.run import build_training_run_prov
from rationai.mlkit.provenance.user import (
    build_user_prov,
    register_new_user,
)


__all__ = [
    "build_dataset_prov",
    "build_training_run_prov",
    "build_user_prov",
    "capture_environment",
    "log_environment",
    "log_provenance",
    "register_dataset",
    "register_new_user",
    "verify_dataset",
]
