"""One-shot dataset provenance logger.

All-in-one helper that captures environment, logs metadata params/tags, and
builds the PROV-O document — collapsing ~50 lines of boilerplate into a
single call inside your ``main()``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from omegaconf import DictConfig

from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.provenance.dataset import build_dataset_prov
from rationai.mlkit.provenance.environment import capture_environment


def log_dataset_provenance(
    dataset: pd.DataFrame,
    logger: MLFlowLogger,
    config: DictConfig,
    *,
    dataset_name: str = "ulcerative-colitis-dysplasia",
    version: str = "1.0.0",
    positive_label: str = "NEGATIVE",
    path_column: str = "slide_path",
    label_column: str = "annot_path",
    snapshot_env: bool = True,
) -> dict[str, Any]:
    """Log dataset metadata + PROV document to the active MLflow run.

    This is a convenience wrapper that handles:
      1. ``capture_environment(snapshot_env=...)``
      2. Logging ``num_samples``, ``num_positive``, ``num_negative`` as params
      3. Logging ``dataset_name``, ``version``, ``file_sizes`` as tags
      4. Building and uploading the PROV-O document as an artifact

    Args:
        dataset: The dataset DataFrame (must have *path_column* for file
            paths and *label_column* for positive/negative labels).
        logger: MLFlowLogger instance from ``autolog``.
        config: Hydra DictConfig.
        dataset_name: Name of the dataset for provenance tags.
        version: Dataset version string.
        positive_label: The sentinel value that indicates a negative sample
            (e.g. ``"NEGATIVE"`` — samples **not** equal to this are positive).
        path_column: Column name containing absolute file paths.
        label_column: Column name containing the label (positive/negative).
        snapshot_env: Whether to capture pip freeze snapshot.

    Returns:
        Dict with ``prov_doc``, ``num_samples``, ``num_positive``,
        ``num_negative``, ``file_sizes`` for downstream use.

    Example:
        >>> from rationai.mlkit.provenance import log_dataset_provenance
        >>>
        >>> @autolog
        >>> def main(config, logger):
        >>>     dataset = create_dataset(...)
        >>>     output_path = ...  # save csv, log artifact
        >>>     log_dataset_provenance(dataset, logger, config)
    """
    # ── Environment ────────────────────────────────────────
    capture_environment(snapshot_env=snapshot_env)

    # ── Metadata ───────────────────────────────────────────
    num_samples = len(dataset)
    num_positive = int((dataset[label_column] != positive_label).sum())
    num_negative = num_samples - num_positive

    file_sizes: dict[str, int] = {}
    for _, row in dataset.iterrows():
        slide_path: str = str(row[path_column])
        basename = os.path.basename(slide_path)
        fpath = Path(slide_path)
        file_sizes[basename] = int(fpath.stat().st_size) if fpath.exists() else -1

    # ── Log params + tags ──────────────────────────────────
    mlflow.log_params(
        {
            "num_samples": num_samples,
            "num_positive": num_positive,
            "num_negative": num_negative,
        }
    )
    mlflow.set_tags(
        {
            "dataset_name": dataset_name,
            "version": version,
            "file_sizes": json.dumps(file_sizes),
        }
    )

    # ── PROV document ──────────────────────────────────────
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run — call inside @autolog")
    run_id = active_run.info.run_id

    dataset_root = str(getattr(config, "data_path", "."))

    prov_doc = build_dataset_prov(
        run_id=run_id,
        dataset_name=dataset_name,
        version=version,
        dataset_root=dataset_root,
        num_samples=num_samples,
        num_positive=num_positive,
        num_negative=num_negative,
        file_sizes=file_sizes,
    )

    # ── Upload PROV artifact ───────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)
        prov_path = prov_dir / "prov.json"
        prov_path.write_text(json.dumps(prov_doc, indent=2))
        logger.log_artifact(str(prov_path), artifact_path="provenance")

    return {
        "prov_doc": prov_doc,
        "num_samples": num_samples,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "file_sizes": file_sizes,
    }
