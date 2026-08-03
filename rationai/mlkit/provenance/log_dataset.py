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
    path_column: str = "slide_path",
    label_column: str | None = None,
    snapshot_env: bool = True,
) -> dict[str, Any]:
    """Log dataset metadata + PROV document to the active MLflow run.

    Universal wrapper — works with binary, multiclass, regression, and
    unlabeled datasets. Label column is optional.

    This handles:
      1. ``capture_environment(snapshot_env=...)``
      2. Computing sample counts + class distribution from the data
      3. Logging params & tags to MLflow
      4. Building and uploading the PROV-O document as an artifact

    Args:
        dataset: The dataset DataFrame (must have *path_column* for file
            paths. *label_column* is optional).
        logger: MLFlowLogger instance from ``autolog``.
        config: Hydra DictConfig.
        dataset_name: Name of the dataset for provenance tags.
        version: Dataset version string.
        path_column: Column name containing absolute file paths.
        label_column: Column name containing labels. If ``None``, auto-
            detected from column names (looks for columns with "label",
            "class", "target", "annot" in the name). Unlabeled datasets
            work fine — class_distribution will be omitted.
        snapshot_env: Whether to capture pip freeze snapshot.

    Returns:
        Dict with ``prov_doc``, ``num_samples``, ``class_distribution``,
        ``file_sizes`` for downstream use.

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

    # ── Auto-detect label column if not specified ──────────
    if label_column is None:
        label_column = _detect_label_column(dataset)

    # ── Class distribution (universal) ─────────────────────
    num_samples = len(dataset)
    class_distribution: dict[str, int] | None = None
    num_positive: int | None = None
    num_negative: int | None = None

    if label_column and label_column in dataset.columns:
        counts = dataset[label_column].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in counts.items()}

        # Backward compat: for binary "0"/"1", extract num_positive/negative
        if set(class_distribution.keys()) == {"0", "1"}:
            num_positive = class_distribution["1"]
            num_negative = class_distribution["0"]

    # ── File sizes (required) ─────────────────────────────
    file_sizes: dict[str, int] = {}
    for _, row in dataset.iterrows():
        file_path: str = str(row[path_column])
        basename = os.path.basename(file_path)
        fpath = Path(file_path)
        file_sizes[basename] = int(fpath.stat().st_size) if fpath.exists() else -1

    # ── Log params + tags ──────────────────────────────────
    params: dict[str, Any] = {"num_samples": num_samples}
    if num_positive is not None:
        params["num_positive"] = num_positive
    if num_negative is not None:
        params["num_negative"] = num_negative
    mlflow.log_params(params)

    tags: dict[str, Any] = {
        "dataset_name": dataset_name,
        "version": version,
        "file_sizes": json.dumps(file_sizes),
    }
    if class_distribution is not None:
        tags["class_distribution"] = json.dumps(class_distribution)
    mlflow.set_tags(tags)

    # ── PROV document ──────────────────────────────────────
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run — call inside @autolog")
    run_id = active_run.info.run_id

    dataset_root = str(getattr(config, "data_path", "."))

    prov_kwargs: dict[str, Any] = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "version": version,
        "dataset_root": dataset_root,
        "num_samples": num_samples,
        "file_sizes": file_sizes,
    }
    if class_distribution is not None:
        prov_kwargs["class_distribution"] = class_distribution
    if num_positive is not None:
        prov_kwargs["num_positive"] = num_positive
    if num_negative is not None:
        prov_kwargs["num_negative"] = num_negative

    prov_doc = build_dataset_prov(**prov_kwargs)

    # ── Upload PROV artifact ───────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)
        prov_path = prov_dir / "prov.json"
        prov_path.write_text(json.dumps(prov_doc, indent=2))
        logger.log_artifact(str(prov_path), artifact_path="provenance")

    result: dict[str, Any] = {
        "prov_doc": prov_doc,
        "num_samples": num_samples,
        "class_distribution": class_distribution,
        "file_sizes": file_sizes,
    }
    if num_positive is not None:
        result["num_positive"] = num_positive
    if num_negative is not None:
        result["num_negative"] = num_negative
    return result


def _detect_label_column(df: pd.DataFrame) -> str | None:
    """Heuristic: find a column that looks like a label."""
    hints = ("label", "class", "target", "annot", "y", "outcome")
    for col in df.columns:
        if any(h in col.lower() for h in hints):
            return col
    return None
