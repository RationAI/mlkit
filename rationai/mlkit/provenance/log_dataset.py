from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import pandas as pd
from omegaconf import DictConfig

from rationai.mlkit.provenance.dataset import build_dataset_prov
from rationai.mlkit.provenance.environment import capture_environment


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger


def log_dataset_provenance(
    dataset: pd.DataFrame,
    logger: MLFlowLogger,
    config: DictConfig,
    *,
    dataset_name: str,
    positive_label: str,
    version: str = "1.0.0",
    path_column: str = "slide_path",
    label_column: str = "annot_path",
    snapshot_env: bool = True,
) -> dict[str, Any]:
    """Log dataset metadata + PROV document + CSV manifest to active MLflow run.

    Args:
        dataset: Dataset dataframe with at least ``path_column`` and ``label_column``.
        logger: MLFlowLogger of the current run.
        config: Hydra config (uses ``data_path`` for the PROV document root).
        dataset_name: Human-readable dataset name (tag + PROV).
        positive_label: Value of ``label_column`` that counts as the
            *negative* class; every other value counts as positive.
        version: Dataset version string.
        path_column: Column with slide file paths.
        label_column: Column with annotation/label values.
        snapshot_env: Whether to snapshot the python environment.
    """
    # ── Environment ────────────────────────────────────────
    capture_environment(snapshot_env=snapshot_env)

    # ── Metadata ───────────────────────────────────────────
    for column in (path_column, label_column):
        if column not in dataset.columns:
            raise ValueError(
                f"Dataset is missing required column {column!r}; "
                f"available columns: {list(dataset.columns)}"
            )
    num_samples = len(dataset)
    num_positive = int((dataset[label_column] != positive_label).sum())
    num_negative = num_samples - num_positive

    file_sizes: dict[str, int] = {}
    file_mtimes: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []

    for _, row in dataset.iterrows():
        slide_path_str: str = str(row[path_column])
        basename = os.path.basename(slide_path_str)
        fpath = Path(slide_path_str)

        if fpath.exists():
            try:
                stat = fpath.stat()
            except OSError:
                size = -1
                mtime_iso = "unknown"
            else:
                size = int(stat.st_size)
                mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        else:
            size = -1
            mtime_iso = "unknown"

        file_sizes[basename] = size
        file_mtimes[basename] = mtime_iso

        manifest_rows.append(
            {
                "filename": basename,
                "path": slide_path_str,
                "size_bytes": size,
                "modified_at": mtime_iso,
            }
        )

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

    # ── Upload Artefacts (PROV JSON + CSV Manifest) ────────
    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)

        # 1. PROV-O JSON document
        prov_path = prov_dir / "prov.json"
        prov_path.write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")

        # 2. Dataset Manifest jako CSV tabulka
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_csv_path = prov_dir / "dataset_manifest.csv"
        manifest_df.to_csv(manifest_csv_path, index=False)

        # Nahrajeme soubory do MLflow
        logger.log_artifact(str(prov_path), artifact_path="provenance")
        logger.log_artifact(str(manifest_csv_path), artifact_path="provenance")

    return {
        "prov_doc": prov_doc,
        "num_samples": num_samples,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "file_sizes": file_sizes,
        "file_mtimes": file_mtimes,
    }
