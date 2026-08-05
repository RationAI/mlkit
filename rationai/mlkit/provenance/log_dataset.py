from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
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
    """Log dataset metadata + PROV document + CSV manifest to active MLflow run."""
    # ── Environment ────────────────────────────────────────
    capture_environment(snapshot_env=snapshot_env)

    # ── Metadata ───────────────────────────────────────────
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
            stat = fpath.stat()
            size = int(stat.st_size)
            mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            print(f"File: {basename}, Size: {size} bytes, Modified: {mtime_iso}")
            print(f"Modification time (raw): {stat.st_mtime}")
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
        print(f"Manifest row added for {basename}: {manifest_rows[-1]}")
        print(manifest_rows[0])

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