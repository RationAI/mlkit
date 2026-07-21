"""Lightning callback that verifies the dataset against MLflow on trainer start."""

from __future__ import annotations

import mlflow

from lightning.pytorch.callbacks import Callback


class DatasetVerificationCallback(Callback):
    """Run dataset verification once at the start of training.

    Auto-detects ``manifest.csv`` under ``data/``, looks up the latest
    ``Dataset_Registry`` run, and checks that per-file sizes still match.

    Logs verification results as MLflow params so they appear on the run
    page alongside metrics and artifacts.

    Example::

        from rationai.mlkit.lightning.callbacks import DatasetVerificationCallback

        trainer = Trainer(
            callbacks=[DatasetVerificationCallback()],
            logger=MLFlowLogger(...),
        )
    """

    def __init__(self, manifest_path: str | None = None):
        self._manifest_path = manifest_path
        self._done = False

    def on_fit_start(self, trainer, pl_module):  # noqa: ARG002
        if self._done:
            return
        self._done = True

        # Import here so the callback doesn't require provenance as a hard dep
        from rationai.mlkit.provenance.register_dataset import (
            _detect_manifest,
            _lookup_dataset_run,
            _verify_dataset,
        )

        manifest_path = self._manifest_path
        data_root = None
        if manifest_path is None:
            manifest_path, data_root = _detect_manifest()

        if manifest_path is None:
            print("  [DatasetVerificationCallback] No manifest.csv found — skipping")
            return

        if data_root is None:
            import os
            data_root = os.path.dirname(os.path.abspath(manifest_path))

        dataset_run_id = _lookup_dataset_run()
        verification = _verify_dataset(manifest_path, data_root, dataset_run_id)

        for detail in verification["details"]:
            print(f"  [DatasetVerificationCallback] {detail}")

        # Log to the active MLflow run (if any)
        active_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        if active_run_id:
            mlflow.log_params({
                "dataset_verified": verification["verified"],
                "dataset_file_sizes_match": verification["file_sizes_match"] is True,
                "dataset_files_missing": verification["files_missing"],
                "dataset_files_total": verification["files_total"],
            })
            if verification["verified"]:
                mlflow.set_tag("dataset_verification", "VERIFIED")
            else:
                mlflow.set_tag("dataset_verification", "MISMATCH")
