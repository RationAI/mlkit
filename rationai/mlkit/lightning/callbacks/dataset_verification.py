"""Lightning callback that verifies the dataset against MLflow on trainer start.

Optionally performs a stratified train/test split and logs it as an MLflow
artifact alongside verification results.

Example::

    from rationai.mlkit.lightning.callbacks import DatasetVerificationCallback

    # Verification only
    trainer = Trainer(callbacks=[DatasetVerificationCallback()])

    # Verification + train/test split
    trainer = Trainer(
        callbacks=[DatasetVerificationCallback(test_size=0.2, random_state=42)],
    )
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from typing import Any

import mlflow
from lightning.pytorch.callbacks import Callback


log = logging.getLogger(__name__)


class DatasetVerificationCallback(Callback):
    """Run dataset verification (and optional train/test split) at training start.

    Auto-detects ``manifest.csv`` under ``data/``, looks up the latest
    ``Dataset_Registry`` run, and checks that per-file sizes still match.

    Stores results on ``self`` so sibling callbacks (e.g. ``ProvenanceCallback``)
    can read them without duplicating work:

        - ``_verification`` (dict | None) — verification result
        - ``_split_data`` (dict | None) — train/test split data

    Args:
        manifest_path: Path to manifest.csv (auto-detected if None).
        test_size: Fraction of data for the test split. Set to 0 to skip splitting.
        random_state: Random seed for train/test split.
        fail_fast: Abort training if dataset verification fails.
    """

    def __init__(
        self,
        manifest_path: str | None = None,
        test_size: float = 0.0,
        random_state: int = 42,
        fail_fast: bool = True,
    ) -> None:
        """Initialise the dataset verification callback.

        Args:
            manifest_path: Path to manifest.csv (auto-detected if None).
            test_size: Fraction of data for the test split. Set to 0 to skip splitting.
            random_state: Random seed for train/test split.
            fail_fast: Abort training if dataset verification fails.
        """
        self._manifest_path = manifest_path
        self.test_size = test_size
        self.random_state = random_state
        self.fail_fast = fail_fast
        self._done = False
        self._verification: dict[str, Any] | None = None
        self._split_data: dict[str, Any] | None = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Verify the dataset and optionally split into train/test.

        Looks up the latest ``Dataset_Registry`` run, checks file integrity,
        logs verification params to MLflow, and (when ``test_size > 0``)
        performs a stratified train/test split saved as an MLflow artifact.
        """
        if self._done:
            return
        self._done = True

        from rationai.mlkit.provenance.dataset import (
            _detect_manifest,
            _lookup_dataset_run,
            _verify_dataset,
        )
        from rationai.mlkit.provenance.dataset import (
            load_manifest as _load_manifest,
        )

        manifest_path = self._manifest_path
        data_root = None
        if manifest_path is None:
            manifest_path, data_root = _detect_manifest()

        if manifest_path is None:
            log.warning(
                "[DatasetVerificationCallback] No manifest.csv found — skipping"
            )
            return

        if data_root is None:
            data_root = os.path.dirname(os.path.abspath(manifest_path))

        # ── Verification ────────────────────────────────────────
        dataset_run_id = _lookup_dataset_run()
        verification = _verify_dataset(manifest_path, data_root, dataset_run_id)
        self._verification = verification

        for detail in verification.get("details", []):
            log.info(f"  [DatasetVerificationCallback] {detail}")

        # ── Log verification results ────────────────────────────
        if mlflow.active_run():
            mlflow.log_params(
                {
                    "dataset_verified": verification["verified"],
                    "dataset_file_sizes_match": verification["file_sizes_match"]
                    is True,
                    "dataset_files_missing": verification["files_missing"],
                    "dataset_files_total": verification["files_total"],
                }
            )
            if verification["verified"]:
                mlflow.set_tag("dataset_verification", "VERIFIED")
            else:
                mlflow.set_tag("dataset_verification", "MISMATCH")
                mlflow.set_tag(
                    "dataset_verification_details",
                    "; ".join(verification.get("details", [])),
                )

            if self.fail_fast and not verification["verified"]:
                raise RuntimeError(
                    "Dataset verification failed — aborting training.\n"
                    + "\n".join(f"  {d}" for d in verification.get("details", []))
                )

        # ── Train/test split (optional) ─────────────────────────
        if self.test_size > 0:
            import pandas as pd
            from sklearn.model_selection import train_test_split

            samples = _load_manifest(manifest_path, data_root)

            train_samples, test_samples = train_test_split(
                samples,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=[s["label"] for s in samples],
            )

            self._split_data = {
                "train": train_samples,
                "test": test_samples,
                "test_size": self.test_size,
                "random_state": self.random_state,
            }

            # Log split as artifact
            if mlflow.active_run():
                split_dir = f"_mlflow_split_{uuid.uuid4().hex[:8]}"
                os.makedirs(split_dir, exist_ok=True)
                for subset_name, subset_samples in [
                    ("train", train_samples),
                    ("test", test_samples),
                ]:
                    split_file = os.path.join(split_dir, f"{subset_name}_split.csv")
                    pd.DataFrame(subset_samples).to_csv(split_file, index=False)

                mlflow.log_artifacts(split_dir, artifact_path="split")
                shutil.rmtree(split_dir, ignore_errors=True)

                # Log split counts
                train_labels = [s["label"] for s in train_samples]
                test_labels = [s["label"] for s in test_samples]
                mlflow.log_params(
                    {
                        "train_samples": len(train_samples),
                        "test_samples": len(test_samples),
                        "train_positive": sum(train_labels),
                        "train_negative": len(train_labels) - sum(train_labels),
                        "test_positive": sum(test_labels),
                        "test_negative": len(test_labels) - sum(test_labels),
                        "split_test_size": self.test_size,
                        "split_random_state": self.random_state,
                        "split_stratified": True,
                    }
                )
