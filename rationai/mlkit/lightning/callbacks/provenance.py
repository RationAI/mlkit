"""Lightning callback that captures PROV-O provenance for MLflow runs.

Slim callback that depends on sibling callbacks for environment and dataset
verification data:

    - :class:`~rationai.mlkit.lightning.callbacks.environment.EnvironmentCallback`
      provides git info, hardware, docker, env snapshot, and user tags.
    - :class:`~rationai.mlkit.lightning.callbacks.dataset_verification.DatasetVerificationCallback`
      provides dataset verification and train/test split results.

When used alone, it falls back to doing its own environment/verification work
so the user gets a single-drop-in experience.

Example::

    from rationai.mlkit.lightning.callbacks import ProvenanceCallback

    trainer = Trainer(
        callbacks=[ProvenanceCallback(model_name="resnet_v1")],
        logger=MLFlowLogger(...),
    )
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd
from lightning.pytorch.callbacks import Callback

# Import shared PROV helpers from prov.py to avoid duplication
from rationai.mlkit.provenance.common import (
    get_prov_prefixes as _get_prov_prefixes,
)
from rationai.mlkit.provenance.run import (
    build_training_run_prov as _build_prov_document,
)


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Model / Optimizer / Scheduler summaries
# ──────────────────────────────────────────────


def _model_summary(model: Any) -> dict[str, str | int]:
    """Extract architecture details from a torch.nn.Module."""
    info: dict[str, str | int] = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_parameters"] = total_params
    info["trainable_parameters"] = trainable_params

    layer_lines = []
    for name, module in model.named_modules():
        if name == "":
            continue
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        children = len(list(module.children()))
        layer_lines.append(
            f"{name}({type(module).__name__}): params={param_count}, "
            f"children={children}",
        )

    layer_summary: str = "\n".join(layer_lines[:20])
    if len(layer_lines) > 20:
        layer_summary += f"\n... ({len(layer_lines)} layers total)"
    info["layer_summary"] = layer_summary

    info["model_class"] = type(model).__name__
    return info


def _optimizer_summary(optimizer: Any) -> dict[str, str | float]:
    """Extract optimizer settings from torch.optim.Optimizer."""
    info: dict[str, str | float] = {}
    info["optimizer_type"] = type(optimizer).__name__
    for name, value in optimizer.defaults.items():
        if isinstance(value, (int, float, bool, str)):
            info[f"opt_{name}"] = value
    return info


def _scheduler_summary(scheduler: Any) -> dict[str, str | float]:
    """Extract scheduler settings."""
    info: dict[str, str | float] = {}
    if scheduler is None:
        info["scheduler_type"] = "none"
        return info

    info["scheduler_type"] = type(scheduler).__name__
    for attr in (
        "step_size",
        "gamma",
        "milestones",
        "factor",
        "patience",
        "min_lr",
        "T_max",
        "eta_min",
    ):
        val = getattr(scheduler, attr, None)
        if val is not None:
            info[f"sch_{attr}"] = (
                str(list(val)) if isinstance(val, (list, tuple)) else str(val)
            )

    if hasattr(scheduler, "optimizer"):
        for name, value in scheduler.optimizer.defaults.items():
            if isinstance(value, (int, float, bool)):
                info[f"sch_opt_{name}"] = value

    return info




class ProvenanceCallback(Callback):
    """Lightning callback that captures PROV document + run summary.

    Reads environment data from :class:`EnvironmentCallback` and dataset
    verification/split data from :class:`DatasetVerificationCallback` when
    present as sibling callbacks.  When used alone, falls back to doing its
    own environment/verification work so the user still gets a complete PROV
    document.

    Args:
        model_name: Identifier for this model (shown in run name).
        experiment_name: MLflow experiment name (default: "Training_Pipeline").
        manifest_path: Path to manifest.csv (auto-detected if None).
        data_root: Root directory for dataset files (auto-detected if None).
        test_size: Fraction of data for the test split.
        random_state: Random seed for train/test split.
        fail_fast: Abort training if dataset verification fails.
        strict: If True, re-raise errors from optional provenance steps.
        register_model: If True, auto-log model summary from pl_module.
        register_optimizer: Log optimizer config (or True to auto-detect).
        register_scheduler: Log scheduler config (or True to auto-detect).
        prov_prefixes: Optional override for PROV namespace prefixes.
    """

    def __init__(
        self,
        model_name: str | None = None,
        experiment_name: str = "Training_Pipeline",
        manifest_path: str | None = None,
        data_root: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        fail_fast: bool = True,
        strict: bool = False,
        register_model: bool = True,
        register_optimizer: bool = True,
        register_scheduler: bool = True,
        prov_prefixes: dict[str, str] | None = None,
    ) -> None:
        """Initialise the provenance callback.

        Args:
            model_name: Name of the model (defaults to ``MODEL_NAME`` env var or "model").
            experiment_name: MLflow experiment name for the training run.
            manifest_path: Path to manifest.csv (auto-detected if None).
            data_root: Root directory of the dataset (auto-detected if None).
            test_size: Fraction of data for the test split. Set to 0 to skip splitting.
            random_state: Random seed for train/test split.
            fail_fast: Abort training if dataset verification fails.
            strict: If True, re-raise errors from optional provenance steps.
            register_model: If True, auto-log model summary from pl_module.
            register_optimizer: Log optimizer config (or True to auto-detect).
            register_scheduler: Log scheduler config (or True to auto-detect).
            prov_prefixes: Optional override for PROV namespace prefixes.
        """
        self.model_name = model_name or os.environ.get("MODEL_NAME", "model")
        self.experiment_name = experiment_name
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.test_size = test_size
        self.random_state = random_state
        self.fail_fast = fail_fast
        self.strict = strict
        self.register_model = register_model
        self.register_optimizer = register_optimizer
        self.register_scheduler = register_scheduler
        self._prov_prefixes = prov_prefixes

        # Internal state (populated by on_fit_start or sibling callbacks)
        self._run_id: str | None = None
        self._temp_dirs: list[str] = []
        self._split_data: dict[str, object] | None = None
        self._verification: dict[str, object] | None = None
        self._frozen_requirements: str | None = None
        self._git_commit: str = "unknown"
        self._git_url: str = "unknown"
        self._git_branch: str = "unknown"

    # ── helpers ──────────────────────────────────────────────

    def _gather_from_siblings(self, trainer: Any) -> None:
        """Read data already collected by sibling callbacks."""
        from rationai.mlkit.lightning.callbacks.dataset_verification import (
            DatasetVerificationCallback,
        )
        from rationai.mlkit.lightning.callbacks.environment import EnvironmentCallback

        for cb in trainer.callbacks:
            if isinstance(cb, EnvironmentCallback):
                self._git_commit = getattr(cb, "_git_commit", "unknown")
                self._git_url = getattr(cb, "_git_url", "unknown")
                self._git_branch = getattr(cb, "_git_branch", "unknown")
                self._frozen_requirements = getattr(cb, "_frozen_requirements", None)
            elif isinstance(cb, DatasetVerificationCallback):
                self._verification = getattr(cb, "_verification", None)
                self._split_data = getattr(cb, "_split_data", None)

    def _fallback_on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Do environment + verification work when no sibling callbacks exist."""
        from rationai.mlkit.lightning.callbacks.environment import (
            _detect_docker,
            _detect_hardware,
            _lookup_user_run,
            _snapshot_environment,
        )
        from rationai.mlkit.provenance.dataset import (
            _detect_manifest,
            _lookup_dataset_run,
            _verify_dataset,
        )

        # ── Git info (read from MLflow tags set by MLFlowLogger) ──
        try:
            run = mlflow.active_run()
            run_tags: dict[str, str] = {}
            if run and run.info and run.info.run_id:
                client = mlflow.tracking.MlflowClient()
                run_data = client.get_run(run.info.run_id)
                run_tags = dict(run_data.data.tags) if run_data.data.tags else {}
            self._git_commit = run_tags.get(
                "mlflow.source.git.commit", run_tags.get("git.commit", "unknown")
            )
            self._git_url = run_tags.get(
                "mlflow.source.git.repoUrl", run_tags.get("git.repo_url", "unknown")
            )
            self._git_branch = run_tags.get(
                "mlflow.source.git.branch", run_tags.get("git.branch", "unknown")
            )
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Git info failed: %s", e)

        # ── User lookup ─────────────────────────────────────────
        try:
            user_run_id, user_tags = _lookup_user_run()
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] User lookup failed: %s", e)
            user_run_id, user_tags = None, {}

        # ── Hardware (skip if MLflow system metrics are on) ─────
        sys_metrics_on = any(
            getattr(logger, "log_system_metrics", False) for logger in trainer.loggers
        )
        hardware = {} if sys_metrics_on else _detect_hardware()
        docker = _detect_docker()

        # ── Dataset verification & split ────────────────────────
        manifest_path = self.manifest_path
        data_root = self.data_root

        if manifest_path is None:
            manifest_path, data_root = _detect_manifest()
        elif data_root is None:
            data_root = os.path.dirname(os.path.abspath(manifest_path))

        if manifest_path and data_root:
            from rationai.mlkit.provenance.dataset import (
                load_manifest,
            )

            # ── Verification (always) ────────────────────────────
            dataset_run_id = _lookup_dataset_run()
            verification = _verify_dataset(manifest_path, data_root, dataset_run_id)
            self._verification = verification or {}
            verification_details: list[str] = verification.get("details", [])

            for detail in verification_details:
                log.info(f"  [ProvenanceCallback] {detail}")

            # Log verification results
            if verification:
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
                        "; ".join(verification["details"]),
                    )

                if self.fail_fast and not verification["verified"]:
                    raise RuntimeError(
                        "Dataset verification failed — aborting training.\n"
                        + "\n".join(f"  {d}" for d in verification["details"]),
                    )

            # ── Train/test split (only if test_size > 0) ─────────
            if self.test_size > 0:
                from sklearn.model_selection import train_test_split

                samples = load_manifest(manifest_path, data_root)
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
                split_dir = f"_mlflow_split_{uuid.uuid4().hex[:8]}"
                os.makedirs(split_dir, exist_ok=True)
                self._temp_dirs.append(split_dir)
                for subset_name, subset_samples in [
                    ("train", train_samples),
                    ("test", test_samples),
                ]:
                    split_file = os.path.join(split_dir, f"{subset_name}_split.csv")
                    pd.DataFrame(subset_samples).to_csv(split_file, index=False)

                mlflow.log_artifacts(split_dir, artifact_path="split")

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
                    }
                )
        else:
            log.warning(
                "[ProvenanceCallback] No manifest.csv found — "
                "train/test split not logged."
            )

        # ── Tags ────────────────────────────────────────────────
        tags: dict[str, str] = {}
        if user_run_id:
            tags["user_run_id"] = user_run_id
            for key in ("username", "real_name", "organization"):
                if key in user_tags:
                    tags[key] = user_tags[key]

        dataset_run_id = _lookup_dataset_run()
        if dataset_run_id:
            tags["dataset_run_id"] = dataset_run_id

        tags.update(
            {
                "git_commit": self._git_commit,
                "git_url": self._git_url,
                "git_branch": self._git_branch,
                "prov_start_time": datetime.now(UTC).isoformat(),
            }
        )
        mlflow.set_tags(tags)

        # ── Params: hardware + docker + split config ────────────
        all_params: dict[str, str | float | int] = {
            "model_name": self.model_name,
            **hardware,
            **docker,
            "split_test_size": self.test_size,
            "split_random_state": self.random_state,
            "split_stratified": True,
        }
        mlflow.log_params(all_params)

        # ── Environment snapshot ────────────────────────────────
        artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
        os.makedirs(artifact_dir, exist_ok=True)
        self._temp_dirs.append(artifact_dir)
        try:
            self._frozen_requirements = _snapshot_environment(artifact_dir)
            mlflow.log_artifacts(artifact_dir, artifact_path="environment")
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Environment snapshot failed: %s", e)

    # ── lightning hooks ───────────────────────────────────────

    def _ensure_active_run(self, trainer: Any) -> str | None:
        """Ensure MLflow has an active run by triggering the logger's experiment.

        Returns the run_id if successful, or None if no MLFlowLogger is present.
        """
        try:
            from rationai.mlkit.lightning.loggers.mlflow import MLFlowLogger
        except ImportError:
            # Standalone mlflow — rely on whatever active_run exists
            run = mlflow.active_run()
            return run.info.run_id if run else None

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                # Access .experiment to trigger lazy init + active-run setup
                _ = logger.experiment
                self._run_id = logger.run_id
                return logger.run_id

        run = mlflow.active_run()
        return run.info.run_id if run else None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Gather environment/verification data from siblings or fall back."""
        from rationai.mlkit.lightning.callbacks.dataset_verification import (
            DatasetVerificationCallback,
        )
        from rationai.mlkit.lightning.callbacks.environment import EnvironmentCallback

        # Ensure the MLFlowLogger has an active run before any fluent API calls
        self._ensure_active_run(trainer)

        # Check if sibling callbacks are present
        has_env = any(isinstance(cb, EnvironmentCallback) for cb in trainer.callbacks)
        has_verify = any(
            isinstance(cb, DatasetVerificationCallback) for cb in trainer.callbacks
        )

        if has_env or has_verify:
            self._gather_from_siblings(trainer)
        else:
            # No siblings — do everything ourselves
            self._fallback_on_fit_start(trainer, pl_module)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Log model/optimizer/scheduler summaries and PROV document."""
        _active_run = mlflow.active_run()
        if _active_run:
            run_id: str = _active_run.info.run_id
        elif self._run_id:
            run_id = self._run_id
        else:
            self._ensure_active_run(trainer)
            if self._run_id:
                run_id = self._run_id
            else:
                return

        # Get a Run object for metadata access
        active_run = mlflow.get_run(run_id)

        # ── Model summary ───────────────────────────────────────
        if self.register_model and pl_module is not None:
            try:
                model_summary = _model_summary(pl_module)
                mlflow.log_params(model_summary)
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Model summary failed: %s", e)

        # ── Optimizer summary ───────────────────────────────────
        if self.register_optimizer and pl_module is not None:
            try:
                for opt in trainer.optimizers:
                    optimizer_info = _optimizer_summary(opt)
                    mlflow.log_params(optimizer_info)
                    break
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Optimizer summary failed: %s", e)

        # ── Scheduler summary ───────────────────────────────────
        if self.register_scheduler and pl_module is not None:
            try:
                for sched in getattr(trainer, "lr_schedulers", []):
                    scheduler_info = _scheduler_summary(sched.get("scheduler"))
                    mlflow.log_params(scheduler_info)
                    break
            except Exception as e:
                if self.strict:
                    raise
                log.warning("[ProvenanceCallback] Scheduler summary failed: %s", e)

        # ── PROV document + run summary ─────────────────────────
        try:
            run_data = mlflow.get_run(run_id)
            params = {k: str(v) for k, v in run_data.data.params.items()}
            metrics = {k: float(v) for k, v in run_data.data.metrics.items()}
            tags = {
                k: v
                for k, v in run_data.data.tags.items()
                if not k.startswith("mlflow.")
            }

            # ── Run summary JSON ────────────────────────────────
            summary_dir = f"_mlflow_summary_{uuid.uuid4().hex[:8]}"
            os.makedirs(summary_dir, exist_ok=True)
            self._temp_dirs.append(summary_dir)
            summary_path = os.path.join(summary_dir, "run_summary.json")

            summary = {
                "model_name": self.model_name,
                "params": dict(run_data.data.params),
                "metrics": {k: float(v) for k, v in run_data.data.metrics.items()},
                "tags": tags,
                "run_id": run_id,
                "experiment_name": self.experiment_name,
                "split": {
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "stratified": True,
                    "train_count": len(self._split_data["train"])
                    if isinstance(self._split_data, dict)
                    else 0,  # type: ignore[arg-type]
                    "test_count": len(self._split_data["test"])
                    if isinstance(self._split_data, dict)
                    else 0,  # type: ignore[arg-type]
                    "train": self._split_data["train"] if self._split_data else None,
                    "test": self._split_data["test"] if self._split_data else None,
                }
                if self._split_data
                else None,
                "dataset_verification": self._verification,
                "requirements": self._frozen_requirements,
                "source": {
                    "git_commit": self._git_commit,
                    "git_branch": self._git_branch,
                    "git_remote": self._git_url,
                },
            }

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            mlflow.log_artifact(summary_path, artifact_path="provenance")
            shutil.rmtree(summary_dir, ignore_errors=True)

            # ── PROV document (§9 — configurable prefixes) ──────
            prov_doc = _build_prov_document(
                run_id=run_id,
                run_name=active_run.info.run_name or f"Training_{self.model_name}",
                params=params,
                metrics=metrics,
                tags=tags,
                start_time_ms=active_run.info.start_time,
                end_time_ms=active_run.info.end_time,
                split_data={
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "train": self._split_data["train"] if self._split_data else None,
                    "test": self._split_data["test"] if self._split_data else None,
                }
                if self._split_data
                else None,
                requirements=self._frozen_requirements,
                verification=self._verification,
                prov_prefixes=_get_prov_prefixes(self._prov_prefixes),
            )

            prov_dir = f"_mlflow_prov_{uuid.uuid4().hex[:8]}"
            os.makedirs(prov_dir, exist_ok=True)
            self._temp_dirs.append(prov_dir)
            prov_path = os.path.join(prov_dir, "prov.json")
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)

            mlflow.log_artifact(prov_path, artifact_path="provenance")
            shutil.rmtree(prov_dir, ignore_errors=True)

            log.info("[ProvenanceCallback] Complete → %s", run_id)
        except Exception as e:
            if self.strict:
                raise
            log.warning(
                "[ProvenanceCallback] Could not write provenance artifacts: %s", e
            )

        # ── Clean up temp dirs ──────────────────────────────────
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
