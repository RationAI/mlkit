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

import contextlib
import json
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from typing import Any, cast

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
        split_uris: Optional mapping of split name → CSV URI
            (``mlflow-artifacts:/...`` or local path) produced by an upstream
            preprocessing pipeline.  When given, these are the authoritative
            splits: they are referenced (not copied) in provenance and no
            train/test split is recomputed.
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
        split_uris: dict[str, str] | None = None,
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
            split_uris: Optional mapping of split name to dataset URI, used to
                reference pre-existing splits instead of recomputing train/test.
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
        self.split_uris = split_uris
        self._prov_prefixes = prov_prefixes

        # Internal state (populated by on_fit_start or sibling callbacks)
        self._run_id: str | None = None
        self._temp_dirs: list[str] = []
        self._split_data: dict[str, object] | None = None
        self._verification: dict[str, object] | None = None
        self._frozen_requirements: str | None = None

    # ── helpers ──────────────────────────────────────────────

    def _gather_from_siblings(self, trainer: Any) -> None:
        """Read data already collected by sibling callbacks."""
        from rationai.mlkit.lightning.callbacks.dataset_verification import (
            DatasetVerificationCallback,
        )
        from rationai.mlkit.lightning.callbacks.environment import EnvironmentCallback

        for cb in trainer.callbacks:
            if isinstance(cb, EnvironmentCallback):
                self._frozen_requirements = cb.frozen_requirements
            elif isinstance(cb, DatasetVerificationCallback):
                self._verification = cb.verification
                self._split_data = cb.split_data

    def _load_provided_splits(self) -> None:
        """Load precomputed split CSVs declared via ``split_uris``.

        The URIs point at artifacts of an upstream preprocessing run
        (``mlflow-artifacts:/...``) or local CSV files.  They are referenced
        from provenance, never copied into this run, and no new split is
        computed.
        """
        from mlflow.artifacts import download_artifacts
        from omegaconf import DictConfig, OmegaConf

        splits_node: Any = self.split_uris
        if isinstance(splits_node, DictConfig):
            splits_node = OmegaConf.to_container(splits_node, resolve=True)

        uris = {str(k): str(v) for k, v in dict(splits_node).items()}
        stats: dict[str, dict[str, Any]] = {}
        for name, uri in uris.items():
            local = (
                download_artifacts(uri) if uri.startswith("mlflow-artifacts:") else uri
            )
            df = pd.read_csv(local)
            entry: dict[str, Any] = {"rows": len(df), "uri": uri}
            if "case_id" in df.columns:
                entry["cases"] = int(df["case_id"].nunique())
            if "fold" in df.columns:
                try:
                    entry["folds"] = sorted(int(f) for f in df["fold"].unique())
                except (TypeError, ValueError):
                    log.warning(
                        "[ProvenanceCallback] split %r: non-integer fold values "
                        "ignored",
                        name,
                    )
            stats[name] = entry
            log.info(
                "[ProvenanceCallback] split %r: %s rows (%s)",
                name,
                entry["rows"],
                uri,
            )

        self._split_data = {
            "source": "artifacts",
            "splits": stats,
        }

        params: dict[str, str | int] = {}
        for name, entry in stats.items():
            params[f"split_{name}_rows"] = entry["rows"]
            if "cases" in entry:
                params[f"split_{name}_cases"] = entry["cases"]
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "split_source": "preprocessing_artifacts",
                "split_uris": json.dumps(uris),
            }
        )

    def _split_summary(self) -> dict[str, Any] | None:
        """Serializable summary of the split used, for prov + run summary."""
        if not self._split_data:
            return None
        if "splits" in self._split_data:
            return {
                "source": "artifacts",
                "splits": self._split_data["splits"],
            }
        train = cast("list[Any]", self._split_data.get("train") or [])
        test = cast("list[Any]", self._split_data.get("test") or [])
        return {
            "source": "manifest_split",
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratified": True,
            "train_count": len(train),
            "test_count": len(test),
            "train": self._split_data.get("train"),
            "test": self._split_data.get("test"),
        }

    def _lookup_user_safe(self) -> tuple[str | None, dict[str, str]]:
        """Best-effort user lookup; returns ``(run_id, tags)`` or ``(None, {})``."""
        from rationai.mlkit.provenance.environment import lookup_user_run

        try:
            return lookup_user_run()
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] User lookup failed: %s", e)
            return None, {}

    def _detect_environment_params(
        self, trainer: Any
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        """Detect hardware / docker / pytorch / seed settings for the run."""
        from rationai.mlkit.provenance.environment import (
            detect_hardware,
            detect_pytorch,
            detect_seeds,
            get_container_image,
        )

        # Hardware is skipped when MLflow system metrics are enabled
        sys_metrics_on = any(
            getattr(logger, "log_system_metrics", False) for logger in trainer.loggers
        )
        hardware = {} if sys_metrics_on else detect_hardware()
        docker: dict[str, str] = {}
        container_image = get_container_image()
        if container_image:
            docker["container_image"] = container_image
        pytorch = detect_pytorch()
        seeds = detect_seeds()
        return hardware, docker, pytorch, seeds

    def _load_provided_splits_safe(self) -> None:
        """Load ``split_uris`` (authoritative splits), respecting ``strict``."""
        if not self.split_uris:
            return
        try:
            self._load_provided_splits()
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Loading provided splits failed: %s", e)

    def _resolve_manifest(self) -> tuple[str | None, str | None]:
        """Resolve ``(manifest_path, data_root)``, auto-detecting when unset."""
        from rationai.mlkit.provenance.dataset import detect_manifest

        manifest_path = self.manifest_path
        data_root = self.data_root

        if manifest_path is None and not self.split_uris:
            manifest_path, data_root = detect_manifest()
        elif data_root is None and manifest_path is not None:
            data_root = os.path.dirname(os.path.abspath(manifest_path))
        return manifest_path, data_root

    def _verify_and_split(self, manifest_path: str, data_root: str) -> None:
        """Verify the dataset manifest and (legacy) compute a train/test split."""
        from rationai.mlkit.provenance.dataset import (
            lookup_dataset_run,
            verify_manifest,
        )

        # ── Verification (always) ────────────────────────────
        dataset_run_id = lookup_dataset_run()
        verification = verify_manifest(manifest_path, data_root, dataset_run_id)
        self._verification = verification or {}
        for detail in verification.get("details", []):
            log.info(f"  [ProvenanceCallback] {detail}")

        if verification:
            mlflow.log_params(
                {
                    "dataset_verified": verification["verified"],
                    "dataset_file_sizes_match": bool(verification["file_sizes_match"]),
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

        # ── Train/test split (only if test_size > 0 and no provided splits) ─
        if self.test_size > 0 and self._split_data is None:
            self._legacy_train_test_split(manifest_path, data_root)

    def _legacy_train_test_split(self, manifest_path: str, data_root: str) -> None:
        """Recompute a stratified train/test split from the manifest."""
        from sklearn.model_selection import train_test_split

        from rationai.mlkit.provenance.dataset import load_manifest

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
        self._log_split_artifacts(train_samples, test_samples)

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

    def _log_split_artifacts(
        self,
        train_samples: list[dict[str, Any]],
        test_samples: list[dict[str, Any]],
    ) -> None:
        """Write train/test split CSVs and log them as MLflow artifacts."""
        split_dir = f"_mlflow_split_{uuid.uuid4().hex[:8]}"
        self._temp_dirs.append(split_dir)
        try:
            os.makedirs(split_dir, exist_ok=True)
            for subset_name, subset_samples in [
                ("train", train_samples),
                ("test", test_samples),
            ]:
                split_file = os.path.join(split_dir, f"{subset_name}_split.csv")
                pd.DataFrame(subset_samples).to_csv(split_file, index=False)

            mlflow.log_artifacts(split_dir, artifact_path="split")
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Split artifact logging failed: %s", e)

    def _log_run_tags(self, user_run_id: str | None, user_tags: dict[str, str]) -> None:
        """Set provenance tags (user / dataset run links, start time)."""
        from rationai.mlkit.provenance.dataset import lookup_dataset_run

        tags: dict[str, str] = {}
        if user_run_id:
            tags["user_run_id"] = user_run_id
            for key in ("username", "real_name", "organization"):
                if key in user_tags:
                    tags[key] = user_tags[key]

        dataset_run_id = lookup_dataset_run()
        if dataset_run_id:
            tags["dataset_run_id"] = dataset_run_id

        tags["prov_start_time"] = datetime.now(UTC).isoformat()
        mlflow.set_tags(tags)

    def _log_env_snapshot(self) -> None:
        """Snapshot the python environment and log it as artifacts."""
        from rationai.mlkit.provenance.environment import snapshot_environment

        artifact_dir = f"_mlflow_env_{uuid.uuid4().hex[:8]}"
        self._temp_dirs.append(artifact_dir)
        try:
            os.makedirs(artifact_dir, exist_ok=True)
            self._frozen_requirements = snapshot_environment(artifact_dir)
            mlflow.log_artifacts(artifact_dir, artifact_path="environment")
        except Exception as e:
            if self.strict:
                raise
            log.warning("[ProvenanceCallback] Environment snapshot failed: %s", e)

    def _fallback_on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Do environment + verification work when no sibling callbacks exist."""
        user_run_id, user_tags = self._lookup_user_safe()
        hardware, docker, pytorch, seeds = self._detect_environment_params(trainer)
        self._load_provided_splits_safe()

        manifest_path, data_root = self._resolve_manifest()
        if manifest_path and data_root:
            self._verify_and_split(manifest_path, data_root)
        else:
            log.warning(
                "[ProvenanceCallback] No manifest.csv found — "
                "train/test split not logged."
            )

        self._log_run_tags(user_run_id, user_tags)

        # ── Params: hardware + docker + pytorch + split config ──
        all_params: dict[str, str | float | int] = {
            "model_name": self.model_name,
            **hardware,
            **docker,
            **pytorch,
            "split_test_size": self.test_size,
            "split_random_state": self.random_state,
            "split_stratified": True,
        }
        mlflow.log_params(all_params)

        # ── Seeds as tags ───────────────────────────────────────
        if seeds:
            mlflow.set_tags({f"seed_{k}": str(v) for k, v in seeds.items()})

        self._log_env_snapshot()

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
                "split": self._split_summary(),
                "dataset_verification": self._verification,
                "requirements": self._frozen_requirements,
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
                split_data=self._split_summary(),
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
            # best-effort cleanup of temporary artifact dirs
            with contextlib.suppress(OSError):
                shutil.rmtree(d)
