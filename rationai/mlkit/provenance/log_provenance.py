"""Universal provenance logging for any pipeline run.

One entry point — ``log_provenance`` — captures everything that describes a
run regardless of its type (mask creation, tiling, filtering, embeddings,
training, inference, …):

* environment + hardware + git + seeds (via ``capture_environment``)
* flattened scalar config leaves as MLflow params
* output dataframe statistics (+ optional file manifest and label breakdown)
* upstream inputs (explicit or auto-detected ``mlflow-artifacts:/`` URIs in
  the config) — links preprocessing steps into a chain
* a self-contained PROV-O JSON document (``provenance/prov.json`` artifact)

Every argument is optional; the function derives what it can from the
Hydra config and the active MLflow run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from rationai.mlkit.provenance.common import (
    get_prov_prefixes,
    iso_timestamp,
    qualified,
    qualified_name,
    safe_id,
    typed_value,
)
from rationai.mlkit.provenance.environment import capture_environment


if TYPE_CHECKING:
    from rationai.mlkit.lightning.loggers import MLFlowLogger

log = logging.getLogger(__name__)

# Columns probed (in order) when looking for file paths / labels.
_PATH_COLUMN_CANDIDATES = (
    "slide_path",
    "wsi_path",
    "image_path",
    "filepath",
    "path",
    "tile_path",
    "mask_path",
    "annot_path",
)
_LABEL_COLUMN_CANDIDATES = ("label", "annotation", "class", "diagnosis")

# Config paths probed (in order) for dataset name / version / label semantics.
_NAME_CONFIG_PATHS = ("dataset.name", "dataset_name")
_VERSION_CONFIG_PATHS = ("dataset.version", "dataset_version")
_POSITIVE_LABEL_CONFIG_PATHS = ("dataset.positive_label", "positive_label")

# URI schemes recognised as references to upstream run artifacts.
_INPUT_URI_SCHEMES = ("mlflow-artifacts:/", "runs:/", "models:/")

_MAX_PARAMS = 200  # safety cap for flattened config leaves
_MAX_LABEL_CLASSES = 40  # per-class count params only for small cardinalities


def _select_config(config: Any, path: str) -> Any:
    """Read a dotted path from a Hydra config / dict, returning None if absent."""
    if config is None:
        return None
    try:
        if isinstance(config, DictConfig):
            return OmegaConf.select(config, path, default=None)
        node: Any = config
        for part in path.split("."):
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        return node
    except Exception:
        return None


def _select_any(config: Any, paths: tuple[str, ...]) -> Any:
    """First non-None value among candidate config paths."""
    for path in paths:
        value = _select_config(config, path)
        if value is not None:
            return value
    return None


def _flatten_scalars(node: Any, prefix: str, out: dict[str, str]) -> None:
    """Flatten config leaves into ``key.subkey: str(value)`` scalar params."""
    if len(out) >= _MAX_PARAMS:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_scalars(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, (list, tuple)):
        if len(node) <= 5 and all(not isinstance(v, (dict, list, tuple)) for v in node):
            out[prefix] = json.dumps([str(v) for v in node])
    elif node is None:
        out[prefix] = "null"
    else:
        out[prefix] = str(node)


def _config_to_container(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=False)
        if isinstance(container, dict):
            return cast("dict[str, Any]", container)
        return {}
    if isinstance(config, dict):
        return config
    return {}


def _find_input_uris(node: Any, found: set[str]) -> None:
    """Collect upstream-artifact URI strings from config leaves."""
    if isinstance(node, dict):
        for value in node.values():
            _find_input_uris(value, found)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _find_input_uris(value, found)
    elif isinstance(node, str) and node.startswith(_INPUT_URI_SCHEMES):
        found.add(node)


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _file_manifest(
    df: pd.DataFrame, path_column: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per-file manifest rows + size map for an existing path column."""
    rows: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for raw in df[path_column]:
        path_str = str(raw)
        basename = os.path.basename(path_str)
        fpath = Path(path_str)
        size = -1
        mtime = "unknown"
        if fpath.exists():
            try:
                stat = fpath.stat()
            except OSError:
                pass
            else:
                size = int(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        sizes[basename] = size
        rows.append(
            {
                "filename": basename,
                "path": path_str,
                "size_bytes": size,
                "modified_at": mtime,
            }
        )
    return rows, sizes


def _label_breakdown(
    df: pd.DataFrame,
    label_column: str,
    positive_label: str | None,
    frame_name: str,
) -> dict[str, int]:
    """Positive/negative counts, or per-class counts for small cardinalities."""
    prefix = f"{frame_name}_" if frame_name != "output" else ""
    if positive_label is not None:
        num_positive = int((df[label_column] != positive_label).sum())
        return {
            f"{prefix}num_positive": num_positive,
            f"{prefix}num_negative": len(df) - num_positive,
        }
    uniques = df[label_column].dropna().unique()
    if 1 < len(uniques) <= _MAX_LABEL_CLASSES:
        counts = df[label_column].value_counts()
        return {
            f"{prefix}label_{safe_id(str(k))}_count": int(v) for k, v in counts.items()
        }
    return {}


def _frame_stats(
    frames: dict[str, pd.DataFrame],
    *,
    positive_label: str | None,
    path_column: str | None,
    label_column: str | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """Compute per-frame stats, manifests and file sizes."""
    stats: dict[str, Any] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    file_size_maps: dict[str, dict[str, int]] = {}
    for frame_name, df in frames.items():
        prefix = f"{frame_name}_" if frame_name != "output" else ""
        stats[f"{prefix}num_rows"] = len(df)
        stats[f"{prefix}num_columns"] = len(df.columns)
        stats[f"{prefix}columns"] = json.dumps([str(c) for c in df.columns])

        pcol = (
            path_column
            if path_column in df.columns
            else _detect_column(df, _PATH_COLUMN_CANDIDATES)
        )
        if pcol is not None:
            rows, sizes = _file_manifest(df, pcol)
            manifests[frame_name] = rows
            file_size_maps[frame_name] = sizes
            stats[f"{prefix}path_column"] = pcol
            stats[f"{prefix}files_existing"] = sum(1 for s in sizes.values() if s >= 0)

        lcol = (
            label_column
            if label_column in df.columns
            else _detect_column(df, _LABEL_COLUMN_CANDIDATES)
        )
        if lcol is not None:
            stats[f"{prefix}label_column"] = lcol
            stats.update(_label_breakdown(df, lcol, positive_label, frame_name))
    return stats, manifests, file_size_maps


# ──────────────────────────────────────────────
# PROV document builder (generic run)
# ──────────────────────────────────────────────


def _get_run_tags() -> dict[str, str]:
    """Tags of the active run (empty dict when unavailable)."""
    try:
        from mlflow.tracking import MlflowClient

        run = mlflow.active_run()
        if run is None:
            return {}
        client = MlflowClient()
        return dict(client.get_run(run.info.run_id).data.tags)
    except Exception:
        return {}


def _input_entity_id(uri: str) -> str:
    return qualified("gen", safe_id(f"input_{uri}"))


def build_run_prov(
    run_id: str,
    *,
    name: str,
    version: str | None,
    activity_name: str,
    stats: dict[str, Any],
    inputs: list[str],
    extra_meta: dict[str, Any],
    frames: list[str] | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a generic PROV-O document for an arbitrary pipeline run.

    Structure matches the other builders: agent (from run tags), one run
    activity, a CPM metadata bundle, output/input entities and the CPM
    wrapper activity.
    """
    prefixes = prov_prefixes or get_prov_prefixes()
    tags = _get_run_tags()

    username = tags.get("username", tags.get("mlflow.user", "unknown"))
    real_name = tags.get("real_name", username)
    organization = tags.get("organization", "")

    agent_id = qualified("gen", safe_id(f"user_{username}"))
    run_act_id = qualified("gen", safe_id(f"run_{run_id}"))
    meta_id = qualified("meta", run_id)
    main_act_id = qualified("blank", f"Run_{run_id[:8]}")
    version_id = (version or "snapshot").replace(".", "_")
    output_id = qualified("gen", safe_id(f"output_{name}_{version_id}"))

    # ── AGENT ────────────────────────────────────────────
    agent_props: dict[str, Any] = {
        "schema:name": typed_value(real_name),
        "prov:type": [qualified_name("schema", "Person")],
    }
    if organization:
        agent_props["schema:affiliation"] = typed_value(organization)

    now = iso_timestamp()

    # ── RUN ACTIVITY ─────────────────────────────────────
    run_activity: dict[str, Any] = {
        "prov:type": [qualified_name("schema", "Action")],
        "prov:startTime": [now],
        "schema:name": typed_value(activity_name),
        "gen:output_name": typed_value(name),
    }
    if version:
        run_activity["gen:output_version"] = typed_value(version)
    git_commit = tags.get("git_commit", tags.get("mlflow.source.git.commit", ""))
    if git_commit:
        run_activity["schema:identifier"] = typed_value(git_commit)

    # ── OUTPUT ENTITIES (one per frame when several) ─────
    output_ids: list[str] = [output_id]
    entities: dict[str, Any] = {}
    if frames and len(frames) > 1:
        output_ids = []
        for frame_name in frames:
            frame_id = qualified(
                "gen", safe_id(f"output_{name}_{frame_name}_{version_id}")
            )
            entities[frame_id] = {
                "schema:name": typed_value(f"{name}/{frame_name}"),
                "prov:type": [qualified_name("sosa", "Sample")],
                "dct:description": typed_value(
                    f"Output {frame_name} of {name} v{version or 'snapshot'} of {activity_name}"
                ),
                "gen:output_name": typed_value(name),
                "gen:frame": typed_value(frame_name),
            }
            output_ids.append(frame_id)
    else:
        entities[output_id] = {
            "schema:name": typed_value(name),
            "prov:type": [qualified_name("sosa", "Sample")],
            "dct:description": typed_value(
                f"Output {name} v{version or 'snapshot'} of {activity_name}"
            ),
        }

    # ── INPUT ENTITIES + USED ────────────────────────────
    used: dict[str, Any] = {}
    rel_no = 0
    for uri in inputs:
        input_id = _input_entity_id(uri)
        entities[input_id] = {
            "schema:name": typed_value(uri),
            "schema:url": typed_value(uri),
            "prov:type": [qualified_name("sosa", "Sample")],
        }
        used[f"_:n{rel_no}"] = {"prov:activity": run_act_id, "prov:entity": input_id}
        rel_no += 1

    # ── CPM METADATA ENTITY ──────────────────────────────
    meta_entity: dict[str, Any] = {
        "prov:type": [qualified_name("cpm", "BundleMetadata")],
        "gen:output_name": typed_value(name),
        "gen:run_name": typed_value(activity_name),
    }
    if version:
        meta_entity["gen:output_version"] = typed_value(version)
    if organization:
        meta_entity["cpm:organization"] = typed_value(organization)
    for key, value in stats.items():
        meta_entity[f"gen:{safe_id(str(key))}"] = typed_value(value)
    if inputs:
        meta_entity["gen:input_uris"] = [json.dumps(sorted(inputs))]
    for key, value in extra_meta.items():
        meta_entity[f"gen:{safe_id(str(key))}"] = typed_value(value)

    entities[meta_id] = meta_entity

    # ── CPM MAIN ACTIVITY ────────────────────────────────
    main_activity: dict[str, Any] = {
        "prov:type": [qualified_name("cpm", "mainActivity")],
        "cpm:referencedMetaBundleId": [{"type": "prov:QUALIFIED_NAME", "$": meta_id}],
        "dct:hasPart": [{"type": "prov:QUALIFIED_NAME", "$": run_act_id}],
    }

    was_generated_by: dict[str, Any] = {}
    for i, out_id in enumerate(output_ids):
        was_generated_by[f"_:n{rel_no + i}"] = {
            "prov:entity": out_id,
            "prov:activity": run_act_id,
        }
    was_associated_with = {
        f"_:n{rel_no + len(output_ids)}": {
            "prov:activity": run_act_id,
            "prov:agent": agent_id,
        }
    }

    inner: dict[str, Any] = {
        "prefix": prefixes,
        "entity": entities,
        "activity": {run_act_id: run_activity, main_act_id: main_activity},
        "agent": {agent_id: agent_props},
        "wasAssociatedWith": was_associated_with,
        "wasGeneratedBy": was_generated_by,
    }
    inner["entity"][meta_id] = meta_entity
    if used:
        inner["used"] = used

    return {"bundle": {f"storage:{run_id}": inner}}


# ──────────────────────────────────────────────
# Universal provenance entry point
# ──────────────────────────────────────────────


def log_provenance(
    output: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    logger: MLFlowLogger | None = None,
    config: DictConfig | dict[str, Any] | None = None,
    *,
    name: str | None = None,
    version: str | None = None,
    positive_label: str | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    inputs: list[str] | None = None,
    snapshot_env: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log end-to-end provenance for the current run to MLflow.

    Works for any pipeline step (dataset creation, masking, tiling,
    filtering, embeddings, training, inference, ...).  Everything is
    optional; values are derived from *config* and the active run whenever
    possible.

    Args:
        output: Produced dataframe, or mapping of split name → dataframe.
        logger: ``MLFlowLogger`` of the run (falls back to the active run).
        config: Hydra config / plain dict (used for name/version derivation,
            scalar param logging and upstream-URI detection).
        name: Output name (default: ``config.dataset.name`` →
            ``config.dataset_name`` → run name).
        version: Output version string (default: config value).
        positive_label: Value of the label column counting as the *negative*
            class; other values count as positive.  When None and the label
            column is categorical (≤ 40 classes), per-class counts are
            logged instead.
        path_column: Column with file paths (auto-detected from common
            names when omitted; file manifest skipped when no match).
        label_column: Column with labels (auto-detected likewise).
        inputs: Upstream references (``mlflow-artifacts:/...`` URIs or
            paths).  Merged with URIs auto-detected inside *config*.
        snapshot_env: Whether to snapshot the python environment.
        extra: Extra key/value pairs for the PROV metadata bundle.

    Returns:
        Dict with the computed stats and the PROV document.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run — call inside @autolog")
    run_id = active_run.info.run_id

    # ── Environment (hardware, git, docker, seeds, freeze) ──
    capture_environment(snapshot_env=snapshot_env)

    # ── Derive identity ───────────────────────────────────
    run_name = active_run.data.tags.get("mlflow.runName", run_id)
    if name is None:
        raw_name = _select_any(config, _NAME_CONFIG_PATHS)
        name = str(raw_name) if raw_name is not None else str(run_name)
    if version is None:
        raw_version = _select_any(config, _VERSION_CONFIG_PATHS)
        version = str(raw_version) if raw_version is not None else None
    if positive_label is None:
        raw_label = _select_any(config, _POSITIVE_LABEL_CONFIG_PATHS)
        positive_label = str(raw_label) if raw_label is not None else None

    # ── Inputs: explicit + config-detected upstream URIs ──
    all_inputs: set[str] = set(inputs or [])
    _find_input_uris(_config_to_container(config), all_inputs)
    input_list = sorted(all_inputs)

    # ── Flatten scalar config leaves to params ────────────
    container = _config_to_container(config)
    if container:
        flat: dict[str, str] = {}
        _flatten_scalars(container, "", flat)
        if flat:
            mlflow.log_params(
                {f"cfg_{safe_id(k)}": v for k, v in list(flat.items())[:_MAX_PARAMS]}
            )

    # ── Output frame stats ────────────────────────────────
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(output, pd.DataFrame):
        frames["output"] = output
    elif isinstance(output, dict):
        frames = {str(k): v for k, v in output.items()}

    stats, manifests, file_size_maps = _frame_stats(
        frames,
        positive_label=positive_label,
        path_column=path_column or "",
        label_column=label_column or "",
    )
    if stats:
        mlflow.log_params(
            {
                k: (v if isinstance(v, (int, float)) else str(v))
                for k, v in stats.items()
            }
        )

    # ── Tags ──────────────────────────────────────────────
    tags: dict[str, str] = {
        "output_name": name,
        "provenance_logged": "true",
    }
    if version:
        tags["output_version"] = version
    if input_list:
        tags["input_uris"] = json.dumps(input_list)
    mlflow.set_tags(tags)

    # ── PROV document + artifacts ─────────────────────────
    prov_doc = build_run_prov(
        run_id,
        name=name,
        version=version,
        activity_name=str(run_name),
        stats=stats,
        inputs=input_list,
        extra_meta=dict(extra or {}),
        frames=list(frames.keys()) if frames else None,
        prov_prefixes=None,
    )

    def _log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        if logger is not None:
            logger.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        prov_dir = Path(tmpdir) / "provenance"
        prov_dir.mkdir(exist_ok=True)

        prov_path = prov_dir / "prov.json"
        prov_path.write_text(json.dumps(prov_doc, indent=2), encoding="utf-8")
        _log_artifact(str(prov_path), artifact_path="provenance")

        for frame_name, rows in manifests.items():
            manifest_csv = prov_dir / f"{frame_name}_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_csv, index=False)
            _log_artifact(str(manifest_csv), artifact_path="provenance")

    return {
        "prov_doc": prov_doc,
        "name": name,
        "version": version,
        "inputs": input_list,
        "stats": stats,
        "file_sizes": file_size_maps,
    }
