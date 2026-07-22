"""Dataset registration, verification, and PROV-O document generation.

Colocates ``build_dataset_prov`` (the PROV document builder) with the
MLflow registration entry points ``register_dataset`` and ``verify_dataset``.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any

import mlflow
import pandas as pd

from rationai.mlkit.provenance.common import (
    _iso_timestamp,
    _qualified,
    _qualified_name,
    _safe_id,
    _typed_value,
    get_prov_prefixes,
)


# ──────────────────────────────────────────────
# PROV document builder
# ──────────────────────────────────────────────


def build_dataset_prov(
    run_id: str,
    dataset_name: str,
    version: str,
    dataset_root: str,
    num_samples: int,
    num_positive: int,
    num_negative: int,
    file_sizes: dict[str, int],
    manifest_path: str | None = None,
    prov_prefixes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a PROV document for a dataset registration run.

    Produces a ``dataset`` entity (``sosa:Sample``) linked to the
    registration activity and metadata bundle.
    """
    prefixes = prov_prefixes or get_prov_prefixes()

    run_act_local = _safe_id(f"run_{run_id}")
    run_act_id = _qualified("gen", run_act_local)

    ds_local = _safe_id(f"dataset_{dataset_name}_{version.replace('.', '_')}")
    ds_id = _qualified("gen", ds_local)

    meta_local = run_id
    meta_id = _qualified("meta", meta_local)

    main_act_local = f"DatasetReg_{run_id[:8]}"
    main_act_id = _qualified("blank", main_act_local)

    entities: dict[str, dict[str, Any]] = {}
    activities: dict[str, dict[str, Any]] = {}
    used: dict[str, dict[str, Any]] = {}
    was_generated_by: dict[str, dict[str, Any]] = {}

    rel_counter = [0]

    def _blank_rel_id() -> str:
        rid = f"_:n{rel_counter[0]}"
        rel_counter[0] += 1
        return rid

    now = _iso_timestamp()

    # ── DATASET ENTITY ───────────────────────────────────
    ds_props: dict[str, list[Any]] = {
        "schema:name": _typed_value(dataset_name),
        "prov:type": [_qualified_name("sosa", "Sample")],
        "dct:description": _typed_value(
            f"Dataset {dataset_name} v{version} ({num_samples} samples)",
        ),
    }
    if manifest_path:
        ds_props["schema:url"] = _typed_value(manifest_path)
    entities[ds_id] = ds_props

    # ── ACTIVITY (the registration action) ────────────────
    run_activity: dict[str, Any] = {}
    run_activity["prov:type"] = [_qualified_name("schema", "Action")]
    run_activity["prov:startTime"] = [now]
    run_activity["prov:endTime"] = [now]
    run_activity["schema:name"] = _typed_value(f"Register dataset {dataset_name}")
    run_activity["gen:dataset_name"] = _typed_value(dataset_name)
    run_activity["gen:dataset_version"] = _typed_value(version)
    activities[run_act_id] = run_activity

    # ── USED (activity consumed the dataset entity) ──────
    used[_blank_rel_id()] = {
        "prov:activity": run_act_id,
        "prov:entity": ds_id,
    }

    # ── CPM METADATA ENTITY ───────────────────────────────
    meta_entity: dict[str, list[Any]] = {}
    meta_entity["prov:type"] = [_qualified_name("cpm", "BundleMetadata")]
    meta_entity["gen:dataset_name"] = _typed_value(dataset_name)
    meta_entity["gen:dataset_version"] = _typed_value(version)
    meta_entity["gen:dataset_root"] = _typed_value(dataset_root)
    meta_entity["gen:num_samples"] = _typed_value(str(num_samples))
    meta_entity["gen:num_positive"] = _typed_value(str(num_positive))
    meta_entity["gen:num_negative"] = _typed_value(str(num_negative))
    if manifest_path:
        meta_entity["gen:manifest_path"] = _typed_value(manifest_path)

    file_sizes_str = json.dumps(file_sizes)
    meta_entity["gen:file_sizes"] = [file_sizes_str]
    entities[meta_id] = meta_entity

    # ── CPM MAIN ACTIVITY ────────────────────────────────
    main_activity: dict[str, Any] = {}
    main_activity["prov:type"] = [_qualified_name("cpm", "mainActivity")]
    main_activity["cpm:referencedMetaBundleId"] = [
        {"type": "prov:QUALIFIED_NAME", "$": meta_id},
    ]
    main_activity["dct:hasPart"] = [
        {"type": "prov:QUALIFIED_NAME", "$": run_act_id},
    ]
    activities[main_act_id] = main_activity

    # ── RELATIONSHIPS ─────────────────────────────────────
    was_generated_by[_blank_rel_id()] = {
        "prov:entity": meta_id,
        "prov:activity": run_act_id,
    }

    # ── ASSEMBLE BUNDLE ───────────────────────────────────
    inner: dict[str, Any] = {"prefix": prefixes}
    if entities:
        inner["entity"] = entities
    if activities:
        inner["activity"] = activities
    if used:
        inner["used"] = used
    if was_generated_by:
        inner["wasGeneratedBy"] = was_generated_by

    bundle_key = f"storage:{run_id}"
    return {"bundle": {bundle_key: inner}}


# ──────────────────────────────────────────────
# MLflow registration & verification helpers
# ──────────────────────────────────────────────


def _lookup_experiment(name: str) -> str | None:
    """Return the MLflow experiment ID for *name*, or None."""
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else None


def _lookup_dataset_run() -> str | None:
    """Return the latest Dataset_Registry run ID.

    Falls back to the most recent run so that workflows with only one
    registered dataset still work.
    """
    exp_id = _lookup_experiment("Dataset_Registry")
    if exp_id is None:
        return None

    runs_df = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"],
    )
    if pd.DataFrame(runs_df).empty:
        return None

    return pd.DataFrame(runs_df).iloc[0]["run_id"]


def _detect_manifest() -> tuple[str | None, str | None]:
    """Walk data/ looking for manifest.csv.

    Returns (manifest_path, data_root) or (None, None).
    """
    for root_dir in ("data", "test_data", "."):
        for dirpath, _, filenames in os.walk(root_dir):
            if "manifest.csv" in filenames:
                return (
                    os.path.join(dirpath, "manifest.csv"),
                    os.path.dirname(
                        os.path.abspath(
                            os.path.join(dirpath, "manifest.csv"),
                        )
                    ),
                )
    return None, None


# ──────────────────────────────────────────────
# Manifest loading (shared with ProvenanceCallback)
# ──────────────────────────────────────────────


def load_manifest(manifest_path: str, data_root: str) -> list[dict[str, Any]]:
    """Load a manifest.csv and resolve WSI paths.

    Returns a list of dicts with keys ``path`` (absolute) and ``label``.

    Shared by ``register_dataset``, ``verify_dataset``, and
    ``ProvenanceCallback`` to avoid duplicating the CSV iteration pattern.
    """
    df = pd.read_csv(manifest_path)
    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rel = row["wsi_path"]
        full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
        samples.append({"path": full, "label": int(row["cancer"])})
    return samples


# ──────────────────────────────────────────────
# Dataset verification
# ──────────────────────────────────────────────


def verify_dataset(
    manifest_path: str | None = None,
    data_root: str | None = None,
) -> dict[str, Any]:
    """Public entry point — verify the current dataset against MLflow.

    Auto-detects the manifest if *manifest_path* is not given.

    Returns a dict with keys::

        {
            "verified": bool,
            "file_sizes_match": bool | None,
            "files_missing": int,
            "files_total": int,
            "details": list[str],
        }
    """
    if manifest_path is None:
        manifest_path, data_root = _detect_manifest()

    if manifest_path is None:
        return {
            "verified": False,
            "dataset_run_id": None,
            "file_sizes_match": None,
            "files_missing": 0,
            "files_total": 0,
            "details": ["No manifest.csv found — skipping verification"],
        }

    if data_root is None:
        data_root = os.path.dirname(os.path.abspath(manifest_path))

    dataset_run_id = _lookup_dataset_run()
    return _verify_dataset(manifest_path, data_root, dataset_run_id)


def _verify_dataset(
    manifest_path: str,
    data_root: str,
    dataset_run_id: str | None,
) -> dict[str, Any]:
    """Verify the current dataset against the registered version in MLflow.

    Checks:
      1. Per-file sizes match (file-level integrity)
      2. All WSI files exist on disk

    Returns a dict with verification results.
    """
    result: dict[str, Any] = {
        "verified": False,
        "dataset_run_id": dataset_run_id,
        "file_sizes_match": None,
        "files_missing": 0,
        "files_total": 0,
        "details": [],
    }

    if not dataset_run_id:
        result["details"].append(
            "No Dataset_Registry run found — skipping verification"
        )
        return result

    # Fetch registered metadata
    try:
        reg_run = mlflow.get_run(dataset_run_id)
        reg_tags = reg_run.data.tags
        reg_file_sizes_str = reg_tags.get("file_sizes", "")
        reg_file_sizes = json.loads(reg_file_sizes_str) if reg_file_sizes_str else {}
    except Exception as e:
        result["details"].append(f"Failed to fetch Dataset_Registry run: {e}")
        return result

    samples = load_manifest(manifest_path, data_root)
    curr_file_sizes: dict[str, int] = {}
    for s in samples:
        basename = os.path.basename(s["path"])
        if os.path.isfile(s["path"]):
            curr_file_sizes[basename] = os.stat(s["path"]).st_size
        else:
            curr_file_sizes[basename] = -1  # missing

    manifest_match = set(curr_file_sizes) == set(reg_file_sizes)
    if not manifest_match:
        result["details"].append(
            f"File manifest mismatch: expected {len(reg_file_sizes)} file(s), found {len(curr_file_sizes)} file(s)"
        )
    sizes_match = all(
        curr_file_sizes.get(k) == reg_file_sizes[k] for k in reg_file_sizes
    )
    result["file_sizes_match"] = manifest_match and sizes_match

    if not result["file_sizes_match"]:
        mismatched = [
            name
            for name in reg_file_sizes
            if curr_file_sizes.get(name) != reg_file_sizes[name]
        ]
        if mismatched:
            result["details"].append(
                f"File size mismatch on {len(mismatched)} file(s): "
                + ", ".join(sorted(mismatched)[:5])
                + ("…" if len(mismatched) > 5 else ""),
            )

    missing = sum(1 for s in samples if not os.path.isfile(s["path"]))
    result["files_total"] = len(samples)
    result["files_missing"] = missing

    if missing > 0:
        result["details"].append(f"{missing}/{len(samples)} WSI files missing on disk")

    result["verified"] = result["file_sizes_match"] and missing == 0

    if result["verified"]:
        result["details"].append("✅ Dataset verified — matches registered version")
    else:
        result["details"].append("❌ Dataset verification FAILED")

    return result


# ──────────────────────────────────────────────
# Hash-based registration (preferred)
# ──────────────────────────────────────────────


def register_dataset(
    dataset_dir: str,
    dataset_name: str | None = None,
    version: str = "1.0.0",
    experiment_name: str = "Dataset_Registry",
) -> str:
    """Register a dataset in MLflow's Dataset_Registry experiment.

    Captures per-file metadata (size, last modified) and stores it as tags
    so future training runs can inspect the exact state of the data.

    Args:
        dataset_dir: Path to the dataset root (must contain manifest.csv).
        dataset_name: Human-readable name (defaults to directory basename).
        version: Dataset version string.
        experiment_name: MLflow experiment for registration.

    Returns:
        The run_id of the registration run.

    Example:
        from rationai.mlkit.provenance import register_dataset

        run_id = register_dataset("data/pato_cohort_01", version="2.0")
        print(f"Registered as {run_id}")
    """
    dataset_dir = os.path.abspath(dataset_dir)
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No manifest.csv found in {dataset_dir}. "
            "Dataset registration requires a manifest.csv file.",
        )

    if dataset_name is None:
        dataset_name = os.path.basename(dataset_dir)

    samples = load_manifest(manifest_path, dataset_dir)
    file_sizes = {}
    file_mtimes = {}

    for s in samples:
        basename = os.path.basename(s["path"])
        if os.path.isfile(s["path"]):
            st = os.stat(s["path"])
            file_sizes[basename] = st.st_size
            file_mtimes[basename] = st.st_mtime
        else:
            file_sizes[basename] = -1
            file_mtimes[basename] = -1

    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=f"Dataset_{dataset_name}_{version}")
    run_id = run.info.run_id
    run_active = True

    try:
        mlflow.log_params(
            {
                "dataset_root": dataset_dir,
                "num_samples": len(samples),
                "num_positive": sum(1 for s in samples if s["label"] == 1),
                "num_negative": sum(1 for s in samples if s["label"] == 0),
            }
        )

        mlflow.set_tags(
            {
                "dataset_name": dataset_name,
                "version": version,
                "file_sizes": json.dumps(file_sizes),
                "file_mtimes": json.dumps(file_mtimes),
            }
        )

        # ── PROV-O document ────────────────────────────────
        prov_doc = build_dataset_prov(
            run_id=run_id,
            dataset_name=dataset_name,
            version=version,
            dataset_root=dataset_dir,
            num_samples=len(samples),
            num_positive=sum(1 for s in samples if s["label"] == 1),
            num_negative=sum(1 for s in samples if s["label"] == 0),
            file_sizes=file_sizes,
            manifest_path=manifest_path,
        )

        prov_dir = f"_dataset_prov_{uuid.uuid4().hex[:8]}"
        os.makedirs(prov_dir, exist_ok=True)
        prov_path = os.path.join(prov_dir, "prov.json")
        try:
            with open(prov_path, "w") as f:
                json.dump(prov_doc, f, indent=2)
            mlflow.log_artifact(prov_path, artifact_path="provenance")
        finally:
            shutil.rmtree(prov_dir, ignore_errors=True)

        # ── Legacy dataset provenance JSON (backward compat) ──
        legacy_prov_dir = f"_dataset_legacy_{uuid.uuid4().hex[:8]}"
        os.makedirs(legacy_prov_dir, exist_ok=True)
        legacy_path = os.path.join(legacy_prov_dir, "dataset_provenance.json")
        try:
            with open(legacy_path, "w") as f:
                json.dump(
                    {
                        "dataset_name": dataset_name,
                        "version": version,
                        "dataset_root": dataset_dir,
                        "file_sizes": file_sizes,
                        "file_mtimes": file_mtimes,
                        "num_samples": len(samples),
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact(legacy_path, artifact_path="provenance")
        finally:
            shutil.rmtree(legacy_prov_dir, ignore_errors=True)
    finally:
        if run_active:
            mlflow.end_run()

    print(f"  [register_dataset] {dataset_name} v{version} → run_id={run_id}")
    return run_id
