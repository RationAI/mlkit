"""Dataset registration, verification, and legacy CSV-based paths."""

from __future__ import annotations

import json
import os
import shutil
import uuid

import mlflow
import pandas as pd


# ── Internal helpers ────────────────────────────────────────────────────

def _lookup_experiment(name):
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
    if runs_df.empty:
        return None

    return runs_df.iloc[0]["run_id"]


def _detect_manifest() -> tuple[str | None, str | None]:
    """Walk data/ looking for manifest.csv.

    Returns (manifest_path, data_root) or (None, None).
    """
    for root_dir in ("data", "test_data", "."):
        for dirpath, _, filenames in os.walk(root_dir):
            if "manifest.csv" in filenames:
                return (
                    os.path.join(dirpath, "manifest.csv"),
                    os.path.dirname(os.path.abspath(
                        os.path.join(dirpath, "manifest.csv")
                    )),
                )
    return None, None


def load_manifest(manifest_path: str, data_root: str) -> list[dict]:
    """Load a manifest.csv and resolve WSI paths.

    Returns a list of dicts with keys ``path`` (absolute) and ``label``.

    Shared by ``register_dataset``, ``_verify_dataset``, and
    ``ProvenanceCallback`` to avoid duplicating the CSV iteration pattern.
    """
    df = pd.read_csv(manifest_path)
    samples: list[dict] = []
    for _, row in df.iterrows():
        rel = row["wsi_path"]
        full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
        samples.append({"path": full, "label": int(row["cancer"])})
    return samples


def verify_dataset(
    manifest_path: str | None = None,
    data_root: str | None = None,
) -> dict:
    """Public entry point — verify the current dataset against MLflow.

    Auto-detects the manifest if *manifest_path* is not given.

    Returns a dict with keys::

        {"verified": bool, "file_sizes_match": bool|None,
         "files_missing": int, "files_total": int, "details": list[str]}
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
) -> dict:
    """Verify the current dataset against the registered version in MLflow.

    Checks:
      1. Per-file sizes match (file-level integrity)
      2. All WSI files exist on disk

    Returns a dict with verification results.
    """
    result: dict = {
        "verified": False,
        "dataset_run_id": dataset_run_id,
        "file_sizes_match": None,
        "files_missing": 0,
        "files_total": 0,
        "details": [],
    }

    if not dataset_run_id:
        result["details"].append("No Dataset_Registry run found — skipping verification")
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
    curr_file_sizes = {}
    for s in samples:
        basename = os.path.basename(s["path"])
        if os.path.isfile(s["path"]):
            curr_file_sizes[basename] = os.stat(s["path"]).st_size
        else:
            curr_file_sizes[basename] = -1  # missing

    # Compare sizes
    result["file_sizes_match"] = curr_file_sizes == reg_file_sizes

    if not result["file_sizes_match"]:
        mismatched = [
            name for name in reg_file_sizes
            if curr_file_sizes.get(name) != reg_file_sizes[name]
        ]
        if mismatched:
            result["details"].append(
                f"File size mismatch on {len(mismatched)} file(s): "
                + ", ".join(sorted(mismatched)[:5])
                + ("…" if len(mismatched) > 5 else "")
            )

    # Check file existence
    missing = sum(1 for s in samples if not os.path.isfile(s["path"]))
    result["files_total"] = len(samples)
    result["files_missing"] = missing

    if missing > 0:
        result["details"].append(f"{missing}/{len(samples)} WSI files missing on disk")

    # Overall verdict
    result["verified"] = result["file_sizes_match"] and missing == 0

    if result["verified"]:
        result["details"].append("✅ Dataset verified — matches registered version")
    else:
        result["details"].append("❌ Dataset verification FAILED")

    return result


# ── Hash-based registration (preferred) ──────────────────────────────────

def register_dataset(
    dataset_dir: str,
    dataset_name: str | None = None,
    version: str = "1.0.0",
    experiment_name: str = "Dataset_Registry",
):
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
            "Dataset registration requires a manifest.csv file."
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

    # Register in MLflow
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=f"Dataset_{dataset_name}_{version}")
    run_id = run.info.run_id

    mlflow.log_params({
        "dataset_root": dataset_dir,
        "num_samples": len(samples),
        "num_positive": sum(1 for s in samples if s["label"] == 1),
        "num_negative": sum(1 for s in samples if s["label"] == 0),
    })

    mlflow.set_tags({
        "dataset_name": dataset_name,
        "version": version,
        "file_sizes": json.dumps(file_sizes),
        "file_mtimes": json.dumps(file_mtimes),
    })

    # ── PROV-O document (W3C PROV-O compatible) ────────────
    from rationai.mlkit.provenance.prov import build_dataset_prov  # noqa: PLC0415

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
    with open(prov_path, "w") as f:
        json.dump(prov_doc, f, indent=2)
    mlflow.log_artifact(prov_path, artifact_path="provenance")
    shutil.rmtree(prov_dir, ignore_errors=True)

    # ── Legacy dataset provenance JSON (kept for backward compat) ──
    legacy_prov_dir = f"_dataset_legacy_{uuid.uuid4().hex[:8]}"
    os.makedirs(legacy_prov_dir, exist_ok=True)
    legacy_path = os.path.join(legacy_prov_dir, "dataset_provenance.json")
    with open(legacy_path, "w") as f:
        json.dump({
            "dataset_name": dataset_name,
            "version": version,
            "dataset_root": dataset_dir,
            "file_sizes": file_sizes,
            "file_mtimes": file_mtimes,
            "num_samples": len(samples),
        }, f, indent=2)
    mlflow.log_artifact(legacy_path, artifact_path="provenance")
    shutil.rmtree(legacy_prov_dir, ignore_errors=True)

    mlflow.end_run()
    print(f"  [register_dataset] {dataset_name} v{version} → run_id={run_id}")
    return run_id



