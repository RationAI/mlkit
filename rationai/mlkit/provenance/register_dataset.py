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


def _lookup_dataset_run(manifest_path: str | None = None) -> str | None:
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

    # Read manifest and check current file sizes
    df = pd.read_csv(manifest_path)
    samples = []
    curr_file_sizes = {}
    for _, row in df.iterrows():
        rel = row["wsi_path"]
        full = os.path.join(data_root, rel) if not os.path.isabs(rel) else rel
        samples.append({"path": full, "label": int(row["cancer"])})
        basename = os.path.basename(full)
        if os.path.isfile(full):
            curr_file_sizes[basename] = os.stat(full).st_size
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

    # Read manifest and collect per-file metadata (size, last_modified)
    df = pd.read_csv(manifest_path)
    samples = []
    file_sizes = {}
    file_mtimes = {}

    for _, row in df.iterrows():
        rel = row["wsi_path"]
        full = os.path.join(dataset_dir, rel) if not os.path.isabs(rel) else rel
        samples.append({"path": full, "label": int(row["cancer"])})

        basename = os.path.basename(full)
        if os.path.isfile(full):
            st = os.stat(full)
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

    # Save provenance JSON as an artifact for offline verification
    prov_dir = f"_dataset_prov_{uuid.uuid4().hex[:8]}"
    os.makedirs(prov_dir, exist_ok=True)
    prov_path = os.path.join(prov_dir, "dataset_provenance.json")
    with open(prov_path, "w") as f:
        json.dump({
            "dataset_name": dataset_name,
            "version": version,
            "dataset_root": dataset_dir,
            "file_sizes": file_sizes,
            "file_mtimes": file_mtimes,
            "num_samples": len(samples),
        }, f, indent=2)
    mlflow.log_artifact(prov_path, artifact_path="provenance")
    shutil.rmtree(prov_dir, ignore_errors=True)

    mlflow.end_run()
    print(f"  [register_dataset] {dataset_name} v{version} → run_id={run_id}")
    return run_id


# ── Legacy CSV-based registration (backward compat) ─────────────────────

def register_dataset_as_provenance(manifest_path, dataset_root, dataset_name, version):
    """Legacy CSV-based dataset registration.

    Stores an enriched manifest as an artifact with a ``manifest_uri`` tag.
    Does NOT set ``manifest_hash`` / ``samples_hash`` tags — use
    :func:`register_dataset` instead for hash-based verification support.
    """
    mlflow.set_experiment("Dataset_Registry")

    with mlflow.start_run(run_name=f"Dataset_{dataset_name}_v{version}") as run:
        # 1. Indexace v MLflow (pro rychlé hledání/filtrování)
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("version", version)

        # 2. Metadata pro tracking
        mlflow.log_param("dataset_root", dataset_root)

        # 3. Zpracování manifestu a obohacení o metadata (size, mtime)
        df = pd.read_csv(manifest_path)
        metadata_list = []

        for path in df["wsi_path"]:
            full_path = os.path.join(dataset_root, path) if not os.path.isabs(path) else path
            if os.path.exists(full_path):
                stat = os.stat(full_path)
                metadata_list.append({"file_size": stat.st_size, "last_modified": stat.st_mtime})
            else:
                metadata_list.append({"file_size": -1, "last_modified": -1})

        df_enriched = pd.concat([df, pd.DataFrame(metadata_list)], axis=1)

        # 4. Uložení artefaktu (Zlatý zdroj pravdy)
        provenance_file = "dataset_provenance.csv"
        df_enriched.to_csv(provenance_file, index=False)
        mlflow.log_artifact(provenance_file, artifact_path="provenance")

        # 5. Uložení odkazu do tagu (velmi důležité pro automatizaci!)
        mlflow.set_tag("manifest_uri", f"runs:/{run.info.run_id}/provenance/{provenance_file}")

        print(f"Dataset '{dataset_name}' (v{version}) úspěšně zaregistrován.")
        print(f"Run ID: {run.info.run_id}")

        os.remove(provenance_file)


if __name__ == "__main__":
    # Příklad použití pro tvůj dataset
    register_dataset_as_provenance(
        manifest_path="data/dummy_dataset_1/manifest.csv",
        dataset_root="data/dummy_dataset_1",
        dataset_name="pato_cohort_01",
        version="1.0.0"
    )
