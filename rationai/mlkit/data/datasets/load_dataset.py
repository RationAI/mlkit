from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset as hf_load_dataset
from mlflow.artifacts import download_artifacts


def load_dataset[T: (HFDataset, HFDatasetDict)](
    *,
    paths: Iterable[Path | str] | None = None,
    uris: Iterable[str] | None = None,
    dataset: T | None = None,
    entities: Iterable[str] | None = None,
    hf_kwargs: dict[str, Any] | None = None,
) -> T:
    """Load parquet-backed HF datasets from local paths and/or MLflow artifact URIs.

    The result is always a ``DatasetDict`` keyed by split name. If the source
    data has no split structure, everything falls back to ``"train"``.
    Multiple sources are concatenated split-by-split.

    Args:
        paths: Local filesystem paths to search for parquet files.
        uris: MLflow artifact URIs to download and load.
        dataset: An already-loaded ``DatasetDict`` to merge with the above.
        entities: If given, only these entity names are loaded. For URIs this
            also limits what gets downloaded. ``None`` loads all entities.
        hf_kwargs: Extra kwargs forwarded to ``datasets.load_dataset``.
            Defaults to ``{"path": "parquet"}``. Any ``"split"`` key is
            ignored — use the ``splits`` parameter instead.
    """
    if not (paths or uris or dataset):
        raise ValueError("At least one of paths, uris or dataset must be provided.")

    if hf_kwargs is None:
        hf_kwargs = {"path": "parquet"}

    _entities = frozenset(entities) if entities is not None else None
    datasets: list[HFDatasetDict] = [dataset] if dataset is not None else []

    if paths or uris:
        datasets.extend(_load_sources(paths or [], uris or [], hf_kwargs, _entities))

    types = {type(ds) for ds in datasets}
    if len(types) > 1:
        raise TypeError(
            f"Cannot merge datasets of mixed types: {', '.join(t.__name__ for t in types)}"
        )

    if all(isinstance(ds, HFDataset) for ds in datasets):
        return concatenate_datasets(datasets)

    return _merge(datasets)


def _merge(datasets: Iterable[HFDatasetDict]) -> HFDatasetDict:
    """Merge a sequence of DatasetDicts split-by-split.

    Splits present in only a subset of the dicts are still included —
    they are concatenated from whichever dicts contain them.
    """
    datasets = list(datasets)
    all_splits = set().union(*[dd.keys() for dd in datasets])
    return HFDatasetDict({
        split: concatenate_datasets([dd[split] for dd in datasets if split in dd])
        for split in all_splits
    })


def _download_uri(uri: str, entities: frozenset[str] | None) -> list[Path]:
    """Download an MLflow artifact URI, returning local paths.

    When ``entities`` is specified only the per-entity subdirectories are
    downloaded (``{uri}/{entity}``), avoiding pulling unused data.
    Returns a list because a single URI may expand to multiple local paths
    (one per entity).
    """
    if entities is None:
        return [Path(download_artifacts(artifact_uri=uri))]

    with ThreadPoolExecutor() as executor:
        return [
            Path(p)
            for p in executor.map(
                lambda entity: download_artifacts(artifact_uri=str(Path(uri, entity))),
                entities,
            )
        ]


def _load_sources(
    paths: Iterable[str | Path],
    uris: Iterable[str],
    hf_kwargs: dict[str, Any],
    entities: frozenset[str] | None,
) -> list[HFDataset | HFDatasetDict]:
    """Load parquet files from local paths and MLflow URIs into DatasetDicts.

    Each source is loaded independently.
    """
    with ThreadPoolExecutor() as executor:
        artifacts_paths = list(
            chain.from_iterable(
                executor.map(lambda uri: _download_uri(uri, entities), uris)
            )
        )

    resolved = [Path(p) for p in (*paths, *artifacts_paths)]

    if not resolved:
        return []

    try:
        return [
            hf_load_dataset(
                **hf_kwargs,
                **({"data_dir": str(p)} if p.is_dir() else {"data_files": str(p)}),
            )
            for p in resolved
        ]
        
    except Exception as e:
        msg = "Failed to load Parquet files."
        raise RuntimeError(msg) from e
