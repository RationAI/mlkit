from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import concatenate_datasets, load_dataset
from mlflow.artifacts import download_artifacts


class DatasetLoader:
    """Loads one or more parquet-backed HF datasets from local paths or MLflow artifact URIs.

    The resulting dataset is always a `DatasetDict` keyed by split name. If the
    source data has no split structure, everything is placed under ``"train"``.
    Multiple sources (paths and/or URIs) are concatenated split-by-split.
    """

    dataset: HFDatasetDict

    def __init__(
        self,
        *,
        paths: Iterable[Path | str] | None = None,
        uris: Iterable[str] | None = None,
        dataset: HFDatasetDict | None = None,
        splits: Iterable[str] | None = None,
        hf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize DatasetLoader.

        Args:
            paths: Local filesystem paths to search for parquet files.
            uris: MLflow artifact URIs to download and load.
            dataset: An already-loaded ``DatasetDict`` to merge with the above.
            splits: If given, only these split names are loaded. For URIs this
                also limits what gets downloaded. ``None`` loads all splits.
            hf_kwargs: Extra kwargs forwarded to ``datasets.load_dataset``.
                Defaults to ``{"path": "parquet"}``. Any ``"split"`` key is
                ignored — use the ``splits`` parameter instead.
        """
        if not (paths or uris or dataset):
            raise ValueError(
                "At least one of paths, uris or dataset must be provided."
            )

        if hf_kwargs is None:
            hf_kwargs = {"path": "parquet"}

        _splits = frozenset(splits) if splits is not None else None
        datasets: list[HFDatasetDict] = [dataset] if dataset is not None else []

        if paths or uris:
            datasets.extend(self.load_dataset(paths or [], uris or [], hf_kwargs, _splits))

        self.dataset = self.concatenate_datasets(datasets)

    @staticmethod
    def concatenate_datasets(datasets: Iterable[HFDatasetDict]) -> HFDatasetDict:
        """Merge a sequence of DatasetDicts split-by-split.

        Splits present in only a subset of the dicts are still included —
        they are concatenated from whichever dicts contain them.
        """
        all_splits = set().union(*[dd.keys() for dd in datasets])
        return HFDatasetDict({
            split: concatenate_datasets([dd[split] for dd in datasets if split in dd])
            for split in all_splits
        })

    @staticmethod
    def _download_uri(uri: str, splits: frozenset[str] | None) -> list[Path]:
        """Download an MLflow artifact URI, returning local paths.

        When ``splits`` is specified only the per-split subdirectories are
        downloaded (``{uri}/{split}``), which avoids pulling unused data.
        Returns a list because a single URI may expand to multiple local paths
        (one per split).
        """
        if splits is None:
            return [Path(download_artifacts(artifact_uri=uri))]

        with ThreadPoolExecutor() as executor:
            return [
                Path(p)
                for p in executor.map(
                    lambda split: download_artifacts(artifact_uri=str(Path(uri, split))),
                    splits,
                )
            ]

    @staticmethod
    def load_dataset(
        paths: Iterable[str | Path],
        uris: Iterable[str],
        hf_kwargs: dict[str, Any],
        splits: frozenset[str] | None = None,
    ) -> list[HFDatasetDict]:
        """Load parquet files from local paths and MLflow URIs into DatasetDicts.

        Each source (path or downloaded URI) is loaded independently and
        returned as a separate ``DatasetDict``. The caller is responsible for
        merging them (see ``concatenate_datasets``).

        A bare ``Dataset`` (no split structure in the files) is normalised to
        ``DatasetDict({"train": ds})`` so the return type is always uniform.
        """
        with ThreadPoolExecutor() as executor:
            # _download_uri returns list[Path] per URI; flatten into one sequence.
            artifacts_paths = list(
                chain.from_iterable(
                    executor.map(lambda uri: DatasetLoader._download_uri(uri, splits), uris)
                )
            )

        resolved_paths = [Path(p) for p in (*paths, *artifacts_paths)]

        if not resolved_paths:
            return []

        try:
            loaded = [
                load_dataset(
                    **{k: v for k, v in hf_kwargs.items() if k != "split"},
                    **({"data_dir": str(p)} if p.is_dir() else {"data_files": str(p)}),
                )
                for p in resolved_paths
            ]
            return [
                HFDatasetDict({"train": ds}) if isinstance(ds, HFDataset) else ds
                for ds in loaded
            ]
        except Exception as e:
            msg = "Failed to load Parquet files."
            raise RuntimeError(msg) from e
