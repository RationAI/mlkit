from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar, cast

import pyarrow.compute as pc
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from mlflow.artifacts import download_artifacts
from torch.utils.data import ConcatDataset, Dataset
import pyarrow as pa
import numpy as np

T = TypeVar("T", covariant=True)


class MetaTiledSlides(ConcatDataset[T], ABC):
    """Abstract base class for creating concatenated datasets from slides and tiles.

    This class provides a factory method to load and concatenate datasets from different
    sources: local storage, preloaded data, or artifacts stored in MLFlow.

    Attributes:
        slides (HFDataset): Dataset containing slide metadata.
        tiles (HFDataset): Dataset containing tile metadata.
    """

    def __init__(
        self,
        *,
        paths: Iterable[Path | str] | None = None,
        uris: Iterable[str] | None = None,
        slides_and_tiles: tuple[HFDataset, HFDataset] | None = None,
    ) -> None:
        """Load slides and tiles from MLFlow artifacts.

        Args:
            paths: List of directories to load slides and tiles from. Each
                directory must include two files: `slides.parquet` and tiles.parquet`.
            uris: List of MLFlow artifact URIs pointing to folders containing
                `slides.parquet` and `tiles.parquet`.
            slides_and_tiles: Tuple containing the slides and tiles Datasets.
        """
        assert paths or uris or slides_and_tiles, (
            "At least one of paths, uris or slides_and_tiles must be provided."
        )

        slides, tiles = self.load_slides_and_tiles(paths or [], uris or [])

        if slides_and_tiles is not None:
            slides = concatenate_datasets([slides, slides_and_tiles[0]])
            tiles = concatenate_datasets([tiles, slides_and_tiles[1]])

        self.slides = slides
        self.tiles = tiles
        self._slide_id_to_indices = self._build_tile_index(self.tiles)

        super().__init__(self.generate_datasets())

    @staticmethod
    def _build_tile_index(tiles: HFDataset) -> dict[str, range]:
        """Creates a fast lookup table for slide indices.

        This function builds a mapping from `slide_id` to the range of indices in the
        `tiles` dataset that correspond to that slide. It assumes that the `tiles` dataset
        is sorted by `slide_id`, which allows for efficient retrieval of tile indices
        for each slide without needing to scan the entire dataset for each slide.

        Args:
            tiles: A dataset containing a `slide_id` column, sorted by `slide_id`.

        Returns:
            A dictionary mapping each `slide_id` to a range of indices in the `tiles` dataset.
        """
        if len(tiles) == 0:
            return {}

        # 1. Grab the column directly from the underlying PyArrow Table
        slide_ids = tiles.data.column("slide_id")
        num_rows = len(slide_ids)

        # 2. Generate sequential row indices
        # np.arange is used here because PyArrow can wrap it instantly with zero-copy overhead
        row_indices = pa.array(np.arange(num_rows, dtype=np.int64))

        # 3. Combine them into a lightweight PyArrow Table
        table = pa.Table.from_arrays(
            [slide_ids, row_indices], 
            names=["slide_id", "idx"]
        )

        # 4. Perform the native Arrow groupby and aggregate
        # The "list" function aggregates all indices for a given slide_id into a single Arrow List scalar
        grouped = table.group_by("slide_id").aggregate([("idx", "list")])

        # 5. Extract to a Python dictionary
        # PyArrow automatically names the aggregated column "idx_list" (pattern: {column}_{agg_func})
        keys = grouped.column("slide_id").to_pylist()
        values = grouped.column("idx_list").to_pylist()

        return dict(zip(keys, values, strict=True))

    @abstractmethod
    def generate_datasets(self) -> Iterable[Dataset[T]]:
        """Factory method to generate datasets from slides and tiles.

        Example:
            ```python
            return (
                SlideTiles(
                    slide_path=slide["path"],
                    level=slide["level"],
                    tile_extent_x=slide["tile_extent_x"],
                    tile_extent_y=slide["tile_extent_y"],
                    tiles=self.filter_tiles_by_slide(slide.id),
                )
                for slide in self.slides
            )
            ```
        """

    def filter_tiles_by_slide(self, slide_id: str) -> HFDataset:
        """Returns a view of the dataset using a slice or indices.

        This function creates a view of the `self.tiles` dataset that contains only
        the tiles belonging to the specified slide. It uses the precomputed
        `_slide_id_to_indices` mapping to efficiently retrieve the relevant tiles
        without copying data.

        Args:
            slide_id: The ID of the slide to filter tiles.

        Returns:
            A view of the tiles dataset containing only the tiles for the specified slide.
        """
        tile_range = self._slide_id_to_indices.get(slide_id, range(0))
        return self.tiles.select(tile_range)

    @staticmethod
    def load_slides_and_tiles(
        paths: Iterable[str | Path], uris: Iterable[str]
    ) -> tuple[HFDataset, HFDataset]:
        """Load slides and tiles parquets from local storage and MLFlow artifacts.

        Args:
            paths: List of directories to load slides and tiles from. Each
                directory must include two files: `slides.parquet` and tiles.parquet`.
            uris: List of MLFlow artifact URIs pointing to folders containing
                `slides.parquet` and `tiles.parquet`.

        Raises:
            FileNotFoundError: If the data cannot be loaded from the specified URIs.

        Returns:
            A tuple containing the slides and tiles Datasets.
        """
        # Parallelize MLFlow downloads (I/O Bound)
        with ThreadPoolExecutor() as executor:
            artifacts_paths = list(
                executor.map(lambda uri: download_artifacts(artifact_uri=uri), uris)
            )

        search_dirs = [Path(p) for p in (*paths, *artifacts_paths)]

        # Extract existing file paths
        slide_files = [
            str(s) for p in search_dirs if (s := p / "slides.parquet").exists()
        ]
        tile_files = [
            str(t) for p in search_dirs if (t := p / "tiles.parquet").exists()
        ]

        # Handle empty datasets
        if not (slide_files and tile_files):
            return HFDataset.from_dict({}), HFDataset.from_dict({})

        try:
            # Load datasets with memory mapping (lazy)
            loader_kwargs = {"path": "parquet", "split": "train"}

            slides_ds = load_dataset(**loader_kwargs, data_files=slide_files)  # pyright: ignore[reportArgumentType, reportCallIssue]
            tiles_ds = load_dataset(**loader_kwargs, data_files=tile_files)  # pyright: ignore[reportArgumentType, reportCallIssue]

            return cast("HFDataset", slides_ds), cast("HFDataset", tiles_ds)

        except Exception as e:
            msg = f"Failed to load Parquet files. Found {len(slide_files)} slides and {len(tile_files)} tiles."
            raise FileNotFoundError(msg) from e
