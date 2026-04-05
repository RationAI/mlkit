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
        self.tiles = tiles.sort("slide_id")
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

        # Get the underlying Arrow table
        table = tiles.data.table
        slide_ids = table.column("slide_id")

        # FIX: Cast to large_string to prevent 32-bit offset overflow (2GB limit)
        # we use pc.cast because slide_ids is a ChunkedArray
        large_slide_ids = pc.cast(slide_ids, pa.large_string())

        # Now combine_chunks will work because it uses 64-bit offsets
        run_ends = pc.run_end_encode(large_slide_ids.combine_chunks())

        values = run_ends.values
        ends = run_ends.run_ends

        index_map = {}
        current_offset = 0

        for sid, end in zip(values, ends, strict=True):
            end_py = end.as_py()
            index_map[sid.as_py()] = range(current_offset, end_py)
            current_offset = end_py

        return index_map

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
