from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import pyarrow as pa
from datasets import Dataset as HFDataset
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
        slides: HFDataset,
        tiles: HFDataset
    ) -> None:
        """Load slides and tiles from MLFlow artifacts.

        Args:
            slides: Dataset containing slide metadata.
            tiles: Dataset containing tile metadata.
        """
        self.slides = slides
        self.tiles = tiles

        self._slide_id_to_indices = self._build_tile_index(self.tiles)

        super().__init__(self.generate_datasets())

    def filter_tiles_by_slide(self, slide_id: str | bytes) -> HFDataset:
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
            tile_indices = self._slide_id_to_indices.get(
                slide_id, pa.scalar([], type=pa.list_(pa.int64()))
            )
            return self.tiles.select(tile_indices.values.to_numpy())


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
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                )
                for slide in self.slides
            )
            ```
        """

    @staticmethod
    def _build_tile_index(tiles: HFDataset) -> dict[str | bytes, pa.ListScalar]:
        """Creates a fast lookup table for slide indices.

        This function builds a mapping from `slide_id` to the list of indices in the
        `tiles` dataset that correspond to that slide.

        Args:
            tiles: A dataset containing a `slide_id` column.

        Returns:
            A dictionary mapping each `slide_id` to a list of indices in the `tiles` dataset.
        """
        if len(tiles) == 0:
            return {}

        slide_ids = tiles.data.column("slide_id")
        num_rows = len(slide_ids)

        # group_by requires the "large" variants for string/binary columns
        current_type = slide_ids.type
        if pa.types.is_string(current_type):
            slide_ids = slide_ids.cast(pa.large_string())
        elif pa.types.is_binary(current_type):
            slide_ids = slide_ids.cast(pa.large_binary())

        # np.arange is used here because PyArrow can wrap it with zero-copy overhead
        row_indices = pa.array(np.arange(num_rows, dtype=np.int64))
        table = pa.Table.from_arrays(
            [slide_ids, row_indices], names=["slide_id", "idx"]
        )

        # "list" aggregates all indices for a given slide_id into a single Arrow List scalar
        grouped = table.group_by("slide_id").aggregate([("idx", "list")])

        # Keep values as PyArrow ListScalars to avoid materializing them in Python
        keys = grouped.column("slide_id").to_numpy()
        values_array = grouped.column("idx_list")
        return {key: values_array[i] for i, key in enumerate(keys)}