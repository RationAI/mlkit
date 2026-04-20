from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset, Dataset

from rationai.mlkit.data.datasets.slides_tiles_loader import SlidesTilesLoader


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
        hf_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Load slides and tiles from MLFlow artifacts.

        Args:
            paths: List of directories to load slides and tiles from. Each
                directory must include either single files (`slides.parquet`
                and `tiles.parquet`) or subdirectories (`slides/` and `tiles/`)
                containing chunked Parquet files.
            uris: List of MLFlow artifact URIs pointing to folders containing
                either single files (`slides.parquet` and `tiles.parquet`) or
                subdirectories (`slides/` and `tiles/`) containing chunked
                Parquet files.
            slides_and_tiles: Tuple containing the slides and tiles Datasets.
            hf_kwargs: Additional keyword arguments to pass to HuggingFace's
                `load_dataset` function. Defaults to `{"path": "parquet", "split": "train"}`.
        """
        self._meta = SlidesTilesLoader(
            paths=paths,
            uris=uris,
            slides_and_tiles=slides_and_tiles,
            hf_kwargs=hf_kwargs,
        )
        self.slides = self._meta.slides
        self.tiles = self._meta.tiles
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
        return self._meta.filter_tiles_by_slide(slide_id)

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
