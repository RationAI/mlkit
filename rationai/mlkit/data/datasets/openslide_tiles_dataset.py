from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset
from numpy.typing import NDArray
from ratiopath.openslide import OpenSlide
from torch.utils.data import Dataset


class OpenSlideTilesDataset(Dataset[NDArray[np.uint8]]):
    """Dataset for reading tiles from a single slide image.

    This dataset reads tiles from an OpenSlide image. The tiles are specified by a
    table with columns ["x", "y"]. The RGBA tiles are converted to RGB before
    being returned.

    Attributes:
        slide (Path): Path to the slide image.
        level (int | str): Level of the slide to read. If int, it is used as the level.
            If str, it is used as the column name in the tiles Dataset.
        tile_extent_x (int | str): Width of the tile. If int, it is used as the width.
            If str, it is used as the column name in the tiles Dataset.
        tile_extent_y (int | str): Height of the tile. If int, it is used as the height.
            If str, it is used as the column name in the tiles Dataset.
        tiles (HFDataset): Lazy Dataset with columns ["x", "y"] specifying the tiles
            to be read.
    """

    def __init__(
        self,
        slide_path: str | Path,
        level: int | str,
        tile_extent_x: int | str,
        tile_extent_y: int | str,
        tiles: HFDataset,
    ) -> None:
        """Initialize OpenSlideTilesDataset dataset.

        Args:
            slide_path: Path to the slide image.
            level: Level of the slide to read. If int, it is used as the level. If str,
                it is used as the column name in the tiles Dataset.
            tile_extent_x: Width of the tile. If int, it is used as the width. If str, it
                is used as the column name in the tiles Dataset.
            tile_extent_y: Height of the tile. If int, it is used as the height. If str,
                it is used as the column name in the tiles Dataset.
            tiles: Dataset with columns ["x", "y"].
        """
        super().__init__()
        self.slide_path = Path(slide_path)
        self.level = level
        self.tile_extent_x = tile_extent_x
        self.tile_extent_y = tile_extent_y
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Returns tile from the slide image at the specified index in RGB format."""
        tile = self.tiles[idx]
        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)

        with OpenSlide(self.slide_path) as slide:
            return slide.read_tile(tile["x"], tile["y"], extent_x, extent_y, level)

    @staticmethod
    def _get_from_tile(tile: dict[str, Any], key: int | str) -> int:
        return tile[key] if isinstance(key, str) else key
