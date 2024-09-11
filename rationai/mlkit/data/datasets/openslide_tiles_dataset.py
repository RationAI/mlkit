from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openslide import OpenSlide
from PIL import Image
from torch.utils.data import Dataset


class OpenSlideTilesDataset(Dataset[NDArray[np.uint8]]):
    """Dataset for reading tiles from a single slide image.

    This dataset reads tiles from an OpenSlide image. The tiles are specified by a
    DataFrame with columns ["x", "y"]. The RGBA tiles are converted to RGB before
    being returned.

    Attributes:
        slide (str | Path): Path to the slide image.
        level (int | str): Level of the slide to read. If int, it is used as the level.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_x (int | str): Width of the tile. If int, it is used as the width.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_y (int | str): Height of the tile. If int, it is used as the height.
            If str, it is used as the column name in the tiles DataFrame.
        tiles (pd.DataFrame): DataFrame with columns ["x", "y"] specifying the tiles
            to be read.
    """

    def __init__(
        self,
        slide_path: str | Path,
        level: int | str,
        tile_extent_x: int | str,
        tile_extent_y: int | str,
        tiles: pd.DataFrame,
    ) -> None:
        """Initialize OpenSlideTilesDataset dataset.

        Args:
            slide_path: Path to the slide image.
            level: Level of the slide to read. If int, it is used as the level. If str,
                it is used as the column name in the tiles DataFrame.
            tile_extent_x: Width of the tile. If int, it is used as the width. If str, it
                is used as the column name in the tiles DataFrame.
            tile_extent_y: Height of the tile. If int, it is used as the height. If str,
                it is used as the column name in the tiles DataFrame.
            tiles: DataFrame with columns ["x", "y"].
        """
        super().__init__()
        self.slide_path = slide_path
        self.level = level
        self.tile_extent_x = tile_extent_x
        self.tile_extent_y = tile_extent_y
        self.tiles = tiles

        self._slide: OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Returns tile from the slide image at the specified index in RGB format."""
        if self._slide is None:
            self._slide = OpenSlide(self.slide_path)

        tile = self.tiles.iloc[idx]

        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)
        x = int(tile["x"] * self._slide.level_downsamples[level])
        y = int(tile["y"] * self._slide.level_downsamples[level])

        rgba_region = self._slide.read_region((x, y), level, (extent_x, extent_y))
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        return np.array(rgb_region)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the OpenSlide file handle."""
        if self._slide is not None:
            self._slide.close()

    def _get_from_tile(self, tile: pd.Series, key: int | str) -> int:
        return tile[key] if isinstance(key, str) else key
