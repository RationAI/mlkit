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
        slide (Path): Path to the slide image.
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
        self.slide_path = Path(slide_path)
        self.level = level
        self.tile_extent_x = tile_extent_x
        self.tile_extent_y = tile_extent_y
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Returns tile from the slide image at the specified index in RGB format."""
        with OpenSlide(self.slide_path) as slide:
            tile = self.tiles.iloc[idx]

            level = self._get_from_tile(tile, self.level)
            extent_x = self._get_from_tile(tile, self.tile_extent_x)
            extent_y = self._get_from_tile(tile, self.tile_extent_y)
            x = int(tile["x"] * slide.level_downsamples[level])
            y = int(tile["y"] * slide.level_downsamples[level])

            bg_tile = Image.new(mode="RGB", size=(extent_x, extent_y), color="#FFFFFF")
            rgba_region = slide.read_region((x, y), level, (extent_x, extent_y))

            # Paste the RGBA region onto the background tile
            # using the alpha channel as a mask to handle transparency.
            bg_tile.paste(im=rgba_region, mask=rgba_region, box=None)

            return np.array(bg_tile)

    def _get_from_tile(self, tile: pd.Series, key: int | str) -> int:
        return tile[key] if isinstance(key, str) else key
