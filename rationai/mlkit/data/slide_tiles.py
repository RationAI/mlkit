from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openslide import OpenSlide
from PIL import Image
from torch.utils.data import Dataset


class SlideTiles(Dataset[NDArray[np.uint8]]):
    def __init__(
        self,
        slide_path: str | Path,
        level: int,
        tile_width: int,
        tile_height: int,
        tiles: pd.DataFrame,
    ) -> None:
        """Initialize SlideTiles dataset.

        Args:
            slide_path: Path to the slide image.
            level: Level of the slide to read.
            tile_width: Width of the tile.
            tile_height: Height of the tile.
            tiles: DataFrame with columns ["x", "y", "width", "height", "level"].
        """
        super().__init__()
        self.slide_path = slide_path
        self.level = level
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tiles = tiles
        self.slide: OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        if self.slide is None:
            self.slide = OpenSlide(self.slide_path)

        tile = self.tiles.iloc[idx]

        x = int(tile["x"] * self.slide.level_downsamples[self.level])
        y = int(tile["y"] * self.slide.level_downsamples[self.level])
        rgba_region = self.slide.read_region(
            (x, y), level=self.level, size=(self.tile_width, self.tile_height)
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        return np.array(rgb_region)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self.slide is not None:
            self.slide.close()
