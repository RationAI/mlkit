from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openslide import OpenSlide
from PIL import Image
from torch.utils.data import Dataset


class SlideTiles(Dataset[NDArray[np.uint8]]):
    def __init__(self, slide_labels: Any, tiles: pd.DataFrame) -> None:
        """Initialize SlideTiles dataset.

        Args:
            slide_labels: SlideLabels object.
            tiles: DataFrame with columns ["x", "y"].
        """
        super().__init__()
        self.slide = OpenSlide(slide_labels.path)
        self.slide_labels = slide_labels
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        tile = self.tiles.iloc[idx]
        rgba_region = self.slide.read_region(
            (tile["x"], tile["y"]),
            level=self.slide_labels.level,
            size=(self.slide_labels.tile_width, self.slide_labels.tile_height),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        return np.asarray(rgb_region)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        self.slide.close()
