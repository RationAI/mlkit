import io

import numpy as np
import pandas as pd
import PIL
from empaia_client import EmpaiaSlide
from empaia_client.clients.synchronous import BaseEmpaiaClient
from numpy.typing import NDArray
from torch.utils.data import Dataset


class EmpaiaTilesDataset(Dataset[NDArray[np.uint8]]):
    """Dataset for reading tiles from a single slide image stored in Empaia.

    This dataset reads tiles of given slide from an Empaia API. The tiles are specified by a
    DataFrame with columns ["x", "y"].

    Attributes:
        case_id: Id of the case containing the slide.
        slide_id: Id of the slide.
        level (int | str): Level of the slide to read. If int, it is used as the level.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_x (int | str): Width of the tile. If int, it is used as the width.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_y (int | str): Height of the tile. If int, it is used as the height.
            If str, it is used as the column name in the tiles DataFrame.
        tiles (pd.DataFrame): DataFrame with columns ["x", "y"] specifying the tiles
            to be read.
        empaia_client (BaseEmpaiaClient): Instance of synchronous EmpaiaClient used to fetch slide images.
    """

    def __init__(
        self,
        case_id: str,  # FUT this can be removed once we do not use the scope api
        slide_id: str,
        level: int | str,
        tile_extent_x: int | str,
        tile_extent_y: int | str,
        tiles: pd.DataFrame,
        empaia_client: BaseEmpaiaClient,
    ) -> None:
        super().__init__()
        self.case_id = case_id
        self.slide_id = slide_id
        self.level = level
        self.tile_extent_x = tile_extent_x
        self.tile_extent_y = tile_extent_y
        self.tiles = tiles
        self._empaia_client = empaia_client

        self._slide: EmpaiaSlide | None = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Returns tile from the slide image at the specified index in RGB format."""
        if self._slide is None:
            self._slide = self._empaia_client.get_slide(self.case_id, self.slide_id)

        tile = self.tiles.iloc[idx]

        level = self._get_from_tile(tile, self.level)
        extent_x = self._get_from_tile(tile, self.tile_extent_x)
        extent_y = self._get_from_tile(tile, self.tile_extent_y)
        x = int(tile["x"] * self._slide.level_downsamples[level])
        y = int(tile["y"] * self._slide.level_downsamples[level])

        tile_bytes = self._empaia_client.get_region(
            self.case_id, self.slide_id, level, x, y, extent_x, extent_y
        )
        bg_tile = PIL.Image.new(mode="RGB", size=(extent_x, extent_y), color="#FFFFFF")
        tile_image = PIL.Image.open(io.BytesIO(tile_bytes))
        bg_tile.paste(im=tile_image, box=None)
        return np.array(bg_tile)

    def _get_from_tile(self, tile: pd.Series, key: int | str) -> int:
        return tile[key] if isinstance(key, str) else key
