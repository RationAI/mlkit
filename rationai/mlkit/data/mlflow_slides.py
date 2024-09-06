from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from mlflow.artifacts import download_artifacts
from torch.utils.data import ConcatDataset, Dataset

from rationai.mlkit.data.slide_tiles import SlideTiles


class MLFlowSlides(ConcatDataset):
    def __init__(self, uris: list[str]) -> None:
        """Load slides and tiles from MLFlow artifacts.

        Args:
            uris: List of MLFlow artifact URIs in the form of `mlflow-artifacts:/<artifact_path>`.
        """
        self.uris = uris
        self.slides, self.tiles = self.load_slides_and_tiles(uris)
        super().__init__(self.generate_datasets())

    def generate_datasets(self) -> Iterable[Dataset]:
        return (
            SlideTiles(
                slide_path=slide.path,
                level=slide.level,
                tile_width=slide.tile_width,
                tile_height=slide.tile_height,
                tiles=self.filter_tiles_by_slide(slide.path),
            )
            for slide in self.slides.itertuples()
        )

    def filter_tiles_by_slide(self, slide_path: str) -> pd.DataFrame:
        return self.tiles[self.tiles["slide_path"] == slide_path]

    @staticmethod
    def load_slides_and_tiles(uris: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        slides_dfs: list[pd.DataFrame] = []
        tiles_dfs: list[pd.DataFrame] = []

        for uri in uris:
            fp = Path(download_artifacts(artifact_uri=uri))
            try:
                slides = pd.read_parquet(fp / "slides.parquet")
                tiles = pd.read_parquet(fp / "tiles.parquet")
            except OSError as e:
                raise FileNotFoundError(f"Cannot load data from {fp}.") from e

            slides_dfs.append(slides)
            tiles_dfs.append(tiles)

        return pd.concat(slides_dfs), pd.concat(tiles_dfs)
