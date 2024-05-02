from abc import abstractmethod

from lightning import LightningDataModule
from ray.data import Dataset


class RayDataModule(LightningDataModule):
    @abstractmethod
    def datasets(self) -> dict[str, Dataset]:
        """Return a dictionary of datasets for distributed training."""
