from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import BatchSampler, Sampler


class MulticlassBatchSampler(BatchSampler):
    def __init__(
        self,
        samplers: list[Sampler[int] | Iterable[int]],
        distribution: list[float],
        epoch_size: int,
        batch_size: int,
    ) -> None:
        self.samplers = samplers
        self.distribution = distribution
        self.epoch_size = epoch_size
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        samplers_iters = [iter(sampler) for sampler in self.samplers]

        for _ in range(self.epoch_size):
            batch = []
            for sampler, count in zip(
                samplers_iters,
                self._random_sample_counts(self.distribution, self.batch_size),
                strict=False,
            ):
                batch.extend([next(sampler) for _ in range(count)])
            yield batch

    def __len__(self) -> int:
        return self.epoch_size

    @staticmethod
    def _random_sample_counts(
        distribution: list[float] | NDArray[np.float64], size: int
    ) -> list[int]:
        real_counts = distribution * size
        counts = np.floor(real_counts).astype(int)
        missing_counts = real_counts - counts

        # If there are missing samples, add randomly weighted samples
        if missing_counts.sum():
            indices_to_add = (
                np.random.rand(len(distribution))
                * missing_counts
                / np.sum(missing_counts)
                # numpy's argsort sorts in ascending order so we need to reverse it
            ).argsort()[-(size - np.sum(counts)) :]
            counts += np.isin(np.arange(len(distribution)), indices_to_add)
        return list(counts)


class SubsetRandomInftySampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        while True:
            for i in self.__single_iteration__():
                yield self.indices[i]

    def __single_iteration__(self) -> Iterator[int]:
        return iter(np.random.permutation(len(self.indices)))

    def __len__(self) -> None:
        return None


class PDMulticlassBatchSampler(MulticlassBatchSampler):
    def __init__(
        self,
        data: pd.DataFrame,
        stratify_by: None,
        distribution: list[float],
        epoch_size: int,
        batch_size: int,
        **kwargs: dict[str, Any],
    ):
        samplers: list[Sampler[int]] = [
            SubsetRandomInftySampler(list(x.index))
            for _, x in data.groupby(by=stratify_by, **kwargs)
        ]
        super().__init__(
            samplers=samplers,
            distribution=distribution,
            epoch_size=epoch_size,
            batch_size=batch_size,
        )
