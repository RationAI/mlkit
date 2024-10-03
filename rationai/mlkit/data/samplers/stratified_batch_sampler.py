import random
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import BatchSampler


class StratifiedBatchSampler(BatchSampler):
    """A batch sampler that selects balanced batches from multiple classes.

    This sampler allows for creating batches with a fixed size by drawing
    samples from a list of class-based indices while maintaining class balance.

    Args:
        data_indices: A list of lists, where each sublist contains the indices
            corresponding to samples from a particular class.
        batch_size: The size of each batch.
    """

    def __init__(self, data_indices: list[list[int]], batch_size: int) -> None:
        """Initializes the StratifiedBatchSampler with data indices and batch size.

        Args:
            data_indices: A list of class-based sample indices.
            batch_size: The number of samples in each batch.
        """
        self.data_indices = data_indices
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Creates an iterator that yields batches of balanced indices.

        Each batch is created by sampling from the available class-based indices,
        while ensuring that the total number of samples in the batch is exactly
        `batch_size`. Indices are shuffled at the start to ensure randomness.

        Yields:
            A list of sample indices for each batch.
        """
        indices = deepcopy(self.data_indices)

        # Shuffle indices
        for group in indices:
            random.shuffle(group)

        while self._indices_size(indices).sum() >= self.batch_size:
            real_counts = self._compute_ratios(indices) * self.batch_size
            counts = np.floor(real_counts).astype(int)
            missing_counts = real_counts - counts

            # If there are missing samples, add randomly weighted samples
            if missing_counts.sum():
                indices_to_add = (
                    np.random.rand(len(indices))
                    * missing_counts
                    / np.sum(missing_counts)
                    # numpy's argsort sorts in ascending order so we need to reverse it
                ).argsort()[-(self.batch_size - np.sum(counts)) :]
                counts += np.isin(np.arange(len(indices)), indices_to_add)

            batch_indices = list(
                np.concatenate(
                    [[indices[i].pop() for _ in range(c)] for i, c in enumerate(counts)]
                )
            )

            # Filter out empty groups efficiently
            indices = [group for group in indices if group]
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self._indices_size(self.data_indices).sum() // self.batch_size

    @staticmethod
    def _indices_size(indices: list[list[int]]) -> NDArray[np.int_]:
        return np.array([len(x) for x in indices])

    @staticmethod
    def _compute_ratios(indices: list[list[int]]) -> NDArray[np.float64]:
        sizes = StratifiedBatchSampler._indices_size(indices)
        return sizes.astype(float) / sizes.sum()


class PDMStratifiedBatchSampler(StratifiedBatchSampler):
    """A subclass of MulticlassBatchSampler that handles Pandas DataFrames.

    This sampler is designed to create balanced batches from a DataFrame by
    stratifying samples based on a specified column.


    """

    def __init__(
        self,
        data: pd.DataFrame,
        stratify_by: None,
        batch_size: int,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initializes the PDMStratifiedBatchSampler with DataFrame and batch size.

        Groups the DataFrame based on the stratification column and generates
        batches by sampling from these groups.

        Args:
            data: A pandas DataFrame containing the data.
            stratify_by: The column name to stratify by. It groups samples according to
                this column.
            batch_size: The size of each batch.
            **kwargs: Additional keyword arguments passed to `pd.groupby`.
        """
        data_indices: list[list[int]] = [
            list(x.index) for _, x in data.groupby(by=stratify_by, **kwargs)
        ]
        super().__init__(data_indices, batch_size)
