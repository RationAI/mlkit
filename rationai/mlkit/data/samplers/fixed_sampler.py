import random
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import BatchSampler, Dataset, Sampler, WeightedRandomSampler


class TargetBatchSampler(BatchSampler):
    """A batch sampler that selects fixed size batches from multiple classes.

    The sampler selects a `batch_size // 2` of target class samples and distribute
    rest of the non-target samples randomly to create balanced batches.
    """

    def __init__(
        self,
        data_indices: list[Sampler[int]],
        target_label: int,
        batch_size: int,
        epoch_size: int,
    ) -> None:
        """Initializes the sampler.

        Args:
            data_indices (list[Sampler[int]]): A list of class-based sample indices.
            target_label (int): The target label to draw samples from.
            batch_size (int): The size of the batch.
            epoch_size (int): The number of batches to yield.
        """
        assert batch_size % 2 == 0, "Batch size must be even."
        assert len(data_indices) > 1, "At least two classes are required."
        assert (
            len(data_indices) < batch_size // 2
        ), "Class samples has to have atleast one sample. Increase batch size."

        self.epoch_size = epoch_size
        self.data_indices = data_indices
        self.batch_size = batch_size
        self.target_label = target_label

    def __iter__(self) -> Iterator[list[int]]:
        """Yields balanced batches of samples from multiple classes based on the distribution.

        Yields:
            A list of sample indices for each batch.
        """
        samplers_iters = [iter(sampler) for sampler in self.data_indices]

        for _ in range(self.epoch_size):
            batch = []
            for sampler, count in zip(
                samplers_iters,
                self._compute_batch_distribution(self.target_label),
                strict=False,
            ):
                batch.extend(next(sampler) for _ in range(count))
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Returns the number of batches per epoch.

        Returns:
            The number of batches per epoch.
        """
        return self.epoch_size

    def _compute_batch_distribution(self, target_label: int) -> NDArray:
        """Computes the distribution of samples to draw from each class.

        Target distribution will be sampled to have `self.batch_size // 2` samples in one batch.
        The rest of the samples are distributed to match given batch size randomly.

        Args:
            target_label (int): The target label to draw samples from.

        Returns:
            NDArray: The distribution of samples to draw from each class.
        """
        num_classes = len(self.data_indices)
        non_target_indices = [i for i in range(num_classes) if i != target_label - 1]
        distribution = np.zeros(num_classes, dtype=int)
        distribution[target_label - 1] = self.batch_size // 2
        remaining_samples = self.batch_size // 2
        remaining_iteration = distribution[target_label - 1] // (num_classes - 1)
        add = np.ones(num_classes, dtype=np.int32) * remaining_iteration

        # Null tge target label bin
        add[target_label - 1] = 0

        # Add samples to the distribution
        distribution += add
        remaining_samples -= sum(add)
        # Fill the rest of the bins
        if remaining_samples:
            for _ in range(remaining_samples):
                random_index = random.choice(non_target_indices)
                distribution[random_index] += 1
        assert (
            sum(distribution) % self.batch_size == 0
        ), "Functionality is corrupted. Report the issue"
        return distribution


class RebalanceSampler(Sampler[int]):
    """A sampler that rebalances the dataset using `torch.utilds.data.WeightedRandomSampler` with replacement."""

    def __init__(self, indices: list[int], num_samples: int) -> None:
        """Initializes the sampler.

        Args:
            indices (list[int]): A list of sample indices.
            num_samples (int): The number of samples to draw.
        """
        self.indices = indices
        self.class_count = len(indices)
        self.weights = [1 / num_samples for _ in indices]
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        """Yields resampled data indices.

        Yields:
            Sample indices.
        """
        while True:
            for i in self.__single_iteration__():
                yield self.indices[i]

    def __single_iteration__(self) -> Iterator[int]:
        """Performs a single iteration of sampling.

        Returns:
            An iterator over sample indices.
        """
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_samples,
            replacement=self.class_count < self.num_samples,
        ).__iter__()

    def __len__(self) -> None:
        """Returns the length of the sampler.

        Returns:
            None
        """
        return None


class DatasetMulticlassSampler(TargetBatchSampler):
    """A sampler that creates balanced batches from a multiclass dataset."""

    def __init__(
        self,
        dataset: Dataset,
        epoch_samples: int,
        stratify_by: str,
        target_label: int,
        batch_size: int,
    ) -> None:
        """Initializes the sampler.

        Args:
            dataset (Dataset): The dataset to sample from.
            epoch_samples (int): The number of samples per epoch.
            stratify_by (str): The attribute to stratify by.
            target_label (int): Target label thah will be represented in the batch with half of the samples.
            batch_size (int): The size of the batch.
        """
        data_indices: list[Sampler[int]] = [
            RebalanceSampler(
                indices=list(group.index),
                num_samples=epoch_samples,
            )
            for _, group in dataset.tiles.groupby(stratify_by)
        ]

        epoch_size = epoch_samples // (batch_size // 2)
        super().__init__(
            data_indices=data_indices,
            target_label=target_label,
            batch_size=batch_size,
            epoch_size=epoch_size,
        )
