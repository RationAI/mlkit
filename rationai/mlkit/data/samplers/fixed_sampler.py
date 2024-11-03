import random
from collections.abc import Iterator
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, Dataset, WeightedRandomSampler


class FixedBatchSampler(BatchSampler):
    """A batch sampler that selects fixed size batches from multiple classes.

    This sampler allows for creating batches with a fixed size by drawing
    samples until minimum number of samples is reached for one of the classes.
    """

    def __init__(
        self, data_indices: list[list[int]], batch_distribution: NDArray
    ) -> None:
        """Initializes the Fixed sampler with data indices and batch size.

        Args:
            data_indices: A list of class-based sample indices.
            batch_distribution: The distribution of samples to draw from each class.
        """
        assert len(data_indices) == len(
            batch_distribution
        ), f"Number of classes must match with distribution : {batch_distribution}"
        self.data_indices = data_indices
        self.batch_size = sum(batch_distribution)
        self.batch_distribution = batch_distribution

    def __iter__(self) -> Iterator[list[int]]:
        """Yields balanced batches of samples from multiple classes based on the distribution.

        Yields:
            A list of sample indices for each batch.
        """
        indices = deepcopy(self.data_indices)
        while all(len(sublist) // self.batch_size > 0 for sublist in indices):
            batch_indices = list(
                np.concatenate(
                    [
                        [indices[i].pop() for _ in range(c)]
                        for i, c in enumerate(self.batch_distribution)
                    ]
                )
            )
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        num_batches_per_class = [
            len(sublist) // num_samples
            for sublist, num_samples in zip(
                self.data_indices, self.batch_distribution, strict=False
            )
        ]
        return min(num_batches_per_class)


class ResampleBatchSampler(FixedBatchSampler):
    """This sampler is designed to create balanced batches from a Dataset by stratifying sampling.

    It leverages WeightedRandomSampler to upsample to given distribution. Then, stratified sampling is performed to create balanced batches.
    Note that when specifying replacement=True, the number of samples must be greater than or equal to the number of classes. Also note that
    the balance could be visible over multiple epochs, as the sampler work with weights and not counts, but its not.
    """

    def __init__(
        self,
        dataset: Dataset,
        sampler_args: DictConfig,
        batch_size: int,
    ) -> None:
        """Initializes the ResampleBatchSampler with dataset and batch distribution."""
        label_counts = dataset.tiles[sampler_args.stratify_by].value_counts().to_dict()

        # Compute class weights to adjust the sampler
        class_weights = {
            label: sampler_args.sampler_distribution.get(label, 0) / count
            for label, count in label_counts.items()
        }

        # Assign weights to each sample based on class weights
        labels = dataset.tiles[sampler_args.stratify_by]
        weights = labels.map(class_weights).values

        replacement = any(
            count < sampler_args.sampler_distribution.get(label, 0)
            for label, count in label_counts.items()
        )

        total_samples = sum(sampler_args.sampler_distribution.values())
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=total_samples, replacement=replacement
        )
        # Group sampled indices by class
        sampled_indices = list(sampler)
        data_indices = [
            (label, list(group.index))
            for label, group in dataset.tiles.iloc[sampled_indices].groupby(
                sampler_args.stratify_by
            )
        ]

        batch_distribution = [
            (np.array(sampler_args.batch_distribution[label]) * batch_size).astype(int)
            for label, _ in data_indices
        ]

        assert (
            sum(batch_distribution) == batch_size
        ), f"Batch distribution must sum to batch size: {batch_distribution}"

        super().__init__([x[1] for x in data_indices], batch_distribution)
