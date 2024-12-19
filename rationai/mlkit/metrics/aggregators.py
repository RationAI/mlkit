from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torchmetrics import Metric

from rationai.masks import HeatmapAssembler


class Aggregator(Metric, ABC):
    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None: ...

    @abstractmethod
    def compute(self) -> tuple[Tensor, Tensor]: ...


class MaxAggregator(Aggregator):
    """Aggregator to compute the maximum value of predictions and targets."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "preds", default=torch.tensor(float("-inf")), dist_reduce_fx="max"
        )
        self.add_state(
            "targets", default=torch.tensor(float("-inf")), dist_reduce_fx="max"
        )

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.preds = torch.max(self.preds, preds)
        self.targets = torch.max(self.targets, targets)

    def compute(self) -> tuple[Tensor, Tensor]:
        return self.preds, self.targets


class MeanAggregator(Aggregator):
    """Aggregator to compute the mean of predictions and targets."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state("preds", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("targets", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.preds += preds
        self.targets += targets
        self.count += 1

    def compute(self) -> tuple[Tensor, Tensor]:
        return self.preds / self.count, self.targets / self.count


class MeanPoolMaxAggregator(Aggregator):
    """Aggregator to compute the max of predictions after average pooling.

    Targets are assumed to be the same for all predictions. Therefore, only the first
    target is used for the aggregation.

    Predictions are transformed into heatmap with `HeatmapAssembler` and then pooled
    using `nn.AvgPool2d`. The aggregated value is the maximum value of the pooled
    heatmap.

    Arguments:
        kernel_size (int): Size of the pooling kernel.
        extent_tile (int): Size of the tile.
        stride (int): Stride of the pooling operation
    """

    def __init__(
        self,
        kernel_size: int,
        extent_tile: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("xs", default=[], dist_reduce_fx="cat")
        self.add_state("ys", default=[], dist_reduce_fx="cat")
        self.pool = nn.AvgPool2d(kernel_size, stride=1)
        self.extent_tile = extent_tile
        self.stride = stride

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.preds.append(preds)
        self.targets.append(targets)
        if "x" not in kwargs or "y" not in kwargs:
            raise ValueError("x and y must be provided as keyword arguments.")
        self.xs.append(kwargs["x"])
        self.ys.append(kwargs["y"])

    def _get_extents(self) -> tuple[int, int]:
        extent_x = max(x + self.extent_tile for x in self.xs)
        extent_y = max(y + self.extent_tile for y in self.ys)

        return extent_x, extent_y

    def compute(self) -> tuple[Tensor, Tensor]:
        extent_x, extent_y = self._get_extents()
        assembler = HeatmapAssembler(
            extent_x,
            extent_y,
            self.extent_tile,
            self.extent_tile,
            self.stride,
            self.stride,
            device=self.preds[0].device if self.preds else "cpu",
        )
        assembler.update(
            torch.cat(self.preds), torch.stack(self.xs), torch.stack(self.ys)
        )
        return (
            self.pool(assembler.compute().unsqueeze(0).unsqueeze(0)).max(),
            torch.stack(self.targets).max(),
        )
