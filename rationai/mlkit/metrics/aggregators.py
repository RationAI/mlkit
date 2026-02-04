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

    max_pred: Tensor
    max_target: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "max_pred", default=torch.tensor(float("-inf")), dist_reduce_fx="max"
        )
        self.add_state(
            "max_target", default=torch.tensor(float("-inf")), dist_reduce_fx="max"
        )

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.max_pred = torch.max(self.max_pred, preds.max())
        self.max_target = torch.max(self.max_target, targets.max())

    def compute(self) -> tuple[Tensor, Tensor]:
        return self.max_pred, self.max_target


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


class HeatmapAggregator(Aggregator):
    preds: list[Tensor]
    targets: list[Tensor]
    xs: list[Tensor]
    ys: list[Tensor]

    """Abstract aggregator covering the prediction heatmap generation.

    Arguments:
        extent_tile (int): Size of the tile.
        stride_tile (int): Tile stride.
    """

    def __init__(
        self,
        extent_tile: int,
        stride_tile: int,
    ) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("xs", default=[], dist_reduce_fx="cat")
        self.add_state("ys", default=[], dist_reduce_fx="cat")
        self.extent_tile = extent_tile
        self.stride_tile = stride_tile

    def update(self, preds: Tensor, targets: Tensor, **kwargs: Any) -> None:
        self.preds.append(preds)
        self.targets.append(targets)
        if "x" not in kwargs or "y" not in kwargs:
            raise ValueError("x and y must be provided as keyword arguments.")
        self.xs.append(kwargs["x"])
        self.ys.append(kwargs["y"])

    def _get_extents(self, xs: Tensor, ys: Tensor) -> tuple[int, int]:
        extent_x = (xs + self.extent_tile).max().item()
        extent_y = (ys + self.extent_tile).max().item()
        return int(extent_x), int(extent_y)

    def _get_heatmap(self) -> Tensor:
        xs = torch.cat(self.xs)
        ys = torch.cat(self.ys)
        extent_x, extent_y = self._get_extents(xs, ys)
        assembler = HeatmapAssembler(
            extent_x,
            extent_y,
            self.extent_tile,
            self.extent_tile,
            self.stride_tile,
            self.stride_tile,
            device=str(self.preds[0].device) if self.preds else "cpu",
        )
        assembler.update(torch.cat(self.preds), xs, ys)
        return assembler.compute()


class MeanPoolMaxAggregator(HeatmapAggregator):
    """Aggregator to compute the max of predictions after average pooling of the prediction heatmap.

    Arguments:
        kernel_size (int): Size of the pooling kernel.
        extent_tile (int): Size of the tile.
        stride_tile (int): Tile stride.
    """

    def __init__(
        self,
        kernel_size: int,
        extent_tile: int,
        stride_tile: int,
    ) -> None:
        super().__init__(extent_tile, stride_tile)
        self.pool = nn.AvgPool2d(kernel_size, stride=1)

    def compute(self) -> tuple[Tensor, Tensor]:
        heatmap = self._get_heatmap()
        return (
            self.pool(heatmap.unsqueeze(0).unsqueeze(0)).max(),
            torch.cat(self.targets).max(),
        )


class TopKAggregator(HeatmapAggregator):
    """Aggregator to compute the mean of top k predictions after average pooling of the prediction heatmap.

    Arguments:
        kernel_size (int): Size of the pooling kernel.
        extent_tile (int): Size of the tile.
        stride_tile (int): Tile stride.
        k (int): Number of top predictions to take the mean from.
    """

    def __init__(
        self, kernel_size: int, extent_tile: int, stride_tile: int, k: int
    ) -> None:
        super().__init__(extent_tile, stride_tile)
        self.pool = nn.AvgPool2d(kernel_size, stride=1)
        self.k = k

    def compute(self) -> tuple[Tensor, Tensor]:
        heatmap = self._get_heatmap()
        pooled_values = self.pool(heatmap.unsqueeze(0).unsqueeze(0)).squeeze().flatten()
        topk_vals, _ = torch.topk(pooled_values, self.k)
        return (
            topk_vals.mean(),
            torch.cat(self.targets).max(),
        )
