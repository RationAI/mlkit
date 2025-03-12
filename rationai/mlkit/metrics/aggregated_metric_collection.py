from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from torch import Tensor
from torchmetrics import Metric, MetricCollection

from rationai.mlkit.metrics.aggregators import Aggregator


class AggregatedMetricCollection(MetricCollection):
    """AggregatedMetricCollection is a customized MetricCollection designed to aggregate tile-level predictions and targets before computing metrics.

    In digital pathology, predictions and targets are often at the tile-level. However,
    some use cases require slide-level, which means that tile-level predictions and
    targets have to be aggregated to one prediction and target per slide.

    This class enables such aggregation by collecting tile-level data and applying
    user-defined aggregation functions before computing slide-level metrics.

    The aggregation functions should take a list of tile-level predictions, a list of
    tile-level targets, and a list of additional arguments, and return a tuple of
    aggregated predictions and targets.

    How it works:
        - Update method: Collect tile-level predictions and targets, but do not update
            the metrics.
        - Compute method: Aggregator is applied to tile-level predictions, targets, and
            additional arguments. Then all metrics are updated with the aggregated
            predictions and targets. Finally, the metrics are computed and the results
            are returned. Note that the metrics are reset after each compute call.

    Arguments:
        metrics
            (Metric | Sequence[Metric] | dict[str, Metric]): Metric(s) to be computed.
        aggregator
            (Aggregator): Aggregator to be applied to tile-level predictions and targets.
        prefix (str): Prefix to add to the metric names.

    Example:
        >>> import torch
        >>> from torchmetrics import Accuracy, Precision, Recall
        >>> from rationai.mlkit.metrics import AggregatedMetricCollection
        >>> from rationai.mlkit.metrics.aggregators import MaxAggregator

        >>> # Define the metrics to compute after aggregation.
        >>> metrics = {
        ...     "accuracy": Accuracy("binary"),
        ...     "precision": Precision("binary"),
        ...     "recall": Recall("binary"),
        ... }

        >>> # Define the aggregatior to use.
        >>> aggregatior = MaxAggregator()

        >>> # Initialize AggregatedMetricCollection with the chosen metrics and aggregator.
        >>> agg_metrics = AggregatedMetricCollection(metrics, aggregatior)

        >>> # Simulate tile-level predictions and targets for two slides.
        >>> preds = torch.tensor([0.1, 0.8, 0.3, 0.6])
        >>> targets = torch.tensor([1, 1, 0, 0])  # Ground truth labels for each tile.
        >>> keys = ["slide1", "slide1", "slide2", "slide2"]  # Tile-to-slide mapping.

        >>> # Update metrics with tile-level data.
        >>> agg_metrics.update(preds, targets, keys)

        >>> # Compute slide-level metrics after aggregation.
        >>> agg_metrics.compute()
        {
            "accuracy": 0.5,    # Slide-level accuracy after aggregation.
            "precision": 0.5,   # Slide-level precision after aggregation.
            "recall": 1.0,      # Slide-level recall after aggregation.
        }
    """

    def __init__(
        self,
        metrics: Metric | Sequence[Metric] | dict[str, Metric],
        aggregator: Aggregator,
        prefix: str | None = None,
    ) -> None:
        super().__init__(metrics, prefix=prefix)

        self.aggregators: dict[str, Aggregator] = defaultdict(aggregator.clone)

    def update(  # pylint: disable=arguments-differ
        self, preds: Tensor, targets: Tensor, keys: list[str], **kwargs: Any
    ) -> None:
        kwargs_t = ({k: v[i] for k, v in kwargs.items()} for i in range(len(preds)))
        for pred, target, k, kwarg in zip(preds, targets, keys, kwargs_t, strict=False):
            self.aggregators[k].update(pred, target, **kwarg)

    def compute(self) -> dict[str, Any]:
        for aggregator in self.aggregators.values():
            pred, target = aggregator.compute()
            super().update(pred.unsqueeze(0), target.unsqueeze(0))

        output = super().compute()
        super().reset()
        return output
