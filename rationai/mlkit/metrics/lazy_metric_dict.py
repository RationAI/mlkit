from copy import deepcopy
from typing import Any

from deprecated import deprecated
from torch.nn import ModuleDict
from torchmetrics import Metric, MetricCollection


@deprecated(
    version="0.2.0",
    reason="LazyMetricDict is deprecated. Use rationai.mlkit.metrics.NestedMetricCollection instead.",
)
class LazyMetricDict(ModuleDict):
    def __init__(self, metric: Metric | MetricCollection) -> None:
        super().__init__()
        self.metric = metric

    def update(self, *args: Any, key: str, **kwargs: Any) -> None:  # type: ignore[override]
        if key not in self:
            self.add_module(key, deepcopy(self.metric))

        self[key].update(*args, **kwargs)

    def compute(self) -> dict[str, Any]:
        return {k: v.compute() for k, v in self.items() if k != "metric"}

    def reset(self) -> None:
        for metric in self.values():
            metric.reset()
