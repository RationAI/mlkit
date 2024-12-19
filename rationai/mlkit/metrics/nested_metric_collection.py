from collections import defaultdict
from copy import deepcopy
from typing import Any

from torch import Tensor
from torchmetrics import Metric, MetricCollection


class NestedMetricCollection(MetricCollection):
    """NestedMetricCollection is a specialized MetricCollection that groups metrics by a unique identifier (e.g., 'slide' ID), allowing for the computation of metrics at multiple levels, such as slides and scenes.

    This class was designed for use in digital pathology, where performance metrics
    (e.g., accuracy, precision, recall) may need to be calculated separately for each
    slide or scene, enabling evaluation at granular levels.

    Attributes:
        metrics (dict[str, Metric]): Dictionary containing the metrics to be computed.
        key_name (str): Name of the key used to group the metrics.
        class_names (list[str] | None): List of class names for multi-class metrics
        sep (str): Separator used to distinguish between the key and metric name. Note
            that the separator cannot be present in the key.

    Example:
        >>> import torch
        >>> from torchmetrics import Accuracy, Precision, Recall
        >>> from rationai.mlkit.metrics import NestedMetricCollection

        >>> # Define a base set of metrics that will be calculated for each slide individually.
        >>> metrics = {
        ...     "accuracy": Accuracy("multiclass", num_classes=3),
        ...     "precision": Precision("multiclass", num_classes=3, average=None),
        ...     "recall": Recall("multiclass", num_classes=3, average=None),
        ... }

        >>> # Create the NestedMetricCollection, setting 'slide' as the unique identifier for grouping. The class names are provided for multi-class metrics.
        >>> nested_metrics = NestedMetricCollection(
        ...     metrics,
        ...     key_name="slide"
        ...     class_names=["A", "B", "C"],
        ... )

        >>> # Simulate predictions and ground truth labels for two pathology slides, with two classes.
        >>> preds = torch.tensor(
        ...     [
        ...         [0.1, 0.2, 0.7],
        ...         [0.5, 0.3, 0.2],
        ...         [0.3, 0.3, 0.4],
        ...     ]
        ... )
        >>> targets = torch.tensor([2, 0, 1])
        >>> keys = ["slide1", "slide2", "slide2"]  # Unique identifiers for each slide.

        >>> # Update the metrics with predictions and targets for each slide.
        >>> nested_metrics.update(preds, targets, keys)

        >>> # Compute the results for each slide.
        >>> nested_metrics.compute()
        {
            'slide': ['slide1', 'slide2'],
            'accuracy': [1.0, 0.5], # Accuracy for each slide.
            'precision/A': [0.0, 1.0], # Precision for class A in each slide.
            'precision/B': [0.0, 0.0],
            'precision/C': [1.0, 0.0],
            'recall/A': [0.0, 1.0], # Recall for class A in each slide.
            'recall/B': [0.0, 0.0],
            'recall/C': [1.0, 0.0],
        }
        >>> # The result is compatible with pandas.DataFrame, allowing for easy logging with mlkit.lightning.loggers.MLFlowLogger.log_dict().
    """

    def __init__(
        self,
        metrics: Metric | MetricCollection | dict[str, Metric],
        key_name: str = "key",
        class_names: list[str] | None = None,
        sep: str = "~",
    ) -> None:
        super().__init__([])
        if isinstance(metrics, Metric):
            metrics = MetricCollection(metrics)

        self.metrics = dict(metrics.items())
        self.key_name = key_name
        self.class_names = class_names
        self.sep = sep

    def update(  # pylint: disable=arguments-differ
        self, preds: Tensor, targets: Tensor, keys: list[str]
    ) -> None:
        for pred, target, key in zip(preds, targets, keys, strict=True):
            if self.sep in key:
                raise ValueError(
                    f"Key cannot contain '{self.sep}', please choose another separator."
                )

            for name, metric in self.metrics.items():
                new_name = f"{key}{self.sep}{name}"
                if new_name not in self:
                    self.add_metrics({new_name: deepcopy(metric).to(preds.device)})

                self[new_name].update(pred.unsqueeze(0), target.unsqueeze(0))

    def compute(self) -> dict[str, Any]:
        divided_metrics = defaultdict(dict)
        for name, value in super().compute().items():
            key, subkey = name.split(self.sep, maxsplit=1)

            assert isinstance(value, Tensor)
            if not value.shape:
                divided_metrics[key][subkey] = value.item()
            else:
                # handle multi-class metrics without averaging
                assert len(value.shape) == 1
                if self.class_names is None:
                    self.class_names = list(range(len(value)))

                if len(value) != len(self.class_names):
                    raise ValueError(
                        f"Expected {len(self.class_names)} classes, got {len(value)}"
                    )
                for class_name, v in zip(self.class_names, value, strict=True):
                    divided_metrics[key][f"{subkey}/{class_name}"] = v.item()

        out = defaultdict(list)
        for key, metrics in divided_metrics.items():
            out[self.key_name].append(key)
            for subkey, value in metrics.items():
                out[subkey].append(value)
        return dict(out)
