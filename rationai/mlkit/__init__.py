"""rationai.mlkit — ML toolkit with provenance tracking."""

from rationai.mlkit.autolog import autolog
from rationai.mlkit.provenance.register_dataset import register_dataset
from rationai.mlkit.stream import StreamCapture, StreamLogger

__all__ = [
    "StreamCapture",
    "StreamLogger",
    "AggregatedMetricCollection",
    "Aggregator",
    "MaxAggregator",
    "MeanAggregator",
    "MeanPoolMaxAggregator",
    "TopKAggregator",
    "NestedMetricCollection",
    "LazyMetricDict",
    "StratifiedBatchSampler",
    "PDMStratifiedBatchSampler",
    "MetaTiledSlides",
    "OpenSlideTilesDataset",
    "Trainer",
    "MLFlowLogger",
    "MultiloaderLifecycle",
    "ProvenanceCallback",
    "autolog",
    "with_cli_args",
    "register_dataset",
]


def __getattr__(name):
    if name in ("Trainer", "MLFlowLogger", "MultiloaderLifecycle", "with_cli_args"):
        import importlib
        _mod = importlib.import_module("rationai.mlkit.lightning")
        return getattr(_mod, name)

    if name == "ProvenanceCallback":
        from rationai.mlkit.lightning.callbacks import ProvenanceCallback
        return ProvenanceCallback

    if name in ("AggregatedMetricCollection", "Aggregator",
                "MaxAggregator", "MeanAggregator", "MeanPoolMaxAggregator",
                "TopKAggregator", "NestedMetricCollection", "LazyMetricDict"):
        import importlib
        _mod = importlib.import_module("rationai.mlkit.metrics")
        return getattr(_mod, name)

    if name in ("StratifiedBatchSampler", "PDMStratifiedBatchSampler"):
        import importlib
        _mod = importlib.import_module("rationai.mlkit.data.samplers")
        return getattr(_mod, name)

    if name in ("MetaTiledSlides", "OpenSlideTilesDataset"):
        import importlib
        _mod = importlib.import_module("rationai.mlkit.data.datasets")
        return getattr(_mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
