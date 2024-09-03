import functools
import logging
from typing import Any

import lightning
import mlflow
import torch

from rationai.mlkit.lightning.callbacks.dataloader_agnostic import (
    DataloaderAgnosticCallback
)
from rationai.visualization.image_builders import ImageBuilder


logger = logging.getLogger("ExAI")

# Captum Types
TargetType = (
    None | int | tuple[int, ...] | torch.Tensor | list[tuple[int, ...]] | list[int]
)
BaselineType = (
    None | torch.Tensor | int | float | tuple[torch.Tensor | int | float, ...]
)
StridesType = None | int | tuple[int, ...] | tuple[int | tuple[int, ...], ...]


class Explainer(DataloaderAgnosticCallback):
    image_builder: ImageBuilder
    partial_image_builder: functools.partial
    explainer: Any | None
    predict_mode: str
    target: TargetType
    save_dir: str

    def __init__(
        self,
        image_builder: functools.partial,
        save_dir: str,
        target: TargetType = None,
        predict_mode="max",
    ) -> None:
        super().__init__()
        self.partial_image_builder = image_builder
        self.predict_mode = predict_mode
        self.save_dir = save_dir
        self.target = target
        self.explainer = None

    def on_test_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        pass

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        logger.debug("Creating new Heatmap visualizer.")
        self.image_builder = self.partial_image_builder(
            metadata=metadata, save_dir=self.save_dir
        )

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        logger.debug("Saving explanation map.")
        save_path = self.image_builder.save()
        mlflow.log_artifact(local_path=save_path, artifact_path=self.save_dir)

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        x, y, metadata = batch
        target = self._resolve_target(y, outputs["outputs"])
        res = self._explain(x, target)
        data = torch.sigmoid(res).mean(dim=1, keepdim=True)
        self.image_builder.update(data=data, metadata=metadata)

    def _explain(self, inputs: torch.Tensor, target: TargetType) -> Any:
        raise NotImplementedError()

    def _resolve_target(
        self, labels: torch.Tensor, outputs: torch.Tensor
    ) -> TargetType:
        match self.target:
            case "label":
                return labels
            case "predict":
                match self.predict_mode:
                    case "max":
                        return torch.argmax(outputs, dim=-1)
                    case "min":
                        return torch.argmin(outputs, dim=-1)
                    case _:
                        raise ValueError(
                            f"Invalid predict_mode argument. Should be 'min' or 'max'; found: '{self.predict_mode}'."
                        )
            case _:
                return self.target
