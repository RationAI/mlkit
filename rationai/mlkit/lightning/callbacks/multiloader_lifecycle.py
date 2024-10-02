from abc import ABC
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class MultiloaderLifecycle(Callback, ABC):
    """An abstract callback class that adds new methods: `on_{stage}_dataloader_start` and `on_{stage}_dataloader_end`.

    `stage` is one of "validation", "test", or "predict".

    The callback monitors the dataloader index and calls the
    `on_{stage}_dataloader_start` and `on_{stage}_dataloader_end` methods when the
    dataloader index changes.

    Note:
        If subclasses override either `on_{stage}_batch_start` or `on_{stage}_epoch_end`,
        they must also call the corresponding parent methods to ensure proper
        functionality.
    """

    def __init__(self) -> None:
        self._dataloader_idxs = {"validation": -1, "test": -1, "predict": -1}

    def on_validation_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_validation_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_test_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_test_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None: ...

    def on_batch_start(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx != self._dataloader_idxs[stage]:
            if self._dataloader_idxs[stage] != -1:
                getattr(self, f"on_{stage}_dataloader_end")(
                    trainer, pl_module, self._dataloader_idxs[stage]
                )

            self._dataloader_idxs[stage] = dataloader_idx
            getattr(self, f"on_{stage}_dataloader_start")(
                trainer, pl_module, dataloader_idx
            )

    def on_epoch_end(
        self, stage: str, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self._dataloader_idxs[stage] != -1:
            getattr(self, f"on_{stage}_dataloader_end")(
                trainer, pl_module, self._dataloader_idxs[stage]
            )
            self._dataloader_idxs[stage] = -1

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("validation", trainer, pl_module, dataloader_idx)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_epoch_end("validation", trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("test", trainer, pl_module, dataloader_idx)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_end("test", trainer, pl_module)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_start("predict", trainer, pl_module, dataloader_idx)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_epoch_end("predict", trainer, pl_module)
