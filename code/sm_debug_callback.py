import logging
from typing import Optional

import pytorch_lightning as pl
import smdebug.pytorch as smd
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback, TQDMProgressBar

logger = logging.getLogger(__name__)


class SmDebugCallback(Callback):
    def __init__(
        self,
        out_dir: Optional[str] = "/opt/ml/output/tensors",
        log_frequency: Optional[int] = 10,
    ) -> None:
        """ """
        self._out_dir = out_dir
        self._log_frequency = log_frequency

    def on_fit_start(self, trainer, pl_module):
        logger.info("Initializing debug hook")
        try:
            self._hook = smd.Hook.create_from_json_file()
        except Exception:
            logger.info("No pre-existing hook found, creating a new one")
            self._hook = smd.Hook(out_dir=self._out_dir, export_tensorboard=True)

        self._hook.register_hook(pl_module)

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger.debug("Switching debug hook to TRAIN mode")
        self._hook.set_mode(smd.modes.TRAIN)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self._log_frequency == 0:
            for k, o in trainer.logged_metrics.items():
                self._hook.save_scalar(name=k, value=o.item())

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger.debug("Switching debug hook to PREDICT mode")
        self._hook.set_mode(smd.modes.PREDICT)

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger.debug("Switching debug hook to EVAL mode")
        self._hook.set_mode(smd.modes.EVAL)


class MeterlessProgressBar(TQDMProgressBar):
    """Remove dynamic meter to the progress bar

    Inspired by https://stackoverflow.com/a/68565522/2109965
    """

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar
