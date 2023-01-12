"""
Amazon SageMaker Experiments Logger
-------------
"""
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from lightning_lite.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers.logger import DummyLogger, Logger, rank_zero_experiment
from sagemaker.experiments import Run, load_run
from time import time

log = logging.getLogger(__name__)


class SmLogger(Logger):
    """TBD"""

    def __init__(self, run: Optional[Run]) -> None:
        super().__init__()
        print("Calling logger init")
        self._run = _DummyRun()
        if run is not None:
            self._time_start = time()
            self._run = run
            self._time_since_last_epoch = self._time_start
            self._current_epoch = 0
            

    @property
    @rank_zero_experiment
    def experiment(self) -> Union[Run, "_DummyRun"]:
        print("Calling self.experiment")
        return self._run

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        [
            self._run.log_metric(
                name=k,
                value=o,
                step=metrics["epoch"],  # type: ignore
            )
            for k, o in metrics.items()
            if k != "epoch"
        ]

    @rank_zero_only
    def log_hyperparams(self, hparams: Dict[str, Union[str, int, float]]) -> None:
        self._run.log_parameters(hparams)

    @property
    @rank_zero_only
    def name(self) -> str:
        """Return the experiment name."""
        if isinstance(self._run, Run):
            return self._run.experiment_config["ExperimentName"]
        return ""

    @property
    @rank_zero_only
    def version(self) -> str:
        """Return the Run name."""
        if isinstance(self._run, Run):
            return self._run.experiment_config["RunName"]
        return ""

    @rank_zero_only
    def finalize(self, status: str) -> None:
        duration = time() - self._time_start
        self._run.log_metric(name="training_duration", value=duration)
        return

    @rank_zero_only
    def log_confusion_matrix(
        self, matrix: List[float], title: Optional[str] = None, is_output: bool = True
    ):
        data = {
            "type": "ConfusionMatrix",
            "version": 0,
            "title": title,
            "confusionMatrix": matrix,
        }
        self._run._log_graph_artifact(
            artifact_name=title,
            data=data,
            graph_type="ConfusionMatrix",
            is_output=is_output,
        )

    @rank_zero_only
    def log_epoch_duration(self, epoch: int) -> None:
        if epoch > self._current_epoch:
            duration = time() - self._time_since_last_epoch
            self._time_since_last_epoch = time()
            self._current_epoch = epoch
            self._run.log_metric(name="epoch_duration", value=duration, step=epoch)

    def __getitem__(self, idx: int) -> "DummyLogger":
        # enables self.logger[0].experiment.add_image(...)
        return self  # type: ignore

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods,
        to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return  # type: ignore


class _DummyRun:
    """Dummy run."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyRun":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass


@contextmanager
def load_dummy_run(*args, **kwargs):
    """Dummy context manager simulating `load_run()`"""
    try:
        yield None
    finally:
        pass


def select_loader():
    """Select if using real or dummy SM experiment context manager"""

    # workaround to be able to run script also in local mode
    if os.getenv("SM_CURRENT_INSTANCE_TYPE") == "local":
        return load_dummy_run

    # workaround to limit the logging to rank 0 process
    if rank_zero_only.rank > 0:
        return load_dummy_run
    return load_run
