import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import lightning_fabric as fl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint

from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils import logging
from nemo.utils.app_state import AppState


@dataclass
class NeMoLogger:
    """Logger for NeMo runs.

    Args:
        name (str): Name of the experiment.
        dir (Optional[str]): Directory to save logs.
        explicit_log_dir (Optional[str]): Explicit log directory.
        version (Optional[str]): Version of the experiment.
        use_datetime_version (bool): Whether to use datetime as version.
        log_local_rank_0_only (bool): Log only on local rank 0.
        log_global_rank_0_only (bool): Log only on global rank 0.
        files_to_copy (Optional[List[str]]): List of files to copy to log directory.
        update_logger_directory (bool): Whether to update logger directory.
        ckpt (Optional[ModelCheckpoint]): Model checkpoint callback.
    """

    name: str = "default"
    dir: Optional[str] = None
    explicit_log_dir: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: bool = True
    log_local_rank_0_only: bool = False
    log_global_rank_0_only: bool = False
    files_to_copy: Optional[List[str]] = None
    update_logger_directory: bool = True
    ckpt: Optional[ModelCheckpoint] = None

    def __post_init__(self):
        if self.log_local_rank_0_only is True and self.log_global_rank_0_only is True:
            raise ValueError(
                f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
            )

    def setup(
        self,
        trainer: Union[pl.Trainer, fl.Fabric],
        resume_if_exists: bool = False,
    ):
        """Setup the logger for the experiment.

        Args:
            trainer (Union[pl.Trainer, fl.Fabric]): Trainer or Fabric instance.
            resume_if_exists (bool): Whether to resume if log directory exists.

        Returns:
            AppState: The application state with updated log directory and other settings.
        """
        from nemo.constants import NEMO_ENV_VARNAME_TESTING, NEMO_ENV_VARNAME_VERSION
        from nemo.utils.env_var_parsing import get_envbool
        from nemo.utils.exp_manager import check_explicit_log_dir
        from nemo.utils.get_rank import is_global_rank_zero
        from nemo.utils.mcore_logger import add_handlers_to_mcore_logger

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = trainer.node_rank * trainer.world_size + local_rank
        logging.rank = global_rank

        if self.explicit_log_dir and isinstance(trainer, pl.Trainer):  # If explicit log_dir was passed, short circuit
            return check_explicit_log_dir(trainer, self.explicit_log_dir, self.dir, self.name, self.version)

        # Default dir to ./nemo_experiments if None was passed
        _dir = self.dir
        if self.dir is None:
            _dir = str(Path.cwd() / 'nemo_experiments')

        if not self.name:
            self.name = "default"

        if isinstance(trainer, pl.Trainer) and trainer.logger is not None:
            if self.update_logger_directory:
                logging.warning(
                    f'"update_logger_directory" is True. Overwriting logger "save_dir" to {_dir} and "name" to {self.name}'
                )
                trainer.logger._root_dir = _dir
                trainer.logger._name = self.name

        version = self.version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)
        if is_global_rank_zero():
            if self.use_datetime_version:
                version = time.strftime('%Y-%m-%d_%H-%M-%S')
        if resume_if_exists:
            logging.warning(
                "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
            )
            version = None
        trainer.logger._version = version or ""
        if version:
            if is_global_rank_zero():
                os.environ[NEMO_ENV_VARNAME_VERSION] = version

        log_dir = Path(_dir) / Path(str(self.name)) / Path("" if version is None else str(version))
        # update app_state with log_dir, exp_dir, etc
        app_state = AppState()
        app_state.log_dir = log_dir
        app_state.exp_dir = _dir
        app_state.name = self.name
        app_state.version = version

        os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
        logging.info(f'Experiments will be logged at {log_dir}')

        if isinstance(trainer, pl.Trainer):
            if self.ckpt:
                _overwrite_i = None
                for i, callback in enumerate(trainer.callbacks):
                    if isinstance(callback, PTLModelCheckpoint):
                        logging.warning(
                            "The Trainer already contains a ModelCheckpoint callback. " "This will be overwritten."
                        )
                        _overwrite_i = i
                        break
                if _overwrite_i is not None:
                    trainer.callbacks[_overwrite_i] = self.ckpt
                else:
                    trainer.callbacks.append(self.ckpt)

                if self.ckpt.monitor and "val" in self.ckpt.monitor:
                    if (
                        trainer.max_epochs is not None
                        and trainer.max_epochs != -1
                        and trainer.max_epochs < trainer.check_val_every_n_epoch
                    ):
                        logging.error(
                            "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                            f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch({trainer.check_val_every_n_epoch}"
                            f"). It is very likely this run will fail with ModelCheckpoint(monitor='{self.ckpt.monitor}') not found "
                            "in the returned metrics. Please ensure that validation is run within trainer.max_epochs."
                        )
                    elif trainer.max_steps is not None and trainer.max_steps != -1:
                        logging.warning(
                            "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                            f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                            f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
                        )

            for callback in trainer.callbacks:
                if isinstance(callback, PTLModelCheckpoint):
                    if callback.dirpath is None:
                        callback.dirpath = Path(log_dir / "checkpoints")
                    if callback.filename is None:
                        callback.filename = f'{self.name}--{{{callback.monitor}:.4f}}-{{epoch}}'
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback.filename + '-last'

        # This is set if the env var NEMO_TESTING is set to True.
        nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

        files_to_move = []
        if Path(log_dir).exists():
            for child in Path(log_dir).iterdir():
                if child.is_file():
                    files_to_move.append(child)

        # Handle logging to file
        log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt'
        if self.log_local_rank_0_only is True and not nemo_testing:
            if local_rank == 0:
                logging.add_file_handler(log_file)
        elif self.log_global_rank_0_only is True and not nemo_testing:
            if global_rank == 0:
                logging.add_file_handler(log_file)
        else:
            # Logs on all ranks.
            logging.add_file_handler(log_file)

        add_handlers_to_mcore_logger()

        app_state.files_to_move = files_to_move
        app_state.files_to_copy = self.files_to_copy
        app_state.cmd_args = sys.argv

        return app_state

    def teardown(self):
        pass
