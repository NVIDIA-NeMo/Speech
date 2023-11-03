# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from shutil import copy, move
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Interval, Timer
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.loops import _TrainingEpochLoop
from pytorch_lightning.strategies.ddp import DDPStrategy

from nemo.collections.common.callbacks import EMA
from nemo.constants import NEMO_ENV_VARNAME_TESTING, NEMO_ENV_VARNAME_VERSION
from nemo.utils import logging, timers
from nemo.utils.app_state import AppState
from nemo.utils.callbacks import NeMoModelCheckpoint, PreemptionCallback
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exceptions import NeMoBaseException
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger
from nemo.utils.loggers import ClearMLLogger, ClearMLParams, DLLogger, DLLoggerParams, MLFlowParams
from nemo.utils.mcore_logger import add_handlers_to_mcore_logger
from nemo.utils.model_utils import uninject_model_parallel_rank


class NotFoundError(NeMoBaseException):
    """ Raised when a file or folder is not found"""


class LoggerMisconfigurationError(NeMoBaseException):
    """ Raised when a mismatch between trainer.logger and exp_manager occurs"""

    def __init__(self, message):
        message = (
            message
            + " You can disable lighning's trainer from creating a logger by passing logger=False to its constructor."
        )
        super().__init__(message)


class CheckpointMisconfigurationError(NeMoBaseException):
    """ Raised when a mismatch between trainer.callbacks and exp_manager occurs"""


@dataclass
class EarlyStoppingParams:
    monitor: str = "val_loss"  # The metric that early stopping should consider.
    mode: str = "min"  # inform early stopping whether to look for increase or decrease in monitor.
    min_delta: float = 0.001  # smallest change to consider as improvement.
    patience: int = 10  # how many (continuous) validation cycles to wait with no improvement and stopping training.
    verbose: bool = True
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: Optional[bool] = None
    log_rank_zero_only: bool = False


@dataclass
class CallbackParams:
    filepath: Optional[str] = None  # Deprecated
    dirpath: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    filename: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    monitor: Optional[str] = "val_loss"
    verbose: Optional[bool] = True
    save_last: Optional[bool] = True
    save_top_k: Optional[int] = 3
    save_weights_only: Optional[bool] = False
    mode: Optional[str] = "min"
    auto_insert_metric_name: bool = True
    every_n_epochs: Optional[int] = 1
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    prefix: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    postfix: str = ".nemo"
    save_best_model: bool = False
    always_save_nemo: bool = False
    save_nemo_on_train_end: Optional[bool] = True  # Whether to automatically save .nemo file durin on_train_end hook
    model_parallel_size: Optional[int] = None  # tensor parallel size * pipeline parallel size
    save_on_train_epoch_end: Optional[bool] = False  # Save after training, not after validation


@dataclass
class StepTimingParams:
    reduction: Optional[str] = "mean"
    # if True torch.cuda.synchronize() is called on start/stop
    sync_cuda: Optional[bool] = False
    # if positive, defines the size of a sliding window for computing mean
    buffer_size: Optional[int] = 1


@dataclass
class EMAParams:
    enable: Optional[bool] = False
    decay: Optional[float] = 0.999
    cpu_offload: Optional[bool] = False
    validate_original_weights: Optional[bool] = False
    every_n_steps: int = 1


@dataclass
class ExpManagerConfig:
    """Experiment Manager config for validation of passed arguments.
    """

    # Log dir creation parameters
    explicit_log_dir: Optional[str] = None
    exp_dir: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: Optional[bool] = True
    resume_if_exists: Optional[bool] = False
    resume_past_end: Optional[bool] = False
    resume_ignore_no_checkpoint: Optional[bool] = False
    resume_from_checkpoint: Optional[str] = None
    # Logging parameters
    create_tensorboard_logger: Optional[bool] = True
    summary_writer_kwargs: Optional[Dict[Any, Any]] = None
    create_wandb_logger: Optional[bool] = False
    wandb_logger_kwargs: Optional[Dict[Any, Any]] = None
    create_mlflow_logger: Optional[bool] = False
    mlflow_logger_kwargs: Optional[MLFlowParams] = field(default_factory=lambda: MLFlowParams())
    create_dllogger_logger: Optional[bool] = False
    dllogger_logger_kwargs: Optional[DLLoggerParams] = field(default_factory=lambda: DLLoggerParams())
    create_clearml_logger: Optional[bool] = False
    clearml_logger_kwargs: Optional[ClearMLParams] = field(default_factory=lambda: ClearMLParams())
    # Checkpointing parameters
    create_checkpoint_callback: Optional[bool] = True
    checkpoint_callback_params: Optional[CallbackParams] = field(default_factory=lambda: CallbackParams())
    create_early_stopping_callback: Optional[bool] = False
    early_stopping_callback_params: Optional[EarlyStoppingParams] = field(
        default_factory=lambda: EarlyStoppingParams()
    )
    create_preemption_callback: Optional[bool] = True
    # Additional exp_manager arguments
    files_to_copy: Optional[List[str]] = None
    # logs timing of train/val/test steps
    log_step_timing: Optional[bool] = True
    step_timing_kwargs: Optional[StepTimingParams] = field(default_factory=lambda: StepTimingParams())
    # Configures creation of log files for different ranks
    log_local_rank_0_only: Optional[bool] = False
    log_global_rank_0_only: Optional[bool] = False
    # disable initial validation when resuming from a checkpoint saved during validation
    disable_validation_on_resume: Optional[bool] = True
    ema: Optional[EMAParams] = field(default_factory=lambda: EMAParams())
    # Wall clock time limit
    max_time_per_run: Optional[str] = None
    # time to sleep non 0 ranks during initialization
    seconds_to_sleep: float = 5


class TimingCallback(Callback):
    """
    Logs execution time of train/val/test steps
    """

    def __init__(self, timer_kwargs={}):
        self.timer = timers.NamedTimer(**timer_kwargs)

    def _on_batch_start(self, name):
        # reset only if we do not return mean of a sliding window
        if self.timer.buffer_size <= 0:
            self.timer.reset(name)

        if self.timer.is_active(name):
            logging.warning(
                f"Timer `{name}` was not correctly stopped, suggesting a "
                "possible issue. The timer will be reset for now."
            )
            self.timer.reset(name)

        self.timer.start(name)

    def _on_batch_end(self, name, pl_module):
        self.timer.stop(name)
        # Set the `batch_size=1` as WAR for `dataloader_iter`, which is not used for any metric
        pl_module.log(
            name + ' in s',
            self.timer[name],
            on_step=True,
            on_epoch=False,
            batch_size=1,
            prog_bar=(name == "train_step_timing"),
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._on_batch_start("train_step_timing")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._on_batch_end("train_step_timing", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start("validation_step_timing")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end("validation_step_timing", pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._on_batch_start("test_step_timing")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._on_batch_end("test_step_timing", pl_module)

    def on_before_backward(self, trainer, pl_module, loss):
        self._on_batch_start("train_backward_timing")

    def on_after_backward(self, trainer, pl_module):
        self._on_batch_end("train_backward_timing", pl_module)


def exp_manager(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm
    of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get exp_dir,
    name, and version from the logger. Otherwise it will use the exp_dir and name arguments to create the logging
    directory. exp_manager also allows for explicit folder creation via explicit_log_dir.

    The version can be a datetime string or an integer. Datestime version can be disabled if use_datetime_version is set
    to False. It optionally creates TensorBoardLogger, WandBLogger, DLLogger, MLFlowLogger, ClearMLLogger,
    ModelCheckpoint objects from pytorch lightning.
    It copies sys.argv, and git information if available to the logging directory. It creates a log file for each
    process to log their output into.

    exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing training from
    the constructed log_dir. When you need to continue the training repeatedly (like on a cluster which you need
    multiple consecutive jobs), you need to avoid creating the version folders. Therefore from v1.0.0, when
    resume_if_exists is set to True, creating the version folders is ignored.

    Args:
        trainer (pytorch_lightning.Trainer): The lightning trainer.
        cfg (DictConfig, dict): Can have the following keys:

            - explicit_log_dir (str, Path): Can be used to override exp_dir/name/version folder creation. Defaults to
                None, which will use exp_dir, name, and version to construct the logging directory.
            - exp_dir (str, Path): The base directory to create the logging directory. Defaults to None, which logs to
                ./nemo_experiments.
            - name (str): The name of the experiment. Defaults to None which turns into "default" via name = name or
                "default".
            - version (str): The version of the experiment. Defaults to None which uses either a datetime string or
                lightning's TensorboardLogger system of using version_{int}.
            - use_datetime_version (bool): Whether to use a datetime string for version. Defaults to True.
            - resume_if_exists (bool): Whether this experiment is resuming from a previous run. If True, it sets
                trainer._checkpoint_connector._ckpt_path so that the trainer should auto-resume. exp_manager will move files
                under log_dir to log_dir/run_{int}. Defaults to False. From v1.0.0, when resume_if_exists is True,
                we would not create version folders to make it easier to find the log folder for next runs.
            - resume_past_end (bool): exp_manager errors out if resume_if_exists is True and a checkpoint matching
                ``*end.ckpt`` indicating a previous training run fully completed. This behaviour can be disabled, in which
                case the ``*end.ckpt`` will be loaded by setting resume_past_end to True. Defaults to False.
            - resume_ignore_no_checkpoint (bool): exp_manager errors out if resume_if_exists is True and no checkpoint
                could be found. This behaviour can be disabled, in which case exp_manager will print a message and
                continue without restoring, by setting resume_ignore_no_checkpoint to True. Defaults to False.
            - resume_from_checkpoint (str): Can be used to specify a path to a specific checkpoint file to load from. This will
                override any checkpoint found when resume_if_exists is True. Defaults to None.
            - create_tensorboard_logger (bool): Whether to create a tensorboard logger and attach it to the pytorch
                lightning trainer. Defaults to True.
            - summary_writer_kwargs (dict): A dictionary of kwargs that can be passed to lightning's TensorboardLogger
                class. Note that log_dir is passed by exp_manager and cannot exist in this dict. Defaults to None.
            - create_wandb_logger (bool): Whether to create a Weights and Baises logger and attach it to the pytorch
                lightning trainer. Defaults to False.
            - wandb_logger_kwargs (dict): A dictionary of kwargs that can be passed to lightning's WandBLogger
                class. Note that name and project are required parameters if create_wandb_logger is True.
                Defaults to None.
            - create_mlflow_logger (bool): Whether to create an MLFlow logger and attach it to the pytorch lightning
                training. Defaults to False
            - mlflow_logger_kwargs (dict): optional parameters for the MLFlow logger
            - create_dllogger_logger (bool): Whether to create an DLLogger logger and attach it to the pytorch lightning
                training. Defaults to False
            - dllogger_logger_kwargs (dict): optional parameters for the DLLogger logger
            - create_clearml_logger (bool): Whether to create an ClearML logger and attach it to the pytorch lightning
                training. Defaults to False
            - clearml_logger_kwargs (dict): optional parameters for the ClearML logger
            - create_checkpoint_callback (bool): Whether to create a ModelCheckpoint callback and attach it to the
                pytorch lightning trainer. The ModelCheckpoint saves the top 3 models with the best "val_loss", the most
                recent checkpoint under ``*last.ckpt``, and the final checkpoint after training completes under ``*end.ckpt``.
                Defaults to True.
            - create_early_stopping_callback (bool): Flag to decide if early stopping should be used to stop training. Default is False.
             See EarlyStoppingParams dataclass above.
            - create_preemption_callback (bool): Flag to decide whether to enable preemption callback to save checkpoints and exit training
             immediately upon preemption. Default is True.
            - files_to_copy (list): A list of files to copy to the experiment logging directory. Defaults to None which
                copies no files.
            - log_local_rank_0_only (bool): Whether to only create log files for local rank 0. Defaults to False.
                Set this to True if you are using DDP with many GPUs and do not want many log files in your exp dir.
            - log_global_rank_0_only (bool): Whether to only create log files for global rank 0. Defaults to False.
                Set this to True if you are using DDP with many GPUs and do not want many log files in your exp dir.
            - max_time (str): The maximum wall clock time *per run*. This is intended to be used on clusters where you want 
                a checkpoint to be saved after this specified time and be able to resume from that checkpoint. Defaults to None.
            - seconds_to_sleep (float): seconds to sleep non rank 0 processes for. Used to give enough time for rank 0 to initialize

    returns:
        log_dir (Path): The final logging directory where logging files are saved. Usually the concatenation of
            exp_dir, name, and version.
    """
    # Add rank information to logger
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = trainer.node_rank * trainer.num_devices + local_rank
    logging.rank = global_rank

    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return
    if trainer.fast_dev_run:
        logging.info("Trainer was called with fast_dev_run. exp_manager will return without any functionality.")
        return

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    error_checks(trainer, cfg)  # Ensures that trainer options are compliant with NeMo and exp_manager arguments

    log_dir, exp_dir, name, version = get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume_if_exists=cfg.resume_if_exists,
    )

    check_resume(
        trainer,
        log_dir,
        cfg.resume_if_exists,
        cfg.resume_past_end,
        cfg.resume_ignore_no_checkpoint,
        cfg.checkpoint_callback_params.dirpath,
        cfg.resume_from_checkpoint,
    )

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == '':
        checkpoint_name = cfg.name or "default"

    # Set mlflow name if it's not set, before the main name is erased
    if cfg.create_mlflow_logger and (not cfg.mlflow_logger_kwargs.get("experiment_name", None)):
        cfg.mlflow_logger_kwargs.experiment_name = cfg.name
        logging.warning(
            'mlflow logger specified but no experiment name set. Using the same as Tensorboard: %s',
            cfg.mlflow_logger_kwargs.experiment_name,
        )

    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # update app_state with log_dir, exp_dir, etc
    app_state = AppState()
    app_state.log_dir = log_dir
    app_state.exp_dir = exp_dir
    app_state.name = name
    app_state.version = version
    app_state.checkpoint_name = checkpoint_name
    app_state.create_checkpoint_callback = cfg.create_checkpoint_callback
    app_state.checkpoint_callback_params = cfg.checkpoint_callback_params

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f'Experiments will be logged at {log_dir}')
    trainer._default_root_dir = log_dir

    if cfg.log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
        raise ValueError(
            f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
        )

    # This is set if the env var NEMO_TESTING is set to True.
    nemo_testing = get_envbool(NEMO_ENV_VARNAME_TESTING, False)

    # Handle logging to file
    log_file = log_dir / f'nemo_log_globalrank-{global_rank}_localrank-{local_rank}.txt'
    if cfg.log_local_rank_0_only is True and not nemo_testing:
        if local_rank == 0:
            logging.add_file_handler(log_file)
    elif cfg.log_global_rank_0_only is True and not nemo_testing:
        if global_rank == 0:
            logging.add_file_handler(log_file)
    else:
        # Logs on all ranks.
        logging.add_file_handler(log_file)

    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if (
        cfg.create_tensorboard_logger
        or cfg.create_wandb_logger
        or cfg.create_mlflow_logger
        or cfg.create_dllogger_logger
        or cfg.create_clearml_logger
    ):
        configure_loggers(
            trainer,
            exp_dir,
            log_dir,
            cfg.name,
            cfg.version,
            cfg.checkpoint_callback_params,
            cfg.create_tensorboard_logger,
            cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
            cfg.create_mlflow_logger,
            cfg.mlflow_logger_kwargs,
            cfg.create_dllogger_logger,
            cfg.dllogger_logger_kwargs,
            cfg.create_clearml_logger,
            cfg.clearml_logger_kwargs,
        )

    # add loggers timing callbacks
    if cfg.log_step_timing:
        timing_callback = TimingCallback(timer_kwargs=cfg.step_timing_kwargs or {})
        trainer.callbacks.insert(0, timing_callback)

    if cfg.ema.enable:
        ema_callback = EMA(
            decay=cfg.ema.decay,
            validate_original_weights=cfg.ema.validate_original_weights,
            cpu_offload=cfg.ema.cpu_offload,
            every_n_steps=cfg.ema.every_n_steps,
        )
        trainer.callbacks.append(ema_callback)

    if cfg.create_early_stopping_callback:
        early_stop_callback = EarlyStopping(**cfg.early_stopping_callback_params)
        trainer.callbacks.append(early_stop_callback)

    if cfg.create_checkpoint_callback:
        configure_checkpointing(
            trainer,
            log_dir,
            checkpoint_name,
            cfg.resume_if_exists,
            cfg.checkpoint_callback_params,
            cfg.create_preemption_callback,
        )

    if cfg.disable_validation_on_resume:
        # extend training loop to skip initial validation when resuming from checkpoint
        configure_no_restart_validation_training_loop(trainer)
    # Setup a stateless timer for use on clusters.
    if cfg.max_time_per_run is not None:
        found_ptl_timer = False
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                # NOTE: PTL does not expose a `trainer.max_time`. By the time we are in this function, PTL has already setup a timer if the user specifies `trainer.max_time` so best we can do is replace that.
                # Working: If only `trainer.max_time` is set - it behaves as a normal PTL timer. If only `exp_manager.max_time_per_run` is set - it behaves as a StateLessTimer. If both are set, it also behaves as a StateLessTimer.
                logging.warning(
                    f'Found a PTL Timer callback, replacing with a StatelessTimer callback. This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run.'
                )
                trainer.callbacks[idx] = StatelessTimer(cfg.max_time_per_run)
                found_ptl_timer = True
                break

        if not found_ptl_timer:
            trainer.max_time = cfg.max_time_per_run
            trainer.callbacks.append(StatelessTimer(cfg.max_time_per_run))

    if is_global_rank_zero():
        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / 'cmd-args.log', 'w', encoding='utf-8') as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / 'git-info.log', 'w', encoding='utf-8') as _file:
                _file.write(f'commit hash: {git_hash}')
                _file.write(get_git_diff())

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / 'nemo_error_log.txt')

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / 'lightning_logs.txt', log_dir / 'nemo_error_log.txt')

    elif trainer.num_nodes * trainer.num_devices > 1:
        # sleep other ranks so rank 0 can finish
        # doing the initialization such as moving files
        time.sleep(cfg.seconds_to_sleep)

    add_handlers_to_mcore_logger()

    return log_dir


def error_checks(trainer: 'pytorch_lightning.Trainer', cfg: Optional[Union[DictConfig, Dict]] = None):
    """
    Checks that the passed trainer is compliant with NeMo and exp_manager's passed configuration. Checks that:
        - Throws error when hydra has changed the working directory. This causes issues with lightning's DDP
        - Throws error when trainer has loggers defined but create_tensorboard_logger or create_wandB_logger
            or create_mlflow_logger or create_dllogger_logger is True
        - Prints error messages when 1) run on multi-node and not Slurm, and 2) run on multi-gpu without DDP
    """
    if HydraConfig.initialized() and get_original_cwd() != os.getcwd():
        raise ValueError(
            "Hydra changed the working directory. This interferes with ExpManger's functionality. Please pass "
            "hydra.run.dir=. to your python script."
        )
    if trainer.logger is not None and (
        cfg.create_tensorboard_logger or cfg.create_wandb_logger or cfg.create_mlflow_logger
    ):
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger, and either "
            f"create_tensorboard_logger: {cfg.create_tensorboard_logger} or create_wandb_logger: "
            f"{cfg.create_wandb_logger} or create_mlflow_logger: {cfg.create_mlflow_logger}"
            f"or create_dllogger_logger: {cfg.create_mlflow_logger} was set to True. "
            "These can only be used if trainer does not already have a logger."
        )
    if trainer.num_nodes > 1 and not check_slurm(trainer):
        logging.error(
            "You are running multi-node training without SLURM handling the processes."
            " Please note that this is not tested in NeMo and could result in errors."
        )
    if trainer.num_devices > 1 and not isinstance(trainer.strategy, DDPStrategy):
        logging.error(
            "You are running multi-gpu without ddp.Please note that this is not tested in NeMo and could result in "
            "errors."
        )


def check_resume(
    trainer: 'pytorch_lightning.Trainer',
    log_dir: str,
    resume_if_exists: bool = False,
    resume_past_end: bool = False,
    resume_ignore_no_checkpoint: bool = False,
    dirpath: str = None,
    resume_from_checkpoint: str = None,
):
    """Checks that resume=True was used correctly with the arguments pass to exp_manager. Sets
    trainer._checkpoint_connector._ckpt_path as necessary.

    Returns:
        log_dir (Path): The log_dir
        exp_dir (str): The base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raises:
        NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
        ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """

    if not log_dir:
        raise ValueError(f"Resuming requires the log_dir {log_dir} to be passed to exp_manager")

    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = resume_from_checkpoint
    if resume_if_exists:
        # Use <log_dir>/checkpoints/ unless `dirpath` is set
        checkpoint_dir = Path(dirpath) if dirpath else Path(Path(log_dir) / "checkpoints")

        # when using distributed checkpointing, checkpoint_dir is a directory of directories
        # we check for this here
        dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
        end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
        last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

        end_checkpoints = end_dist_checkpoints if end_dist_checkpoints else list(checkpoint_dir.rglob("*end.ckpt"))
        last_checkpoints = last_dist_checkpoints if last_dist_checkpoints else list(checkpoint_dir.rglob("*last.ckpt"))

        if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
            if resume_ignore_no_checkpoint:
                warn = f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. "
                if checkpoint is None:
                    warn += "Training from scratch."
                elif checkpoint == resume_from_checkpoint:
                    warn += f"Training from {resume_from_checkpoint}."
                logging.warning(warn)
            else:
                raise NotFoundError(
                    f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
                )
        elif len(end_checkpoints) > 0:
            if resume_past_end:
                if len(end_checkpoints) > 1:
                    if 'mp_rank' in str(end_checkpoints[0]):
                        checkpoint = end_checkpoints[0]
                    else:
                        raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
            else:
                raise ValueError(
                    f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                )
        elif len(last_checkpoints) > 1:
            if 'mp_rank' in str(last_checkpoints[0]) or 'tp_rank' in str(last_checkpoints[0]):
                checkpoint = last_checkpoints[0]
                checkpoint = uninject_model_parallel_rank(checkpoint)
            else:
                raise ValueError(f"Multiple checkpoints {last_checkpoints} that matches *last.ckpt.")
        else:
            checkpoint = last_checkpoints[0]

    # PTL 2.0 supports ckpt_path instead of resume_from_checkpoint as the trainer flag
    if checkpoint is not None:
        trainer.ckpt_path = str(checkpoint)
        logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    if is_global_rank_zero():
        # Check to see if any files exist that need to be moved
        files_to_move = []
        if Path(log_dir).exists():
            for child in Path(log_dir).iterdir():
                if child.is_file():
                    files_to_move.append(child)

        if len(files_to_move) > 0:
            # Move old files to a new folder
            other_run_dirs = Path(log_dir).glob("run_*")
            run_count = 0
            for fold in other_run_dirs:
                if fold.is_dir():
                    run_count += 1
            new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
            new_run_dir.mkdir()
            for _file in files_to_move:
                move(str(_file), str(new_run_dir))


def check_explicit_log_dir(
    trainer: 'pytorch_lightning.Trainer', explicit_log_dir: Union[Path, str], exp_dir: str, name: str, version: str
) -> Tuple[Path, str, str, str]:
    """ Checks that the passed arguments are compatible with explicit_log_dir.

    Returns:
        log_dir (Path): the log_dir
        exp_dir (str): the base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment

    Raise:
        LoggerMisconfigurationError
    """
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and explicit_log_dir: "
            f"{explicit_log_dir} was pass to exp_manager. Please remove the logger from the lightning trainer."
        )
    # Checking only (explicit_log_dir) vs (exp_dir and version).
    # The `name` will be used as the actual name of checkpoint/archive.
    if exp_dir or version:
        logging.error(
            f"exp_manager received explicit_log_dir: {explicit_log_dir} and at least one of exp_dir: {exp_dir}, "
            f"or version: {version}. Please note that exp_dir, name, and version will be ignored."
        )
    if is_global_rank_zero() and Path(explicit_log_dir).exists():
        logging.warning(f"Exp_manager is logging to {explicit_log_dir}, but it already exists.")
    return Path(explicit_log_dir), str(explicit_log_dir), "", ""


def get_log_dir(
    trainer: 'pytorch_lightning.Trainer',
    exp_dir: str = None,
    name: str = None,
    version: str = None,
    explicit_log_dir: str = None,
    use_datetime_version: bool = True,
    resume_if_exists: bool = False,
) -> Tuple[Path, str, str, str]:
    """
    Obtains the log_dir used for exp_manager.

    Returns:
        log_dir (Path): the log_dir
        exp_dir (str): the base exp_dir without name nor version
        name (str): The name of the experiment
        version (str): The version of the experiment
        explicit_log_dir (str): The explicit path to the log folder. Defaults to False.
        use_datetime_version (bool): Uses date and time as the version of the log folder. Defaults to True.
        resume_if_exists (bool): if resume_if_exists of the exp_manager's config is enabled or not. When enabled, the
            version folders would not get created.

    Raise:
        LoggerMisconfigurationError: If trainer is incompatible with arguments
        NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
        ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """
    if explicit_log_dir:  # If explicit log_dir was passed, short circuit
        return check_explicit_log_dir(trainer, explicit_log_dir, exp_dir, name, version)

    # Default exp_dir to ./nemo_experiments if None was passed
    _exp_dir = exp_dir
    if exp_dir is None:
        _exp_dir = str(Path.cwd() / 'nemo_experiments')

    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if exp_dir:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and exp_dir ({exp_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and exp_dir "
                    "must be None."
                )
            _exp_dir = trainer.logger.save_dir
        if name:
            raise LoggerMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name: "
                f"{name} was also passed to exp_manager. If the trainer contains a "
                "logger, exp_manager will use trainer.logger.name, and name passed to exp_manager must be None."
            )
        name = trainer.logger.name
        version = f"version_{trainer.logger.version}"
    # Use user-defined exp_dir, project_name, exp_name, and versioning options
    else:
        name = name or "default"
        version = version or os.environ.get(NEMO_ENV_VARNAME_VERSION, None)

        if not version:
            if resume_if_exists:
                logging.warning(
                    "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
                )
                version = None
            elif is_global_rank_zero():
                if use_datetime_version:
                    version = time.strftime('%Y-%m-%d_%H-%M-%S')
                else:
                    tensorboard_logger = TensorBoardLogger(save_dir=Path(_exp_dir), name=name, version=version)
                    version = f"version_{tensorboard_logger.version}"
                os.environ[NEMO_ENV_VARNAME_VERSION] = "" if version is None else version

    log_dir = Path(_exp_dir) / Path(str(name)) / Path("" if version is None else str(version))
    return log_dir, str(_exp_dir), name, version


def get_git_hash():
    """
    Helper function that tries to get the commit hash if running inside a git folder

    returns:
        Bool: Whether the git subprocess ran without error
        str: git subprocess output or error message
    """
    try:
        return (
            True,
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode(),
        )
    except subprocess.CalledProcessError as err:
        return False, "{}\n".format(err.output.decode("utf-8"))


def get_git_diff():
    """
    Helper function that tries to get the git diff if running inside a git folder

    returns:
        Bool: Whether the git subprocess ran without error
        str: git subprocess output or error message
    """
    try:
        return subprocess.check_output(['git', 'diff'], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return "{}\n".format(err.output.decode("utf-8"))


def configure_loggers(
    trainer: 'pytorch_lightning.Trainer',
    exp_dir: [Path, str],
    log_dir: [Path, str],
    name: str,
    version: str,
    checkpoint_callback_params: dict,
    create_tensorboard_logger: bool,
    summary_writer_kwargs: dict,
    create_wandb_logger: bool,
    wandb_kwargs: dict,
    create_mlflow_logger: bool,
    mlflow_kwargs: dict,
    create_dllogger_logger: bool,
    dllogger_kwargs: dict,
    create_clearml_logger: bool,
    clearml_kwargs: dict,
):
    """
    Creates TensorboardLogger and/or WandBLogger / MLFlowLogger / DLlogger / ClearMLLogger and attach them to trainer.
    Raises ValueError if summary_writer_kwargs or wandb_kwargs are misconfigured.
    """
    # Potentially create tensorboard logger and/or WandBLogger / MLFlowLogger / DLLogger
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}
        elif "log_dir" in summary_writer_kwargs:
            raise ValueError(
                "You cannot pass `log_dir` as part of `summary_writer_kwargs`. `log_dir` is handled by lightning's "
                "TensorBoardLogger logger."
            )
        tensorboard_logger = TensorBoardLogger(save_dir=exp_dir, name=name, version=version, **summary_writer_kwargs)
        logger_list.append(tensorboard_logger)
        logging.info("TensorboardLogger has been set up")

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")

        # Update the wandb save_dir
        if wandb_kwargs.get('save_dir', None) is None:
            wandb_kwargs['save_dir'] = exp_dir
        os.makedirs(wandb_kwargs['save_dir'], exist_ok=True)
        wandb_logger = WandbLogger(version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)
        logging.info("WandBLogger has been set up")

    if create_mlflow_logger:
        mlflow_logger = MLFlowLogger(run_name=version, **mlflow_kwargs)

        logger_list.append(mlflow_logger)
        logging.info("MLFlowLogger has been set up")

    if create_dllogger_logger:
        dllogger_logger = DLLogger(**dllogger_kwargs)

        logger_list.append(dllogger_logger)
        logging.info("DLLogger has been set up")

    if create_clearml_logger:
        clearml_logger = ClearMLLogger(
            clearml_cfg=clearml_kwargs,
            log_dir=log_dir,
            prefix=name,
            save_best_model=checkpoint_callback_params.save_best_model,
        )

        logger_list.append(clearml_logger)
        logging.info("ClearMLLogger has been set up")

    trainer._logger_connector.configure_logger(logger_list)


def configure_checkpointing(
    trainer: 'pytorch_lightning.Trainer',
    log_dir: Path,
    name: str,
    resume: bool,
    params: 'DictConfig',
    create_preemption_callback: bool,
):
    """ Adds ModelCheckpoint to trainer. Raises CheckpointMisconfigurationError if trainer already has a ModelCheckpoint
    callback
    """
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            raise CheckpointMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                "to False, or remove ModelCheckpoint from the lightning trainer"
            )
    # Create the callback and attach it to trainer
    if "filepath" in params:
        if params.filepath is not None:
            logging.warning("filepath is deprecated. Please switch to dirpath and filename instead")
            if params.dirpath is None:
                params.dirpath = Path(params.filepath).parent
            if params.filename is None:
                params.filename = Path(params.filepath).name
        with open_dict(params):
            del params["filepath"]
    if params.dirpath is None:
        params.dirpath = Path(log_dir / 'checkpoints')
    if params.filename is None:
        params.filename = f'{name}--{{{params.monitor}:.4f}}-{{epoch}}'
    if params.prefix is None:
        params.prefix = name
    NeMoModelCheckpoint.CHECKPOINT_NAME_LAST = params.filename + '-last'

    logging.debug(params.dirpath)
    logging.debug(params.filename)
    logging.debug(params.prefix)

    if "val" in params.monitor:
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs != -1
            and trainer.max_epochs < trainer.check_val_every_n_epoch
        ):
            logging.error(
                "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch({trainer.check_val_every_n_epoch}"
                f"). It is very likely this run will fail with ModelCheckpoint(monitor='{params.monitor}') not found "
                "in the returned metrics. Please ensure that validation is run within trainer.max_epochs."
            )
        elif trainer.max_steps is not None and trainer.max_steps != -1:
            logging.warning(
                "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
            )

    checkpoint_callback = NeMoModelCheckpoint(n_resume=resume, **params)
    checkpoint_callback.last_model_path = trainer.ckpt_path or ""
    if 'mp_rank' in checkpoint_callback.last_model_path or 'tp_rank' in checkpoint_callback.last_model_path:
        checkpoint_callback.last_model_path = uninject_model_parallel_rank(checkpoint_callback.last_model_path)
    trainer.callbacks.append(checkpoint_callback)
    if create_preemption_callback:
        # Check if cuda is avialable as preemption is supported only on GPUs
        if torch.cuda.is_available():
            ## By default PreemptionCallback handles SIGTERM. To handle other signals pass the signal in the call as below:
            ## PreemptionCallback(checkpoint_callback, signal.SIGCHLD)
            preemption_callback = PreemptionCallback(checkpoint_callback)
            trainer.callbacks.append(preemption_callback)
        else:
            logging.info("Preemption is supported only on GPUs, disabling preemption")


def check_slurm(trainer):
    try:
        return trainer.accelerator_connector.is_slurm_managing_tasks
    except AttributeError:
        return False


class StatelessTimer(Timer):
    """Extension of PTL timers to be per run."""

    def __init__(self, duration: timedelta = None, interval: str = Interval.step, verbose: bool = True,) -> None:
        super().__init__(duration, interval, verbose)

    # Override PTL Timer's state dict to not store elapsed time information so that we can restore and continue training.
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return

    def _check_time_remaining(self, trainer: "pl.Trainer") -> None:
        super()._check_time_remaining(trainer)
        if trainer.should_stop:
            checkpoint_callback: Optional[NeMoModelCheckpoint] = trainer.checkpoint_callback
            if checkpoint_callback:
                monitor_candidates = checkpoint_callback._monitor_candidates(trainer)
                checkpoint_callback._save_last_checkpoint(trainer, monitor_candidates)
            # Throw this exception to signal to Lightning to terminate gracefully.
            from pytorch_lightning.utilities.exceptions import _TunerExitException

            raise _TunerExitException()


def configure_no_restart_validation_training_loop(trainer: pytorch_lightning.Trainer) -> None:
    if type(trainer.fit_loop.epoch_loop) != _TrainingEpochLoop:
        warnings.warn("Detected custom epoch loop. Skipping no validation on restart support.", UserWarning)
        return
    ## Pass trainer object to avoid trainer getting overwritten as None
    loop = SkipResumeTrainingValidationLoop(trainer, trainer.min_steps, trainer.max_steps)
    trainer.fit_loop.epoch_loop = loop


class SkipResumeTrainingValidationLoop(_TrainingEpochLoop):
    """
    Extend the PTL Epoch loop to skip validating when resuming.
    This happens when resuming a checkpoint that has already run validation, but loading restores
    the training state before validation has run.
    """

    def _should_check_val_fx(self) -> bool:
        if self.restarting and self.global_step % self.trainer.val_check_batch == 0:
            return False
        return super()._should_check_val_fx()


def clean_exp_ckpt(exp_log_dir: Union[str, Path], remove_ckpt: bool = True, remove_nemo: bool = False):
    """
    Helper method that removes Pytorch Lightning .ckpt files or NeMo .nemo files from the checkpoint directory

    Args:
        exp_log_dir: str path to the root directory of the current experiment.
        remove_ckpt: bool, whether to remove all *.ckpt files in the checkpoints directory.
        remove_nemo: bool, whether to remove all *.nemo files in the checkpoints directory.
    """
    exp_log_dir = str(exp_log_dir)

    if remove_ckpt:
        logging.info("Deleting *.ckpt files ...")
        ckpt_files = glob.glob(os.path.join(exp_log_dir, "checkpoints", "*.ckpt"))
        for filepath in ckpt_files:
            os.remove(filepath)
            logging.info(f"Deleted file : {filepath}")

    if remove_nemo:
        logging.info("Deleting *.nemo files ...")
        nemo_files = glob.glob(os.path.join(exp_log_dir, "checkpoints", "*.nemo"))
        for filepath in nemo_files:
            os.remove(filepath)
            logging.info(f"Deleted file : {filepath}")
