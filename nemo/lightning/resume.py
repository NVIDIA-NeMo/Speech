# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import lightning.fabric as fl
import lightning.pytorch as pl

from nemo.lightning import io
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import uninject_model_parallel_rank
from nemo.utils.msc_utils import import_multistorageclient, is_multistorageclient_url

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == "nt":
    BasePath = WindowsPath
else:
    BasePath = PosixPath


def _try_restore_tokenizer(model, ckpt_path):
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.lightning.io import load_context

    try:
        tokenizer = load_context(ckpt_path, "model.tokenizer")
    except ValueError as e:
        logging.warning(
            f"Encountered error while trying to restore tokenizer. Tokenizer is not restored. " f"Original error: {e}"
        )
        return model

    if isinstance(tokenizer, TokenizerSpec):
        model.tokenizer = tokenizer
        model.__io__.tokenizer = tokenizer.__io__
    else:
        # Ignore if the ckpt doesn't have a tokenizer. type(tokenizer)==TrainerContext in this case.
        logging.warning("Checkpoint does not have model.tokenizer field. Tokenizer is not restored.")

    return model


@dataclass(kw_only=True)
class AutoResume:
    """Class that handles the logic for setting checkpoint paths and restoring from
    checkpoints in NeMo.

    Attributes:
        restore_config (Optional[RestoreConfig]): Optional config for selectively restoring specific parts like model
            weights, optimizer states, etc.
            If the config contains a path from HF or another non-NeMo checkpoint format, the checkpoint will be
            automatically converted to a NeMo compatible format.
            resume_from_folder or the run's log_dir takes precedence over restore_config.
        resume_from_directory (str): Path to the checkpointing directory to restore from.
        resume_from_path (str): Path to a specific checkpoint to restore from.
        resume_if_exists (bool): Whether this experiment is resuming from a previous run. If
            True, it sets trainer._checkpoint_connector._ckpt_path so that the trainer should
            auto-resume. exp_manager will move files under log_dir to log_dir/run_{int}.
            Defaults to False.
        resume_past_end (bool): By default, AutoResume throws an error if resume_if_exists is
            True and a checkpoint matching ``*end.ckpt`` indicating a previous training run
            fully completed. Setting resume_past_end=True disables this behavior and loads the
            last checkpoint.
        resume_ignore_no_checkpoint (bool): AutoResume throws an error if resume_if_exists is
            True and no checkpoint could be found. Setting resume_ignore_no_checkpoint=True
            disables this behavior, in which case exp_manager will print a message and
            continue without restoring.
    """

    restore_config: Optional[RestoreConfig] = None
    resume_from_directory: Optional[str] = None
    resume_from_path: Optional[str] = None
    resume_if_exists: bool = False
    resume_past_end: bool = False
    resume_ignore_no_checkpoint: bool = False

    WEIGHTS_PATH = "weights"

    def get_weights_path(self, path) -> Path:
        """Returns the path to the weights directory within the specified path.

        Args:
            path: The checkpoint directory path

        Returns:
            Path: A Path object pointing to the weights directory
        """
        return path / self.WEIGHTS_PATH

    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], model=None):
        """Sets up checkpoint restoration for the Pytorch Lightning trainer.

        This method configures the trainer with the appropriate checkpoint path for resuming
        training and handles loading model artifacts like tokenizers when specified.

        Args:
            trainer: The PyTorch Lightning trainer or Fabric instance
            model: Optional model instance to load artifacts into

        Raises:
            NotImplementedError: If trainer is a Fabric instance (not yet supported)
        """
        if isinstance(trainer, fl.Fabric):
            raise NotImplementedError("Fabric is not supported yet.")

        trainer_ckpt_path = self.get_trainer_ckpt_path(model)
        if trainer_ckpt_path:
            trainer.ckpt_path = trainer_ckpt_path
            trainer.checkpoint_callback.last_model_path = trainer_ckpt_path
            # Load artifacts
            if getattr(self.restore_config, "load_artifacts", False):
                if isinstance(trainer_ckpt_path, AdapterPath):
                    # load tokenizer from the base model during peft resume, in case the first peft checkpoint
                    # is deleted before the current peft checkpoint is saved
                    context_path = trainer_ckpt_path.base_model_path / "context"
                    if not context_path.exists():
                        context_path = trainer_ckpt_path.base_model_path
                else:
                    context_path = self.get_context_path(model)
                model = _try_restore_tokenizer(model, context_path)

        elif self.restore_config:
            new_path = self._extract_path(
                path=self.restore_config.path,
            )
            assert not isinstance(new_path, AdapterPath), "AdapterPath is not supported for restore_config"
            self.restore_config.path = str(new_path)
            trainer.strategy.restore_config = self.restore_config
            # Load artifacts
            if self.restore_config.load_artifacts:
                if isinstance(new_path, AdapterPath):
                    context_path = Path(new_path.base_model_path) / "context"
                else:
                    context_path = new_path / "context"
                if not context_path.is_dir():
                    context_path = new_path

                _try_restore_tokenizer(model, context_path)

    def _extract_path(self, path: str) -> BasePath:
        if "://" in path:
            assert path.startswith("nemo://"), "Only NeMo based paths starting with nemo:// are currently supported."
            _, _path = path.split("://")
            new_path = os.path.join(NEMO_MODELS_CACHE, _path)
        else:
            new_path = path

        if isinstance(new_path, str):
            new_path = Path(new_path)

        return new_path

    def _get_base_model_path_for_adapter(self, adapter_meta_path, model):
        with open(adapter_meta_path, "r") as f:
            metadata = json.load(f)

        # Use the model_ckpt_path from metadata directly
        base_model_path = Path(metadata["model_ckpt_path"])

        # If base_model_path points to a specific checkpoint file, use its parent directory
        if not base_model_path.is_dir() and base_model_path.exists():
            base_model_path = base_model_path.parent

        return base_model_path

    def _find_trainer_ckpt_path(self) -> Optional[Path]:
        from nemo.utils.exp_manager import NotFoundError, _filter_out_unfinished_checkpoints

        app_state = AppState()
        log_dir = app_state.log_dir

        checkpoint = None

        # Use <log_dir>/checkpoints/ unless `dirpath` is set
        if self.resume_from_directory:
            if is_multistorageclient_url(self.resume_from_directory):
                msc = import_multistorageclient()
                checkpoint_dir = msc.Path(self.resume_from_directory)
            else:
                checkpoint_dir = Path(self.resume_from_directory)
        elif log_dir is not None:
            checkpoint_dir = Path(Path(log_dir) / "checkpoints")
        else:  # ie. if log_dir is None
            return None

        # when using distributed checkpointing, checkpoint_dir is a directory of directories
        # we check for this here
        dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
        end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
        last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

        end_chkpt_cnt = len(end_dist_checkpoints)
        end_checkpoints = _filter_out_unfinished_checkpoints(end_dist_checkpoints)
        finished_end_chkpt_cnt = len(end_checkpoints)
        if end_chkpt_cnt > 0 and finished_end_chkpt_cnt == 0:
            raise ValueError(
                "End checkpoint is unfinished and cannot be used to resume the training."
                " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                " restarting from scratch."
            )

        last_chkpt_cnt = len(last_dist_checkpoints)
        last_checkpoints = _filter_out_unfinished_checkpoints(last_dist_checkpoints)
        finished_last_chkpt_cnt = len(last_checkpoints)
        if last_chkpt_cnt > 0 and finished_last_chkpt_cnt == 0:
            raise ValueError(
                "Last checkpoint is unfinished and cannot be used to resume the training."
                " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                " restarting from scratch. Hint: Iteration number can be added to the checkpoint name pattern"
                " to maximize chance that there is at least one finished last checkpoint to resume from."
            )

        if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
            if self.resume_ignore_no_checkpoint:
                message = (
                    f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir "
                    f":{checkpoint_dir}. "
                )
                if not self.restore_config:
                    logging.warning(message + "Training from scratch.")
                else:
                    logging.info(message + "Trying to resume from RestoreConfig.")
            else:
                if self.restore_config:
                    # resume_if_exists is True but run is not resumable. Do not fail and try to do selective restore
                    # later instead.
                    return None
                else:
                    raise NotFoundError(
                        f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir "
                        f":{checkpoint_dir}. Cannot resume."
                    )
        elif len(end_checkpoints) > 0:
            if not self.resume_past_end:
                raise ValueError(
                    f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                )

            if len(end_checkpoints) > 1:
                if "mp_rank" in str(end_checkpoints[0]):
                    checkpoint = end_checkpoints[0]
                else:
                    raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
        elif len(last_checkpoints) > 1:
            if any([s for s in ["mp_rank", "tp_rank", "fsdp_shard"] if s in str(last_checkpoints[0])]):
                checkpoint = last_checkpoints[0]
                checkpoint = uninject_model_parallel_rank(checkpoint)
            else:
                # Select the checkpoint with the latest modified time
                checkpoint = sorted(last_checkpoints, key=lambda pth: pth.lstat().st_mtime, reverse=True)[0]
                logging.warning(
                    f"Multiple checkpoints {last_checkpoints} matches *last.ckpt. Selecting one with the latest "
                    f"modified time."
                )
        else:
            checkpoint = last_checkpoints[0]

        return checkpoint

    def get_context_path(self, model: Optional[io.ConnectorMixin] = None) -> Optional[Path]:
        """Retrieves the path to the context directory of a checkpoint.

        The context directory contains serialized objects like tokenizers. This method
        handles both cases where the context is directly in the checkpoint directory
        or in a subdirectory called "context".

        Args:
            model: Optional model instance

        Returns:
            Optional[Path]: Path to the context directory if found, None otherwise
        """
        checkpoint = None
        app_state = AppState()
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            checkpoint = self._find_trainer_ckpt_path()

        if checkpoint:
            maybe_context_path = checkpoint / "context"
            if maybe_context_path.is_dir():
                checkpoint = maybe_context_path
        return checkpoint

    def get_trainer_ckpt_path(self, model: Optional[io.ConnectorMixin] = None) -> Optional[Path]:
        """Resolves the path to a checkpoint for resuming training.

        This method handles various checkpoint sources with the following priority:
        1. Explicit path specified in resume_from_path
        2. Automatic discovery in the checkpoint directory when resume_if_exists=True

        For adapter checkpoints (PEFT), it also retrieves the base model path from metadata.

        Args:
            model: Optional model instance

        Returns:
            Optional[Path]: Path to the checkpoint if found, or AdapterPath for PEFT checkpoints,
                           or None if no checkpoint is found or needed
        """
        if self.resume_from_path:
            if is_multistorageclient_url(self.resume_from_path):
                msc = import_multistorageclient()
                resume_from_path = msc.Path(self.resume_from_path)
            else:
                resume_from_path = Path(self.resume_from_path)

            maybe_weights_path = self.get_weights_path(resume_from_path)
            if maybe_weights_path.is_dir():
                adapter_meta_path = maybe_weights_path / ADAPTER_META_FILENAME
                if adapter_meta_path.exists():
                    # the resume_from_path is an adapter checkpoint
                    base_model_path = self._get_base_model_path_for_adapter(adapter_meta_path, model)
                    return AdapterPath(Path(self.resume_from_path), base_model_path=base_model_path)
                else:
                    # the resume_from_path is not PEFT checkpoint
                    return maybe_weights_path
            else:
                return self.resume_from_path

        checkpoint = None
        app_state = AppState()
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            checkpoint = self._find_trainer_ckpt_path()

        if checkpoint:
            maybe_weights_path = self.get_weights_path(checkpoint)
            if maybe_weights_path.is_dir():
                checkpoint = maybe_weights_path

        if checkpoint:
            adapter_meta_path = checkpoint / ADAPTER_META_FILENAME
            if adapter_meta_path.exists():
                base_model_path = self._get_base_model_path_for_adapter(adapter_meta_path, model)
                return AdapterPath(checkpoint, base_model_path=base_model_path)
            else:
                return checkpoint

        return None


class AdapterPath(BasePath):
    """Path object for adapter paths which include a field for the base model the adapters are trained on
    to facilitate model loading."""

    base_model_path: Optional[Path]

    def __new__(cls, *args, base_model_path: Optional[Path] = None, **kwargs):
        output = super().__new__(cls, *args, **kwargs)
        output.base_model_path = base_model_path
        return output

    def __repr__(self):
        return "{}({!r}, base_model_path={})".format(self.__class__.__name__, self.as_posix(), self.base_model_path)
