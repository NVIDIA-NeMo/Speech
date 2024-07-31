# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pytorch_lightning
import torch
from _weakref import proxy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as PTLModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import _is_local_file_protocol
from pytorch_lightning.utilities import rank_zero_info

from nemo.lightning.io.pl import TrainerContext
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import ckpt_to_dir


class ModelCheckpoint(PTLModelCheckpoint):

    UNFINISHED_CHECKPOINT_SUFFIX = "-unfinished"

    def __init__(
        self,
        monitor: Optional[str] = "val_loss",
        verbose: bool = True,
        save_last: Optional[bool] = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,  ## TODO: check support
        mode: str = "min",
        every_n_epochs: int = None,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        save_best_model: bool = False,
        save_on_train_epoch_end: Optional[bool] = False,  # Save after training, not after validation
        enable_nemo_ckpt_io: bool = True,
        try_restore_best_ckpt: bool = True,
        **kwargs,
    ):
        self.save_best_model = save_best_model
        self.previous_best_path = ""
        self.enable_nemo_ckpt_io = enable_nemo_ckpt_io
        # Checkpoints which removal is deferred until async save is done.
        # Each element of `deferred_ckpts_to_remove` is a growing list
        # that `self._remove_checkpoint` adds to. Once `self._save_checkpoint`
        # is called, the last element is frozen and a new element is added.
        self.deferred_ckpts_to_remove: List[List[str]] = []
        self.try_restore_best_ckpt = try_restore_best_ckpt

        # Call the parent class constructor with the remaining kwargs.
        super().__init__(
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs,
        )

    def on_train_start(self, trainer, pl_module):
        from nemo.utils.exp_manager import get_git_diff, get_git_hash
        from nemo.utils.get_rank import is_global_rank_zero
        from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger

        app_state = AppState()
        if self.save_top_k != -1 and app_state.restore:
            logging.debug("Checking previous runs")
            self.nemo_topk_check_previous_run()

        if is_global_rank_zero():
            log_dir = app_state.log_dir

            # Check to see if any files exist that need to be moved
            files_to_move = app_state.files_to_move

            if len(files_to_move) > 0:
                # Move old files to a new folder
                other_run_dirs = Path(log_dir).glob("run_*")
                run_count = 0
                for fold in other_run_dirs:
                    if fold.is_dir():
                        run_count += 1
                new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
                if not new_run_dir.exists():
                    new_run_dir.mkdir()
                    for _file in files_to_move:
                        shutil.move(str(_file), str(new_run_dir))

            # Move files_to_copy to folder and add git information if present
            if app_state.files_to_copy:
                for _file in app_state.files_to_copy:
                    src_path = Path(_file)
                    dst_path = Path(log_dir) / src_path.name
                    if not dst_path.exists():
                        shutil.copy(src_path, dst_path)

            # Create files for cmd args and git info
            if app_state.cmd_args:
                cmd_args_file = log_dir / 'cmd-args.log'
                if not cmd_args_file.exists():
                    with open(cmd_args_file, 'w', encoding='utf-8') as _file:
                        _file.write(" ".join(app_state.cmd_args))

            # Try to get git hash
            git_repo, git_hash = get_git_hash()
            if git_repo:
                git_info_file = log_dir / 'git-info.log'
                if not git_info_file.exists():
                    with open(git_info_file, 'w', encoding='utf-8') as _file:
                        _file.write(f'commit hash: {git_hash}\n')
                        _file.write(get_git_diff())

            # Add err_file logging to global_rank zero
            logging.add_err_file_handler(log_dir / 'nemo_error_log.txt')

            # Add lightning file logging to global_rank zero
            add_filehandlers_to_pl_logger(log_dir / 'lightning_logs.txt', log_dir / 'nemo_error_log.txt')
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def nemo_topk_check_previous_run(self):
        try:
            self.best_k_models
            self.kth_best_model_path
            self.best_model_score
            self.best_model_path
        except AttributeError:
            raise AttributeError(
                "Lightning's ModelCheckpoint was updated. NeMo's ModelCheckpoint will need an update."
            )
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""

        checkpoints = list(path for path in self._saved_checkpoint_paths if not self._is_ema_filepath(path))
        for checkpoint in checkpoints:
            checkpoint = str(checkpoint)
            if checkpoint[-10:] == '-last.ckpt' or checkpoint[-5:] == '-last':
                continue
            index = checkpoint.find(self.monitor) + len(self.monitor) + 1  # Find monitor in str + 1 for '='
            if index != len(self.monitor):
                match = re.search('[A-z]', checkpoint[index:])
                if match:
                    value = checkpoint[index : index + match.start() - 1]  # -1 due to separator hyphen
                    self.best_k_models[checkpoint] = float(value)
        if len(self.best_k_models) < 1:
            return  # No saved checkpoints yet

        _reverse = False if self.mode == "min" else True

        best_k_models = sorted(self.best_k_models, key=self.best_k_models.get, reverse=_reverse)

        # This section should be ok as rank zero will delete all excess checkpoints, since all other ranks are
        # instantiated after rank zero. models_to_delete should be 0 for all other ranks.
        models_to_delete = len(best_k_models) - self.save_top_k
        models_to_delete = max(0, models_to_delete)
        logging.debug(f'Number of models to delete: {models_to_delete}')

        # If EMA enabled, delete the additional EMA weights
        ema_enabled = self._has_ema_ckpts(self._saved_checkpoint_paths)

        for _ in range(models_to_delete):
            model = best_k_models.pop(-1)
            self.best_k_models.pop(model)
            self._del_model_without_trainer(model)
            if ema_enabled and self._fs.exists(self._ema_format_filepath(model)):
                self._del_model_without_trainer(self._ema_format_filepath(model))
            logging.debug(f"Removed checkpoint: {model}")

        self.kth_best_model_path = best_k_models[-1]
        self.best_model_path = best_k_models[0]
        self.best_model_score = self.best_k_models[self.best_model_path]

    def _remove_invalid_entries_from_topk(self):
        # Removes invalid (incomplete or not existing) checkpoints from topk checkpoints.
        # This might be needed if the checkpointing was abruptly terminated.
        def __is_ckpt_ok(ckpt_path: str) -> bool:
            exists = os.path.isdir(ckpt_path.removesuffix('.ckpt'))
            return exists and not self.is_checkpoint_unfinished(ckpt_path)

        self.best_k_models = {k: v for k, v in self.best_k_models.items() if __is_ckpt_ok(k)}
        if len(self.best_k_models) > 0:
            reverse_arr = self.mode != "min"
            best_k_models_arr = sorted(self.best_k_models, key=self.best_k_models.get, reverse=reverse_arr)
            self.kth_best_model_path = best_k_models_arr[-1]
            self.kth_value = self.best_k_models[self.kth_best_model_path]
            self.best_model_path = best_k_models_arr[0]
            self.best_model_score = self.best_k_models[self.best_model_path]
        else:
            self.kth_best_model_path = ""
            self.kth_value = None
            self.best_model_path = ""
            self.best_model_score = None

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._remove_invalid_entries_from_topk()

    def setup(self, trainer, *args, **kwargs) -> None:
        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero():
            logging.debug("Removing unfinished checkpoints if any...")
            ModelCheckpoint._remove_unfinished_checkpoints(self.dirpath)
        # Ensure that all ranks continue with unfinished checkpoints removed
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.async_save = getattr(trainer.strategy, "async_save", False)
        super().setup(trainer, *args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        output = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        return output

    def on_train_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return None

        # check if we need to save a last checkpoint manually as validation isn't always run based on the interval
        if self.save_last and trainer.val_check_interval != 0:
            should_save_last_checkpoint = False
            if isinstance(trainer.val_check_interval, float) and trainer.val_check_interval % trainer.global_step != 0:
                should_save_last_checkpoint = True
            if isinstance(trainer.val_check_interval, int) and trainer.global_step % trainer.val_check_interval != 0:
                should_save_last_checkpoint = True
            if should_save_last_checkpoint:
                monitor_candidates = self._monitor_candidates(trainer)
                if self.last_model_path == self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST):
                    logging.debug(f'Last checkpoint {self.last_model_path} already saved')
                else:
                    super()._save_last_checkpoint(trainer, monitor_candidates)
        # Call parent on_train_end() to save the -last checkpoint
        super().on_train_end(trainer, pl_module)

        # Load the best model and then re-save it
        if self.save_best_model:
            # wait for all processes
            trainer.strategy.barrier("SaveBestCheckpointConnector.resume_end")
            if self.best_model_path == "":
                logging.warning(
                    f"{self} was told to save the best checkpoint at the end of training, but no saved checkpoints "
                    "were found. Saving latest model instead."
                )

            else:
                if os.path.isdir(self.best_model_path.split('.ckpt')[0]):
                    self.best_model_path = self.best_model_path.split('.ckpt')[0]
                if self.try_restore_best_ckpt:
                    self.best_model_path = trainer.strategy.broadcast(self.best_model_path)
                    trainer._checkpoint_connector.restore(self.best_model_path)

    def _del_model_without_trainer(self, filepath: str) -> None:
        from nemo.utils.get_rank import is_global_rank_zero

        filepath = Path(filepath)

        if is_global_rank_zero():
            try:
                dist_ckpt = ckpt_to_dir(filepath)
                shutil.rmtree(dist_ckpt, ignore_errors=True)
                logging.info(f"Removed distributed checkpoint: {dist_ckpt}")
            except:
                logging.info(f"Tried to remove distributed checkpoint: {dist_ckpt} but failed.")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _ema_callback(self, trainer: 'pytorch_lightning.Trainer'):
        from nemo.collections.common.callbacks import EMA

        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    @staticmethod
    def format_checkpoint_unfinished_marker_path(checkpoint_path: Union[Path, str]) -> Path:
        """Format the path to the unfinished checkpoint marker file.

        If the marker file exists, corresponding checkpoint is considered unfinished/incomplete.
        NOTE: Marker path for the EMA checkpoint part is the same as for the original checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.

        Returns:
            Path to the unfinished checkpoint marker file.
        """
        marker_filepath = str(checkpoint_path).removesuffix(".ckpt")
        marker_filepath = marker_filepath.removesuffix("-EMA")
        return Path(marker_filepath + ModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX)

    @staticmethod
    def is_checkpoint_unfinished(checkpoint_path: Union[Path, str]) -> bool:
        """Check if the checkpoint is unfinished.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.

        Returns:
            True if the checkpoint is unfinished, False otherwise.
        """
        return ModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path).exists()

    @staticmethod
    def set_checkpoint_unfinished_marker(checkpoint_path: Union[Path, str], barrier_after=False) -> None:
        """Marks given checkpoint as unfinished.

        Args:
            checkpoint_filepath: Path to the checkpoint file or dir.
              Does not need to exist.
            barrier_after: Synchronize ranks after writing the marker file.
              Defaults to False.
        """
        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero():
            marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
        if barrier_after and torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def remove_checkpoint_unfinished_marker(checkpoint_path: Union[Path, str], barrier_before=False) -> None:
        """Clear unfinished marker for given checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file or dir.
              Does not need to exist.
            barrier_before: Synchronize ranks before removing the marker file.
              Defaults to False.
        """
        from nemo.utils.get_rank import is_global_rank_zero

        try:
            if barrier_before and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_global_rank_zero():
                marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)
                if marker_path.exists():
                    marker_path.unlink()
        except:
            return

    def file_exists(self, filepath: str, trainer: "pytorch_lightning.Trainer", check_dist_ckpt: bool = True) -> bool:
        """Checks if a file or a file without a suffix (distributed checkpoint) exists."""
        exists = self._fs.exists(filepath) or (check_dist_ckpt and self._fs.exists(ckpt_to_dir(filepath)))
        return trainer.strategy.broadcast(exists)

    def _save_checkpoint(self, trainer: 'pytorch_lightning.Trainer', filepath: str) -> None:
        # barrier_after=True, so all ranks continue after the unfinished checkpoint marker is placed.
        # if anything goes wrong during checkpointing, we should be able to detect that data is incomplete.
        self.set_checkpoint_unfinished_marker(filepath, barrier_after=True)
        ema_callback = self._ema_callback(trainer)

        self._last_global_step_saved = trainer.global_step

        if ema_callback is not None:
            if self.async_save:
                raise ValueError('async_save with EMA not supported')
            with ema_callback.save_original_optimizer_state(trainer):
                super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
            self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)
        else:
            # Async save passes the finalization function to checkpoint_io,
            # sync save calls the finalization function immediately after save.
            finalize_fn = self._get_finalize_save_checkpoint_callback(trainer, filepath, trainer.global_step)
            if self.async_save:
                checkpoint_io = trainer.strategy.checkpoint_io
                from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO

                if not isinstance(checkpoint_io, AsyncFinalizableCheckpointIO):
                    raise ValueError('Async save requires async compatible CheckpointIO')
                storage_options = dict(finalize_fn=finalize_fn)
                # Each upcoming ckpt removal request will be executed as part of this save finalization
                self.deferred_ckpts_to_remove.append([])
            else:
                storage_options = None
            trainer.save_checkpoint(filepath, self.save_weights_only, storage_options=storage_options)

            ## NOTE: saving context happens synchronously always
            from nemo.utils.get_rank import is_global_rank_zero

            if self.enable_nemo_ckpt_io and is_global_rank_zero():
                TrainerContext.from_trainer(trainer).io_dump(ckpt_to_dir(filepath))
            if self.async_save:
                logging.info(f'Scheduled async checkpoint save for {filepath}')
            else:
                finalize_fn()

    def _get_finalize_save_checkpoint_callback(
        self, trainer: 'pytorch_lightning.Trainer', filepath: str, global_step: int
    ):
        """Creates a callback that can be used to finalize async (and sync) ckpt saves."""

        def _cb():
            logging.debug(f'Finalize callback called for step {global_step}, filepath {filepath}')
            self._last_checkpoint_saved = filepath

            # notify loggers
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.after_save_checkpoint(proxy(self))

            # barrier_before=True, so all ranks synchronize before removing the unfinished checkpoint marker
            # we don't want to remove the marker until all checkpointing is done.
            self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)

            if not self.async_save:
                return

            logging.info(f'Async checkpoint save for step {global_step} ({filepath}) finalized successfully.')

            # Remove checkpoints marked for removal by `self._remove_checkpoint`
            # For each finalization there is exactly one entry in self.deferred_ckpts_to_remove
            assert self.deferred_ckpts_to_remove
            ckpts_to_remove = self.deferred_ckpts_to_remove.pop(0)
            logging.debug(f'Checkpoints to remove: {ckpts_to_remove}')
            for ckpt_to_remove in ckpts_to_remove:
                self._remove_checkpoint(trainer, ckpt_to_remove, override_async=True)

        return _cb

    def _remove_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str, override_async=False) -> None:
        """Performs checkpoint removal.

        With async save, `self._remove_checkpoint` is called before the checkpoint
        is actually finished so we can't remove it. Instead we add it to
        `self.deferred_ckpts_to_remove` for future removal.
        """
        if self.async_save and not override_async:
            # Register checkpoint removal in the last (active) checkpoint removal list
            self.deferred_ckpts_to_remove[-1].append(filepath)
            return
        # barrier_after=True, so all ranks continue after the unfinished checkpoint marker is placed.
        # if anything goes wrong during removal, we should be able to detect that data is incomplete.
        self.set_checkpoint_unfinished_marker(filepath, barrier_after=True)
        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.

            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)
        # barrier_before=True, so all ranks synchronize before removing the unfinished checkpoint marker
        # we don't want to remove the marker until the checkpoint is actually removed.
        self.remove_checkpoint_unfinished_marker(filepath, barrier_before=True)

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f'-EMA{self.FILE_EXTENSION}')

    def _has_ema_ckpts(self, checkpoints: Iterable[Path]) -> bool:
        return any(self._is_ema_filepath(checkpoint_path) for checkpoint_path in checkpoints)

    def _is_ema_filepath(self, filepath: Union[Path, str]) -> bool:
        return str(filepath).endswith(f'-EMA{self.FILE_EXTENSION}')

    @property
    def _saved_checkpoint_paths(self) -> Iterable[Path]:
        # distributed checkpoints are directories so we check for them here
        # we filter out unfinished checkpoints, these should be deleted during next cleanup
        dist_checkpoints = [d for d in Path(self.dirpath).glob("*") if d.is_dir()]
        if dist_checkpoints:
            return filter(lambda p: not self.is_checkpoint_unfinished(p), dist_checkpoints)
        else:
            checkpoint_files = [f for f in Path(self.dirpath).rglob("*.ckpt")]
            return filter(lambda p: not self.is_checkpoint_unfinished(p), checkpoint_files)

    @staticmethod
    def _remove_unfinished_checkpoints(checkpoint_dir: Union[Path, str]) -> None:
        from nemo.utils.get_rank import is_global_rank_zero

        # Delete unfinished checkpoints from the filesystems.
        # "Unfinished marker" files are removed as well.

        if not is_global_rank_zero():
            raise AssertionError("_remove_unfinished_checkpoints should run only on rank 0")

        checkpoint_dir = Path(checkpoint_dir)

        existing_marker_filepaths = {
            f.resolve() for f in checkpoint_dir.glob(f"*{ModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}") if f.is_file()
        }

        checkpoint_filepaths = {f.resolve() for f in checkpoint_dir.rglob("*.ckpt")}
        for ckpt_filepath in checkpoint_filepaths:
            possible_marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt_filepath)
            if possible_marker_path in existing_marker_filepaths:
                logging.warning(f'Removing unfinished checkpoint: {ckpt_filepath}')
                os.remove(ckpt_filepath)

        # some directories might be distributed checkpoints, we remove these if they have a unfinished marker
        all_dirpaths = {d.resolve() for d in checkpoint_dir.glob("*") if d.is_dir()}
        for ckpt_dirpath in all_dirpaths:
            possible_marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt_dirpath)
            if possible_marker_path in existing_marker_filepaths:
                logging.warning(f'Removing unfinished dist checkpoint: {ckpt_dirpath}')
                shutil.rmtree(ckpt_dirpath)

        # delete markers
        for marker_path in existing_marker_filepaths:
            os.remove(marker_path)

    def _should_remove_checkpoint(self, trainer: "pl.Trainer", previous: str, current: str) -> bool:
        """Checks if the previous checkpoint should be deleted.
        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local
            and the resumed from checkpoint is not the last checkpoint
        """
        if previous == current:
            return False
        if not _is_local_file_protocol(previous):
            return True
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None

        if resume_path is not None and previous == resume_path:
            if str(current).endswith("-last.ckpt") and resume_path.name.endswith("-last.ckpt"):
                # delete the previous `-last.ckpt` checkpoint when current saved checkpoint is also `-last.ckpt`, if they're in the same directory
                pass
            else:
                return False
        if self.dirpath is None:
            raise ValueError(f"{self.__class__}.dirpath is None.")
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents
