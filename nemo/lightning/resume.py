import os
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import lightning_fabric as fl
import pytorch_lightning as pl

from nemo.lightning import io
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import uninject_model_parallel_rank

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == 'nt':
    BasePath = WindowsPath
else:
    BasePath = PosixPath


class Resume(IOMixin):
    def nemo_path(self, model=None) -> Optional[Path]:
        """Returns the checkpoint to resume from."""

    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], model=None):
        if isinstance(trainer, fl.Fabric):
            raise NotImplementedError("Fabric is not supported yet.")

        ckpt_path = self.nemo_path(model)
        if ckpt_path:
            trainer.ckpt_path = ckpt_path
            trainer.checkpoint_callback.last_model_path = ckpt_path


class AutoResume(Resume, io.IOMixin):
    """Class that handles the logic for setting checkpoint paths and restoring from
    checkpoints in NeMo.
    """

    def __init__(
        self,
        path: Optional[str] = None,  ## old resume_from_checkpoint
        dirpath: Optional[str] = None,  ## optional path to checkpoint directory
        import_path: Optional[str] = None,  ## for importing from hf or other checkpoint formats
        adapter_path: Optional[str] = None,
        resume_if_exists: bool = False,
        resume_past_end: bool = False,
        resume_ignore_no_checkpoint: bool = False,
    ):
        """
        Args:
            path (str): Can be used to specify a path to a specific checkpoint file to load from.
                This will override any checkpoint found when resume_if_exists is True.
                Defaults to None
            dirpath (str): Path to the checkpointing directory to restore from. Defaults to <log_dir>/checkpoints
            import_path (str): Path to specify if importing a checkpoint from HF or
                another non-NeMo checkpoint format. If import_path is provided, other arguments
                are unused.
            adapter_path (str): Path to any adapter checkpoints.
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
        if path and import_path:
            raise ValueError("Only one of path or import_path can be set")

        self.path = path
        self.dirpath = dirpath
        self.import_path = import_path
        self.adapter_path = adapter_path
        self.resume_if_exists = resume_if_exists
        self.resume_past_end = resume_past_end
        self.resume_ignore_no_checkpoint = resume_ignore_no_checkpoint

    def nemo_path(self, model=None) -> Optional[Path]:
        from nemo.utils.exp_manager import NotFoundError, _filter_out_unfinished_checkpoints

        if self.import_path:
            if model is None:
                raise ValueError("Model is needed to import checkpoint from HF or other non-NeMo checkpoint format.")
            output = model.import_ckpt(self.import_path)
            if self.adapter_path:
                return AdapterPath(output, adapter_path=Path(self.adapter_path))
            return output

        ### refactored from exp_manager
        checkpoint = None
        app_state = AppState()
        log_dir = app_state.log_dir
        app_state.restore = self.resume_if_exists
        if self.path:
            checkpoint = self.path
        if self.resume_if_exists:
            # Use <log_dir>/checkpoints/ unless `dirpath` is set
            checkpoint_dir = Path(self.dirpath) if self.dirpath else Path(Path(log_dir) / "checkpoints")

            # when using distributed checkpointing, checkpoint_dir is a directory of directories
            # we check for this here
            dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
            end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
            last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

            end_checkpoints = _filter_out_unfinished_checkpoints(end_dist_checkpoints)
            last_checkpoints = _filter_out_unfinished_checkpoints(last_dist_checkpoints)

            if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
                if self.resume_ignore_no_checkpoint:
                    warn = f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. "
                    if checkpoint is None:
                        warn += "Training from scratch."
                    elif checkpoint == self.path:
                        warn += f"Training from {self.path}."
                    logging.warning(warn)
                else:
                    raise NotFoundError(
                        f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
                    )
            elif len(end_checkpoints) > 0:
                if self.resume_past_end:
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
                if any([s for s in ['mp_rank', 'tp_rank', 'fsdp_shard'] if s in str(last_checkpoints[0])]):
                    checkpoint = last_checkpoints[0]
                    checkpoint = uninject_model_parallel_rank(checkpoint)
                else:
                    # Select the checkpoint with the latest modified time
                    checkpoint = sorted(last_checkpoints, key=lambda pth: pth.lstat().st_mtime, reverse=True)[0]
                    logging.warning(
                        f"Multiple checkpoints {last_checkpoints} matches *last.ckpt. Selecting one with the latest modified time."
                    )
            else:
                checkpoint = last_checkpoints[0]

        if checkpoint:
            if self.adapter_path:
                return AdapterPath(checkpoint, adapter_path=Path(self.adapter_path))
            return Path(checkpoint)

        return None


class AdapterPath(BasePath):
    adapter_path: Optional[Path]

    def __new__(cls, *args, adapter_path: Optional[Path] = None, **kwargs):
        output = super().__new__(cls, *args, **kwargs)
        output.adapter_path = adapter_path
        return output
