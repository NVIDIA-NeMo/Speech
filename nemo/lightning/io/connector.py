import logging
import os
import shutil
from pathlib import Path, PosixPath, PurePath, WindowsPath
from typing import Generic, Optional, Tuple, TypeVar

import pytorch_lightning as pl
from filelock import FileLock, Timeout
from pytorch_lightning.trainer.states import TrainerFn

from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ckpt_to_weights_subdir

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == 'nt':
    BasePath = WindowsPath
else:
    BasePath = PosixPath


SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")


class Connector(BasePath, Generic[SourceT, TargetT]):
    """
    A generic connector class that provides a framework for transforming a source type (SourceT)
    to a target type (TargetT) while handling file paths based on the operating system.

    Attributes
    ----------
        default_path (Optional[Path]): A default path used when no path is explicitly provided.

    Methods
    -------
        init() -> TargetT:
            Should be implemented to initialize the target type from the source type.

        apply(output_path: Path) -> Path:
            Should be implemented to apply the transformation and save the result at the output path.

        __new__(cls, *args, **kwargs) -> 'Connector':
            Creates a new instance of the connector, using default_path if no path is provided.

        __call__(output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
            Processes the transformation and handles file operations like overwriting.

        local_path(base_path: Optional[Path] = None) -> Path:
            Computes the local path for storage based on a base path or a default cache home.

        is_in_cache(base_path: Optional[Path] = None) -> bool:
            Checks if the transformed data is already cached at the specified base path.
    """

    default_path = None
    LOCK_TIMEOUT = 1200

    def init(self) -> TargetT:
        raise NotImplementedError()

    def apply(self, output_path: Path) -> Path:
        raise NotImplementedError()

    def __new__(cls, *args, **kwargs):
        if cls.default_path is not None and not args and 'path' not in kwargs:
            # If default_path is set and no arguments are provided, use default_path as the argument
            return super().__new__(cls, cls.default_path)

        return super().__new__(cls, *args, **kwargs)

    def __call__(self, output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
        _output_path = output_path or self.local_path()
        lock_path = _output_path.with_suffix(_output_path.suffix + '.lock')
        lock = FileLock(lock_path)

        # Check if the lock file exists and set overwrite to False if it does
        if lock_path.exists():
            overwrite = False

        try:
            with lock.acquire(timeout=self.LOCK_TIMEOUT):
                if overwrite and _output_path.exists():
                    shutil.rmtree(_output_path)

                if not _output_path.exists():
                    to_return = self.apply(_output_path)
                    _output_path = to_return or _output_path

        except Timeout:
            logging.error(f"Timeout occurred while trying to acquire the lock for {_output_path}")
            raise

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

        finally:
            # Delete the lock file if it exists
            if lock_path.exists():
                try:
                    os.remove(lock_path)
                except OSError as e:
                    logging.warning(f"Failed to remove lock file {lock_path}: {e}")

        return _output_path

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = base_path
        else:
            from nemo.lightning.base import NEMO_CACHE_HOME

            _base = Path(NEMO_CACHE_HOME)

        return _base / str(self).replace("://", "/")

    def is_in_cache(self, base_path: Optional[Path] = None) -> bool:
        return self.local_path(base_path=base_path).exists()


class ModelConnector(Connector, Generic[SourceT, TargetT]):
    """
    A specialized connector that extends the generic Connector to handle model-specific operations
    such as setup, save, and load using the Lightning framework.

    Methods
    -------
        nemo_setup(model: pl.LightningModule, trainer: Optional[pl.Trainer] = None) -> pl.Trainer:
            Sets up the model and trainer using a specified strategy, preparing it for training or inference.

        nemo_save(output_path: Path, trainer: pl.Trainer):
            Saves the model's state to the specified path using the trainer's current strategy.

        nemo_load(path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True) -> Tuple[Any, pl.Trainer]:
            Loads a model from the specified path, optionally using a CPU-focused strategy, and returns the model and trainer.
    """

    def nemo_setup(self, model: pl.LightningModule, trainer: Optional[pl.Trainer] = None) -> pl.Trainer:
        """
        Sets up the model and trainer using a specified strategy, preparing it for training or inference.

        Args:
            model (pl.LightningModule): The model to be set up.
            trainer (Optional[pl.Trainer]): The trainer to be used, if not provided a new one will be created.
        Returns
        -------
            pl.Trainer: The trainer configured with the model and strategy.
        """
        from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib

        _trainer = trainer or Trainer(
            devices=1,
            accelerator="cpu",
            strategy=MegatronStrategy(ckpt_save_optimizer=False, always_save_context=True),
        )
        # Note: set trainer to fitting state to avoid the following code path. Feel free to refactor if we no longer
        #  need to avoid this:
        #  https://github.com/NVIDIA/NeMo/blob/e35a6592f53ee34b1ec2fc3f1e009dd1ebc79e65/nemo/lightning/pytorch/strategies/megatron_strategy.py#L346-L349
        _trainer.state.fn = TrainerFn.FITTING  # needed for proper save.

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()

        if not model.state_dict():
            _trainer.strategy.lazy_init = True
            with _trainer.init_module(), _strategy_lib.megatron_lazy_init_context(model.config):
                model.configure_model()
        return _trainer

    def nemo_save(self, output_path: Path, trainer: pl.Trainer, dump_io: bool = True) -> None:
        """
        Saves the model's state to the specified path using the trainer's current strategy.

        Args:
            output_path (Path): The path where the model checkpoint will be saved.
            trainer (pl.Trainer): The trainer with the strategy to save the model.
            dump_io (bool): If True, the IO configuration will be saved to the output path.
        """
        # Import here to avoid circular import

        trainer.strategy._setup_optimizers = False
        trainer.strategy._init_model_parallel = False
        trainer.strategy.setup(trainer)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(ckpt_to_weights_subdir(output_path))

        from nemo.lightning.io.pl import TrainerContext
        from nemo.utils.get_rank import is_global_rank_zero

        if is_global_rank_zero() and dump_io:
            TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(output_path), yaml_attrs=["model"])

    def nemo_load(
        self, path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True
    ) -> Tuple[pl.LightningModule, pl.Trainer]:
        """
        Loads a model from the specified path.

        Args:
            path (Path): The path from which the model will be loaded.
            trainer (Optional[pl.Trainer]): The trainer to be used, if not provided a new one will be created.
            cpu (bool): If True, the model will be loaded with a CPU-focused strategy.

        Returns
        -------
            Tuple[pl.LightningModule, pl.Trainer]: The loaded model and the trainer configured with the model.
        """
        from nemo.lightning import MegatronStrategy, Trainer, _strategy_lib
        from nemo.lightning.io.api import load_context

        model = load_context(path).model
        _trainer = trainer or Trainer(
            devices=1, accelerator="cpu" if cpu else "gpu", strategy=MegatronStrategy(ddp="pytorch")
        )

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()
        # TODO: Fix cpu initialization
        if not model.state_dict():
            if cpu:
                # TODO: Make this more generic
                with _strategy_lib.megatron_cpu_init_context(model.config):
                    model.configure_model()
            else:
                model.configure_model()

        _trainer.strategy.setup(_trainer)
        _trainer.strategy.load_checkpoint(path)

        return model, _trainer

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = Path(base_path)
        else:
            from nemo.lightning.base import NEMO_MODELS_CACHE

            _base = Path(NEMO_MODELS_CACHE)

        # If the useu supplied `hf:///path/to/downloaded/my-model/`
        # then extract the last dir-name (i.e. my-model) and append it to _base
        if str(self).startswith('/'):
            return _base / PurePath((str(self))).name
        return _base / str(self).replace("://", "/")

    def on_import_ckpt(self, model: pl.LightningModule):
        if hasattr(self, "tokenizer"):
            model.tokenizer = self.tokenizer
            if hasattr(model, "__io__") and hasattr(self.tokenizer, '__io__'):
                model.__io__.tokenizer = self.tokenizer.__io__

    def save_hf_tokenizer_assets(self, tokenizer_name_or_path, save_path="/tmp/nemo_tokenizer"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        # Save tokenizer assets to save_path.
        tok.save_pretrained(save_path)
        return save_path
