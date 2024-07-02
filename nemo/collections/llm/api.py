from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from typing_extensions import Annotated

from nemo.collections.llm.utils import Config, task
from nemo.lightning import AutoResume, MegatronStrategy, NeMoLogger, OptimizerModule, Trainer, io, teardown


@task(namespace="llm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[str] = None,
    # TODO: Fix export export: Optional[str] = None,
) -> Path:
    """
    Trains a model using the specified data and trainer, with optional tokenizer, source, and export.

    Args:
        model (pl.LightningModule): The model to be trained.
        data (pl.LightningDataModule): The data module containing training data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[Union[AutoResume, Resume]]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[str]): Tokenizer setting to be applied. Can be 'data' or 'model'.
        export (Optional[str]): Filename to save the exported checkpoint after training.

    Returns
    -------
        Path: The directory path where training artifacts are saved.

    Raises
    ------
        ValueError: If the trainer's strategy is not MegatronStrategy.

    Examples
    --------
        >>> model = MyModel()
        >>> data = MyDataModule()
        >>> trainer = Trainer(strategy=MegatronStrategy())
        >>> train(model, data, trainer, tokenizer='data', source='path/to/ckpt.ckpt', export='final.ckpt')
        PosixPath('/path/to/log_dir')
    """
    _log = log or NeMoLogger()
    app_state = _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=getattr(train, "__io__", None),
    )
    if resume is not None:
        resume.setup(model, trainer)
    if optim:
        optim.connect(model)
    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)

    trainer.fit(model, data)

    _log.teardown()

    return app_state.exp_dir


@task(namespace="llm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    source: Optional[str] = None,
    # export: Optional[str] = None
) -> Path:
    return train(model=model, data=data, trainer=trainer, tokenizer="data", source=source)


@task(namespace="llm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    tokenizer: Optional[str] = None,
    source: Optional[str] = None,
    export: Optional[str] = None,
) -> Path:
    if not isinstance(trainer.strategy, MegatronStrategy):
        raise ValueError("Only MegatronStrategy is supported")

    validate_kwargs = {}
    run_dir = Path(trainer.logger.log_dir)
    export_dir = run_dir / "export"

    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)
    if source:
        _add_ckpt_path(source, model, validate_kwargs)

    trainer.validate(model, data, **validate_kwargs)
    trainer.save_checkpoint(export_dir)
    if export:
        teardown(trainer)
        del trainer, model, data
        export_ckpt(export_dir, export)

    return run_dir


@task(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    return io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite)


def load_connector_from_trainer_ckpt(path: Path, target: str) -> io.ModelConnector:
    return io.load_context(path).model.exporter(target, path)


@task(name="export", namespace="llm")
def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
) -> Path:
    return io.export_ckpt(path, target, output_path, overwrite, load_connector)


def _use_tokenizer(model: pl.LightningModule, data: pl.LightningDataModule, tokenizer: str) -> None:
    if tokenizer == "data":
        model.tokenizer = data.tokenizer
        if hasattr(model, "__io__"):
            model.__io__.tokenizer = data.tokenizer
    elif tokenizer == "model":
        data.tokenizer = model.tokenizer
        if hasattr(data, "__io__"):
            data.__io__.tokenizer = model.tokenizer


def _add_ckpt_path(source, model, kwargs) -> None:
    if io.is_distributed_ckpt(source):
        kwargs["ckpt_path"] = source
    else:
        kwargs["ckpt_path"] = model.import_ckpt(source)


def _save_config_img(*args, **kwargs):
    try:
        from nemo_sdk.utils import save_config_img

        save_config_img(*args, **kwargs)
    except ImportError:
        pass
