from typing import Optional

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from nemo import lightning as nl
from nemo.collections.llm.utils import Config


def tensorboard_logger(name: str, save_dir: str = "tb_logs") -> Config[TensorBoardLogger]:
    return Config(TensorBoardLogger, save_dir=save_dir, name=name)


def wandb_logger(project: str, name: str) -> Config[WandbLogger]:
    return Config(
        WandbLogger,
        project=project,
        name=name,
        config={},
    )


def default_log(
    ckpt_dir: str,
    name: str,
    tensorboard_logger: Optional[Config[TensorBoardLogger]] = None,
    wandb_logger: Optional[Config[WandbLogger]] = None,
) -> Config[nl.NeMoLogger]:
    ckpt = Config(
        nl.ModelCheckpoint,
        save_best_model=False,
        save_last=True,
        save_top_k=10,
        every_n_train_steps=200,
        filename="{model_name}--{val_loss:.2f}-{step}-{consumed_samples}",
    )

    return Config(
        nl.NeMoLogger,
        ckpt=ckpt,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        dir=ckpt_dir,
    )


def default_resume() -> Config[nl.AutoResume]:
    return Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )
