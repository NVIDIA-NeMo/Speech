from unittest.mock import patch

import pytest
from pytorch_lightning.callbacks import ModelCheckpoint as PTLModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl


class TestNeMoLogger:
    @pytest.fixture
    def trainer(self):
        return nl.Trainer(accelerator="cpu")

    def test_loggers(self):
        trainer = nl.Trainer(accelerator="cpu")
        logger = nl.NeMoLogger(
            update_logger_directory=True,
            wandb=WandbLogger(save_dir="test", offline=True),
        )

        logger.setup(trainer)
        assert logger.tensorboard is None
        assert len(logger.extra_loggers) == 0
        assert len(trainer.loggers) == 2
        assert isinstance(trainer.loggers[1], WandbLogger)
        assert str(trainer.loggers[1].save_dir).endswith("nemo_experiments")
        assert trainer.loggers[1]._name == "default"

    def test_explicit_log_dir(self, trainer):
        explicit_dir = "explicit_test_dir"
        logger = nl.NeMoLogger(name="test", explicit_log_dir=explicit_dir)

        with patch("nemo.utils.exp_manager.check_explicit_log_dir") as mock_check:
            logger.setup(trainer)
            mock_check.assert_called_once_with(trainer, explicit_dir, None, "test", None)

    def test_custom_version(self, trainer):
        custom_version = "v1.0"
        logger = nl.NeMoLogger(name="test", version=custom_version, use_datetime_version=False)

        app_state = logger.setup(trainer)
        assert app_state.version == custom_version

    def test_file_logging_setup(self, trainer):
        logger = nl.NeMoLogger(name="test")

        with patch("nemo.lightning.nemo_logger.logging.add_file_handler") as mock_add_handler:
            logger.setup(trainer)
            mock_add_handler.assert_called_once()

    def test_model_checkpoint_setup(self, trainer):
        ckpt = PTLModelCheckpoint(dirpath="test_ckpt", filename="test-{epoch:02d}-{val_loss:.2f}")
        logger = nl.NeMoLogger(name="test", ckpt=ckpt)

        logger.setup(trainer)
        assert any(isinstance(cb, PTLModelCheckpoint) for cb in trainer.callbacks)
        ptl_ckpt = next(cb for cb in trainer.callbacks if isinstance(cb, PTLModelCheckpoint))
        assert str(ptl_ckpt.dirpath).endswith("test_ckpt")
        assert ptl_ckpt.filename == "test-{epoch:02d}-{val_loss:.2f}"
