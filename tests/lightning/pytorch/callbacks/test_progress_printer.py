import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

from lightning.pytorch import LightningModule, Trainer

from nemo.lightning.pytorch.callbacks.progress_printer import ProgressPrinter


class DummyTestModel(LightningModule):
    """
    Minimal LightningModule for testing ProgressPrinter during test phase.
    No optimizer or training loop needed.
    """

    def test_step(self, batch, batch_idx):
        return {}

    def test_dataloader(self):
        x = torch.randn(4, 2)
        y = torch.zeros(4)
        return DataLoader(TensorDataset(x, y), batch_size=1)


def test_progress_printer_test_phase_does_not_crash():
    """
    Regression test for:
    https://github.com/NVIDIA-NeMo/NeMo/issues/15064

    Ensures ProgressPrinter does not reference validation-only state
    during trainer.test().
    """

    model = DummyTestModel()
    callback = ProgressPrinter(log_interval=1)

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        enable_checkpointing=False,
        logger=False,
    )

    # Mock Megatron microbatch logic so the test runs without Megatron strategy
    with patch(
        "nemo.lightning.pytorch.callbacks.progress_printer.get_num_microbatches",
        return_value=1,
    ):
        trainer.test(model)
