import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import megatron
import pytest
import pytorch_lightning as pl
import torch
from megatron.core import ModelParallelConfig, parallel_state
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor

import nemo.lightning as nl
from nemo.lightning.io.mixin import IOMixin
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch.plugins import MegatronDataSampler


### model environment related utilities
def _reset_megatron_parallel_state():
    """Resets _GLOBAL_NUM_MICROBATCHES_CALCULATOR in megatron which is used in NeMo to initialized model parallel in
    nemo.collections.nlp.modules.common.megatron.megatron_init.initialize_model_parallel_for_nemo
    """  # noqa: D205, D415
    megatron.core.num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    # Clean up any process groups created in testing
    torch.cuda.empty_cache()
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def reset_megatron_parallel_state() -> Iterator[None]:
    """Puts you into a clean parallel state, and again tears it down at the end."""
    try:
        _reset_megatron_parallel_state()
        yield
    finally:
        _reset_megatron_parallel_state()


class RandomDataset(pl.LightningDataModule):
    def __init__(self, size, length):
        super().__init__()
        self.len = length
        self.data = torch.randn(length, size)
        self.data_sampler = MegatronDataSampler(
            seq_len=size,
            micro_batch_size=2,
            global_batch_size=2,
            rampup_batch_size=None,
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.data, batch_size=2)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.data, batch_size=2)


class PassThroughLossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: DataT, forward_out: Tensor) -> Tuple[Tensor, ReductionT]:

        return forward_out, forward_out

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


class ExampleModel(pl.LightningModule, IOMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()

        ## keeps track of number of validation steps
        self.count = torch.zeros((1,))

    def configure_model(self):

        class NestedModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.modules.Linear(in_features=32, out_features=32)
                self.bn = torch.nn.BatchNorm1d(32)
                self.model_type = "test"
                self.validation_step_outputs = []

                class DummyConfig(ModelParallelConfig):
                    calculate_per_token_loss: bool = False
                    fp8: bool = False

                self.config = DummyConfig()

        self.module = NestedModel()

    def forward(self, batch):
        return self.l1(self.bn(batch)).sum()

    def train_dataloader(self):
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = RandomDataset(32, 16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_dataloader(self):
        dataset = RandomDataset(32, 16)
        dl = torch.utils.data.DataLoader(dataset, batch_size=2)
        self._test_names = ['test_{}_'.format(idx) for idx in range(len(dl))]
        return dl

    def training_step(self, batch):
        return self(batch)

    def validation_step(self, batch):
        ## use a dummy validation loss to ensure that loss is decreasing at each step
        ## which guarantees that the -last checkpoints will be symlinks if specified
        self.count += 1
        self.validation_step_outputs.append(-self.count)
        return -self.count

    def test_step(self, batch):
        loss = self(batch)
        self.test_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack(self.validation_step_outputs).mean())
        self.validation_step_outputs.clear()  # free memory

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        pass

    def training_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        return PassThroughLossReduction()

    def validation_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return PassThroughLossReduction()


def setup_test(path, async_save=False, max_epochs=3):
    model = ExampleModel()

    data = RandomDataset(32, 64)

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )

    nemo_logger = nl.NeMoLogger(
        log_dir=path,
        use_datetime_version=False,
    )

    strategy = nl.MegatronStrategy(
        ckpt_async_save=async_save,
        replace_progress_bar=False,
    )

    trainer = nl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        val_check_interval=6,
        log_every_n_steps=4,
        callbacks=nl.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            save_on_train_epoch_end=True,
            save_context_on_train_end=False,
            filename=f'{{step}}-{{epoch}}-{{val_loss}}-{{consumed_samples}}',
            save_last="link",
        ),
        strategy=strategy,
    )
    nemo_logger.setup(trainer)
    resume.setup(trainer)

    return data, model, trainer


def get_final_checkpoint(checkpoint_dir):
    dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
    last_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

    assert len(last_checkpoints) == 1  ## should only have one -last checkpoint
    final_ckpt = last_checkpoints[0]

    top_k_checkpoints = [d for d in dist_checkpoints if d not in last_checkpoints]

    return final_ckpt, top_k_checkpoints


class TestLinkCheckpoint:

    @pytest.mark.unit
    @pytest.mark.run_only_on("GPU")
    def test_link_ckpt(self, tmpdir):
        """Test to ensure that we always keep top_k checkpoints, even after resuming."""

        with reset_megatron_parallel_state():
            tmp_path = tmpdir / "link_ckpt_test"
            data, model, trainer = setup_test(tmp_path, async_save=False)

            trainer.fit(model, data)

            checkpoint_dir = Path(tmp_path / "default" / "checkpoints")
            final_ckpt, top_k_checkpoints = get_final_checkpoint(checkpoint_dir)
            assert os.path.islink(final_ckpt)

            ## make sure we're saving the expected number of checkpoints
            assert len(top_k_checkpoints) == 3

            link = final_ckpt.resolve()
            assert str(final_ckpt).replace("-last", "") == str(link)

    @pytest.mark.unit
    @pytest.mark.run_only_on("GPU")
    def test_link_ckpt_async(self, tmpdir):
        """Test to ensure that we always keep top_k checkpoints, even after resuming."""

        with reset_megatron_parallel_state():
            tmp_path = tmpdir / "async_link_ckpt_test"
            data, model, trainer = setup_test(tmp_path, async_save=True)

            trainer.fit(model, data)

            checkpoint_dir = Path(tmp_path / "default" / "checkpoints")
            final_ckpt, top_k_checkpoints = get_final_checkpoint(checkpoint_dir)
            assert os.path.islink(final_ckpt)
            assert len(top_k_checkpoints) == 3

            link = final_ckpt.resolve()
            assert str(final_ckpt).replace("-last", "") == str(link)

    @pytest.mark.unit
    @pytest.mark.run_only_on("GPU")
    def test_restore_async(self, tmpdir):
        """Test to ensure that we always keep top_k checkpoints, even after resuming."""

        with reset_megatron_parallel_state():
            tmp_path = tmpdir / "async_link_ckpt_test"
            data, model, trainer = setup_test(tmp_path, async_save=True, max_epochs=3)

            trainer.fit(model, data)

            ## reinitialize
            data, model, trainer = setup_test(tmp_path, async_save=True, max_epochs=6)

            trainer.fit(model, data)

            checkpoint_dir = Path(tmp_path / "default" / "checkpoints")
            final_ckpt, top_k_checkpoints = get_final_checkpoint(checkpoint_dir)
            assert os.path.islink(final_ckpt)
            assert len(top_k_checkpoints) == 3

            epoch = str(final_ckpt).split('epoch=')[1][0]
            assert int(epoch) == 5  ## make sure we're running the correct number of epochs
