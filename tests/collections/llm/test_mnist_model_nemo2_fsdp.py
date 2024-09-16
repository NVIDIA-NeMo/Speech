# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

import megatron.core.num_microbatches_calculator
import pytest
import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning import NeMoLogger, io, resume
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction, ReductionT
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler

TokenizerType = Any

"""This is intended to be a minimal self-container NeMo2 example."""


T = TypeVar("T")


@dataclass
class ExampleConfig(ModelParallelConfig):
    """ExampleConfig is a dataclass that is used to configure the model.

    Timers from ModelParallelConfig are required for megatron forward compatibility.
    """

    calculate_per_token_loss: bool = False

    def configure_model(self) -> nn.Module:
        """This function is called by the strategy to construct the model.

        Note: Must pass self into Model since model requires having a config object.

        Returns:
            The model object.
        """
        return ExampleModel(self)


class MSELossReduction(MegatronLossReduction):
    """A class used for calculating the loss, and for logging the reduced loss across micro batches."""

    def forward(self, batch: DataT, forward_out: Tensor) -> Tuple[Tensor, ReductionT]:
        """Calculates the loss within a micro-batch. A micro-batch is a batch of data on a single GPU.

        Args:
            batch: A batch of data that gets passed to the original forward inside LitAutoEncoder.
            forward_out: the output of the forward method inside LitAutoEncoder.

        Returns:
            A tuple containing [<loss_tensor>, ReductionT] where the loss tensor will be used for
                backpropagation and the ReductionT will be passed to the reduce method
                (which currently only works for logging.).
        """
        x = batch["data"]
        outputs = forward_out
        x_hat = outputs["x_hat"]
        # you could also put a latent loss on z here.
        xview = x.view(x.size(0), -1)
        loss = nn.functional.mse_loss(x_hat, xview)

        return loss, {"avg": loss}

    def reduce(self, losses_reduced_per_micro_batch: Sequence[ReductionT]) -> Tensor:
        """Works across micro-batches. (data on single gpu).

        Note: This currently only works for logging and this loss will not be used for backpropagation.

        Args:
            losses_reduced_per_micro_batch: a list of the outputs of forward

        Returns:
            A tensor that is the mean of the losses. (used for logging).
        """
        mse_losses = torch.stack([loss["avg"] for loss in losses_reduced_per_micro_batch])
        return mse_losses.mean()


def some_first(seq: Iterable[Optional[T]]) -> T:
    """Returns the first non-None value from the sequence or fails"""  # noqa: D415
    for s in seq:
        if s is not None:
            return s
    raise ValueError("non-None value not found")


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:  # noqa: D103
    match torch_object:
        case []:
            raise ValueError("Looking up dtype on an empty list")
        case {**data} if not data:
            raise ValueError("Looking up dtype on an empty dict")
        case torch.Tensor(dtype=dtype, device=device):
            return dtype, device
        case torch.nn.Module() as m:
            try:
                p = next(m.parameters())
            except StopIteration as e:
                raise ValueError("Cannot get dtype on a torch module with no parameters.") from e
            return p.dtype, p.device
        case dict(keys=_, values=values):
            val = some_first(values())
            return get_dtype_device(val)
        case list() as l:
            val = some_first(l)
            return get_dtype_device(val)
        case _:
            raise TypeError("Got something we didnt expect")


# NOTE(SKH): These types are all wrong, but are close. The inner type must always be a torch.Tensor, but the outer container should be generic.
def batch_collator(batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]]) -> Optional[ReductionT]:
    """Takes a sequence of batches and collates them into a single batch.
        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    IMPORTANT: The underlying data primitive _must_ be a torch Tensor. The input to this function is a recurisve type,
    there can be any amount of nesting between dictionaries, tuples, and lists, as long as the inner type is a n-d torch.Tensor.

    Examples:
        Outer container = Dict:
            [{'a': torch.tensor([1]), 'b': torch.tensor([2])}, {'a': torch.tensor([2]), 'b': torch.tensor([3])}] -> {'a': torch.tensor([1, 2]), 'b': torch.tensor([2, 3])}
        Outer container = List:
            [[torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]] -> [torch.tensor([1, 2]), torch.tensor([2, 3])]
        Outer container = Tuple:
            ([torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]) -> (torch.tensor([1, 2]), torch.tensor([2, 3]))

    Args:
        batches (Optional[Sequence[ReductionT]]): sequence of batches to collate into a single batch.

    Returns:
        A single batch of the same type as the elements of your input sequence.
    """  # noqa: D205
    match batches:
        case [torch.Tensor(), *_]:
            return torch.cat(batches, dim=0)
        case [dict(), *_]:
            return {key: batch_collator([batch[key] for batch in batches]) for key in batches[0]}
        case [tuple(), *_]:
            return tuple(batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0])))
        case [list(), *_]:
            return [batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        case None:
            return None
        case []:
            raise ValueError("Cannot process an empty sequence")
        case _:
            raise ValueError("Unsupported input structure in batch_collator")


class PassthroughLossReduction(MegatronLossReduction):
    """Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is expected to return a loss.
    This class hijacks that mechanism to instead pass through the forward output unperturbed as the loss (to enable inference in the predict step), and then the
    reduce method is used to collate the batch of forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple,
    or list of tensors. The inner type _must always be a torch.Tensor_.
    """  # noqa: D205

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[torch.Tensor, DataT]:
        """_summary_

        Args:
            batch (DataT): The batch of data that was passed through the model to generate output.
            forward_out (torch.Tensor): The output from your model's forward pass.

        Returns:
            Tuple[torch.Tensor, ReductionT]: A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """  # noqa: D415
        dtype, device = get_dtype_device(forward_out)
        return torch.zeros(1, device=device, dtype=dtype), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        """This overrides the standard reduce with a simplified version that just takes a list of your model's forward outputs
            and collates them togehter into a single output.

        Args:
            forward_out (List[ReductionT]): _description_

        Returns:
            ReductionT: _description_
        """  # noqa: D205
        return batch_collator(forward_out)


class TorchAdam(OptimizerModule):
    def __init__(self, config, lr_scheduler=None):
        self.conf = config

        super().__init__(lr_scheduler)

    def optimizers(self, model):
        return [
            Adam(
                model.parameters(),
                lr=self.conf.lr,
                betas=(self.conf.adam_beta1, self.conf.adam_beta2),
                eps=self.conf.adam_eps,
                weight_decay=self.conf.weight_decay,
            )
        ]


class LitAutoEncoder(pl.LightningModule, io.IOMixin, io.ConnectorMixin):
    """A very basic lightning module for testing the megatron strategy and the megatron-nemo2-bionemo contract."""

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: a Config object necessary to construct the actual nn.Module (the thing that has the parameters).
        """
        super().__init__()
        self.config = config
        self.optim = TorchAdam(
            config=OptimizerConfig(lr=1e-4, optimizer="adam"),
        )
        # Bind the configure_optimizers method to the model
        self.optim.connect(self)

    def forward(self, batch: Dict, batch_idx: Optional[int] = None) -> Any:
        """This forward will be called by the megatron scheduler and it will be wrapped.

        !!! note

            The `training_step` defines the training loop and is independent of the `forward` method here.

        Args:
            batch: A dictionary of data.
            batch_idx: The index of the batch.

        Returns:
            The output of the model.
        """
        x = batch["data"]
        return self.module(x)

    def training_step(self, batch, batch_idx: Optional[int] = None):
        """The training step is where the loss is calculated and the backpropagation is done.

        Background:
        - NeMo's Strategy overrides this method.
        - The strategies' training step will call the forward method of the model.
        - That forward method then calls the wrapped forward step of MegatronParallel which wraps the forward method of the model.
        - That wrapped forward step is then executed inside the Mcore scheduler, which calls the `_forward_step` method from the
            MegatronParallel class.
        - Which then calls the training_step function here.

        In this particular use case, we simply call the forward method of this class, the lightning module.

        Args:
            batch: A dictionary of data. requires `batch_idx` as default None.
            batch_idx: The index of the batch.
        """
        return self(batch, batch_idx)

    @property
    def training_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        # This is the function that takes batch['loss_mask'] and the logits output by the model and reduces the loss
        return MSELossReduction()

    @property
    def validation_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return MSELossReduction()

    @property
    def test_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        return MSELossReduction()

    @property
    def predict_loss_reduction(self) -> MegatronLossReduction:  # noqa: D102
        # This allows us to do inference (not output the loss)
        return PassthroughLossReduction()

    def configure_model(self) -> None:  # noqa: D102
        self.module = self.config.configure_model()


class ExampleModel(MegatronModule):  # noqa: D101
    def __init__(self, config: ModelParallelConfig) -> None:
        """Constructor of the model.

        Args:
            config: The config object is responsible for telling the strategy what model to create.
        """
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.linear1 = nn.Linear(28 * 28, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 3)
        self.linear3 = nn.Linear(3, 64)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(64, 28 * 28)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            x_hat: The result of the last linear layer of the network.
        """
        x = x.view(x.size(0), -1)
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        x_hat = self.linear3(z)
        x_hat = self.relu2(x_hat)
        x_hat = self.linear4(x_hat)
        return {"x_hat": x_hat, "z": z}

    def set_input_tensor(self, input_tensor: Optional[Tensor]) -> None:
        """This is needed because it is a megatron convention. Even if it is a no-op for single GPU testing.

        See megatron.model.transformer.set_input_tensor()

        Note: Currently this is a no-op just to get by an mcore function.

        Args:
            input_tensor: Input tensor.
        """
        pass


class MnistItem(TypedDict):
    data: Tensor
    label: Tensor
    idx: int


class MNISTCustom(MNIST):  # noqa: D101
    def __getitem__(self, index: int) -> MnistItem:
        """Wraps the getitem method of the MNIST dataset such that we return a Dict
        instead of a Tuple or tensor.

        Args:
            index: The index we want to grab, an int.

        Returns:
            A dict containing the data ("x"), label ("y"), and index ("idx").
        """  # noqa: D205
        x, y = super().__getitem__(index)

        return {
            "data": x,
            "label": y,
            "idx": index,
        }


# TODO: remove this callback after `val` loss is logged by default in training in NeMo2
class LossLoggingCallback(pl.Callback):  # noqa: D101
    def __init__(self):
        """Log the loss at the end of each batch. For training do not reduce across the epoch but do so for validation/test."""
        self.val_losses = []
        self.test_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            loss = outputs
            pl_module.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D102
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            loss = outputs
            self.test_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D102
        # Assuming the loss is computed internally and stored in pl_module
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if isinstance(outputs, dict):
                outputs = outputs["loss"]
            loss = outputs
            self.val_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: D102
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.val_losses) > 0:
                avg_val_loss = torch.stack(self.val_losses).mean()
                pl_module.log("val_loss", avg_val_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.val_losses.clear()

    def on_test_epoch_end(self, trainer, pl_module):  # noqa: D102
        if torch.distributed.get_rank() == 0 and parallel_state.is_pipeline_last_stage():
            if len(self.test_losses) > 0:
                avg_test_loss = torch.stack(self.test_losses).mean()
                pl_module.log("test_loss", avg_test_loss, prog_bar=True, logger=True, rank_zero_only=True)
                self.test_losses.clear()


class MNISTDataModule(pl.LightningDataModule):  # noqa: D101
    def __init__(self, data_dir: str = "./", batch_size: int = 32) -> None:  # noqa: D107
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.micro_batch_size = 8
        self.global_batch_size = 8
        self.max_len = 100
        self.rampup_batch_size = None

        #  Note that this sampler is sequential, meaning it does not do any shuffling. Let's wrap our data in a shuffler.
        # Wraps the datasampler with the MegatronDataSampler. The MegatronDataSampler is a wrapper that allows the sampler
        # to be used with megatron. It sets up the capability to utilize micro-batching and gradient accumulation. It is also
        # the place where the global batch size is constructed.
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
        )

    def setup(self, stage: str) -> None:
        """Sets up the datasets

        Args:
            stage: can be one of train / test / predict.
        """  # noqa: D415
        self.mnist_test = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False)
        self.mnist_predict = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=False)
        mnist_full = MNISTCustom(self.data_dir, download=True, transform=transforms.ToTensor(), train=True)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)


### Begin model environment related utilities
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


@pytest.mark.run_only_on("GPU")
@pytest.mark.integration
def test_train_mnist_litautoencoder_with_fsdp_strategy_single_gpu():
    path = os.path.abspath(__file__)
    call = f"python {path}"
    # Raises a CalledProcessError if there is a failure in the subprocess
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)


def run_train_mnist_litautoencoder_with_fsdp_strategy_single_gpu():
    """This is the actual test that will get run in a subprocess so it does not contaminate the state of other tests."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        assert tmpdir.exists()
        assert tmpdir.is_dir()
        with reset_megatron_parallel_state():
            # Configure our custom Checkpointer
            name = "test_experiment"
            checkpoint_callback = nl_callbacks.ModelCheckpoint(
                save_last=True,
                monitor="val_loss",
                save_top_k=1,
                every_n_train_steps=5,
                # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
                always_save_context=True,
            )
            root_dir = tmpdir
            save_dir = root_dir / name
            tb_logger = TensorBoardLogger(save_dir=str(save_dir), name=name)
            # Setup the logger and train the model
            nemo_logger = NeMoLogger(
                log_dir=str(root_dir),  # WARNING: passing a path in here results in mutating the Path class.
                name=name,
                tensorboard=tb_logger,
                ckpt=checkpoint_callback,
            )
            # Needed so that the trainer can find an output directory for the profiler
            # nemo_logger.save_dir = tmpdir

            model = LitAutoEncoder(config=ExampleConfig())
            strategy = nl.FSDPStrategy()
            trainer = nl.Trainer(
                accelerator="gpu",
                devices=1,
                strategy=strategy,
                limit_val_batches=5,
                val_check_interval=5,
                max_steps=20,
                num_nodes=1,
                log_every_n_steps=5,
                callbacks=[io.track_io(LossLoggingCallback)()],
            )
            data_module = MNISTDataModule(data_dir=tmpdir)
            llm.train(
                model=model,
                data=data_module,
                trainer=trainer,
                log=nemo_logger,
                resume=resume.AutoResume(
                    resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
                    resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
                ),
            )
            trainer._teardown()
        with reset_megatron_parallel_state():
            pred_strategy = nl.FSDPStrategy(
                data_sampler=MegatronDataSampler(
                    seq_len=28 * 28,
                    micro_batch_size=2,
                    global_batch_size=2,
                    output_log=False,  # Disable logs to support predict_step
                ),
            )
            predict_trainer = nl.Trainer(
                accelerator="gpu",
                devices=1,
                strategy=pred_strategy,
                default_root_dir=str(root_dir),  # WARNING: passing a path in here results in mutating the Path class.
            )
            ckpt_path = checkpoint_callback.last_model_path.replace(
                ".ckpt", ""
            )  # strip .ckpt off the end of the last path
            ckpt_path = (
                Path(ckpt_path) / "weights"
            )  ## weights are saved to the "weights" directory within the checkpoint

            assert Path(
                ckpt_path
            ).exists(), f"checkpoint {ckpt_path} not found in {os.listdir(Path(ckpt_path).parent)}"
            # FIXME: the below checkpoint loading strategy and manual module unwrapping probably only works in single GPU
            #  and maybe DDP.
            unwrapped_trained_model = trainer.model.module  # TODO clean this up. Would be good not to have to unwrap.
            forward_output = batch_collator(
                predict_trainer.predict(
                    unwrapped_trained_model, dataloaders=data_module.test_dataloader(), ckpt_path=ckpt_path
                )
            )

            assert set(forward_output.keys()) == {
                "z",
                "x_hat",
            }, f"We expect forward output from predit_step, not the loss, got: {forward_output}"
            assert forward_output["x_hat"].shape == (len(data_module.mnist_test), 28 * 28)
            assert forward_output["z"].shape == (len(data_module.mnist_test), 3)  # latent bottleneck in model of dim 3
            predict_trainer._teardown()


if __name__ == "__main__":
    # Have the test run this one item as a subprocess call
    run_train_mnist_litautoencoder_with_fsdp_strategy_single_gpu()
