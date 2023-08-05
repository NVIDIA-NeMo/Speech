# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os.path
from typing import Any, Dict, Union

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.collections.common.callbacks import EMA
from nemo.collections.common.callbacks.ema import EMAOptimizer
from nemo.core import ModelPT
from nemo.utils.exp_manager import exp_manager

DEVICE_CAPABILITY = None
if torch.cuda.is_available():
    DEVICE_CAPABILITY = torch.cuda.get_device_capability()


def extract_ema_weights(pl_module, trainer):
    ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
    ema_callback.swap_model_weights(trainer)
    weights = extract_weights(pl_module)
    ema_callback.swap_model_weights(trainer)
    return weights


def extract_weights(pl_module):
    return [w.detach().clone() for w in pl_module.parameters()]


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        self.l1 = torch.nn.modules.Linear(in_features=32, out_features=32)
        self.bn = torch.nn.BatchNorm1d(32)

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

    def forward(self, batch):
        return self.l1(self.bn(batch)).sum()

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.test_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def list_available_models(self):
        pass

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        pass

    def setup_test_data(self, val_data_config: Union[DictConfig, Dict]):
        pass

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack(self.validation_step_outputs).mean())
        self.validation_step_outputs.clear()  # free memory


class TestEMAConfig:
    @pytest.mark.unit
    def test_ema_value(self):
        with pytest.raises(MisconfigurationException, match="between 0 and 1"):
            EMA(decay=2)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_ema_saved_state(self, tmpdir, caplog):
        """Test to ensure that when we re-load the EMA callback, it loads the EMA weights correctly."""
        temp_path = os.path.join(tmpdir, 'saved_state')

        class TerminateCallback(Callback):
            def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                self.saved_ema_weights = extract_ema_weights(pl_module, trainer)
                self.pl_module_weights = extract_weights(pl_module)
                raise SystemExit

        model = ExampleModel()
        terminate_callback = TerminateCallback()

        trainer = Trainer(
            max_epochs=2,
            limit_val_batches=1,
            limit_train_batches=16,
            logger=False,
            val_check_interval=0.5,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
            callbacks=[terminate_callback],
        )
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(temp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        with pytest.raises(SystemExit):
            trainer.fit(model=model)
        resume_path = os.path.join(temp_path, 'checkpoints/epoch=0-step=8.ckpt')

        model = ExampleModel()

        class CheckStateCallback(Callback):
            def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                weights = extract_weights(pl_module)
                for x, y in zip(weights, terminate_callback.pl_module_weights):
                    assert torch.allclose(x.cpu(), y.cpu())
                current_ema_weights = extract_ema_weights(pl_module, trainer)
                for x, y in zip(current_ema_weights, terminate_callback.saved_ema_weights):
                    assert torch.allclose(x.cpu(), y.cpu())

                for optimizer in trainer.optimizers:
                    assert isinstance(optimizer, EMAOptimizer)
                    assert optimizer.current_step == 8

        trainer = Trainer(
            max_epochs=2,
            limit_val_batches=0,
            limit_train_batches=16,
            logger=False,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
        )
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(temp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        # add the callback after the exp manager has made modifications.
        trainer.callbacks.append(CheckStateCallback())
        trainer.fit(model, ckpt_path=resume_path)

        # ensure we can resume from the EMA weights
        ema_path = os.path.join(temp_path, 'checkpoints/epoch=0-step=8-EMA.ckpt')

        trainer = Trainer(
            max_epochs=1,
            limit_val_batches=0,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
        )
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(temp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        trainer.fit(model, ckpt_path=ema_path)

        # ensure that we warn when the EMA weights do not exist
        os.remove(ema_path)

        trainer = Trainer(
            max_epochs=1,
            limit_val_batches=0,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
        )
        exp_manager(
            trainer,
            {
                "ema": {"enable": True, "validate_original_weights": True},
                "explicit_log_dir": str(temp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        with pytest.raises(
            MisconfigurationException, match="Unable to find the associated EMA weights when re-loading"
        ):
            trainer.fit(model, ckpt_path=resume_path)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_exp_manager_ema_weights(self, tmpdir):
        """Test to ensure that the exp manager adds the EMA callback, and we save an additional EMA checkpoint."""
        tmp_path = tmpdir / "exp_manager_test"
        model = ExampleModel()
        trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False, accelerator='gpu', devices=1)
        exp_manager(
            trainer,
            {
                "ema": {"enable": True, "validate_original_weights": True},
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        assert any(isinstance(callback, EMA) for callback in trainer.callbacks)
        trainer.fit(model)
        ema_weights = extract_ema_weights(model, trainer)

        assert os.path.exists(tmp_path / "checkpoints/epoch=0-step=8.ckpt")
        ema_path = tmp_path / "checkpoints/epoch=0-step=8-EMA.ckpt"
        assert os.path.exists(ema_path)

        duplicate_model = ExampleModel.load_from_checkpoint(str(ema_path))
        for saved_weight, ema_weight in zip(duplicate_model.state_dict().values(), ema_weights):
            assert torch.allclose(saved_weight.cpu(), ema_weight.cpu())

    @pytest.mark.unit
    def test_exp_manager_ema_weights_topk(self, tmpdir):
        """Test to ensure that EMA correctly ensures we only keep topk checkpoints."""
        tmp_path = tmpdir / "exp_manager_test"
        model = ExampleModel()
        save_top_k = 3

        trainer = Trainer(max_epochs=10, enable_checkpointing=False, logger=False, devices=1)
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"save_top_k": save_top_k},
            },
        )
        trainer.fit(model)

        # we save 3 checkpoints for the model, 3 accompanied EMA weights, the last checkpoint and nemo model.
        assert len(os.listdir(tmp_path / "checkpoints/")) == (save_top_k + 1) * 2 + 1

    @pytest.mark.unit
    def test_exp_manager_ema_weights_topk_resume(self, tmpdir):
        """Test to ensure that we always keep top_k checkpoints, even after resuming."""
        tmp_path = tmpdir / "exp_manager_test"
        model = ExampleModel()
        save_top_k = 3

        trainer = Trainer(max_epochs=10, enable_checkpointing=False, logger=False, devices=1)
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"save_top_k": save_top_k},
            },
        )
        trainer.fit(model)

        # we save 3 checkpoints for the model, 3 accompanied EMA weights, the last checkpoint and nemo model.
        assert len(os.listdir(tmp_path / "checkpoints/")) == (save_top_k + 1) * 2 + 1

        # reduce the top_k number when resuming, we should see only 2 top_k checkpoints now (one is deleted).
        save_top_k = 2

        trainer = Trainer(max_epochs=10, enable_checkpointing=False, logger=False, devices=1)
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(tmp_path),
                "resume_if_exists": True,
                "checkpoint_callback_params": {"save_top_k": save_top_k},
            },
        )
        trainer.fit(model)

        # we save 2 checkpoints for the model, 2 accompanied EMA weights, the last checkpoint and nemo model.
        assert len(os.listdir(tmp_path / "checkpoints/")) == (save_top_k + 1) * 2 + 1


class TestEMATrain:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "precision",
        [
            32,
            16,
            pytest.param(
                "bf16",
                marks=pytest.mark.skipif(
                    not DEVICE_CAPABILITY or DEVICE_CAPABILITY[0] < 8,
                    reason='bfloat16 is not supported on this device',
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
    @pytest.mark.parametrize("validate_original_weights", [True, False])
    @pytest.mark.run_only_on('GPU')
    def test_ema_run_cuda(
        self, test_data_dir, precision, accumulate_grad_batches, validate_original_weights, tmpdir,
    ):
        self.run_training_test(
            accumulate_grad_batches=accumulate_grad_batches,
            validate_original_weights=validate_original_weights,
            accelerator='gpu',
            precision=precision,
            tmpdir=tmpdir,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
    @pytest.mark.parametrize("validate_original_weights", [True, False])
    def test_ema_run_cpu(self, test_data_dir, accumulate_grad_batches, validate_original_weights, tmpdir):
        self.run_training_test(
            accumulate_grad_batches=accumulate_grad_batches,
            validate_original_weights=validate_original_weights,
            accelerator='cpu',
            precision=32,
            tmpdir=tmpdir,
        )

    def run_training_test(self, accumulate_grad_batches, validate_original_weights, accelerator, precision, tmpdir):
        pl.seed_everything(123)
        model = ExampleModel()
        trainer = Trainer(
            max_epochs=1,
            precision=precision,
            limit_train_batches=10,
            limit_val_batches=10,
            logger=False,
            accumulate_grad_batches=accumulate_grad_batches,
            num_sanity_val_steps=0,
            enable_model_summary=False,
            enable_checkpointing=False,
            accelerator=accelerator,
            devices=1,
        )
        exp_manager(
            trainer,
            {
                "ema": {"enable": True, "validate_original_weights": validate_original_weights, "decay": 0.999},
                "explicit_log_dir": str(tmpdir),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        # add the check callback after the exp manager has made modifications.
        trainer.callbacks.append(EMAAssertCallback())
        trainer.callbacks.insert(0, EMAValidationAssertCallback())
        trainer.fit(model=model, val_dataloaders=model.train_dataloader())

    @pytest.mark.unit
    def test_ema_run_with_save_best_model(self, tmpdir):
        """Test to ensure that we save the model correctly when save best model is set to True."""
        tmp_path = tmpdir / "exp_manager_test"
        model = ExampleModel()

        trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False, devices=1, limit_train_batches=1)
        exp_manager(
            trainer,
            {
                "ema": {"enable": True},
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"save_best_model": True},
            },
        )
        trainer.fit(model)


class EMAAssertCallback(Callback):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model_weights = extract_weights(pl_module)
        self.ema_weights = extract_ema_weights(pl_module, trainer)
        for x, y in zip(model_weights, self.ema_weights):
            assert torch.allclose(x, y)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            # skip assertion as ema weights are not updated.
            return
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        decay = ema_callback.decay
        expected_ema_weights = []

        new_weights = extract_weights(pl_module)

        for ema_weight, new_weight in zip(self.ema_weights, new_weights):
            expected_ema_weight = ema_weight * decay
            expected_ema_weight += new_weight * (1 - decay)
            expected_ema_weights.append(expected_ema_weight)
        ema_weights = extract_ema_weights(pl_module, trainer)
        for actual_ema_weight, expected_ema_weight in zip(ema_weights, expected_ema_weights):
            assert torch.allclose(actual_ema_weight, expected_ema_weight)
        self.ema_weights = expected_ema_weights


class EMAValidationAssertCallback(Callback):
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        self._original_weights = extract_weights(pl_module)
        self._ema_weights = extract_ema_weights(pl_module, trainer)
        # call original EMA function
        super().on_validation_start(trainer, pl_module)
        if not ema_callback.validate_original_weights:
            if ema_callback._ema_initialized:
                # check model weights are now EMA weights
                for ema_weights, module_weights in zip(self._ema_weights, extract_weights(pl_module)):
                    torch.allclose(ema_weights, module_weights)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if not ema_callback.validate_original_weights:
            model_weights = extract_weights(pl_module)
            if ema_callback._ema_initialized:
                for orig_weights, module_weights in zip(self._original_weights, model_weights):
                    torch.allclose(orig_weights.cpu(), module_weights.cpu())
