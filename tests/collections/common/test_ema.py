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
from copy import deepcopy
from typing import Any, Dict, Union
from unittest import mock

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

from nemo.collections.common.callbacks import EMA
from nemo.core import ModelPT
from nemo.utils.exp_manager import exp_manager
from tests.collections.nlp.test_gpt_model import DEVICE_CAPABILITY


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(16)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def forward(self, batch):
        output = self.l1(batch)
        return torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        pass

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        pass

    def validation_epoch_end(self, loss):
        self.log("val_loss", torch.stack(loss).mean())


class TestEMAConfig:
    @pytest.mark.unit
    def test_ema_value(self):
        with pytest.raises(MisconfigurationException, match="between 0 and 1"):
            EMA(decay=2)

    @mock.patch('nemo.collections.common.callbacks.ema.apex_available', False)
    def test_ema_apex_unavailable(self):
        with pytest.warns(UserWarning, match="EMA has better performance when Apex is installed"):
            EMA(decay=0.999)

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_ema_saved_state(self, tmpdir, caplog):
        """Test to ensure that when we re-load the EMA callback, it loads the EMA weights correctly."""
        temp_path = os.path.join(tmpdir, 'saved_state')

        class TerminateCallback(Callback):
            def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
                self.saved_ema_weights = ema_callback._ema_model_weights
                self.pl_module_weights = list(pl_module.state_dict().values())
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
                "ema": {"enable": True, "evaluate_ema_weights_instead": True},
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
                ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
                weights = list(pl_module.state_dict().values())
                for x, y in zip(weights, terminate_callback.pl_module_weights):
                    assert torch.allclose(x.cpu(), y.cpu())
                for x, y in zip(ema_callback._ema_model_weights, terminate_callback.saved_ema_weights):
                    assert torch.allclose(x.cpu(), y.cpu())
                assert ema_callback._cur_step == 8

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
                "ema": {"enable": True, "evaluate_ema_weights_instead": True},
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
                "ema": {"enable": True, "evaluate_ema_weights_instead": True},
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
                "ema": {"enable": True, "evaluate_ema_weights_instead": True},
                "explicit_log_dir": str(temp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        with pytest.warns(UserWarning, match="we were unable to find the associated EMA weights when re-loading"):
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
                "ema": {"enable": True, "evaluate_ema_weights_instead": True},
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        assert any(isinstance(callback, EMA) for callback in trainer.callbacks)
        trainer.fit(model)

        assert os.path.exists(tmp_path / "checkpoints/epoch=0-step=8.ckpt")
        ema_path = tmp_path / "checkpoints/epoch=0-step=8-EMA.ckpt"
        assert os.path.exists(ema_path)

        duplicate_model = ExampleModel.load_from_checkpoint(str(ema_path))
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        for saved_weight, ema_weight in zip(duplicate_model.state_dict().values(), ema_callback._ema_model_weights):
            assert torch.allclose(saved_weight.cpu(), ema_weight.cpu())

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_ema_save_in_callback(self, tmpdir):
        """Test to ensure when `save_ema_weights_in_callback_state` is enabled, we save to the callback state."""
        temp_path = os.path.join(tmpdir, 'saved_state')

        model = ExampleModel()

        trainer = Trainer(
            max_epochs=2,
            limit_val_batches=1,
            limit_train_batches=16,
            logger=False,
            val_check_interval=0.5,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
            callbacks=[EMA(decay=0.999, save_ema_weights_in_callback_state=True, evaluate_ema_weights_instead=True)],
        )
        exp_manager(
            trainer,
            {"explicit_log_dir": str(temp_path), "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},},
        )
        trainer.fit(model=model)

        resume_path = os.path.join(temp_path, "checkpoints/epoch=0-step=8.ckpt")
        callback = EMA(decay=0.999, save_ema_weights_in_callback_state=True)

        class AssertCallback(Callback):
            def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                assert callback._ema_model_weights is not None

        model = ExampleModel()

        trainer = Trainer(
            max_epochs=2,
            limit_val_batches=1,
            limit_train_batches=16,
            logger=False,
            val_check_interval=0.5,
            enable_checkpointing=False,
            accelerator='gpu',
            devices=1,
            callbacks=[callback, AssertCallback()],
        )
        trainer.fit(model, ckpt_path=resume_path)


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
    @pytest.mark.parametrize("evaluate_ema_weights_instead", [True, False])
    @pytest.mark.parametrize("apex_available_mock", [True, False])
    @pytest.mark.run_only_on('GPU')
    def test_ema_run_cuda(
        self,
        test_data_dir,
        precision,
        accumulate_grad_batches,
        evaluate_ema_weights_instead,
        apex_available_mock,
        tmpdir,
    ):
        with mock.patch('nemo.collections.common.callbacks.ema.apex_available', apex_available_mock):
            self.run_training_test(
                accumulate_grad_batches=accumulate_grad_batches,
                evaluate_ema_weights_instead=evaluate_ema_weights_instead,
                accelerator='gpu',
                precision=precision,
                tmpdir=tmpdir,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
    @pytest.mark.parametrize("evaluate_ema_weights_instead", [True, False])
    @pytest.mark.run_only_on('GPU')
    def test_ema_run_cpu(self, test_data_dir, accumulate_grad_batches, evaluate_ema_weights_instead, tmpdir):
        self.run_training_test(
            accumulate_grad_batches=accumulate_grad_batches,
            evaluate_ema_weights_instead=evaluate_ema_weights_instead,
            accelerator='cpu',
            precision=32,
            tmpdir=tmpdir,
        )

    def run_training_test(self, accumulate_grad_batches, evaluate_ema_weights_instead, accelerator, precision, tmpdir):
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
                "ema": {"enable": True, "evaluate_ema_weights_instead": evaluate_ema_weights_instead},
                "explicit_log_dir": str(tmpdir),
                "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"},
            },
        )
        # add the check callback after the exp manager has made modifications.
        trainer.callbacks.append(EMAAssertCallback())
        trainer.fit(model=model, val_dataloaders=model.train_dataloader())


class EMAAssertCallback(Callback):
    def __init__(self):
        self._before_calc_ema_weights = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model_weights = list(pl_module.state_dict().values())
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        for x, y in zip(model_weights, ema_callback._ema_model_weights):
            assert torch.allclose(x, y)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        # saved for manual calculation of ema to compare against implementation
        self._before_calc_ema_weights = deepcopy(ema_callback._ema_model_weights)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            # skip assertion as ema weights are not updated.
            return
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        decay = ema_callback.decay
        expected_ema_weights = []
        for orig_weight, ema_weight in zip(list(pl_module.state_dict().values()), self._before_calc_ema_weights):
            expected_ema_weight = orig_weight * (1 - decay) + ema_weight * decay
            expected_ema_weights.append(expected_ema_weight)

        for actual_ema_weight, expected_ema_weight in zip(ema_callback._ema_model_weights, expected_ema_weights):
            assert torch.allclose(actual_ema_weight, expected_ema_weight)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if ema_callback.evaluate_ema_weights_instead:
            # todo (sean): shouldn't use the weights buffer to check original weights
            self._original_weights = list(x.detach().clone() for x in ema_callback._weights_buffer)
            if ema_callback.ema_initialized:
                for ema_weights, module_weights in zip(
                    ema_callback._ema_model_weights, pl_module.state_dict().values()
                ):
                    torch.allclose(ema_weights, module_weights)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ema_callback = [x for x in trainer.callbacks if isinstance(x, EMA)][0]
        if ema_callback.evaluate_ema_weights_instead:
            model_weights = list(pl_module.state_dict().values())
            if ema_callback.ema_initialized:
                for orig_weights, module_weights in zip(self._original_weights, model_weights):
                    torch.allclose(orig_weights, module_weights.cpu())
