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
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pytorch_lightning import Callback
from pytorch_lightning.loops import _TrainingEpochLoop

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.constants import NEMO_ENV_VARNAME_VERSION
from nemo.core.classes import ModelPT
from nemo.utils.callbacks import NeMoModelCheckpoint
from nemo.utils.exp_manager import (
    CheckpointMisconfigurationError,
    LoggerMisconfigurationError,
    NotFoundError,
    exp_manager,
)


class MyTestOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if self._step == 0:
                    p.data = 0.1 * torch.ones(p.shape)
                elif self._step == 1:
                    p.data = 0.0 * torch.ones(p.shape)
                else:
                    p.data = 0.01 * torch.ones(p.shape)
        self._step += 1
        return loss


class DoNothingOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        return loss


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
        pl.seed_everything(1234)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(2)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    def forward(self, batch):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))
        return output

    def validation_step(self, batch, batch_idx):
        self.loss = self(batch)
        return self.loss

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return MyTestOptimizer(self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack([self.loss]).mean())


class ExampleMCoreModel(ExampleModel):
    def sharded_state_dict(self):
        return {'a': 3}


class DoNothingModel(ExampleModel):
    def configure_optimizers(self):
        return DoNothingOptimizer(self.parameters())


class TestExpManager:
    @pytest.mark.unit
    def test_omegaconf(self):
        """Ensure omegaconf raises an error when an unexcepted argument is passed"""
        with pytest.raises(OmegaConfBaseException):
            exp_manager(pl.Trainer(accelerator='cpu'), {"unused": 1})

    @pytest.mark.unit
    def test_trainer_loggers(self, tmp_path):
        """Test that a trainer with logger errors out with a number of arguments. Test that it works with
        create_tensorboard_logger set to False
        """
        test_trainer = pl.Trainer(accelerator='cpu')  # Should create logger and modelcheckpoint

        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"exp_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"resume_if_exists": True})

        # Check that exp_manager uses trainer.logger, it's exp_dir, name, and version
        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert log_dir.resolve() == Path("./lightning_logs/version_0").resolve()
        assert Path("./lightning_logs").exists()
        assert Path("./lightning_logs/version_0").exists()

        # Check that a trainer without a logger gets a logger attached to it
        test_trainer = pl.Trainer(accelerator='cpu', logger=False)
        log_dir = exp_manager(
            test_trainer,
            {"create_tensorboard_logger": True, "create_checkpoint_callback": False, "exp_dir": str(tmp_path)},
        )
        assert isinstance(test_trainer.logger, pl.loggers.TensorBoardLogger)

        test_trainer = pl.Trainer(accelerator='cpu', logger=False)
        # Check that a create_wandb_logger=True errors out unless wandb_logger_kwargs is passed.
        with pytest.raises(ValueError):
            log_dir = exp_manager(
                test_trainer,
                {
                    "create_tensorboard_logger": False,
                    "create_checkpoint_callback": False,
                    "exp_dir": str(tmp_path),
                    "create_wandb_logger": True,
                },
            )
        # Check that a WandbLogger is attached to logger if create_wandb_logger=True and wandb_logger_kwargs has name
        # and project
        log_dir = exp_manager(
            test_trainer,
            {
                "create_tensorboard_logger": False,
                "create_checkpoint_callback": False,
                "exp_dir": str(tmp_path),
                "create_wandb_logger": True,
                "wandb_logger_kwargs": {"name": "", "project": "", "offline": True},
            },
        )
        assert isinstance(test_trainer.logger, pl.loggers.WandbLogger)

    @pytest.mark.unit
    def test_trainer_neptune_logger(self, tmp_path):
        pytest.importorskip("neptune", reason="could not import `neptune`, use `pip install neptune` to run this test")

        test_trainer = pl.Trainer(accelerator='cpu', logger=False)
        # Check that a create_neptune_logger=True errors out unless neptune_logger_kwargs is passed.
        with pytest.raises(ValueError):
            _ = exp_manager(
                test_trainer,
                {
                    "create_tensorboard_logger": False,
                    "create_checkpoint_callback": False,
                    "exp_dir": str(tmp_path),
                    "create_neptune_logger": True,
                },
            )
        # Check that a NeptuneLogger is attached to logger if create_neptune_logger=True and neptune_logger_kwargs has name
        # and project
        _ = exp_manager(
            test_trainer,
            {
                "create_tensorboard_logger": False,
                "create_checkpoint_callback": False,
                "exp_dir": str(tmp_path),
                "create_neptune_logger": True,
                "neptune_logger_kwargs": {"name": "", "project": "", "api_key": ""},
            },
        )
        assert isinstance(test_trainer.logger, pl.loggers.NeptuneLogger)

    @pytest.mark.unit
    def test_checkpoint_configurations(self):
        """Test that trainer creating modelcheckpoint and asking exp_manager to do it too results in errors, but
        is error free if only one is asked to do so.
        """
        disable_tb_logger = {"create_tensorboard_logger": False}
        test_trainer = pl.Trainer(accelerator='cpu')  # Should create logger and modelcheckpoint
        with pytest.raises(CheckpointMisconfigurationError):  # Fails because both try to create modelcheckpoint
            exp_manager(test_trainer, disable_tb_logger)

        # Should succeed without error
        exp_manager(test_trainer, {"create_checkpoint_callback": False, "create_tensorboard_logger": False})

        test_trainer_2 = pl.Trainer(accelerator='cpu', enable_checkpointing=False)
        exp_manager(test_trainer_2, disable_tb_logger)  # Should succeed without error

    @pytest.mark.unit
    def test_default_log_dir(self):
        """Check the default of ./nemo_experiments/default/datetime works as intended"""
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)

        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert (log_dir / "..").resolve() == Path("./nemo_experiments/default/").resolve()
        assert Path("./nemo_experiments").exists()
        assert Path("./nemo_experiments/default/").exists()
        sub_dirs = [x for x in Path("./nemo_experiments/default/").iterdir() if x.is_dir()]
        assert len(sub_dirs) == 1
        assert re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}", sub_dirs[0].name)

    @pytest.mark.unit
    def test_log_dir_overrides(self, monkeypatch, tmp_path):
        """Check a variety of trainer options with exp_manager"""
        # Checks that explicit_log_dir ignores exp_dir, name, and version
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path / "test_log_dir_overrides")})
        assert log_dir.resolve() == (tmp_path / "test_log_dir_overrides").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_log_dir_overrides").exists()

        # Checks that exp_manager uses exp_dir, default name, and explicit version
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "version": 957})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "957").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "957").exists()

        monkeypatch.delenv(NEMO_ENV_VARNAME_VERSION)
        # Checks that use_datetime_version False toggle works
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "use_datetime_version": False})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "version_0").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "version_0").exists()

        monkeypatch.delenv(NEMO_ENV_VARNAME_VERSION)
        # Checks that use_datetime_version False toggle works and version increments
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "use_datetime_version": False})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "version_1").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "version_1").exists()

    @pytest.mark.unit
    def test_resume(self, tmp_path):
        """Tests the resume capabilities of exp_manager"""
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)

        # Error because explicit_log_dir does not exist
        with pytest.raises(NotFoundError):
            exp_manager(
                test_trainer,
                {
                    "exp_dir": str(tmp_path / "test_resume"),
                    "resume_if_exists": True,
                    "explicit_log_dir": "Does_not_exist",
                },
            )

        # Error because checkpoints folder does not exist
        with pytest.raises(NotFoundError):
            exp_manager(test_trainer, {"resume_if_exists": True, "exp_dir": str(tmp_path / "test_resume")})

        # No error because we tell exp_manager to ignore notfounderror
        exp_manager(
            test_trainer,
            {
                "resume_if_exists": True,
                "exp_dir": str(tmp_path / "test_resume_2"),
                "resume_ignore_no_checkpoint": True,
            },
        )

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints").mkdir(parents=True)
        # Error because checkpoints do not exist in folder
        with pytest.raises(NotFoundError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end.ckpt").touch()
        # Error because *end.ckpt is in folder indicating that training has already finished
        with pytest.raises(ValueError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end.ckpt").unlink()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt").touch()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last.ckpt").touch()
        # Error because multiple *last.ckpt is in folder. If more than one, don't know which to restore
        with pytest.raises(ValueError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        # Finally succeed
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last.ckpt").unlink()
        log_dir = exp_manager(
            test_trainer,
            {"resume_if_exists": True, "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0")},
        )
        checkpoint = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt")
        assert Path(test_trainer.ckpt_path).resolve() == checkpoint.resolve()

        # Succeed again and make sure that run_0 exists and previous log files were moved
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        exp_manager(test_trainer, {"resume_if_exists": True, "explicit_log_dir": str(log_dir)})
        checkpoint = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt")
        assert Path(test_trainer.ckpt_path).resolve() == checkpoint.resolve()
        prev_run_dir = Path(tmp_path / "test_resume" / "default" / "version_0" / "run_0")
        assert prev_run_dir.exists()
        prev_log = Path(tmp_path / "test_resume" / "default" / "version_0" / "run_0" / "lightning_logs.txt")
        assert prev_log.exists()

        # Error becasue `dirpath` specified and has no checkpoint
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        dirpath_checkpoint_dir = Path(tmp_path / "test_resume" / "dirpath_test" / "ckpts")
        dirpath_checkpoint_dir.mkdir(parents=True)
        with pytest.raises(NotFoundError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "checkpoint_callback_params": {"dirpath": str(dirpath_checkpoint_dir)},
                    "explicit_log_dir": str(log_dir),
                },
            )

        # Check that model loads from `dirpath` and not <log_dir>/checkpoints
        dirpath_log_dir = Path(tmp_path / "test_resume" / "dirpath_test" / "logs")
        dirpath_log_dir.mkdir(parents=True)
        dirpath_checkpoint = Path(dirpath_checkpoint_dir / "mymodel--last.ckpt")
        dirpath_checkpoint.touch()
        exp_manager(
            test_trainer,
            {
                "resume_if_exists": True,
                "checkpoint_callback_params": {"dirpath": str(dirpath_checkpoint_dir)},
                "explicit_log_dir": str(dirpath_log_dir),
            },
        )
        assert Path(test_trainer.ckpt_path).resolve() == dirpath_checkpoint.resolve()

    @pytest.mark.unit
    def test_nemo_checkpoint_save_best_model_1(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {"checkpoint_callback_params": {"save_best_model": True}, "explicit_log_dir": str(tmp_path / "test")},
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert float(model(torch.tensor([1.0, 1.0], device=model.device))) == 0.0

    @pytest.mark.unit
    def test_nemo_checkpoint_save_best_model_2(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {"explicit_log_dir": str(tmp_path / "test")},
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5

    @pytest.mark.unit
    def test_nemo_checkpoint_always_save_nemo(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_best_model": True, "always_save_nemo": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert float(model(torch.tensor([1.0, 1.0], device=model.device))) == 0.0

    @pytest.mark.unit
    def test_nemo_checkpoint_doesnt_produce_too_many_nemo_ckpts(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_best_model": True, "always_save_nemo": True, "save_top_k": 2},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()
        assert (
            len(list((tmp_path / "test" / "checkpoints").glob("default*.nemo"))) == 1
        )  # check number of `.nemo` checkpoints

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert float(model(torch.tensor([1.0, 1.0], device=model.device))) == 0.0

    @pytest.mark.unit
    def test_nemo_checkpoint_make_checkpoint_dir(self, tmp_path):
        test_trainer = pl.Trainer(
            accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4, check_val_every_n_epoch=5
        )
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_best_model": True, "always_save_nemo": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

    @pytest.mark.unit
    def test_nemo_checkpoint_restore_model(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 1, "save_last": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        checkpoint = list(Path(str(tmp_path / "test" / "checkpoints")).glob("*.ckpt"))
        # Make sure that only the best and last checkpoint is saved
        assert len(checkpoint) == 2
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=5)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 1, "save_last": False},
                "explicit_log_dir": str(tmp_path / "test"),
                "resume_if_exists": True,
                "resume_past_end": True,
            },
        )
        model = DoNothingModel()
        model.l1.weight = torch.nn.Parameter(torch.tensor((0.0, 0.0)).unsqueeze(0))
        model.l1.bias = torch.nn.Parameter(torch.tensor(1.0))
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 1.0) < 1e-5

        test_trainer.fit(model)
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize('test_dist_ckpt', [False, True])
    def test_base_checkpoints_are_not_overwritten(self, tmp_path, test_dist_ckpt):
        """Simulates already existing checkpoints in the ckpt directory and tests non-nemo ckpt versioning"""
        strategy = NLPDDPStrategy() if test_dist_ckpt else 'auto'
        test_trainer = pl.Trainer(
            accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4, strategy=strategy
        )
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_nemo_on_train_end": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleMCoreModel() if test_dist_ckpt else ExampleModel()

        ckpt_dir = Path(tmp_path / "test" / "checkpoints")
        assert not ckpt_dir.exists()

        # Fake existing 1st and last checkpoint
        suffix = '' if test_dist_ckpt else '.ckpt'
        ckpt_dir.mkdir(parents=True)
        ckpt_1 = ckpt_dir / f'default--val_loss=0.0000-epoch=1{suffix}'
        ckpt_2 = ckpt_dir / f'default--val_loss=0.0300-epoch=2{suffix}'

        if test_dist_ckpt:
            ckpt_1.mkdir()
            with open(ckpt_1 / 'metadata.json', 'w') as f:
                json.dump({'sharded_backend': 'xxx'}, f)
        else:
            ckpt_1.touch()
        # don't create 2nd checkpoint
        ckpt_nemo = ckpt_dir / 'default.nemo'
        ckpt_nemo.touch()

        # Train
        test_trainer.fit(model)

        # Check base checkpoint (without versioning)
        all_checkpoints = [p.name for p in Path(str(tmp_path / "test" / "checkpoints")).glob("*")]
        assert ckpt_1.exists(), all_checkpoints  # existed before
        assert ckpt_2.exists(), all_checkpoints
        assert ckpt_nemo.exists(), all_checkpoints  # existed before

        # Versioned checkpoints
        def _get_versioned_name(ckpt_name: Path, nemo: bool = False):
            if test_dist_ckpt and not nemo:
                # no suffix at all
                return ckpt_name.with_name(ckpt_name.name + '-v1')
            return ckpt_name.with_stem(ckpt_name.stem + '-v1')

        assert _get_versioned_name(ckpt_1).exists(), all_checkpoints
        assert not _get_versioned_name(ckpt_2).exists(), all_checkpoints  # ckpt2 didn't exist before
        # .nemo checkpoints are not versioned:
        assert not _get_versioned_name(ckpt_nemo, nemo=True).exists(), all_checkpoints

    @pytest.mark.unit
    def test_last_checkpoint_saved(self, tmp_path):
        max_steps = 64
        tmp_path = tmp_path / "test_1"

        class TestModel(ExampleModel):
            def train_dataloader(self):
                dataset = OnesDataset(64)
                return torch.utils.data.DataLoader(dataset, batch_size=1)

        trainer = pl.Trainer(
            accelerator='cpu', enable_checkpointing=False, logger=False, max_steps=max_steps, val_check_interval=0.33
        )
        exp_manager(
            trainer,
            {
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"filename": f"{{val_loss:.4f}}-{{epoch}}-{{step}}"},
            },
        )
        model = TestModel()
        trainer.fit(model)

        checkpoint_dir = Path(str(tmp_path / "checkpoints"))
        model_path = checkpoint_dir / "val_loss=0.0300-epoch=1-step=64-last.ckpt"
        last_saved_checkpoint = torch.load(model_path)
        assert max_steps == last_saved_checkpoint['global_step']

        # restart training, ensure global step starts correctly
        class AssertCallback(Callback):
            def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                assert trainer.global_step == max_steps

            def on_train_batch_end(
                self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int
            ) -> None:
                # we should only be running for one more step.
                assert trainer.global_step == max_steps + 1

        trainer = pl.Trainer(
            accelerator='cpu',
            enable_checkpointing=False,
            logger=False,
            max_steps=65,
            val_check_interval=0.33,
            callbacks=AssertCallback(),
        )
        exp_manager(
            trainer,
            {
                "explicit_log_dir": str(tmp_path),
                "checkpoint_callback_params": {"filename": f"{{val_loss:.4f}}-{{epoch}}-{{step}}"},
            },
        )
        model = TestModel()
        trainer.fit(model, ckpt_path=model_path)

    @pytest.mark.unit
    def test_resume_checkpoint_skip_validation(self, tmp_path):
        """Test to ensure that when we resume from a checkpoint, we do not re-run validation unnecessarily."""
        tmp_path = tmp_path / "test_2"

        def run_training(resume_path=None):
            class TestModel(ExampleModel):
                def train_dataloader(self):
                    dataset = OnesDataset(10)
                    return torch.utils.data.DataLoader(dataset, batch_size=1)

            class AssertCallback(Callback):
                recorded_validations = 0
                recorded_train_steps = 0

                def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                    self.recorded_validations += 1

                def on_train_batch_end(
                    self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch: Any, batch_idx: int
                ) -> None:
                    self.recorded_train_steps += 1

                def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                    if resume_path is not None:
                        # we should only run validation at the end of training.
                        assert self.recorded_validations == 1
                        # we continue from half way
                        assert self.recorded_train_steps == len(pl_module.train_dataloader()) // 2
                    else:
                        # we've run validation within the middle of training and at the end of training.
                        assert self.recorded_validations == 2
                        assert self.recorded_train_steps == len(pl_module.train_dataloader())

            model = TestModel()
            trainer = pl.Trainer(
                accelerator='cpu',
                enable_checkpointing=False,
                logger=False,
                callbacks=[AssertCallback()],
                val_check_interval=0.5,
                num_sanity_val_steps=0,
                max_epochs=1,
            )
            exp_manager(
                trainer,
                {"explicit_log_dir": str(tmp_path), "checkpoint_callback_params": {"filename": f"{{epoch}}-{{step}}"}},
            )
            trainer.fit(model, ckpt_path=resume_path)

        run_training()
        resume_path = tmp_path / 'checkpoints/epoch=0-step=5.ckpt'
        run_training(resume_path)

    def test_warning_validation_skipping_when_custom_epoch_loop(self, tmp_path):
        """When using validation skipping on restart with a custom epoch loop, we warn the user that we skip
        support to not interfere with their custom logic.
        """
        tmp_path = tmp_path / "test_3"

        class CustomLoop(_TrainingEpochLoop): ...

        trainer = pl.Trainer(
            accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=1, val_check_interval=0.33
        )
        ## _TrainingEpochLoop in PTL 2.0 takes trainer as an arg
        loop = CustomLoop(trainer)
        trainer.fit_loop.epoch_loop = loop
        with pytest.warns(UserWarning, match="Detected custom epoch loop"):
            exp_manager(trainer, {"explicit_log_dir": str(tmp_path)})

    def _write_fake_checkpoint(self, path, isdir, add_unfinished_marker):
        path = Path(path)
        if isdir:
            # fake distributed checkpoint
            path.mkdir(parents=True, exist_ok=True)
            (path / "dummy.txt").touch()
        else:
            # fake checkpoint file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        if add_unfinished_marker:
            NeMoModelCheckpoint.set_checkpoint_unfinished_marker(path)

    @pytest.mark.unit
    def test_skipped_unfinished_checkpoints_when_restoring(self, tmp_path):
        """
        Check if unfinished checkpoints are skipped during last checkpoint lookup.
        Logic of the test:
        - write multiple last checkpoints, some of them incomplete
        - ensure that the last complete checkpoint is found
        """

        test_dir = tmp_path / "test"
        checkpoints_dir = test_dir / "checkpoints"

        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=900-consumed_samples=1000.0.ckpt",
            isdir=False,
            add_unfinished_marker=False,
        )  # not last
        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=900-consumed_samples=1000.0-last.ckpt",
            isdir=False,
            add_unfinished_marker=True,
        )  # incomplete
        self._write_fake_checkpoint(
            checkpoints_dir
            / "mp_rank_00"
            / "megatron_gpt--val_loss=5.01-step=1100-consumed_samples=17600.0-last.ckpt",
            isdir=False,
            add_unfinished_marker=True,
        )  # incomplete
        self._write_fake_checkpoint(
            checkpoints_dir
            / "mp_rank_01"
            / "megatron_gpt--val_loss=5.01-step=1100-consumed_samples=17600.0-last.ckpt",
            isdir=False,
            add_unfinished_marker=True,
        )  # incomplete
        self._write_fake_checkpoint(
            checkpoints_dir
            / "mp_rank_00"
            / "megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0-last.ckpt",
            isdir=False,
            add_unfinished_marker=False,
        )  # ok
        self._write_fake_checkpoint(
            checkpoints_dir
            / "mp_rank_01"
            / "megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0-last.ckpt",
            isdir=False,
            add_unfinished_marker=False,
        )  # ok

        restored_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        exp_manager(
            restored_trainer,
            {"resume_if_exists": True, "explicit_log_dir": str(test_dir)},
        )

        # Check that last complete (w/o unifinished marker) checkpoint was found
        assert (
            Path(restored_trainer.ckpt_path).name
            == 'megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0-last.ckpt'
        )

    @pytest.mark.unit
    def test_skipped_unfinished_dist_checkpoints_when_restoring(self, tmp_path):
        """
        Check if unfinished distributed checkpoints are skipped during last checkpoint lookup.
        Logic of the test:
        - write multiple last checkpoints, some of them incomplete
        - ensure that the last complete checkpoint is found
        """

        test_dir = tmp_path / "test"
        checkpoints_dir = test_dir / "checkpoints"

        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0",
            isdir=True,
            add_unfinished_marker=False,
        )
        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0-last",
            isdir=True,
            add_unfinished_marker=False,
        )
        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=1100-consumed_samples=17600.0",
            isdir=True,
            add_unfinished_marker=False,
        )
        self._write_fake_checkpoint(
            checkpoints_dir / "megatron_gpt--val_loss=5.01-step=1100-consumed_samples=17600.0-last",
            isdir=True,
            add_unfinished_marker=True,
        )

        restored_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        exp_manager(
            restored_trainer,
            {"resume_if_exists": True, "explicit_log_dir": str(test_dir)},
        )

        # Check that last complete (w/o unifinished marker) checkpoint was found
        assert (
            Path(restored_trainer.ckpt_path).name
            == 'megatron_gpt--val_loss=5.01-step=1000-consumed_samples=16000.0-last'
        )

    @pytest.mark.unit
    def test_incomplete_checkpoints_cleanup(self, tmp_path):
        """
        Check if unfinished checkpoints are cleaned up when training starts
        Complete checkpoints should be left intact.
        """
        test_dir = tmp_path / "test"
        checkpoints_dir = test_dir / "checkpoints"

        complete_ckpts = {
            checkpoints_dir / "step=1-epoch=0.ckpt",
            checkpoints_dir / "step=2-epoch=0-last.ckpt",
            checkpoints_dir / "mp_rank_00" / "step=3-epoch=0-last.ckpt",
            checkpoints_dir / "tp_rank_00_pp_rank_000" / "step=4-epoch=0-last.ckpt",
            checkpoints_dir / "tp_rank_00_pp_rank_001" / "step=4-epoch=0-last.ckpt",
        }
        for ckpt_filepath in complete_ckpts:
            self._write_fake_checkpoint(ckpt_filepath, isdir=False, add_unfinished_marker=False)

        incomplete_ckpts = {
            checkpoints_dir / "step=11-epoch=1.ckpt",
            checkpoints_dir / "step=12-epoch=1-last.ckpt",
            checkpoints_dir / "mp_rank_00" / "step=13-epoch=1-last.ckpt",
            checkpoints_dir / "tp_rank_00_pp_rank_000" / "step=14-epoch=1-last.ckpt",
            checkpoints_dir / "tp_rank_00_pp_rank_001" / "step=14-epoch=1-last.ckpt",
        }
        for ckpt_filepath in incomplete_ckpts:
            self._write_fake_checkpoint(ckpt_filepath, isdir=False, add_unfinished_marker=True)

        # sanity check
        remaining_ckpts = {f for f in (test_dir / "checkpoints").rglob("*.ckpt") if f.is_file()}
        assert remaining_ckpts == (complete_ckpts | incomplete_ckpts)

        # marker without corresponding checkpoint should be removed during cleanup in exp_manager
        (checkpoints_dir / f"orphan-marker001-{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}").touch()

        # unfinished checkpoint with EMA part, both parts should be removed
        self._write_fake_checkpoint(
            checkpoints_dir / "incomplete01-EMA.ckpt",
            isdir=False,
            add_unfinished_marker=False,
        )
        self._write_fake_checkpoint(checkpoints_dir / "incomplete01.ckpt", isdir=False, add_unfinished_marker=True)

        # just EMA part - should be removed. NOTE marker path is the same for base part and for EMA part
        self._write_fake_checkpoint(
            checkpoints_dir / "incomplete02-EMA.ckpt",
            isdir=False,
            add_unfinished_marker=False,
        )
        (checkpoints_dir / f"incomplete02{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}").touch()

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=1)

        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 0, "save_last": False},
                "explicit_log_dir": str(test_dir),
            },
        )

        model = ExampleModel()
        test_trainer.fit(model)

        remaining_ckpts = {f for f in (test_dir / "checkpoints").rglob("*.ckpt") if f.is_file()}
        assert remaining_ckpts == complete_ckpts
        remaining_markers = list(checkpoints_dir.rglob(f"*{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}"))
        assert remaining_markers == []

    @pytest.mark.unit
    def test_incomplete_dist_checkpoints_cleanup(self, tmp_path):
        """
        Check if unfinished distributed checkpoints are cleaned up when training starts.
        Complete distributed checkpoints should be left intact.
        """

        test_dir = tmp_path / "test"
        checkpoints_dir = test_dir / "checkpoints"

        complete_dist_ckpts = {
            checkpoints_dir / "step=5-epoch=0",
            checkpoints_dir / "step=6-epoch=0-last",
        }
        for ckpt_dirpath in complete_dist_ckpts:
            self._write_fake_checkpoint(ckpt_dirpath, isdir=True, add_unfinished_marker=False)

        incomplete_dist_ckpts = {
            checkpoints_dir / "step=15-epoch=1",
            checkpoints_dir / "step=16-epoch=1-last",
        }
        for ckpt_dirpath in incomplete_dist_ckpts:
            self._write_fake_checkpoint(ckpt_dirpath, isdir=True, add_unfinished_marker=True)

        # marker without corresponding checkpoint should be removed during cleanup in exp_manager
        (checkpoints_dir / f"orphan-marker001-{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}").touch()

        remaining_dist_ckpts = {f for f in (test_dir / "checkpoints").glob("*") if f.is_dir()}
        assert remaining_dist_ckpts == (complete_dist_ckpts | incomplete_dist_ckpts)

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=1)

        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 0, "save_last": False},
                "explicit_log_dir": str(test_dir),
            },
        )

        model = ExampleModel()
        test_trainer.fit(model)

        remaining_dist_ckpts = {f for f in (test_dir / "checkpoints").glob("*") if f.is_dir()}
        assert remaining_dist_ckpts == complete_dist_ckpts
        remaining_markers = list(checkpoints_dir.rglob(f"*{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}"))
        assert remaining_markers == []

    _chkpt_path_and_marker_path_pairs = [
        ('a=1_b=1.c.d.e', f'a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
        ('a=1_b=1.c.d.e-last', f'a=1_b=1.c.d.e-last{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
        ('.ckpt/a=1_b=1.c.d.e.ckpt', f'.ckpt/a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
        ('.ckpt/a=1_b=1.c.d.e-EMA.ckpt', f'.ckpt/a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
        (
            '.ckpt/a=1_b=1.c.d.e-last.ckpt',
            f'.ckpt/a=1_b=1.c.d.e-last{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}',
        ),
        (
            '/tmp/mp_rank_00/a=1_b=1.c.d.e.ckpt',
            f'/tmp/a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}',
        ),
        (
            '/tmp/tp_rank_00_pp_rank_000/a=1_b=1.c.d.e.ckpt',
            f'/tmp/a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}',
        ),
        ('nemo/a=1_b=1.c.d.e.nemo', f'nemo/a=1_b=1.c.d.e{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
        ('nemo/a=1_b=1.c.d.e-last.nemo', f'nemo/a=1_b=1.c.d.e-last{NeMoModelCheckpoint.UNFINISHED_CHECKPOINT_SUFFIX}'),
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("chkpt_path, expected_marker_path", _chkpt_path_and_marker_path_pairs)
    def test_incomplete_checkpoints_marker_path(self, chkpt_path, expected_marker_path):
        """
        Ensure that unfinished checkpoint marker path is correctly formed.
        """
        marker_path = NeMoModelCheckpoint.format_checkpoint_unfinished_marker_path(chkpt_path)
        assert str(marker_path) == str(expected_marker_path)

    @pytest.mark.unit
    def test_invalid_checkpoints_removed_from_topk(self, tmp_path):
        """
        Ensure that invalid (unfinished, deleted) checkpoints are removed from topk when resuming.
        - Do few training steps and save checkpoints
        - Delete some checkpoints, mark some as unfinished
        - Resume training and verify that topk checkpoints are correct
        """
        test_dir = tmp_path / "test"
        checkpoints_dir = test_dir / "checkpoints"

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=7)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {
                    "save_top_k": 3,
                    "save_last": True,
                    "mode": 'max',
                    "monitor": 'epoch',
                    "filename": f"{{epoch}}",
                },
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        ckpt_filenames = {f.name for f in checkpoints_dir.rglob("*.ckpt") if f.is_file()}
        assert len(ckpt_filenames) == 4  # 3 top + 1 last
        assert 'epoch=7-last.ckpt' in ckpt_filenames
        assert 'epoch=6.ckpt' in ckpt_filenames
        assert 'epoch=5.ckpt' in ckpt_filenames
        assert 'epoch=4.ckpt' in ckpt_filenames

        # Mark 6th epoch checkpoint as unfinished and remove 5th epoch checkpoint,
        # so last valid candidate for topk is 4th epoch checkpoint
        NeMoModelCheckpoint.set_checkpoint_unfinished_marker(checkpoints_dir / 'epoch=6.ckpt')
        (checkpoints_dir / 'epoch=5.ckpt').unlink()

        test_trainer2 = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=9)
        exp_manager(
            test_trainer2,
            {
                "resume_if_exists": True,
                "checkpoint_callback_params": {
                    "save_top_k": 3,
                    "save_last": True,
                    "mode": 'max',
                    "monitor": 'epoch',
                    "filename": f"{{epoch}}",
                },
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer2.fit(model)

        ckpt_filenames = {f.name for f in checkpoints_dir.rglob("*.ckpt") if f.is_file()}
        # 3 top + 1 last
        assert len(ckpt_filenames) == 4
        assert 'epoch=9-last.ckpt' in ckpt_filenames
        assert 'epoch=8.ckpt' in ckpt_filenames
        assert 'epoch=7.ckpt' in ckpt_filenames
        assert 'epoch=4.ckpt' in ckpt_filenames
