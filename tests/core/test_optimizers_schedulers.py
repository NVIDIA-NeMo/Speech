# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import math
import random
from dataclasses import dataclass

import lightning.pytorch as pl
import omegaconf
import pytest
import torch
import torch.optim
from lightning.pytorch.utilities import rank_zero_only

from nemo.core import config, optim
from nemo.core.config.modelPT import ModelConfig, ModelConfigBuilder
from nemo.core.config.schedulers import ReduceLROnPlateauParams
from nemo.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS, PiecewiseLinear
from nemo.core.optim.optimizers import AVAILABLE_OPTIMIZERS
from nemo.utils import logging


class TempModel(torch.nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.layer = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


class OptCounter(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            group.setdefault('count', 0)

    def step(self, closure=None):
        for group in self.param_groups:
            group['count'] += 1
        super().step(closure)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.randn(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(pl.LightningModule):
    def __init__(self, batch_size, dataset_len, drop_last, max_steps):
        super().__init__()
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)
        self.batch_size = batch_size
        self.dataset_len = dataset_len
        self.drop_last = drop_last
        self.max_steps = max_steps

    def train_dataloader(self):
        dataset = RandomDataset(self.dataset_len)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=self.drop_last)

    def training_step(self, batch, batch_idx):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.ones(output.size()).to(output.device))
        return {"loss": output}

    def configure_optimizers(self):
        self.my_opt = OptCounter(self.parameters(), lr=0.02)
        return self.my_opt


class Callback(pl.callbacks.Callback):
    @rank_zero_only
    def on_train_end(self, trainer, module):
        count = module.my_opt.param_groups[0]['count']
        if trainer.global_step != count or trainer.global_step != module.max_steps:
            logging.debug(f"max_epochs: {trainer.max_epochs}")
            logging.debug(f"accumulate_grad_batches: {trainer.accumulate_grad_batches}")
            logging.debug(f"limit_train_batches: {trainer.limit_train_batches}")
            logging.debug(f"num_devices: {trainer.num_devices}")
            logging.debug(f"batch_size: {module.batch_size}")
            logging.debug(f"dataset_len: {module.dataset_len}")
            logging.debug(f"drop_last: {module.drop_last}")
            logging.debug(f"{len(trainer.train_dataloader)}")
            logging.debug(f"{trainer.num_training_batches }")

        self.assert_counts(trainer, module, count)

    def assert_counts(self, trainer, module, count):
        assert trainer.global_step == count, f"{trainer.global_step} != {count} != {module.max_steps}"
        assert trainer.global_step == module.max_steps, f"{trainer.global_step} != {count} != {module.max_steps}"


class SchedulerNoOpCallback(Callback):
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
        # pl_module.max_steps is "original" max steps without trainer extra steps.
        if (trainer.global_step + 1) % 3 == 0 and (trainer.global_step + 1) < pl_module.max_steps:
            schedulers = trainer.lr_scheduler_configs

            for scheduler in schedulers:
                # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                # as well as update the optimizer lr in all param groups
                scheduler.scheduler.last_epoch -= 2
                scheduler.scheduler.step()

            # Increase the max step count by 1
            trainer.fit_loop.epoch_loop.max_steps = trainer.fit_loop.epoch_loop.max_steps + 1

    def assert_counts(self, trainer, module, count):
        num_skips = module.max_steps // 3
        extra_steps = module.max_steps + num_skips
        assert trainer.global_step == count, f"{trainer.global_step} != {count} != {extra_steps}"
        assert trainer.global_step == extra_steps, f"{trainer.global_step} != {count} != {extra_steps}"


class TestOptimizersSchedulers:
    INITIAL_LR = 0.1
    MIN_LR = 1e-3
    MAX_STEPS = 10
    D_MODEL = 16

    # Apex optimizers require CUDA and this test is being run on CPU only tests
    @pytest.mark.unit
    def test_get_optimizer(self):
        model = TempModel()
        if torch.cuda.is_available():
            model.cuda()

        for opt_name in AVAILABLE_OPTIMIZERS.keys():
            if opt_name == 'fused_adam' or opt_name == 'megatron_fused_adam':
                if not torch.cuda.is_available():
                    continue
            if opt_name == 'distributed_fused_adam':
                # TODO: this test fails when run with all other tests, we need to move this test to nightly or CI
                continue
                # if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
                #     continue
                # if not torch.distributed.is_initialized():
                #     torch.distributed.init_process_group(
                #         'nccl', world_size=1, rank=0, store=torch.distributed.HashStore(),
                #     )
            opt_cls = optim.get_optimizer(opt_name)
            if opt_name == 'adafactor':
                # Adafactor's default mode uses relative_step without any lr.
                opt = opt_cls(model.parameters())
            else:
                opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

            assert isinstance(opt, AVAILABLE_OPTIMIZERS[opt_name])

    @pytest.mark.unit
    def test_register_optimizer(self):
        class TempOpt(torch.optim.SGD):
            pass

        class TempOptParams(config.optimizers.SGDParams):
            pass

        optim.register_optimizer('TempOpt', TempOpt, TempOptParams)

        model = TempModel()
        opt_cls = optim.get_optimizer('TempOpt')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        assert isinstance(opt, TempOpt)

    @pytest.mark.unit
    def test_optim_config_parse_bypass(self):
        basic_optim_config = {'weight_decay': 0.001, 'betas': [0.8, 0.5]}
        parsed_params = optim.parse_optimizer_args('novograd', basic_optim_config)
        assert parsed_params['weight_decay'] == basic_optim_config['weight_decay']
        assert parsed_params['betas'][0] == basic_optim_config['betas'][0]
        assert parsed_params['betas'][1] == basic_optim_config['betas'][1]

        dict_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = optim.parse_optimizer_args('novograd', dict_config)
        assert parsed_params['weight_decay'] == dict_config['weight_decay']
        assert parsed_params['betas'][0] == dict_config['betas'][0]
        assert parsed_params['betas'][1] == dict_config['betas'][1]

    @pytest.mark.unit
    def test_optim_config_parse_arg_by_name(self):
        basic_optim_config = {'name': 'auto', 'weight_decay': 0.001, 'betas': [0.8, 0.5]}
        parsed_params = optim.parse_optimizer_args('novograd', basic_optim_config)
        assert parsed_params['weight_decay'] == basic_optim_config['weight_decay']
        assert parsed_params['betas'][0] == basic_optim_config['betas'][0]
        assert parsed_params['betas'][1] == basic_optim_config['betas'][1]

        dict_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = optim.parse_optimizer_args('novograd', dict_config)
        assert parsed_params['weight_decay'] == dict_config['weight_decay']
        assert parsed_params['betas'][0] == dict_config['betas'][0]
        assert parsed_params['betas'][1] == dict_config['betas'][1]

        with pytest.raises(omegaconf.errors.ConfigKeyError):
            optim.parse_optimizer_args('sgd', dict_config)

    @pytest.mark.unit
    def test_optim_config_parse_arg_by_target(self):
        basic_optim_config = {
            '_target_': 'nemo.core.config.NovogradParams',
            'params': {'weight_decay': 0.001, 'betas': [0.8, 0.5]},
        }
        basic_optim_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = optim.parse_optimizer_args('novograd', basic_optim_config)
        assert parsed_params['weight_decay'] == basic_optim_config['params']['weight_decay']
        assert parsed_params['betas'][0] == basic_optim_config['params']['betas'][0]
        assert parsed_params['betas'][1] == basic_optim_config['params']['betas'][1]

        dict_config = omegaconf.OmegaConf.create(basic_optim_config)
        parsed_params = optim.parse_optimizer_args('novograd', dict_config)
        assert parsed_params['weight_decay'] == dict_config['params']['weight_decay']
        assert parsed_params['betas'][0] == dict_config['params']['betas'][0]
        assert parsed_params['betas'][1] == dict_config['params']['betas'][1]

        # Names are ignored when passing class path
        # This will be captured during optimizer instantiation
        output_config = optim.parse_optimizer_args('sgd', dict_config)
        sgd_config = vars(config.SGDParams())
        novograd_config = vars(config.NovogradParams())

        assert set(output_config.keys()) != set(sgd_config.keys())
        assert set(output_config.keys()) == set(novograd_config)

    @pytest.mark.unit
    def test_get_scheduler(self):
        model = TempModel()
        optimizer = optim.Novograd(model.parameters(), lr=self.INITIAL_LR)

        for sched_name in AVAILABLE_SCHEDULERS.keys():
            sched_cls = optim.lr_scheduler.get_scheduler(sched_name)

            try:
                sched = sched_cls(optimizer)
                assert isinstance(sched, AVAILABLE_SCHEDULERS[sched_name])
                continue
            except Exception:
                pass

            try:
                sched = sched_cls(optimizer, max_steps=self.MAX_STEPS)
                assert isinstance(sched, AVAILABLE_SCHEDULERS[sched_name])
                continue
            except Exception:
                pass

    @pytest.mark.unit
    def test_register_scheduler(self):
        class TempSched(optim.lr_scheduler.CosineAnnealing):
            pass

        class TempSchedParams(config.schedulers.CosineAnnealingParams):
            pass

        optim.lr_scheduler.register_scheduler('TempSched', TempSched, TempSchedParams)

        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)
        sched_cls = optim.lr_scheduler.get_scheduler('TempSched')
        sched = sched_cls(opt, max_steps=self.MAX_STEPS)

        assert isinstance(sched, TempSched)

    @pytest.mark.unit
    def test_sched_config_parse_simple(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        basic_sched_config = {'name': 'CosineAnnealing', 'max_steps': 10}
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, basic_sched_config)
        assert isinstance(scheduler_setup['scheduler'], optim.lr_scheduler.CosineAnnealing)

        dict_config = omegaconf.OmegaConf.create(basic_sched_config)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, dict_config)
        assert isinstance(scheduler_setup['scheduler'], optim.lr_scheduler.CosineAnnealing)

    @pytest.mark.unit
    def test_sched_config_parse_from_cls(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        basic_sched_config = {
            '_target_': 'nemo.core.config.CosineAnnealingParams',
            'params': {'min_lr': 0.1},
            'max_steps': self.MAX_STEPS,
        }
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, basic_sched_config)
        assert isinstance(scheduler_setup['scheduler'], optim.lr_scheduler.CosineAnnealing)

        dict_config = omegaconf.OmegaConf.create(basic_sched_config)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, dict_config)
        assert isinstance(scheduler_setup['scheduler'], optim.lr_scheduler.CosineAnnealing)

    @pytest.mark.unit
    def test_sched_config_parse_from_args(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # the override arguments next to `name`, as the example configs write them
        basic_sched_config = {
            'name': 'CosineAnnealing',
            'max_steps': self.MAX_STEPS,
            'args': {'name': 'auto', 'warmup_steps': None, 'warmup_ratio': 0.1, 'min_lr': 0.1, 'last_epoch': -1},
        }
        for sched_config in (basic_sched_config, omegaconf.OmegaConf.create(basic_sched_config)):
            scheduler = optim.lr_scheduler.prepare_lr_scheduler(opt, sched_config)['scheduler']
            assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealing)
            assert scheduler.warmup_steps == int(0.1 * self.MAX_STEPS)
            assert scheduler.min_lr == 0.1

        # the override arguments under `params`, and the parameter config selected by its name
        for args in (
            {'name': 'auto', 'params': {'min_lr': 0.1}},
            {'name': 'CosineAnnealingParams', 'params': {'min_lr': 0.1}},
            {'name': 'CosineAnnealingParams', 'min_lr': 0.1},
        ):
            sched_config = {'name': 'CosineAnnealing', 'max_steps': self.MAX_STEPS, 'args': args}
            scheduler = optim.lr_scheduler.prepare_lr_scheduler(opt, sched_config)['scheduler']
            assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealing)
            assert scheduler.min_lr == 0.1

        # `params` is normally flattened by the config version update before the overrides are read;
        # it stays nested when that update skips the config, here because of a value OmegaConf cannot hold
        sched_config = {
            'name': 'CosineAnnealing',
            'max_steps': self.MAX_STEPS,
            'args': {'name': 'auto', 'warmup_ratio': 0.1, 'params': {'min_lr': 0.1}},
            'unrelated_object': object(),
        }
        scheduler = optim.lr_scheduler.prepare_lr_scheduler(opt, sched_config)['scheduler']
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealing)
        assert scheduler.warmup_steps == int(0.1 * self.MAX_STEPS)
        assert scheduler.min_lr == 0.1

    @pytest.mark.unit
    def test_sched_config_from_model_config_builder(self):
        # the builder's scheduler config is the parameter config plus `name`, without `SchedConfig.min_lr`,
        # which not every scheduler takes: here torch's StepLR through a parameter config of its own
        @dataclass
        class StepLRParams(config.SchedulerParams):
            max_steps: int = 20
            step_size: int = 2
            gamma: float = 0.5

        model = TempModel()
        opt = torch.optim.SGD(model.parameters(), lr=self.INITIAL_LR)
        builder = ModelConfigBuilder(ModelConfig())
        builder.set_optim(config.SGDParams(lr=self.INITIAL_LR), StepLRParams())
        cfg = omegaconf.OmegaConf.structured(builder.model_cfg)  # as ModelPT holds the config
        scheduler = optim.lr_scheduler.prepare_lr_scheduler(opt, cfg.optim.sched)['scheduler']
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 2

        # a scheduler that takes `min_lr` keeps the value of its parameter config
        builder.set_optim(config.SGDParams(lr=self.INITIAL_LR), config.CosineAnnealingParams(max_steps=20, min_lr=0.1))
        cfg = omegaconf.OmegaConf.structured(builder.model_cfg)
        scheduler = optim.lr_scheduler.prepare_lr_scheduler(opt, cfg.optim.sched)['scheduler']
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealing)
        assert scheduler.min_lr == 0.1

        # a registered parameter config that does not derive from `SchedulerParams` converts as well
        builder.set_optim(config.SGDParams(lr=self.INITIAL_LR), ReduceLROnPlateauParams(patience=2))
        cfg = omegaconf.OmegaConf.structured(builder.model_cfg)
        assert cfg.optim.sched.name == 'ReduceLROnPlateau'
        assert cfg.optim.sched.patience == 2
        assert 'last_epoch' not in cfg.optim.sched

        builder.set_optim(config.SGDParams(lr=self.INITIAL_LR))
        assert omegaconf.OmegaConf.structured(builder.model_cfg).optim.sched is None

    @pytest.mark.unit
    def test_sched_config_parse_reduce_on_plateau(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)
        reduce_on_plateau_parameters = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 1,
            'threshold': 1e-4,
            'threshold_mode': 'rel',
            'min_lr': 1e-6,
            'eps': 1e-7,
            'cooldown': 1,
        }
        basic_sched_config = {
            'name': 'ReduceLROnPlateau',
            'monitor': 'val_loss',
            'reduce_on_plateau': True,
            'max_steps': self.MAX_STEPS,
        }
        basic_sched_config.update(reduce_on_plateau_parameters)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, basic_sched_config)
        assert isinstance(scheduler_setup['scheduler'], torch.optim.lr_scheduler.ReduceLROnPlateau)
        for k, v in reduce_on_plateau_parameters.items():
            if k == 'min_lr':
                k += 's'
                v = [v]
            found_v = getattr(scheduler_setup['scheduler'], k)
            assert (
                found_v == v
            ), f"Wrong value `{repr(found_v)}` for `ReduceLROnPlateau` parameter `{k}`. Expected `{repr(v)}`."
        dict_config = omegaconf.OmegaConf.create(basic_sched_config)
        scheduler_setup = optim.lr_scheduler.prepare_lr_scheduler(opt, dict_config)
        assert isinstance(scheduler_setup['scheduler'], torch.optim.lr_scheduler.ReduceLROnPlateau)
        for k, v in reduce_on_plateau_parameters.items():
            if k == 'min_lr':
                k += 's'
                v = [v]
            found_v = getattr(scheduler_setup['scheduler'], k)
            assert (
                found_v == v
            ), f"Wrong value `{repr(found_v)}` for `ReduceLROnPlateau` parameter `{k}`. Expected `{repr(v)}`."

    @pytest.mark.unit
    def test_WarmupPolicy(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupPolicy(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] == self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupPolicy(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 4:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] == self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_WarmupHoldPolicy(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupHoldPolicy(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] == self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 4:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] == self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 4:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] == self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_WarmupAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.WarmupAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.WarmupAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.WarmupHoldPolicy(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 4:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] == self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_SquareAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.SquareAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.SquareAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_SquareRootAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.SquareRootAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.SquareRootAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_CosineAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.CosineAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.CosineAnnealing(opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup + Constant steps available
        policy = optim.lr_scheduler.CosineAnnealing(
            opt, warmup_steps=3, constant_steps=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 3:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR + 1e-5
            elif i > 3 and i <= 8:
                assert policy.get_last_lr()[0] == policy._get_lr(i)[0]
            else:
                assert policy.get_last_lr()[0] == self.MIN_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    # Noam scheduler should decay past MAX_STEPS - run two schedulers in parallel to test it
    @pytest.mark.unit
    def test_NoamAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt1 = opt_cls(model.parameters(), lr=self.INITIAL_LR)
        opt2 = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy1 = optim.lr_scheduler.NoamAnnealing(
            opt1, d_model=self.D_MODEL, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        policy2 = optim.lr_scheduler.NoamAnnealing(
            opt2, d_model=self.D_MODEL, max_steps=self.MAX_STEPS * 2, min_lr=self.MIN_LR
        )
        initial_lr = policy1.get_last_lr()[0]

        assert initial_lr == self.D_MODEL ** (-0.5) * self.INITIAL_LR

        for i in range(self.MAX_STEPS * 2):
            assert self.MIN_LR < policy1.get_last_lr()[0] <= self.INITIAL_LR
            assert policy1.get_last_lr()[0] == policy2.get_last_lr()[0]
            opt1.step()
            opt2.step()
            policy1.step()
            policy2.step()

        # Warmup steps available
        policy1 = optim.lr_scheduler.NoamAnnealing(
            opt1, d_model=self.D_MODEL, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        policy2 = optim.lr_scheduler.NoamAnnealing(
            opt2, d_model=self.D_MODEL, warmup_steps=5, max_steps=self.MAX_STEPS * 2, min_lr=self.MIN_LR
        )
        initial_lr = policy1.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS * 2):
            if i <= 5:
                assert policy1.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert self.MIN_LR < policy1.get_last_lr()[0] < self.INITIAL_LR
                assert policy1.get_last_lr()[0] == policy2.get_last_lr()[0]

            opt1.step()
            opt2.step()
            policy1.step()
            policy2.step()

    @pytest.mark.unit
    def test_PolynomialDecayAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.PolynomialDecayAnnealing(
            opt, power=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.PolynomialDecayAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_PolynomialHoldDecayAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, power=2, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, power=2, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup + Hold steps available
        policy = optim.lr_scheduler.PolynomialHoldDecayAnnealing(
            opt, warmup_steps=5, hold_steps=3, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR, power=2
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 4:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            elif i <= 8:
                assert policy.get_last_lr()[0] == self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_InverseSquareRootAnnealing(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.InverseSquareRootAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR

        # Warmup steps available
        policy = optim.lr_scheduler.InverseSquareRootAnnealing(
            opt, warmup_steps=5, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr < self.INITIAL_LR

        for i in range(self.MAX_STEPS):
            if i <= 5:
                assert policy.get_last_lr()[0] <= self.INITIAL_LR
            else:
                assert policy.get_last_lr()[0] < self.INITIAL_LR

            opt.step()
            policy.step()

        policy.step()
        final_lr = policy.get_last_lr()[0]

        assert final_lr == self.MIN_LR


class TestWarmupHoldAnnealSchedulers:
    INITIAL_LR = 0.1
    MIN_LR = 0.01
    MAX_STEPS = 100

    @pytest.mark.unit
    def test_WarmupHoldAnnealOneMinusSquareRoot(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # Test case 1: No warmup, no hold
        policy = optim.lr_scheduler.WarmupHoldAnnealOneMinusSquareRoot(
            opt, warmup_ratio=None, hold_ratio=None, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]
        assert initial_lr == self.INITIAL_LR

        # Simulate training steps
        lrs = []
        for i in range(self.MAX_STEPS):
            current_lr = policy.get_last_lr()[0]
            lrs.append(current_lr)
            assert current_lr <= self.INITIAL_LR
            opt.step()
            policy.step()

        # Check final LR
        policy.step()
        final_lr = policy.get_last_lr()[0]
        assert final_lr == self.MIN_LR

        # Test case 2: With warmup and hold
        warmup_ratio = 0.1  # 10% warmup
        hold_ratio = 0.2  # 20% hold
        warmup_steps = int(warmup_ratio * self.MAX_STEPS)
        hold_steps = int(hold_ratio * self.MAX_STEPS)

        policy = optim.lr_scheduler.WarmupHoldAnnealOneMinusSquareRoot(
            opt, warmup_ratio=warmup_ratio, hold_ratio=hold_ratio, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )

        initial_lr = policy.get_last_lr()[0]
        assert initial_lr < self.INITIAL_LR  # Should start at a lower LR

        # Simulate training steps
        lrs = []
        for i in range(self.MAX_STEPS):
            current_lr = policy.get_last_lr()[0]
            lrs.append(current_lr)

            # During warmup, LR should increase
            if i < warmup_steps:
                if i > 0:
                    assert current_lr >= lrs[i - 1]
                assert current_lr <= self.INITIAL_LR

            # During hold, LR should remain constant
            elif i < warmup_steps + hold_steps:
                assert abs(current_lr - self.INITIAL_LR) < 1e-6

            # During annealing, LR should decrease
            else:
                if i > warmup_steps + hold_steps:
                    assert current_lr <= lrs[i - 1]

            opt.step()
            policy.step()

        # Check final LR
        policy.step()
        final_lr = policy.get_last_lr()[0]
        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_WarmupHoldAnnealLinear(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # Test case 1: No warmup, no hold
        policy = optim.lr_scheduler.WarmupHoldAnnealLinear(
            opt, warmup_ratio=None, hold_ratio=None, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )
        initial_lr = policy.get_last_lr()[0]
        assert initial_lr == self.INITIAL_LR

        # Simulate training steps
        lrs = []
        for i in range(self.MAX_STEPS):
            current_lr = policy.get_last_lr()[0]
            lrs.append(current_lr)
            assert current_lr <= self.INITIAL_LR
            opt.step()
            policy.step()

        # Check final LR
        policy.step()
        final_lr = policy.get_last_lr()[0]
        assert final_lr == self.MIN_LR

        # Test case 2: With warmup and hold
        warmup_ratio = 0.1  # 10% warmup
        hold_ratio = 0.2  # 20% hold
        warmup_steps = int(warmup_ratio * self.MAX_STEPS)
        hold_steps = int(hold_ratio * self.MAX_STEPS)

        policy = optim.lr_scheduler.WarmupHoldAnnealLinear(
            opt, warmup_ratio=warmup_ratio, hold_ratio=hold_ratio, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR
        )

        initial_lr = policy.get_last_lr()[0]
        assert initial_lr < self.INITIAL_LR  # Should start at a lower LR

        # Simulate training steps
        lrs = []
        for i in range(self.MAX_STEPS):
            current_lr = policy.get_last_lr()[0]
            lrs.append(current_lr)

            # During warmup, LR should increase
            if i < warmup_steps:
                if i > 0:
                    assert current_lr >= lrs[i - 1]
                assert current_lr <= self.INITIAL_LR

            # During hold, LR should remain constant
            elif i < warmup_steps + hold_steps:
                assert abs(current_lr - self.INITIAL_LR) < 1e-6

            # During annealing, LR should decrease
            else:
                if i > warmup_steps + hold_steps:
                    assert current_lr <= lrs[i - 1]

            opt.step()
            policy.step()

        # Check final LR
        policy.step()
        final_lr = policy.get_last_lr()[0]
        assert final_lr == self.MIN_LR

    @pytest.mark.unit
    def test_CosineAnnealing_with_noop_steps(self):
        model = TempModel()
        opt_cls = optim.get_optimizer('novograd')
        opt = opt_cls(model.parameters(), lr=self.INITIAL_LR)

        # No warmup case
        policy = optim.lr_scheduler.CosineAnnealing(opt, max_steps=self.MAX_STEPS, min_lr=self.MIN_LR)
        initial_lr = policy.get_last_lr()[0]

        assert initial_lr == self.INITIAL_LR

        update_steps = 0
        for i in range(self.MAX_STEPS):
            assert policy.get_last_lr()[0] <= self.INITIAL_LR
            opt.step()
            policy.step()

            # Perform a No-Op for scheduler every 2 steps
            if i % 2 == 0:
                policy.last_epoch -= 1
            else:
                update_steps += 1

        policy.step()
        update_steps += 1

        assert update_steps < self.MAX_STEPS

        final_lr = policy.get_last_lr()[0]
        assert final_lr > self.MIN_LR

        # update step = true number of updates performed after some number of skipped steps
        true_end_lr = policy._get_lr(step=update_steps)[0]
        assert final_lr == true_end_lr

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    def test_max_step_computation(self):
        def train(
            max_epochs, accumulate_grad_batches, limit_train_batches, devices, batch_size, dataset_len, drop_last
        ):
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                strategy="ddp_spawn",
                accelerator="cpu",
                devices=devices,
                accumulate_grad_batches=accumulate_grad_batches,
                limit_train_batches=limit_train_batches,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            max_steps = optim.lr_scheduler.compute_max_steps(
                max_epochs,
                accumulate_grad_batches,
                limit_train_batches,
                devices,
                dataset_len,
                batch_size,
                drop_last,
            )
            model = ExampleModel(batch_size, dataset_len, drop_last, max_steps)
            trainer.callbacks.append(Callback())
            trainer.fit(model)

        # This test will break once we and lightning upgrade to pytorch 1.7.0 due to a bug fix in pytorch 1.7.0
        train(
            31,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            devices=9,
            batch_size=60,
            dataset_len=1613,
            drop_last=True,
        )
        train(
            5,
            accumulate_grad_batches=1,
            limit_train_batches=0.5,
            devices=4,
            batch_size=97,
            dataset_len=498,
            drop_last=False,
        )
        train(
            5,
            accumulate_grad_batches=8,
            limit_train_batches=0.5,
            devices=4,
            batch_size=54,
            dataset_len=629,
            drop_last=True,
        )
        train(
            5,
            accumulate_grad_batches=1,
            limit_train_batches=0.5,
            devices=1,
            batch_size=68,
            dataset_len=488,
            drop_last=False,
        )
        for _ in range(5):
            drop_last = bool(random.randint(0, 1))
            accumulate_grad_batches = random.randint(1, 10)

            limit_train_batches_int = random.randint(1, 10)
            limit_train_batches_float = random.uniform(0.5, 1)
            limit_train_batches = random.choice([limit_train_batches_int, limit_train_batches_float])
            max_epochs = random.randint(4, 20)
            devices = random.randint(1, 5)
            dataset_len = random.randint(20, devices * 500)
            batch_size = random.randint(math.ceil(5.0 / devices), min(dataset_len // devices, 128))
            train(
                max_epochs,
                accumulate_grad_batches,
                limit_train_batches,
                devices,
                batch_size,
                dataset_len,
                drop_last,
            )

    @pytest.mark.unit
    @pytest.mark.run_only_on('CPU')
    def test_max_step_computation_with_sched_no_ops(self):
        def train(
            max_steps, accumulate_grad_batches, limit_train_batches, devices, batch_size, dataset_len, drop_last
        ):
            trainer = pl.Trainer(
                max_steps=max_steps,
                strategy="ddp_spawn",
                accelerator="cpu",
                devices=devices,
                accumulate_grad_batches=accumulate_grad_batches,
                limit_train_batches=limit_train_batches,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            model = ExampleModel(batch_size, dataset_len, drop_last, max_steps)
            trainer.callbacks.append(SchedulerNoOpCallback())
            trainer.fit(model)

        # This test will break once we and lightning upgrade to pytorch 1.7.0 due to a bug fix in pytorch 1.7.0
        train(
            max_steps=20,
            accumulate_grad_batches=1,
            limit_train_batches=1.0,
            devices=4,
            batch_size=60,
            dataset_len=2000,
            drop_last=True,
        )


def _sgd_with_group_lrs(*lrs: float) -> torch.optim.Optimizer:
    groups = [{'params': [torch.nn.Parameter(torch.zeros(2))], 'lr': lr} for lr in lrs]
    return torch.optim.SGD(groups, lr=lrs[0])


def _learning_rates_per_step(scheduler, optimizer: torch.optim.Optimizer, steps: int) -> list:
    """The learning rate the optimizer uses at each of ``steps`` optimizer steps."""
    rates = []
    for _ in range(steps):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return rates


class TestPiecewiseLinear:
    @pytest.mark.unit
    def test_registry_and_config(self):
        assert AVAILABLE_SCHEDULERS['PiecewiseLinear'] is PiecewiseLinear
        sched_config = {
            'name': 'PiecewiseLinear',
            'max_steps': 20,
            'relative': True,
            'boundaries': [0.0, 0.5, 1.0],
            'values': [0.1, 1.0, 0.0],
        }
        for cfg in (sched_config, omegaconf.OmegaConf.create(sched_config)):
            setup = optim.lr_scheduler.prepare_lr_scheduler(_sgd_with_group_lrs(1.0), cfg)
            scheduler = setup['scheduler']
            assert isinstance(scheduler, PiecewiseLinear)
            assert setup['interval'] == 'step'
            assert scheduler.multiplier(0.25) == pytest.approx(0.55)

        by_class = {
            '_target_': 'nemo.core.config.PiecewiseLinearParams',
            'params': {'boundaries': [0, 10], 'values': [0.0, 1.0]},
            'max_steps': 20,
        }
        scheduler = optim.lr_scheduler.prepare_lr_scheduler(_sgd_with_group_lrs(1.0), by_class)['scheduler']
        assert isinstance(scheduler, PiecewiseLinear)
        assert not scheduler.relative
        assert scheduler.multiplier(5) == 0.5

        assert config.schedulers.PiecewiseLinearParams().boundaries is None

    @pytest.mark.unit
    def test_steps(self):
        optimizer = _sgd_with_group_lrs(1.0)
        scheduler = PiecewiseLinear(optimizer, boundaries=[0, 4, 8], values=[0.1, 1.0, 0.5])
        rates = _learning_rates_per_step(scheduler, optimizer, steps=11)
        assert rates == pytest.approx([0.1, 0.325, 0.55, 0.775, 1.0, 0.875, 0.75, 0.625, 0.5, 0.5, 0.5])

    @pytest.mark.unit
    def test_relative_to_max_steps(self):
        optimizer = _sgd_with_group_lrs(1.0)
        scheduler = PiecewiseLinear(
            optimizer, boundaries=[0.0, 0.5, 1.0], values=[0.0, 1.0, 0.0], relative=True, max_steps=20
        )
        rates = _learning_rates_per_step(scheduler, optimizer, steps=21)
        assert rates[0] == 0.0
        assert rates[5] == pytest.approx(0.5)
        assert rates[10] == pytest.approx(1.0)
        assert rates[15] == pytest.approx(0.5)
        assert rates[20] == pytest.approx(0.0)
        # the same schedule for a longer training
        longer = PiecewiseLinear(
            _sgd_with_group_lrs(1.0), boundaries=[0.0, 0.5, 1.0], values=[0.0, 1.0, 0.0], relative=True, max_steps=200
        )
        assert longer.multiplier(100 / 200) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_values_before_and_after_the_boundaries(self):
        optimizer = _sgd_with_group_lrs(1.0)
        scheduler = PiecewiseLinear(optimizer, boundaries=[2, 4], values=[0.5, 1.0])
        rates = _learning_rates_per_step(scheduler, optimizer, steps=7)
        assert rates == pytest.approx([0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0])

    @pytest.mark.unit
    def test_multiplies_each_group_learning_rate(self):
        optimizer = _sgd_with_group_lrs(1.0, 0.1)
        scheduler = PiecewiseLinear(optimizer, boundaries=[0, 10], values=[0.0, 1.0])
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        assert scheduler.get_last_lr() == pytest.approx([0.5, 0.05])
        assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([0.5, 0.05])

    @pytest.mark.unit
    def test_state_dict_round_trip(self):
        optimizer = _sgd_with_group_lrs(1.0)
        scheduler = PiecewiseLinear(optimizer, boundaries=[0, 10], values=[0.0, 1.0])
        _learning_rates_per_step(scheduler, optimizer, steps=3)
        state = scheduler.state_dict()
        restored = PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[0, 10], values=[0.0, 1.0])
        restored.load_state_dict(state)
        assert restored.last_epoch == 3
        assert restored.get_last_lr() == pytest.approx([0.3])

    @pytest.mark.unit
    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="needs `boundaries` and `values`"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=None, values=None)
        with pytest.raises(ValueError, match="same length"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[0, 1], values=[1.0])
        with pytest.raises(ValueError, match="same length"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[], values=[])
        with pytest.raises(ValueError, match="strictly increasing"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[0, 1, 1], values=[1.0, 1.0, 1.0])
        for max_steps in (None, 0, -5, 10.5, "10"):
            with pytest.raises(ValueError, match="positive integer `max_steps`"):
                PiecewiseLinear(
                    _sgd_with_group_lrs(1.0), boundaries=[0, 1], values=[1.0, 1.0], relative=True, max_steps=max_steps
                )
        with pytest.raises(ValueError, match="finite"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[0, float('nan')], values=[1.0, 1.0])
        with pytest.raises(ValueError, match="finite"):
            PiecewiseLinear(_sgd_with_group_lrs(1.0), boundaries=[0, 1], values=[1.0, float('inf')])

    @pytest.mark.unit
    def test_production_schedule(self):
        # a training of 88,004 steps with a warmup to 45%, a decay to 90% and a final decay,
        # as a trainer with `max_steps` and relative boundaries configures it
        max_steps = 88_004
        optimizer = _sgd_with_group_lrs(1.0)
        scheduler = PiecewiseLinear(
            optimizer,
            boundaries=[0.0, 0.45, 0.9, 1.0],
            values=[1e-5, 4e-3, 1e-5, 1e-6],
            relative=True,
            max_steps=max_steps,
        )
        rates = _learning_rates_per_step(scheduler, optimizer, steps=max_steps + 1)
        assert rates[0] == 1e-5
        assert rates[-1] == pytest.approx(1e-6)
        peak = max(range(len(rates)), key=rates.__getitem__)
        assert peak in (39601, 39602)  # 0.45 * 88004 = 39601.8
        assert rates[peak] == pytest.approx(4e-3, rel=1e-4)
        assert all(later >= earlier for earlier, later in zip(rates[:peak], rates[1 : peak + 1]))
        assert all(later <= earlier for earlier, later in zip(rates[peak:], rates[peak + 1 :]))
        assert rates[79204] == pytest.approx(1e-5, rel=1e-3)  # 0.9 * 88004 = 79203.6
