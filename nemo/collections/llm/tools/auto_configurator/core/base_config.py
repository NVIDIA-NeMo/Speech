# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer, SentencePieceTokenizer
from nemo.collections.llm import PreTrainingDataModule
from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


class BaseConfig:
    def __init__(self, config=None):
        """
        Args:
            config (AutoConfigurator): auto configurator runner config.
        """

        self.config = config

        self.model = self.get_model()
        self.optim = self.get_optim()
        self.trainer = self.get_trainer()
        self.data = self.get_data()
        self.log = self.get_logger()
        self.run = self.get_run_config()
        self.tokenizer = self.get_tokenizer(config.tokenizer_type, config.tokenizer_path)

    def get_model(self):
        """Function that returns model config.

        Returns:
            Config: model config.
        """

        self.config.model.seq_length = self.config.seq_length

        return self.config.model

    def get_optim(self) -> Config[OptimizerConfig]:
        """Function that returns optimizer config.

        Returns:
            Config[OptimizerConfig]: optimizer config.
        """
        optim_params = {
            "optimizer": "adam",
            "lr": 1e-4,
            "min_lr": 1e-5,
            "use_distributed_optimizer": True,
            "bf16": True,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "clip_grad": 1.0,
            "adam_eps": 1e-5,
        }

        optim_config = Config(
            OptimizerConfig,
            **optim_params,
        )

        sched = Config(
            CosineAnnealingScheduler,
            warmup_steps=10,
            constant_steps=0,
            min_lr=optim_config.min_lr,
        )

        return Config(
            MegatronOptimizerModule,
            config=optim_config,
            lr_scheduler=sched,
        )

    def get_trainer(self) -> Config[nl.Trainer]:
        """Function that returns config for PTL trainer.

        Returns:
            Config[nl.Trainer]: trainer config.
        """

        trainer_config = {
            "accelerator": "gpu",
            "enable_checkpointing": False,
            "use_distributed_sampler": False,
            "max_epochs": None,
            "log_every_n_steps": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "accumulate_grad_batches": 1,
            "num_nodes": self.config.num_nodes,
            "devices": self.config.num_gpus,
            "max_steps": self.config.max_steps_per_run,
            "val_check_interval": self.config.max_steps_per_run,
        }

        strategy = Config(
            nl.MegatronStrategy,
            pipeline_dtype=torch.bfloat16,
        )

        return Config(
            nl.Trainer,
            **trainer_config,
            strategy=strategy,
            plugins=Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
            callbacks=[Config(TimingCallback)],
        )

    def get_tokenizer(self, tokenizer_type: str, tokenizer_path: str) -> Config:
        """Function that returns the tokenizer config.

        Args:
            tokenizer_type (str): tokenizer type.
            tokenizer_path (str): path to the tokenizer.

        Returns:
            Config: tokenizer config.
        """

        if tokenizer_type == "sentencepiece":
            return Config(SentencePieceTokenizer, model_path=tokenizer_path)
        else:
            return Config(AutoTokenizer, pretrained_model_name=tokenizer_path)

    def get_data(self) -> Config[PreTrainingDataModule]:
        """Function that returns dataset config.

        Returns:
            Config[PreTrainingDataModule]: data config.
        """

        # Data config
        data_config = {
            "paths": self.config.data_paths,
            "seq_length": self.config.seq_length,
            "global_batch_size": self.config.global_batch_size,
            "num_workers": 2,
            "index_mapping_dir": None,
        }

        # Define the tokenizer
        tokenizer = self.get_tokenizer(
            self.config.tokenizer_type,
            self.config.tokenizer_path,
        )

        return Config(
            PreTrainingDataModule,
            **data_config,
            tokenizer=tokenizer,
        )

    def get_logger(self) -> Config[nl.NeMoLogger]:
        """Function that returns the training strategy.

        Returns:
            Config[nl.NeMoLogger]: NeMo Logger config.
        """

        # Define TensorBoard Logger
        tb_logger = Config(TensorBoardLogger, save_dir="tb_logs")

        ckpt = Config(
            nl.ModelCheckpoint,
            monitor="reduced_train_loss",
            save_last=False,
            save_top_k=0,
        )

        return Config(
            nl.NeMoLogger,
            ckpt=ckpt,
            tensorboard=tb_logger,
            wandb=None,
            log_dir=self.config.path_to_logs,
        )

    def get_run_config(self) -> dict:
        """Function that returns config for cluster job.

        Returns:
            dict: cluster job config.
        """

        run_config = {
            "name": self.config.model.__class__.__name__,
            "time_limit": f"0-00:{self.config.max_minutes_per_run}:00",
        }

        return run_config


def calculate_model_size(
    gpu_count: int,
    max_training_days: float,
    model_size_in_b: float = None,
    tflops_per_gpu: int = 140,
    num_tokens_in_b: int = 300,
    model_name: str = "gpt3",
) -> float:
    """Estimates a model size to be trained given the constraints. If the
       model_size is provided, it estimates the time to train it with the given
       constraints.

    Example:
        output 5B params to train for 7 days with 160 GPUs.

    Args:
        gpu_count (int): number of gpus to use (num_nodes * gpus_per_node).
        max_training_days (float): number of days to train the model for.
        model_size_in_b (float): number of parameters in the model, if known.
        tflops_per_gpu (int): estimated number of TFLOPS/s per GPU.
        num_tokens_in_b (int): number of tokens to train the model for.
        model_name (str): name of the model.

    Returns:
        float: number of parameters to use for training.
    """

    # Model size is not known, must be estimated.
    if model_size_in_b is None:
        model_size_in_b = _estimate_model_size(
            max_training_days=max_training_days,
            gpu_count=gpu_count,
            tflops_per_gpu=tflops_per_gpu,
            num_tokens_in_b=num_tokens_in_b,
            model_name=model_name,
        )
    # Model size is known, so only time to train estimate is needed.
    else:
        max_training_days = _estimate_training_time(
            model_size_in_b=model_size_in_b,
            gpu_count=gpu_count,
            tflops_per_gpu=tflops_per_gpu,
            num_tokens_in_b=num_tokens_in_b,
            model_name=model_name,
        )

    print(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. This result assumes "
        f"you are training to {num_tokens_in_b}B tokens, and each GPU achieves "
        f"{tflops_per_gpu} TFLOPS."
    )
    return model_size_in_b


def _estimate_model_size(
    max_training_days: float,
    gpu_count: int,
    tflops_per_gpu: int,
    num_tokens_in_b: int,
    model_name: str,
) -> float:
    """Estimates model size given time and hardware constraints. It's only used if the model size is not provided by the user.

    Args:
        max_training_days (float): number of days to train the model for.
        gpu_count (int): number of gpus to use (num_nodes * gpus_per_node).
        tflops_per_gpu (int): estimated number of TFLOPS/s per GPU.
        num_tokens_in_b (int): number of tokens to train the model for.
        model_name (str): name of the model, such as gpt3, t5, mt5...

    Returns:
        float: number of parameters to use for training.

    Raises:
        NotImplementedError: if the model_name is not one of the supported models.
    """

    model_penalty = 0.87 if model_name == "mt5" else 1.0
    valid_models = ["gpt3", "t5", "mt5", "bert", "llama", "mixtral", "mistral", "gemma", "nemotron"]
    try:
        if model_name in valid_models:
            return round(
                model_penalty
                * (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
                / (8 * num_tokens_in_b * 1e9)
                / 1e9,
                2,
            )
        else:
            raise NotImplementedError
    except ValueError as err:
        print(f"Input values were not valid: {err}")
    except ZeroDivisionError as err:
        print(f"Cannot divide by zero. This can happen if num_tokens_in_b is zero: {err}")
    except NotImplementedError as err:
        print(f"Model size estimation is only available for {valid_models}: {err}")
    return None


def _estimate_training_time(
    model_size_in_b: float,
    gpu_count: int,
    tflops_per_gpu: int,
    num_tokens_in_b: int,
    model_name: str,
) -> float:
    """Estimates training time for a given model size and hardware constraint. To be used when a model size is provided by the user.

    Args:
        model_size_in_b (float): number of parameters to use for training.
        gpu_count (int): number of gpus to use (num_nodes * gpus_per_node).
        tflops_per_gpu (int): estimated number of TFLOPS/s per GPU.
        num_tokens_in_b (int): number of tokens to train the model for.
        model_name (str): name of the model, such as gpt3, t5, mt5...

    Returns:
        float: number of days it will take to train the model.

    Raises:
        NotImplementedError: if the model_name is not one of the supported models.
    """

    model_penalty = 1.15 if model_name == "mt5" else 1.0
    valid_models = ["gpt3", "t5", "mt5", "bert", "llama", "mixtral", "mistral", "gemma", "nemotron"]
    try:
        if model_name in valid_models:
            return round(
                model_penalty
                * (model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9)
                / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12),
                2,
            )
        else:
            raise NotImplementedError
    except ValueError as err:
        print(f"Input values were not valid: {err}")
    except ZeroDivisionError as err:
        print(f"Cannot divide by zero. This can happen if gpu_count or tflops_per_gpu are zero: {err}")
    except NotImplementedError as err:
        print(f"Training time estimation is only available for {valid_models}: {err}")
    return None
