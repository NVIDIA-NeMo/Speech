# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
from collections import OrderedDict
from typing import List, Optional

import numpy as np

from nemo.collections.llm.api import pretrain
from nemo.collections.llm.tools.auto_configurator.core.training_config import generate_grid_search_configs
from nemo.collections.llm.tools.auto_configurator.core.utils import _calculate_model_size, generic_base_config
from nemo.collections.llm.utils import Config, Partial
from nemo.utils import logging

SUPPORTED_MODELS = OrderedDict(
    [
        ("gpt3", "GPT"),
        ("bert", "Bert"),
        ("t5", "T5"),
        ("llama", "Llama"),
        ("mixtral", "Mixtral"),
        ("mistral", "Mistral"),
        ("gemma", "Gemma"),
        ("nemotron", "Nemotron"),
        ("starcoder", "Starcoder"),
        ("qwen", "Qwen"),
    ]
)


class AutoConfigurator:
    """Auto Configurator runner config class."""

    def __init__(
        self,
        recipe: Partial = None,
        path_to_logs: str = None,
        gpu_memory_gb: Optional[int] = 80,
        tensor_parallel_sizes: Optional[List[int]] = "auto",
        pipeline_parallel_sizes: Optional[List[int]] = "auto",
        micro_batch_sizes: Optional[List[int]] = "auto",
        context_parallel_sizes: Optional[List[int]] = [1],
        expert_parallel_sizes: Optional[List[int]] = [1],
        min_model_parallel_size: Optional[int] = "auto",
        max_model_parallel_size: Optional[int] = "auto",
        num_tokens_in_b: Optional[int] = 1400,
        tflops_per_gpu: Optional[int] = 140,
        max_minutes_per_run: Optional[int] = 30,
        max_training_days: Optional[int] = 2,
        max_steps_per_run: Optional[int] = 50,
        vocab_size: Optional[int] = 32000,
        calculate_model_size: Optional[bool] = False,
    ):
        """
        Args:
            recipe (Partial): recipe to be used for training.
            path_to_logs (str): path to the directory where the logs will be stored.
            gpu_memory_gb (Optional[int]): memory per GPU, in GB. Currently 40GB and 80GB A100s/H100s supported.
            tensor_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation,
                or a list, such as [1, 2, 4, 8].
            pipeline_parallel_sizes (Optional[List[int]]): set to "auto" to use our recommendation,
                or a list, such as [1, 2, 4, 8].
            micro_batch_sizes (Optional[List[int]]): set to "auto" to use our recommendation,
                or a list, such as [1, 2, 4, 8].
            context_parallel_sizes (Optional[List[int]]): model context parallel size. A list, such as [1, 2, 4, 8].
            expert_parallel_sizes (Optional[List[int]]): model expert parallel size. A list, such as [1, 2, 4, 8].
            min_model_parallel_size (Optional[int]): set to "auto" to use our recommendation,
                or a value for the minimum desired parallelism.
            max_model_parallel_size (Optional[int]): set to "auto" to use our recommendation,
                or a value for the maximum desired parallelism.
            num_tokens_in_b (Optional[int]): number of tokens in billions in train dataset.
            tflops_per_gpu (Optional[int]): estimated tflops per GPU.
            max_minutes_per_run (Optional[int]): maximum number of minutes per run for the grid search.
            max_training_days (Optional[int]): number of days expected model to be trained.
            max_steps_per_run (Optional[int]): maximum number of steps per run for the grid search.
            vocab_size (Optional[int]): size of tokenizer vocabulary.
            calculate_model_size (Optional[bool]): whether the AutoConfigurator should calculate the model size or not.
        """

        # Print out the config
        config = locals()
        config.pop('self')
        for key, value in config.items():
            setattr(self, key, value)
        logging.info(self._get_message(config))

        model_type = self._get_model_type(recipe.model.config)
        assert model_type in SUPPORTED_MODELS, f"model_type must be set to one of {list(SUPPORTED_MODELS.keys())}."

        if model_type in ["bert", "t5"]:
            assert (
                recipe.data.seq_length <= 2048
            ), "seq_length higher than 2048 is not supported for bert and t5 models."
        else:
            assert recipe.data.seq_length in [
                2048,
                4096,
                8192,
                16384,
                32768,
            ], "Available seq_length list for GPT-based models: [2048, 4096, 8192, 16384, 32768]."
        assert path_to_logs, "path_to_logs parameter must be specified."
        assert num_tokens_in_b > 0, "num_tokens_in_b must be an int larger than zero."
        assert tflops_per_gpu > 0, "tflops_per_gpu must be an int larger than zero."
        self.num_gpus = recipe.trainer.devices
        self.num_nodes = recipe.trainer.num_nodes
        gpu_count = self.num_nodes * self.num_gpus
        assert gpu_count > 0, "num_nodes * gpus_per_node must be an int larger than zero."
        assert gpu_memory_gb in (
            40,
            80,
        ), "gpu_memory_gb can only be 40 or 80."
        assert max_minutes_per_run >= 10, "max_minutes_per_run must be an int and be at least 10 minutes."
        assert max_steps_per_run >= 10, "max_steps_per_run must be an int and be at least 10 minutes."

        self.model_type = model_type
        self.model_size_in_b = self._get_model_size(
            recipe.model.config,
            model_type,
            config['vocab_size'],
            config['calculate_model_size'],
        )
        self.gpu_count = gpu_count
        self.seq_length = recipe.data.seq_length
        self.global_batch_size = recipe.data.global_batch_size

    def _get_message(self, config: dict) -> str:
        """
        Function that returns runner config line by line.

        Args:
            config (dict): runner config.

        Returns:
            str: runner config params.
        """

        message = "AutoConfigurator runner config:\n"
        for key, value in config.items():
            message += f"{key}: {value}\n"

        return message

    def _get_model_type(self, model: Config) -> str:
        """
        Function that returns model type from model class name.

        Args:
            models (Config): model object.

        Returns:
            str: model type.
        """

        for k, v in SUPPORTED_MODELS.items():
            if v in str(model):
                return k

        return None

    def _get_model_size(
        self,
        model: Config,
        model_type: str,
        vocab_size: int,
        calculate_model_size: bool,
    ) -> int:
        """
        Function that returns model size from model class name.

        Args:
            model (Config): model class name.
            model_type (str): model type.
            vocab_size (int): vocab size.
            calculate_model_size (bool): whether the AutoConfigurator should calculate the model size or not.

        Returns:
            float: model size.
        """

        if calculate_model_size:
            return None

        if model_type != "bert":
            match = re.search(r'(\d+)([BM])', str(model))
            if match:
                size = int(match.group(1))
                measure = match.group(2)
                if measure == 'B':
                    return size
                elif measure == 'M':
                    return size / 1000  # Convert millions to billions
        elif model_type == "bert":
            return np.round(
                _calculate_model_size(
                    vocab_size=vocab_size,
                    seq_length=model.seq_length,
                    hidden_size=model.hidden_size,
                    num_layers=model.num_layers,
                    ffn_size=model.ffn_hidden_size,
                    att_heads=model.num_attention_heads,
                    model_name=model_type,
                ),
                3,
            )
        return None


def generate_configs(runner_config: AutoConfigurator = None) -> dict:
    """
    Function that returns a dictionary of Partial configs.

    Args:
        config (AutoConfigurator): Auto Configurator object.

    Returns:
        dict: dictionary of Partial configs.
    """

    # Generate base config for the given model size
    base_config, train_config = generic_base_config(runner_config)

    # Launch grid search for training constraints
    base_config, train_configs = generate_grid_search_configs(base_config, train_config)

    configs = {}
    for name, config in train_configs.items():
        trainer = copy.deepcopy(base_config.trainer)
        data = copy.deepcopy(base_config.data)
        log = copy.deepcopy(base_config.log)

        # Set data params
        data.micro_batch_size = config.get("micro_batch_size")
        data.global_batch_size = config.get("global_batch_size")

        # Set strategy params
        trainer.strategy.tensor_model_parallel_size = config.get("tensor_model_parallel_size")
        trainer.strategy.pipeline_model_parallel_size = config.get("pipeline_model_parallel_size")
        trainer.strategy.context_parallel_size = config.get("context_parallel_size")
        trainer.strategy.expert_model_parallel_size = config.get("expert_model_parallel_size")
        trainer.strategy.virtual_pipeline_model_parallel_size = config.get(
            "virtual_pipeline_model_parallel_size", None
        )
        if config.get("tensor_model_parallel_size") > 1:
            trainer.strategy.sequence_parallel = True
        trainer.max_steps = config.get("max_steps")
        trainer.log_every_n_steps = 1

        log.log_dir = os.path.join(config.get("path_to_logs"), name)
        log.ckpt.save_last = False

        # Set the directory where to save the logs
        configs[name] = Partial(
            pretrain,
            model=base_config.model,
            trainer=trainer,
            data=data,
            optim=base_config.optim,
            log=log,
            resume=None,
        )

    return base_config, configs
