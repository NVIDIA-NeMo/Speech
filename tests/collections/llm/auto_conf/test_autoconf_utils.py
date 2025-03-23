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

from collections import OrderedDict
from functools import partial

import numpy as np

from nemo.collections.llm.tools.auto_configurator.core.base_config import _estimate_training_time, calculate_model_size
from nemo.collections.llm.tools.auto_configurator.core.training_config import (
    BertGridSearch,
    GPT3GridSearch,
    T5GridSearch,
)
from nemo.collections.llm.tools.auto_configurator.core.utils import ModelSizeParams, _calculate_model_size, modify_cfg

GPT_PARAMS = OrderedDict(
    [
        (0.24, (768, 12, 6e-4)),
        (0.45, (1024, 16, 3e-4)),
        (0.9, (1536, 16, 2.5e-4)),
        (1, (2048, 16, 2e-4)),
        (2, (2560, 32, 1.6e-4)),
        (4, (3072, 32, 1.4e-4)),
        (7, (4096, 32, 1.2e-4)),
        (14, (5120, 40, 1e-4)),
        (24, (6144, 48, 1e-4)),
        (50, (8192, 64, 0.8e-4)),
        (100, (10240, 80, 0.7e-4)),
        (200, (12288, 96, 0.6e-4)),
        (400, (20480, 128, 0.5e-4)),
        (800, (20480, 128, 0.4e-4)),
        (1100, (25600, 160, 0.3e-4)),
        (1200, (25600, 160, 0.3e-4)),
    ]
)

GPT_PARALLELISM = OrderedDict(
    [
        (0.24, (768, 12, 6e-4)),
        (0.45, (1024, 16, 3e-4)),
        (0.9, (1536, 16, 2.5e-4)),
        (1, (2048, 16, 2e-4)),
        (2, (2560, 32, 1.6e-4)),
        (4, (3072, 32, 1.4e-4)),
        (7, (4096, 32, 1.2e-4)),
        (14, (5120, 40, 1e-4)),
        (24, (6144, 48, 1e-4)),
        (50, (8192, 64, 0.8e-4)),
        (100, (10240, 80, 0.7e-4)),
        (200, (12288, 96, 0.6e-4)),
        (400, (20480, 128, 0.5e-4)),
        (800, (20480, 128, 0.4e-4)),
        (1100, (25600, 160, 0.3e-4)),
        (1200, (25600, 160, 0.3e-4)),
    ]
)

BERT_PARAMS = OrderedDict(
    [
        (0.24, (768, 12, 2e-4)),
        (0.45, (1024, 16, 2e-4)),
        (0.9, (1536, 16, 1e-4)),
        (1, (2048, 16, 1e-4)),
        (2, (2560, 32, 1e-4)),
        (4, (2560, 32, 1e-4)),
        (7, (4096, 32, 1e-4)),
        (14, (5120, 40, 1e-4)),
        (24, (6144, 48, 1e-4)),
        (45, (7680, 48, 1e-4)),
        (85, (9216, 96, 1e-4)),
        (150, (9216, 96, 1e-4)),
        (250, (12288, 96, 1e-4)),
        (1000, (12288, 96, 1e-4)),
    ]
)

T5_PARAMS = OrderedDict(
    [
        (0.05, (512, 6, 1024, 64, 1e-4)),
        (0.2, (768, 12, 2048, 64, 1e-4)),
        (0.9, (1024, 16, 2816, 64, 1e-4)),
        (3, (2048, 32, 5120, 64, 1e-4)),
        (11, (4096, 64, 10240, 64, 1e-4)),
        (22, (5120, 80, 10880, 64, 1e-4)),
        (42, (6144, 96, 10880, 64, 1e-4)),
        (85, (6144, 96, 16384, 64, 1e-4)),
        (163, (7680, 96, 20480, 64, 1e-4)),
        (249, (12288, 96, 32768, 64, 1e-4)),
        (1111, (12288, 96, 32768, 64, 1e-4)),
    ]
)


class TestUtils:
    def test_calculate_model_size(self):
        # GPT
        model_size = calculate_model_size(
            8,
            7,
            None,
            140,
            300,
            "gpt3",
        )
        assert model_size == 0.28, f"expected model_size is 0.28 but got {model_size}."

        # Llama
        model_size = calculate_model_size(
            128,
            30,
            None,
            100,
            3000,
            "llama",
        )
        assert model_size == 1.38, f"expected model_size is 1.38 but got {model_size}."

        # Mixtral
        model_size = calculate_model_size(
            256,
            20,
            None,
            140,
            600,
            "mixtral",
        )
        assert model_size == 12.9, f"expected model_size is 12.9 but got {model_size}."

        # Mistral
        model_size = calculate_model_size(
            1028,
            30,
            None,
            240,
            100,
            "mistral",
        )
        assert model_size == 799.37, f"expected model_size is 799.37 but got {model_size}."

        # Gemma
        model_size = calculate_model_size(
            512,
            30,
            None,
            240,
            100,
            "gemma",
        )
        assert model_size == 398.13, f"expected model_size is 398.13 but got {model_size}."

        # Nemotron
        model_size = calculate_model_size(
            256,
            15,
            None,
            240,
            120,
            "gemma",
        )
        assert model_size == 82.94, f"expected model_size is 82.94 but got {model_size}."

        # Qwen
        model_size = calculate_model_size(
            128,
            2,
            None,
            1000,
            100,
            "qwen",
        )
        assert model_size == 27.65, f"expected model_size is 27.65 but got {model_size}."

        # Starcoder
        model_size = calculate_model_size(
            8,
            1,
            None,
            10,
            10,
            "starcoder",
        )
        assert model_size == 0.09, f"expected model_size is 27.65 but got {model_size}."

        # T5
        model_size = calculate_model_size(
            1024,
            14,
            None,
            1400,
            340,
            "t5",
        )
        assert model_size == 637.53, f"expected model_size is 637.53 but got {model_size}."

        # Bert
        model_size = calculate_model_size(
            512,
            7,
            None,
            140,
            100,
            "bert",
        )
        assert model_size == 54.19, f"expected model_size is 54.19 but got {model_size}."

    def test_calculate_train_time(self):
        # GPT
        train_time = _estimate_training_time(
            175,
            1024,
            140,
            300,
            "gpt3",
        )
        assert train_time == 33.91, f"expected train_time is 33.91 but got {train_time}."

        # Llama
        train_time = _estimate_training_time(
            35,
            512,
            60,
            3000,
            "llama",
        )
        assert train_time == 316.48, f"expected train_time is 316.48 but got {train_time}."

        # Mixtral
        train_time = _estimate_training_time(
            0.8,
            128,
            140,
            1000,
            "mixtral",
        )
        assert train_time == 4.13, f"expected train_time is 4.13 but got {train_time}."

        # Mistral
        train_time = _estimate_training_time(
            11,
            24,
            60,
            250,
            "mistral",
        )
        assert train_time == 176.83, f"expected train_time is 176.83 but got {train_time}."

        # Gemma
        train_time = _estimate_training_time(
            7,
            8,
            55,
            100,
            "gemma",
        )
        assert train_time == 147.31, f"expected train_time is 147.31 but got {train_time}."

        # Nemotron
        train_time = _estimate_training_time(
            14,
            12,
            11,
            55,
            "nemotron",
        )
        assert train_time == 540.12, f"expected train_time is 540.12 but got {train_time}."

        # Qwen
        train_time = _estimate_training_time(
            7,
            64,
            512,
            1000,
            "qwen",
        )

        assert train_time == 19.78, f"expected train_time is 19.78 but got {train_time}."

        # Starcoder
        train_time = _estimate_training_time(
            77,
            128,
            1000,
            1000,
            "starcoder",
        )
        assert train_time == 55.7, f"expected train_time is 19.78 but got {train_time}."

        # Bert
        train_time = _estimate_training_time(
            0.123,
            8,
            100,
            1000,
            "bert",
        )
        assert train_time == 14.24, f"expected train_time is 14.24 but got {train_time}."

        # T5
        train_time = _estimate_training_time(
            0.01,
            1,
            32,
            346,
            "t5",
        )
        assert train_time == 10.01, f"expected train_time is 10.01 but got {train_time}."

    def test_modify_cfg(self):
        from nemo.collections import llm

        base_cfg = partial(llm.bert_110m.pretrain_recipe, num_nodes=1, num_gpus_per_node=1)()
        config = modify_cfg(
            base_cfg=base_cfg,
            act=None,
            num_mbs_act=None,
            act_per_pipe=None,
            tp=11,
            pp=4,
            cp=5,
            ep=1,
            virtual_pipelines=None,
            mbs=6,
            max_steps=10,
            num_nodes=11,
            model_name="bert",
            path_to_logs="/",
            model_size=8,
        )

        assert config is None

    def test_calculate_model_size_utils(self):
        # Qwen
        model_size = np.round(
            _calculate_model_size(
                vocab_size=32000,
                seq_length=4096,
                hidden_size=1024,
                num_layers=24,
                ffn_size=4096,
                kv_channels=8,
                att_heads=32,
                model_name="qwen",
            ),
            2,
        )

        assert model_size == 0.34, f"expected model_size is 0.34 but got {model_size}."

        # Starcoder
        model_size = np.round(
            _calculate_model_size(
                vocab_size=32000,
                seq_length=8192,
                hidden_size=4608,
                num_layers=32,
                ffn_size=18432,
                kv_channels=None,
                att_heads=36,
                model_name="starcoder",
            ),
            2,
        )

        assert model_size == 8.34, f"expected model_size is 8.34 but got {model_size}."

        # Bert
        model_size = np.round(
            _calculate_model_size(
                vocab_size=128000,
                seq_length=512,
                hidden_size=768,
                num_layers=16,
                ffn_size=2048,
                kv_channels=None,
                att_heads=12,
                model_name="bert",
            ),
            2,
        )

        assert model_size == 0.19, f"expected model_size is 8.34 but got {model_size}."

        # Bert
        model_size = np.round(
            _calculate_model_size(
                vocab_size=128000,
                seq_length=512,
                hidden_size=768,
                num_layers=16,
                ffn_size=2048,
                kv_channels=None,
                att_heads=12,
                model_name="bert",
            ),
            2,
        )

        assert model_size == 0.19, f"expected model_size is 8.34 but got {model_size}."

        # T5
        model_size = np.round(
            _calculate_model_size(
                vocab_size=50000,
                seq_length=512,
                hidden_size=4096,
                num_layers=24,
                ffn_size=10240,
                kv_channels=32,
                att_heads=64,
                model_name="t5",
            ),
            2,
        )

        assert model_size == 8.67, f"expected model_size is 8.67 but got {model_size}."

    def test_model_size_params(self):
        # Nemotron
        for model_size, model_params in GPT_PARAMS.items():
            try:
                params = ModelSizeParams(
                    model_size_in_b=model_size,
                    vocab_size=32000,
                    seq_length=8192,
                    model_name="nemotron",
                )
                params.init_params()

                assert (
                    params.hs,
                    params.att_h,
                    params.lr,
                ) == model_params, (
                    f"expected model params are {model_params} but got {(params.hs, params.att_h, params.lr)}."
                )
            except ValueError:
                assert True

        # Bert
        for model_size, model_params in BERT_PARAMS.items():
            try:
                params = ModelSizeParams(
                    model_size_in_b=model_size,
                    vocab_size=52000,
                    seq_length=512,
                    model_name="bert",
                )
                params.init_params()

                assert (
                    params.hs,
                    params.att_h,
                    params.lr,
                ) == model_params, (
                    f"expected model params are {model_params} but got {(params.hs, params.att_h, params.lr)}."
                )
            except ValueError:
                assert True

        # T5
        for model_size, model_params in T5_PARAMS.items():
            try:
                params = ModelSizeParams(
                    model_size_in_b=model_size,
                    vocab_size=52000,
                    seq_length=512,
                    model_name="t5",
                )
                params.init_params()

                gen_model_params = (params.hs, params.att_h, params.ffn, params.kv, params.lr)
                assert (
                    gen_model_params == model_params
                ), f"expected model params are {model_params} but got {gen_model_params}."
            except ValueError:
                assert True

    def test_gpt_grid_search(self):
        # GPT 80GB GPU
        seq_lengths = [2**i for i in range(11, 16)]
        model_sizes = [1, 4, 8, 13, 23, 45, 95, 130, 195, 395, 790, 1100]

        for seq_length in seq_lengths:
            for model_size in model_sizes:
                params = GPT3GridSearch(
                    model_size_in_b=model_size,
                    valid_pp=[1, 2, 4, 8],
                    seq_length=seq_length,
                    gpu_memory_gb=80,
                )
                params.init_params()

        # GPT 40GB GPU
        model_sizes = [1, 4, 8, 13, 23, 45, 95, 130, 195, 395, 790, 1100]

        for model_size in model_sizes:
            params = GPT3GridSearch(
                model_size_in_b=model_size,
                valid_pp=[1, 2, 4, 8],
                seq_length=2048,
                gpu_memory_gb=40,
            )
            params.init_params()

    def test_bert_grid_search(self):
        # Bert 80GB GPU
        model_sizes = [1, 4, 8, 13, 23, 45, 87, 165, 250]

        for model_size in model_sizes:
            params = BertGridSearch(
                model_size_in_b=model_size,
                valid_pp=[1, 2, 4, 8],
                seq_length=512,
                gpu_memory_gb=80,
            )
            params.init_params()

        # Bert 40GB GPU
        for model_size in model_sizes:
            params = BertGridSearch(
                model_size_in_b=model_size,
                valid_pp=[1, 2, 4, 8],
                seq_length=512,
                gpu_memory_gb=40,
            )
            params.init_params()

    def test_t5_grid_search(self):
        # T5 80GB GPU
        model_sizes = [1, 4, 8, 14, 25, 40, 85, 160, 248]

        for model_size in model_sizes:
            params = T5GridSearch(
                model_size_in_b=model_size,
                valid_pp=[1, 2, 4, 8],
                seq_length=512,
                gpu_memory_gb=80,
            )
            params.init_params()

        # T5 40GB GPU
        for model_size in model_sizes:
            params = T5GridSearch(
                model_size_in_b=model_size,
                valid_pp=[1, 2, 4, 8],
                seq_length=512,
                gpu_memory_gb=40,
            )
            params.init_params()
