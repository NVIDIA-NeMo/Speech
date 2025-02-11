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

import gc
import json
import logging
import os.path
from pathlib import Path
from typing import Any, Dict

import numpy
import safetensors.torch

# needed to register 'bfloat16' dtype with numpy for zarr compatibility
import tensorstore  # noqa: F401 pylint: disable=unused-import
import torch
import zarr
from vllm.config import ModelConfig
from vllm.model_executor.model_loader.loader import BaseModelLoader, _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

from nemo.export.tarutils import TarPath, ZarrPathStore
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_sharded_metadata_torch_dist
from nemo.export.utils import is_nemo2_checkpoint
from nemo.export.vllm.model_config import NemoModelConfig

LOGGER = logging.getLogger("NeMo")


class NemoModelLoader(BaseModelLoader):
    """
    Implements a custom ModelLoader for vLLM that reads the weights from a Nemo checkpoint
    and converts them to a vLLM compatible format at load time.

    Also supports an ahead-of-time conversion that stores new weights in a Safetensors file,
    see convert_and_store_nemo_weights(...)
    """

    @staticmethod
    def _load_nemo_checkpoint_state(nemo_file: str):
        LOGGER.info(f'Loading weights from {nemo_file}...')

        if is_nemo2_checkpoint(nemo_file):
            nemo2_weights_path = Path(nemo_file) / 'weights'
            return load_sharded_metadata_torch_dist(nemo2_weights_path)

        sharded_state_dict = {}
        with (TarPath(nemo_file) / 'model_weights' / 'metadata.json').open(mode='r') as f:
            config_dict = json.load(f)

        if config_dict['sharded_backend'] == 'torch_dist':
            return load_sharded_metadata_torch_dist(TarPath(nemo_file) / 'model_weights')

        with TarPath(nemo_file) as archive:
            for subdir in archive.iterdir():
                if not subdir.is_dir() or not (subdir / '.zarray').exists():
                    continue
                key = subdir.name

                zstore = ZarrPathStore(subdir)
                arr = zarr.open(zstore, 'r')

                if arr.dtype.name == "bfloat16":
                    sharded_state_dict[key] = torch.from_numpy(arr[:].view(numpy.int16)).view(torch.bfloat16)
                else:
                    sharded_state_dict[key] = torch.from_numpy(arr[:])

                arr = None
                gc.collect()

                LOGGER.debug(f'Loaded tensor "{key}": {sharded_state_dict[key].shape}')

        return sharded_state_dict

    def download_model(self, model_config: ModelConfig) -> None:  # pylint: disable=missing-function-docstring
        raise NotImplementedError

    def load_model(
        self,
        *,
        vllm_config: NemoModelConfig,
    ) -> torch.nn.Module:
        """
        Overrides the load_model function from BaseModelLoader to convert Nemo weights at load time.
        """
        model_config = vllm_config.model_config
        device_config = vllm_config.device_config

        assert isinstance(model_config, NemoModelConfig)
        state_dict = NemoModelLoader._load_nemo_checkpoint_state(model_config.nemo_checkpoint)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(vllm_config)

            config = model_config.nemo_model_config
            if 'config' in config:
                config = config['config']
            state_dict = NemoModelLoader._standardize_nemo2_naming(state_dict)

            weights_iterator = model_config.model_converter.convert_weights(config, state_dict)
            model.load_weights(weights_iterator)

        return model.eval()

    @staticmethod
    def convert_and_store_nemo_weights(model_config: NemoModelConfig, safetensors_file: str):
        """
        Converts Nemo weights and stores the converted weights in a Safetensors file.
        """

        assert isinstance(model_config, NemoModelConfig)
        assert os.path.exists(model_config.model)

        state_dict = NemoModelLoader._load_nemo_checkpoint_state(model_config.nemo_checkpoint)

        config = model_config.nemo_model_config

        # NeMo2 checkpoint loads the whole TrainerContext where the config is stored under 'config' key
        if 'config' in config:
            config = config['config']
        state_dict = NemoModelLoader._standardize_nemo2_naming(state_dict)

        tensors = {name: tensor for name, tensor in model_config.model_converter.convert_weights(config, state_dict)}

        LOGGER.info(f'Saving weights to {safetensors_file}...')
        safetensors.torch.save_file(tensors, safetensors_file)

    @staticmethod
    def _standardize_nemo2_naming(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {k.replace('module', 'model'): v for k, v in state_dict.items()}
