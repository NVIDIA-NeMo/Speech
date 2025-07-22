# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Union

import torch

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

try:
    from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin, replace_prefix
except (ImportError, ModuleNotFoundError):
    from abc import ABC

    NLPAdapterModelMixin = ABC

from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP, PEFTConfig
from nemo.utils import logging


class SpeechLLMAdapterMixin(NLPAdapterModelMixin):
    def load_adapters(
        self,
        filepath: str,
        peft_cfgs: Optional[Union[PEFTConfig, List[PEFTConfig]]] = None,
        map_location: str = None,
    ):
        """
        Utility method that restores only the adapter module(s), and not the entire model itself.
        This allows the sharing of adapters which are often just a fraction of the size of the full model,
        enabling easier delivery.

        .. note::

            During restoration, assumes that the model does not currently already have one or more adapter modules.

        Args:
            filepath: Filepath of the .ckpt or .nemo file.
            peft_cfgs: One or more PEFTConfig objects that specify the PEFT method configuration.
                If none, will infer from the .nemo checkpoint
            map_location: Pytorch flag, where to place the adapter(s) state dict(s).
        """

        # Determine device
        if map_location is None:
            if torch.cuda.is_available():
                map_location = 'cuda'
            else:
                map_location = 'cpu'

        if filepath.endswith('.nemo'):
            conf, state_dict = self._get_config_and_state_dict_from_nemo(filepath, map_location)
        elif filepath.endswith('.ckpt'):
            state_dict = torch.load(filepath, map_location, weights_only=False)['state_dict']
        else:
            raise RuntimeError(f"{filepath} is not nemo file or ckpt file")
        if not peft_cfgs:
            assert filepath.endswith(
                '.nemo'
            ), "Inferring peft scheme is only supported for .nemo checkpoints. Please supply the `peft_cfgs` argument."
            peft_cfgs = [PEFT_CONFIG_MAP[conf.peft.peft_scheme](conf)]
        if self.cfg.megatron_amp_O2:
            state_dict = {replace_prefix(k, 'model.', 'model.module.'): v for k, v in state_dict.items()}
        self.add_adapter(peft_cfgs)
        if not self.ptuning_only_and_non_first_stage:
            target_keys = self.adapter_keys.union(self.tunable_base_param_keys)
            if set(state_dict.keys()) != target_keys:
                logging.warning(
                    f"Unexpected keys found in state_dict: {set(state_dict.keys()) - target_keys}, missing keys in state_dict: {target_keys - set(state_dict.keys())}"
                )
        # compatible with legacy checkpoints without get_peft_state_dict overwrite below
        for i in self.get_peft_state_dict().keys():
            if i not in state_dict:
                i_no_model = i.replace("model.", "")
                if i_no_model in state_dict:
                    logging.warning(f"Key {i} not found in state_dict, trying {i_no_model}")
                    state_dict[i] = state_dict[i_no_model]
                    del state_dict[i_no_model]

        super(MegatronGPTModel, self).load_state_dict(state_dict, strict=False)

    def get_peft_state_dict(self):
        """
        Gets the keys associated with the adapters only.
        Add prefix "model." to the keys.
        """
        peft_state_dict = super().get_peft_state_dict()
        peft_state_dict_with_prefix = {"model." + k: v for k, v in peft_state_dict.items()}
        return peft_state_dict_with_prefix
