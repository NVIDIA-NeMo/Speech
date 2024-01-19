# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin
from nemo.collections.nlp.parts.peft_config import (
    PEFT_CONFIG_MAP,
    CanonicalAdaptersPEFTConfig,
    LoraPEFTConfig,
    PEFTConfig,
    PtuningPEFTConfig,
)
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging, model_utils
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import swap_mcore_mixin

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class MultimodalAdapterModelMixin(NLPAdapterModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_all_keys(self,):
        # TODO (yuya): p-tuning need additional handle, check peft models.
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        if self.megatron_amp_O2:
            k = [key.replace("model.module.", "model.", 1) for key in k]
        return set(k)

    def add_adapter(self, peft_cfgs: Union[PEFTConfig, List[PEFTConfig]]):
        if not isinstance(peft_cfgs, List):
            peft_cfgs = [peft_cfgs]

        self.base_keys = getattr(self, "base_keys", self._get_all_keys())
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")

        self.use_ptuning_only = len(peft_cfgs) == 1 and isinstance(peft_cfgs[0], PtuningPEFTConfig)

        for peft_cfg in peft_cfgs:
            if self.use_ptuning_only:
                if not self.first_stage_of_pipeline():
                    # There are no params to add if we are not in the first state of the pipeline
                    continue
                self.virtual_tokens = peft_cfg.virtual_tokens

            self._check_and_add_peft_cfg(peft_cfg)

        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        self.adapter_keys = self._get_all_keys() - self.base_keys
        if self.megatron_amp_O2:
            self.adapter_keys = set(key.replace("model.module.", "model.", 1) for key in self.adapter_keys)

        for cfg in peft_cfgs:
            if cfg.weight_tying:
                self.tie_weights(cfg)
        self.use_peft = True

    def load_adapters(
        self, filepath: str, peft_cfgs: Optional[Union[PEFTConfig, List[PEFTConfig]]] = None, map_location: str = None,
    ):
        """
        Utility method that restores only the adapter module(s), and not the entire model itself.
        This allows the sharing of adapters which are often just a fraction of the size of the full model,
        enabling easier deliver.

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
            state_dict = torch.load(filepath, map_location)['state_dict']
        else:
            raise RuntimeError(f"{filepath} is not nemo file or ckpt file")
        if not peft_cfgs:
            assert filepath.endswith(
                '.nemo'
            ), "Inferring peft scheme is only supported for .nemo checkpoints. Please supply the `peft_cfgs` argument."
            peft_cfgs = [PEFT_CONFIG_MAP[conf.peft.peft_scheme](conf)]
        self.add_adapter(peft_cfgs)
        assert set(state_dict.keys()) == self.adapter_keys

        if self.megatron_amp_O2:
            state_dict = {k.replace("model.", "model.module.", 1): v for k, v in state_dict.items()}

        missing_keys, unexpected_keys = NLPModel.load_state_dict(self, state_dict, strict=False)

        if len(missing_keys) > 0:
            logging.warning('Missing keys were detected during the load. Please double check.')
            if len(missing_keys) > 10:
                logging.warning(f'Missing keys: {missing_keys[:10]} and {len(missing_keys) - 10} more.')
            else:
                logging.warning(f'Missing keys: {missing_keys}')
        if len(unexpected_keys) > 0:
            logging.critical('Unexpected keys were detected during the load. Please double check.')
            logging.critical(f'Unexpected keys: \n{unexpected_keys}')
            raise ValueError('Unexpected keys were detected during the load. Please double check.')

    def _check_and_add_adapter(
        self, name, module, peft_name, peft_cfg, name_key_to_mcore_mixins=None, autocast_dtype=None
    ):
        if name_key_to_mcore_mixins is not None:
            for mcore_target, mcore_mixin in name_key_to_mcore_mixins[peft_name]:
                if name in [
                    mcore_target,
                    f'model.{mcore_target}',
                    f'model.module.{mcore_target}',
                ]:  # simple string match for now
                    swap_mcore_mixin(module, mcore_mixin)
                    if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                        module.add_adapter(
                            name=peft_name,
                            cfg=peft_cfg,
                            base_model_cfg=self.cfg,
                            model_parallel_config=self.model_parallel_config,
                        )
                        if autocast_dtype is not None:
                            module.adapter_layer[peft_name] = module.adapter_layer[peft_name].to(autocast_dtype)
        elif isinstance(module, AdapterModuleMixin):
            if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=peft_name,
                    cfg=peft_cfg,
                    base_model_cfg=self.cfg,
                    model_parallel_config=self.model_parallel_config,
                )
                if autocast_dtype is not None:
                    module.adapter_layer[peft_name] = module.adapter_layer[peft_name].to(autocast_dtype)
