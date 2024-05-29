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

from typing import List, Optional, Union

import torch

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.mixins.nlp_adapter_mixins import NLPAdapterModelMixin, replace_prefix
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP, PEFTConfig, PtuningPEFTConfig
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging, model_utils

try:
    from megatron.core import parallel_state

    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import swap_mcore_mixin

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class MultimodalAdapterModelMixin(NLPAdapterModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_all_keys(
        self,
    ):
        # TODO (yuya): p-tuning need additional handle, check peft models.
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        if self.megatron_amp_O2:
            k = [key.replace("model.module.", "model.", 1) for key in k]
        return set(k)

    def add_adapter(self, peft_cfgs: Union[PEFTConfig, List[PEFTConfig]]):
        if self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is not supported when using PEFT')
        if self.cfg.optim.name == "distributed_fused_adam":
            raise ValueError('distributed_fused_adam is not supported for PEFT. Please use fused_adam')

        self.use_peft = True
        if not isinstance(peft_cfgs, List):
            peft_cfgs = [peft_cfgs]

        # @chcui crucial to set self.virtual_tokens and self.use_peft for all PP ranks
        for peft_cfg in peft_cfgs:
            if isinstance(peft_cfg, PtuningPEFTConfig):
                self.virtual_tokens = peft_cfg.virtual_tokens
        ptuning_only = len(peft_cfgs) == 1 and isinstance(peft_cfgs[0], PtuningPEFTConfig)
        self.ptuning_only_and_non_first_stage = ptuning_only and not self.first_stage_of_pipeline()
        if self.ptuning_only_and_non_first_stage:
            # There are no params to add if we are not in the first state of the pipeline
            return

        self.base_keys = getattr(self, "base_keys", self._get_all_keys())
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")

        for peft_cfg in peft_cfgs:
            self._check_and_add_peft_cfg(peft_cfg)

        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        self.adapter_keys = self._get_all_keys() - self.base_keys
        self.tunable_base_param_keys = set()

        for cfg in peft_cfgs:
            if hasattr(cfg, "weight_tying") and cfg.weight_tying:
                self.tie_weights(cfg)

            if hasattr(cfg, "tunable_base_param_names") and cfg.tunable_base_param_names:
                self.set_tunable_base_params(cfg)

        if self.megatron_amp_O2:
            self.adapter_keys = set(key.replace("model.module.", "model.", 1) for key in self.adapter_keys)

    def load_adapters(
        self,
        filepath: str,
        peft_cfgs: Optional[Union[PEFTConfig, List[PEFTConfig]]] = None,
        map_location: str = None,
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

        # TODO (yuya): this logic needs to change for dist ckpt because after
        # adding adapaters the checkpoint will change
        if not peft_cfgs:
            assert filepath.endswith(
                '.nemo'
            ), "Inferring peft scheme is only supported for .nemo checkpoints. Please supply the `peft_cfgs` argument."
            peft_cfgs = [PEFT_CONFIG_MAP[conf.peft.peft_scheme](conf)]
        self.add_adapter(peft_cfgs)
        if filepath.endswith('.nemo'):
            sharded_state_dict = None
            if getattr(self, "sharded_state_dict", None) is not None:
                sharded_state_dict = self.sharded_state_dict(prefix="model.")
            conf, state_dict = self._get_config_and_state_dict_from_nemo(filepath, map_location, sharded_state_dict)
        elif filepath.endswith('.ckpt'):
            state_dict = torch.load(filepath, map_location)['state_dict']
        else:
            raise RuntimeError(f"{filepath} is not nemo file or ckpt file")
        if not self.ptuning_only_and_non_first_stage:
            assert set(state_dict.keys()) == self.adapter_keys.union(self.tunable_base_param_keys)
        if self.cfg.megatron_amp_O2:
            state_dict = {replace_prefix(k, 'model.', 'model.module.'): v for k, v in state_dict.items()}

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
