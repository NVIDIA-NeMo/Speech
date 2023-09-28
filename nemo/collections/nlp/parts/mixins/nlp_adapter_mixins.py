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

from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import swap_mcore_mixin

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import PromptEncoderAdapterConfig
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

try:
    from megatron.core import parallel_state
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


class NLPAdapterModelMixin:
    """ NLP Adapter Mixin that can augment any transformer-based model with Adapter module support.
    This mixin class should be used only with a top level ModelPT subclass, that includes either a `model` or an `enc_dec_model` submodule.
    This mixin class adds several utility methods to add, load and save adapters.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter yields the original output.

    This mixin class aims to integrate with PEFT, which is one or more adapters modules.
    The two features of PEFT, layer selection and weight tying, are also supported in this mixin class.
    """

    def __init__(self, *args, **kwargs):
        self.use_peft = False
        self.setup_complete = False
        self.use_ptuning_only = False
        super().__init__(*args, **kwargs)
        if hasattr(self, "enc_dec_model"):
            self.model_prefix = "enc_dec_model."  # for T5
        else:
            self.model_prefix = "model.module." if self.cfg.megatron_amp_O2 else "model."

        self.use_mcore_gpt = hasattr(self, 'mcore_gpt') and self.mcore_gpt
        if self.use_mcore_gpt:
            assert HAVE_MEGATRON_CORE, "You set `mcore_gpt` as True but megatron core is not found."

    def first_stage_of_pipeline(self):
        if hasattr(self, "model") and hasattr(self.model, "pre_process"):
            return self.model.pre_process
        elif hasattr(self, "model") and hasattr(self.model, "module") and hasattr(self.model.module, "pre_process"):
            # (guyueh1): this if condition is used to handle amp O2
            # when amp_O2 is on, self.model will be wrapped by the Float16Module class
            return self.model.module.pre_process
        logging.warning("no attribute named model or no model.pre_process found. Can not detect stage of pipeline...")
        return False

    def _get_all_keys(self,):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        b = [n for n, p in self.named_buffers() if n.replace("model.module.", "model.", 1) in self.state_dict().keys()]
        # we include buffers because ptuning representations are cached in a buffer and saved to state_dict for inference time use.
        return set(k + b)

    def _check_and_add_adapter(self, name, module, peft_name, peft_cfg, name_key_to_mcore_mixins=None):
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
        elif isinstance(module, AdapterModuleMixin):
            if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                module.add_adapter(
                    name=peft_name,
                    cfg=peft_cfg,
                    base_model_cfg=self.cfg,
                    model_parallel_config=self.model_parallel_config,
                )

    def _check_and_add_peft_cfg(self, peft_cfg):

        layer_selection = peft_cfg.layer_selection

        assert not self.use_mcore_gpt or hasattr(
            peft_cfg, 'name_key_to_mcore_mixins'
        ), f"{peft_cfg.__class__.__name__} is not supported in megatron core mode yet."
        name_key_to_mcore_mixins = peft_cfg.name_key_to_mcore_mixins if self.use_mcore_gpt else None

        for adapter_name, adapter_cfg in peft_cfg.get_config_dict().items():
            # self.mcore_gpt means is GPT and not T5
            if hasattr(self, 'mcore_gpt') and not isinstance(adapter_cfg, PromptEncoderAdapterConfig):
                if layer_selection is not None:
                    logging.info(
                        f"Layer selection {layer_selection} is enabled for the current model ("
                        f"{self.__class__.__name__} + {adapter_name})"
                    )
                if self.use_mcore_gpt:
                    if self.cfg.megatron_amp_O2:
                        layers = self.model.module.decoder.layers
                    else:
                        layers = self.model.decoder.layers
                else:
                    if self.cfg.megatron_amp_O2:
                        layers = self.model.module.language_model.encoder.layers
                    else:
                        layers = self.model.language_model.encoder.layers
                if layer_selection is None:
                    layer_selection = list(range(1, self.cfg.num_layers + 1))
                for layer in layers:
                    if layer.layer_number in layer_selection:
                        for name, module in layer.named_modules():
                            self._check_and_add_adapter(
                                name, module, adapter_name, adapter_cfg, name_key_to_mcore_mixins
                            )
            else:
                # Non GPT models, as well as GPT+PTuning do not support layer selection
                if layer_selection is not None:
                    logging.warning(
                        "Layer selection is specified, but it is not supported for either "
                        f"{self.__class__.__name__} or {adapter_name})"
                    )
                for name, module in self.named_modules():
                    self._check_and_add_adapter(name, module, adapter_name, adapter_cfg, name_key_to_mcore_mixins)

    def add_adapter(self, peft_cfgs: Union[PEFTConfig, List[PEFTConfig]]):
        """
        High level API to add one or more adapter modules to the model, and freeze the base weights
        This method supports adding adapter modules from PEFTConfig or list of PEFTConfig. It would add
        corresponding adapter modules. Layer selection and weight tying would be applied if it's in PEFTConfig

        Args:
            peft_cfgs: One or more PEFTConfig objects that specify the PEFT method configuration
        """

        if not isinstance(peft_cfgs, List):
            peft_cfgs = [peft_cfgs]

        self.base_keys = self._get_all_keys()
        self.freeze()
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

        for cfg in peft_cfgs:
            if cfg.weight_tying:
                self.tie_weights(cfg)
        self.use_peft = True

    def _get_config_and_state_dict_from_nemo(self, filepath, map_location):
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                SaveRestoreConnector._unpack_nemo_file(filepath, tmpdir)

                os.chdir(tmpdir)

                config_yaml = "model_config.yaml"
                model_weights_ckpt = "model_weights.ckpt"

                conf = OmegaConf.load(config_yaml)

                os.chdir(cwd)
                model_weights = os.path.join(tmpdir, model_weights_ckpt)
                model_weights = inject_model_parallel_rank(model_weights)
                state_dict = torch.load(model_weights, map_location=map_location)

                return conf, state_dict
            finally:
                os.chdir(cwd)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups.
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder
        and/or prompt table will use the learning rate set by the user.
        """
        if self.use_peft:
            self.freeze()  # Freeze the entire model
            opt_params = []
            for _, module in self.named_modules():
                if isinstance(module, AdapterModuleMixin) and module.is_adapter_available():
                    module.set_enabled_adapters(enabled=True)
                    module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                    opt_params += [p for p in module.parameters() if p.requires_grad]
            self._optimizer_param_groups = ({"params": opt_params},)
            logging.info(f"Optimizer groups set:\n{self.summarize()}")
        else:
            super().setup_optimizer_param_groups()

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
        super().load_state_dict(state_dict, strict=False)

    def tie_weights(self, peft_cfg):
        pos_idx = 0

        if self.use_mcore_gpt:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.decoder.layers
            else:
                layers = self.model.decoder.layers
        else:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.language_model.encoder.layers
            else:
                layers = self.model.language_model.encoder.layers

        if isinstance(peft_cfg, LoraPEFTConfig):
            layer0 = layers[0].self_attention
        elif isinstance(peft_cfg, CanonicalAdaptersPEFTConfig):
            layer0 = layers[0]
        else:
            raise RuntimeError(f"{peft_cfg} is not supported for tied weights")

        for adapter_name in layer0.adapter_layer:
            adapter = layer0.get_adapter_module(adapter_name)
            print(adapter_name, pos_idx)
            adapter.set_position(pos_idx)
            pos_idx += 1

        for layer in layers[1:]:
            if isinstance(peft_cfg, LoraPEFTConfig):
                layer = layer.self_attention
            for adapter_name in layer.adapter_layer:
                print(adapter_name, pos_idx)
                adapter_l = layer.get_adapter_module(adapter_name)
                adapter_0 = layer0.get_adapter_module(adapter_name)
                adapter_l.tie_weights(pos_idx, adapter_0)
                pos_idx += 1

    def get_peft_state_dict(self):
        """
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix=self.model_prefix)
        peft_state_dict = {}
        for k in self.adapter_keys:
            # state_dict keys needs to be in non-O2 format and will be corrected in PEFTSaveRestoreConnector if O2=True
            new_k = k.replace("model.module.", "model.", 1)
            peft_state_dict[new_k] = state_dict[k]
        return peft_state_dict

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.use_peft and self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix=self.model_prefix)

    def sharded_state_dict(self, prefix: str = ''):
        use_mcore_gpt = hasattr(self, 'mcore_gpt') and self.mcore_gpt
        if not use_mcore_gpt or (self.use_peft and self.setup_complete):
            return None
        else:
            return self.model.sharded_state_dict(prefix=self.model_prefix)

    def load_state_dict(self, state_dict, strict: bool = True):
        if len(state_dict) == 0:
            return  # checkpoint is loaded in on_load_checkpoint()
        if self.use_peft and self.setup_complete:
            # at this stage only adapter params will appear in the state_dict arg
            # so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            # explicitly check if state_dict.keys matches all the expected self.adapter_keys since we don't have the
            # safety in strict=True anymore.
            assert set(state_dict.keys()) == self.adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if self.use_peft and self.setup_complete:
            if not self.use_ptuning_only or self.first_stage_of_pipeline():
                # same as super().on_load_checkpoint() but strict=False and only check unexpected keys
                # mcore uses distributed checkpointing
                if hasattr(self, 'mcore_gpt') and self.mcore_gpt:
                    for index, module in enumerate(self.get_gpt_module_list()):
                        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                            checkpoint_state_dict = checkpoint['state_dict'][f'model_{index}']
                        else:
                            checkpoint_state_dict = checkpoint['state_dict']
                        # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                        checkpoint_state_dict = {
                            key.replace('model.', ''): checkpoint_state_dict.pop(key)
                            for key in list(checkpoint_state_dict.keys())
                        }
                        missing_keys, unexpected_keys = module.load_state_dict(checkpoint_state_dict, strict=False)

                        assert len(unexpected_keys) == 0, 'Unexpected key(s) in state_dict: {}. '.format(
                            ', '.join('"{}"'.format(k) for k in unexpected_keys)
                        )

                # legacy checkpointing for interleaved
                else:
                    if isinstance(self.model, list):
                        for i in range(len(self.model)):
                            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                            self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        else:
            super().on_load_checkpoint(checkpoint)

    @classmethod
    def merge_cfg_with(cls, path: str, cfg: DictConfig) -> DictConfig:
        """
        Merge a given configuration dictionary `cfg` with the configuration dictionary
        obtained from restoring a MegatronGPTSFTModel or MegatronT5SFTModel at the specified `path`.

        Args:
            path (str): The path to the SFT model checkpoint to be restored.
            cfg (DictConfig): The configuration dictionary to merge.

        Returns:
            DictConfig: The merged configuration dictionary.

        Examples:
            >>> path = "/path/to/model/checkpoint"
            >>> cfg = DictConfig({"model": {"key": "value"}, "trainer": {"precision": 16}})
            >>> merged_cfg = merge_cfg_with(path, cfg)

        Notes:
            - The function resolves variables within the `cfg` dictionary using `OmegaConf.resolve`.
            - Keys in `cfg.model` will override the corresponding keys in the output dictionary.
            - If "train_ds" exists in `cfg.model.data`, it updates `micro_batch_size` and `global_batch_size`.
            - If `cfg.trainer` contains a "precision" key, it updates `output.precision`.

        """

        base_cfg = cls.restore_from(path, return_config=True)

        OmegaConf.resolve(cfg)
        with open_dict(base_cfg):
            for key, val in cfg.model.items():
                base_cfg[key] = val
            if "train_ds" in cfg.model.data:
                base_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
                base_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
            if cfg.get("trainer", None) and cfg.trainer.get("precision"):
                base_cfg.precision = cfg.trainer.precision

        return base_cfg

    @classmethod
    def merge_inference_cfg(cls, path: str, cfg: DictConfig) -> DictConfig:
        """
        Generate a configuration dictionary by a given configuration dictionary `cfg` with
        the configuration dictionary obtained from restoring a MegatronGPTSFTModel or MegatronT5SFTModel
        at the specified `path` and modify `cfg` for inference

        Args:
            path (str): The path to the SFT model checkpoint to be restored.
            cfg (DictConfig): The configuration dictionary to modify for inference.

        Returns:
            DictConfig: The configuration dictionary for inference.

        Examples:
            >>> path = "/path/to/model/checkpoint"
            >>> cfg = DictConfig({"model": {"key": "value"}, "trainer": {"precision": 16}})
            >>> merged_cfg = merge_inference_cfg(path, cfg)

        Notes:
            - "precision" and "test_ds" from `cfg` will override the corresponding keys in the output dictionary
            - "activations_checkpoint" will be ovrrided to None in the output dictionary
            - "use_flash_attention" will be True if in one of the configuration dictionarys is True
            - "seq_len_interpolation_factor" will be overrided from `cfg` if it's not None from checkpoint
        """

        peft_cfg = cls.restore_from(path, return_config=True)
        with open_dict(peft_cfg):
            # update the model config of the trained model with params we want to set at inference time.
            peft_cfg.precision = cfg.trainer.precision
            for key, val in cfg.model.items():
                if key != 'data':
                    peft_cfg[key] = val
            peft_cfg.data.test_ds = cfg.model.data.test_ds

        with open_dict(cfg):
            cfg.inference.add_BOS = peft_cfg.data.test_ds.add_bos
            cfg.inference.tokens_to_generate = peft_cfg.data.test_ds.tokens_to_generate

        return peft_cfg
