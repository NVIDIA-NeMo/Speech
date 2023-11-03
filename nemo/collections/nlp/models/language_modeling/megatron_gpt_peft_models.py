# coding=utf-8
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

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.adapters.mcore_mixins import (
    MCoreGPTEmbeddingMixin,
    MCoreSelfAttentionMixin,
    MCoreTransformerLayerMixin,
    swap_mcore_mixin,
)
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKQVAdapterWeightTyingConfig,
    MLPInfusedAdapterConfig,
    ParallelLinearAdapterConfig,
    ParallelLinearAdapterWeightTyingConfig,
    PromptEncoderAdapterConfig,
)
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, model_utils
from nemo.utils.decorators import deprecated

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


@deprecated(
    explanation="Please use MegatronGPTSFTModel.add_adapter() for PEFT features."
    "See the updated `megatron_gpt_peft_tuning.py` for an example."
)
class MegatronGPTPEFTModel(MegatronGPTSFTModel):
    """
    base class for all mixin based adapter models
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.setup_complete = False
        self.base_keys = self.get_all_keys()
        self.freeze()
        self.init_peft_modules()
        self.adapter_keys = self.get_all_keys() - self.base_keys

    def first_stage_of_pipeline(self):
        if hasattr(self, "model") and hasattr(self.model, "pre_process"):
            return self.model.pre_process
        elif hasattr(self, "model") and hasattr(self.model, "module") and hasattr(self.model.module, "pre_process"):
            # (guyueh1): this if condition is used to handle amp O2
            # when amp_O2 is on, self.model will be wrapped by the Float16Module class
            return self.model.module.pre_process
        logging.warning("no attribute named model or no model.pre_process found. Can not detect stage of pipeline...")
        return False

    def init_peft_modules(self):
        """
        Randomly initialize the peft params and add them to the appropriate modules.
        """
        assert len(self.peft_name_keys) > 0, "peft_name_keys have not been set no PEFT modules will be added"
        assert not self.mcore_gpt or hasattr(
            self, 'name_key_to_mcore_mixins'
        ), f"{self.__class__.__name__} is not supported in megatron core mode yet."
        assert len(self.name_key_to_cfg) > 0, "name_key_to_cfg has not been set no PEFT modules will be added"
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")
        for name, module in self.named_modules():
            if self.mcore_gpt:
                for peft_key in self.peft_name_keys:
                    for mcore_target, mcore_mixin in self.name_key_to_mcore_mixins[peft_key]:
                        if name in [
                            f'model.{mcore_target}',
                            f'model.module.{mcore_target}',
                        ]:  # simple string match for now
                            swap_mcore_mixin(module, mcore_mixin)
                            peft_cfg = self.name_key_to_cfg[peft_key]
                            if (
                                model_utils.import_class_by_path(peft_cfg._target_)
                                in module.get_accepted_adapter_types()
                            ):
                                module.add_adapter(
                                    name=peft_key,
                                    cfg=peft_cfg,
                                    base_model_cfg=self.cfg,
                                    model_parallel_config=self.model_parallel_config,
                                )
            else:
                if isinstance(module, adapter_mixins.AdapterModuleMixin):
                    for peft_key in self.peft_name_keys:
                        peft_cfg = self.name_key_to_cfg[peft_key]
                        if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                            module.add_adapter(
                                name=peft_key,
                                cfg=peft_cfg,
                                base_model_cfg=self.cfg,
                                model_parallel_config=self.model_parallel_config,
                            )
                if self.megatron_amp_O2:
                    for adapter_name in getattr(module, 'adapter_layer', []):
                        module.adapter_layer[adapter_name] = module.adapter_layer[adapter_name].to(self.autocast_dtype)
        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        return True

    def setup(self, stage=None):
        super().setup(stage)
        self.setup_complete = True

    def get_all_keys(self,):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        b = [n for n, p in self.named_buffers() if n.replace("model.module.", "model.", 1) in self.state_dict().keys()]
        # we include buffers because ptuning representations are cached in a buffer and saved to state_dict for inference time use.
        return set(k + b)

    def get_peft_state_dict(self,):
        """
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix="model.module." if self.cfg.megatron_amp_O2 else "model.")
        peft_state_dict = {}
        for k in self.adapter_keys:
            # state_dict keys needs to be in non-O2 format and will be corrected in PEFTSaveRestoreConnector if O2=True
            new_k = k.replace("model.module.", "model.", 1)
            peft_state_dict[new_k] = state_dict[k]
        return peft_state_dict

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix="model.")

    def sharded_state_dict(self, prefix: str = ''):
        if self.setup_complete:
            return None
        else:
            return self.model.sharded_state_dict(prefix="model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        if len(state_dict) == 0:
            return  # checkpoint is loaded in on_load_checkpoint()
        if self.setup_complete:
            # at this stage only PEFT params will appear in the state_dict arg
            # so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            # explicitly check if state_dict.keys matches all the expected self.adapter_keys since we don't have the
            # safety in strict=True anymore.
            if self.megatron_amp_O2:
                adapter_keys = set(key.replace("model.", "model.module.", 1) for key in self.adapter_keys)
            else:
                adapter_keys = self.adapter_keys
            assert set(state_dict.keys()) == adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
         https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
         """
        if self.setup_complete:
            # same as super().on_load_checkpoint() but strict=False and only check unexpected keys
            # mcore uses distributed checkpointing
            print('enter peft loading')
            if self.mcore_gpt:
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
        self.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters() if p.requires_grad]
        self._optimizer_param_groups = ({"params": opt_params},)
        logging.info(f"Optimizer groups set:\n{self.summarize()}")


class MegatronGPTLayerwisePEFTModel(MegatronGPTPEFTModel):
    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        super().__init__(cfg, trainer)

    def init_peft_modules(self):
        """
        Randomly initialize the peft params and add them to the appropriate modules.
        """
        assert len(self.peft_name_keys) > 0, "peft_name_keys have not been set no PEFT modules will be added"
        assert not self.mcore_gpt or hasattr(
            self, 'name_key_to_mcore_mixins'
        ), f"{self.__class__.__name__} is not supported in megatron core mode yet."
        assert len(self.name_key_to_cfg) > 0, "name_key_to_cfg has not been set no PEFT modules will be added"
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")
        if self.mcore_gpt:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.decoder.layers
            else:
                layers = self.model.decoder.layers
        else:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.language_model.encoder.layers
            else:
                layers = self.model.language_model.encoder.layers
        for layer in layers:
            if layer.layer_number in self.layer_selection:
                for name, module in layer.named_modules():
                    if self.mcore_gpt:
                        for peft_key in self.peft_name_keys:
                            for mcore_target, mcore_mixin in self.name_key_to_mcore_mixins[peft_key]:
                                if name == mcore_target:
                                    swap_mcore_mixin(module, mcore_mixin)
                                    peft_cfg = self.name_key_to_cfg[peft_key]
                                    if (
                                        model_utils.import_class_by_path(peft_cfg._target_)
                                        in module.get_accepted_adapter_types()
                                    ):
                                        module.add_adapter(
                                            name=peft_key,
                                            cfg=peft_cfg,
                                            model_parallel_config=self.model_parallel_config,
                                        )
                    else:
                        if isinstance(module, adapter_mixins.AdapterModuleMixin):
                            for peft_key in self.peft_name_keys:
                                peft_cfg = self.name_key_to_cfg[peft_key]
                                if (
                                    model_utils.import_class_by_path(peft_cfg._target_)
                                    in module.get_accepted_adapter_types()
                                ):
                                    module.add_adapter(
                                        name=peft_key,
                                        cfg=peft_cfg,
                                        base_model_cfg=self.cfg,
                                        model_parallel_config=self.model_parallel_config,
                                    )
        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        return True


class MegatronGPTAdapterModel(MegatronGPTLayerwisePEFTModel):
    """
    MegatronGPTAdapterLearningModel is a model that combines a base model (GPTSFTModel) with a adapters.
    This class only supports the canonical Adapter training described in Houlsby et al. (https://arxiv.org/pdf/1902.00751.pdf)

    Two adapter's are inserted into each Transformer layer in the base GPT Model.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded
    and infused into the same GPT Model for inference.
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.PRE_ATTN_ADAPTER,
            AdapterName.POST_ATTN_ADAPTER,
        ]
        adapter_tuning_cfg = cfg.peft.adapter_tuning
        adapter_cfg = ParallelLinearAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=cfg.hidden_size,
            dim=adapter_tuning_cfg.adapter_dim,
            norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
            norm_type=adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
            column_init_method=adapter_tuning_cfg.get("column_init_method", "xavier"),
            row_init_method=adapter_tuning_cfg.get("row_init_method", "zero"),
            dropout=adapter_tuning_cfg.adapter_dropout,
        )

        self.name_key_to_cfg = {}
        self.name_key_to_mcore_mixins = {}
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg
            self.name_key_to_mcore_mixins[k] = [("", MCoreTransformerLayerMixin)]

        self.layer_selection = adapter_tuning_cfg.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))
        super().__init__(cfg, trainer)


class MegatronGPTAdapterModelWeightTying(MegatronGPTLayerwisePEFTModel):
    """
    TODO
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.PRE_ATTN_ADAPTER,
            AdapterName.POST_ATTN_ADAPTER,
        ]
        adapter_tuning_cfg = cfg.peft.adapter_tuning

        adapter_cfg = ParallelLinearAdapterWeightTyingConfig(
            in_features=cfg.hidden_size,
            out_features=cfg.hidden_size,
            dim=adapter_tuning_cfg.adapter_dim,
            norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
            norm_type=adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
            column_init_method=adapter_tuning_cfg.get("column_init_method", "xavier"),
            row_init_method=adapter_tuning_cfg.get("row_init_method", "zero"),
            dropout=adapter_tuning_cfg.adapter_dropout,
            num_position_embeddings=cfg.num_layers * 2,
            dim_position_embeddings=cfg.hidden_size,
            position_embedding_strategy=adapter_tuning_cfg.get("position_embedding_strategy", None),
        )

        self.name_key_to_cfg = {}
        self.name_key_to_mcore_mixins = {}
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg
            self.name_key_to_mcore_mixins[k] = [("", MCoreTransformerLayerMixin)]

        self.layer_selection = adapter_tuning_cfg.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))
        super().__init__(cfg, trainer)
        self.tie_weights()

    def tie_weights(self,):
        pos_idx = 0

        if self.mcore_gpt:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.decoder.layers
            else:
                layers = self.model.decoder.layers
        else:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.language_model.encoder.layers
            else:
                layers = self.model.language_model.encoder.layers

        layer0 = layers[0]
        for adapter_name in layer0.adapter_layer:
            adapter = layer0.get_adapter_module(adapter_name)
            print(adapter_name, pos_idx)
            adapter.set_position(pos_idx)
            pos_idx += 1

        for layer in layers[1:]:
            for adapter_name in layer.adapter_layer:
                print(adapter_name, pos_idx)
                adapter_l = layer.get_adapter_module(adapter_name)
                adapter_0 = layer0.get_adapter_module(adapter_name)
                if hasattr(adapter_0, "layer_norm"):
                    lnorm = adapter_0.layer_norm
                else:
                    lnorm = None
                adapter_l.tie_weights(pos_idx, adapter_0)
                pos_idx += 1


class MegatronGPTIA3Model(MegatronGPTLayerwisePEFTModel):
    """
    MegatronGPTInfusedAdapterModel is a model that combines a base model (GPTSFTModel) with a "Infused Adapter that can Inhibiting and Amplify Inner Activations", known as IA3.
    This class supports the addition of IA3 into a transformer based LM as described in Liu et al. (https://arxiv.org/pdf/2205.05638.pdf)

    Three adapter's are inserted into each Transformer layer in the base GPT Model. Each adapter is basically a vector that simply scales the key, value or ffn hidden representations.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded
    and infused into the same GPT Model for inference.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.peft_name_keys = [AdapterName.KEY_INFUSED, AdapterName.VALUE_INFUSED, AdapterName.MLP_INFUSED]

        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=cfg.ffn_hidden_size // cfg.tensor_model_parallel_size
        )
        infused_adapter_cfg = InfusedAdapterConfig(in_features=cfg.hidden_size // cfg.tensor_model_parallel_size)

        self.name_key_to_cfg = {}
        for k in self.peft_name_keys:
            if k == AdapterName.MLP_INFUSED:
                self.name_key_to_cfg[k] = mlp_infused_adapter_cfg
            elif k in [
                AdapterName.KEY_INFUSED,
                AdapterName.VALUE_INFUSED,
            ]:
                self.name_key_to_cfg[k] = infused_adapter_cfg
            else:
                raise ValueError(f"PEFT Key {k} is unknown.")

        self.layer_selection = cfg.peft.ia3_tuning.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))

        super().__init__(cfg, trainer)


class MegatronGPTPTuningModel(MegatronGPTPEFTModel):
    """
    MegatronGPTPTuningModel is a model that combines a base model (GPTSFTModel) with a p-tuning prefix in the
    input word embedding representations using a prompt-encoder as descripted in Liu et al. https://arxiv.org/pdf/2103.10385.pdf

    The mixin framework adds the output of prompt-encoder (i.e. the virtual embeddings) inside
    nemo/collections/nlp/modules/common/megatron/language_model.py
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.peft_name_keys = [AdapterName.PTUNING_ADAPTER]

        adapter_cfg = PromptEncoderAdapterConfig(
            cfg.peft.p_tuning.virtual_tokens,
            cfg.peft.p_tuning.bottleneck_dim,
            cfg.peft.p_tuning.embedding_dim,
            cfg.peft.p_tuning.init_std,
            cfg.hidden_size,
        )
        self.name_key_to_cfg = {AdapterName.PTUNING_ADAPTER: adapter_cfg}
        self.name_key_to_mcore_mixins = {AdapterName.PTUNING_ADAPTER: [('embedding', MCoreGPTEmbeddingMixin)]}
        super().__init__(cfg, trainer)
        self.virtual_tokens = cfg.peft.p_tuning.virtual_tokens

    def init_peft_modules(self,):
        """
        Initialize the p-tuning prompt encoder in the mixin.
        This should only happen in the first stage of the pipeline unlike other PEFT methods like Lora or Adapters
        because p-tuning only adds params at input to the encoder layer.
        """
        if not self.first_stage_of_pipeline():
            # There are no params to add if we are not in the first state of the pipeline
            return True
        super().init_peft_modules()
        return True

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Reimplement state_dict for ptuning because we also need to check the stage of the pipeline.
        The check is required to make pp>1 to work.
        """
        if self.setup_complete:
            if self.first_stage_of_pipeline():
                return self.get_peft_state_dict()
            # if we are not in the first state of pipeline after setup is done
            # there should be no params in the state_dict
            return {}
        else:
            return self.model.state_dict(prefix="model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Reimplement load_state_dict for ptuning because we also need to check the stage of the pipeline.
        The check is required to make pp>1 to work.
        """
        if len(state_dict) == 0:
            return  # checkpoint is loaded in on_load_checkpoint()
        if self.setup_complete:
            if self.first_stage_of_pipeline():
                # if we are not in the first state of pipeline after setup is done
                # there should be no params to load...
                assert set(state_dict.keys()) == self.adapter_keys
                super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if self.setup_complete:
            if self.first_stage_of_pipeline():
                super().on_load_checkpoint(checkpoint)
        else:
            super().on_load_checkpoint(checkpoint)

    def setup_optimizer_param_groups(self):
        if self.first_stage_of_pipeline():
            super().setup_optimizer_param_groups()
        else:
            self.freeze()  # Freeze the entire model
            self._optimizer_param_groups = ({"params": []},)
        logging.info(f"Optimizer groups set:\n{self.summarize()}")


class MegatronGPTAdapterPTuningModel(MegatronGPTPEFTModel):
    """
    Want to combine adapters and p-tuning? Why not? they are orthogonal methods.
    This class includes both sets of params.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.peft_name_keys = [
            AdapterName.PRE_ATTN_ADAPTER,
            AdapterName.POST_ATTN_ADAPTER,
            AdapterName.PTUNING_ADAPTER,
        ]
        ptuning_cfg = PromptEncoderAdapterConfig(
            cfg.peft.p_tuning.virtual_tokens,
            cfg.peft.p_tuning.bottleneck_dim,
            cfg.peft.p_tuning.embedding_dim,
            cfg.peft.p_tuning.init_std,
            cfg.hidden_size,
        )
        adapter_tuning_cfg = cfg.peft.adapter_tuning
        adapter_cfg = ParallelLinearAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=cfg.hidden_size,
            dim=adapter_tuning_cfg.adapter_dim,
            norm_position=adapter_tuning_cfg.get("norm_position", "pre"),
            norm_type=adapter_tuning_cfg.get("norm_type", "mixedfusedlayernorm"),
            column_init_method=adapter_tuning_cfg.get("column_init_method", "xavier"),
            row_init_method=adapter_tuning_cfg.get("row_init_method", "zero"),
            dropout=adapter_tuning_cfg.adapter_dropout,
        )

        self.name_key_to_cfg = {
            AdapterName.PRE_ATTN_ADAPTER: adapter_cfg,
            AdapterName.POST_ATTN_ADAPTER: adapter_cfg,
            AdapterName.PTUNING_ADAPTER: ptuning_cfg,
        }
        logging.warning("AdapterPTuning doesn't support mcore for now. need to use regex to match target.")
        self.name_key_to_mcore_mixins = {
            AdapterName.PRE_ATTN_ADAPTER: [('', MCoreTransformerLayerMixin)],
            AdapterName.POST_ATTN_ADAPTER: [('', MCoreTransformerLayerMixin)],
            AdapterName.PTUNING_ADAPTER: [('embedding', MCoreGPTEmbeddingMixin)],
        }
        super().__init__(cfg, trainer)
        self.virtual_tokens = cfg.peft.p_tuning.virtual_tokens


class MegatronGPTLoRAModel(MegatronGPTLayerwisePEFTModel):
    """
    MegatronGPTLoRAModel is a model that combines a base model (GPTSFTModel) with a low-rank adapters.
    The lora adapters will be added in `nemo/collections/nlp/modules/common/megatron/attention.py`
    The implementation is based on Hu et al. nemo/collections/nlp/modules/common/megatron/attention.py

    A single low-rank feedfowrad layer is used in parallel with the KQV projection layer.
    TODO: Add support to also include an option to adda low-rank adapter in the output projection layer.
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
        ]
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads
        num_query_groups = cfg.get("num_query_groups", None)
        if num_query_groups is None:
            num_query_groups = cfg.num_attention_heads
        qkv_projection_size = projection_size + 2 * kv_channels * num_query_groups

        adapter_cfg = LoraKQVAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=qkv_projection_size,
            dim=lora_cfg.adapter_dim,
            norm_position=None,
            norm_type=None,
            activation="identity",
            column_init_method=lora_cfg.get("column_init_method", "normal"),
            row_init_method=lora_cfg.get("row_init_method", "zero"),
            gather_output=False,
            dropout=lora_cfg.adapter_dropout,
        )

        self.name_key_to_cfg = {}
        self.name_key_to_mcore_mixins = {}  # maps peft_key to a list of tuples (mcore_target, mcore_mixin)
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg
            self.name_key_to_mcore_mixins[k] = [("self_attention", MCoreSelfAttentionMixin)]
        self.layer_selection = lora_cfg.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))
        super().__init__(cfg, trainer)


class MegatronGPTLoRAModelWeightTying(MegatronGPTLayerwisePEFTModel):
    """
    TODO
    """

    def __init__(
        self, cfg: DictConfig, trainer: Trainer,
    ):
        self.peft_name_keys = [
            AdapterName.LORA_KQV_ADAPTER,
        ]
        lora_cfg = cfg.peft.lora_tuning
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads
        num_query_groups = cfg.get("num_query_groups", None)
        if num_query_groups is None:
            num_query_groups = cfg.num_attention_heads
        qkv_projection_size = projection_size + 2 * kv_channels * num_query_groups
        position_embedding_strategy = lora_cfg.get("position_embedding_strategy", None)
        if position_embedding_strategy is None:
            dim_position_embeddings = 0
        elif position_embedding_strategy == "add":
            dim_position_embeddings = cfg.hidden_size
        elif position_embedding_strategy == "biasadd":
            dim_position_embeddings = 3 * projection_size
        elif position_embedding_strategy == "concat":
            dim_position_embeddings = lora_cfg.adapter_dim
        elif position_embedding_strategy == "mlpconcat":
            dim_position_embeddings = lora_cfg.adapter_dim
        else:
            raise RuntimeError(f"Unknown position embedding strategy {position_embedding_strategy} for tied weights")

        adapter_cfg = LoraKQVAdapterWeightTyingConfig(
            in_features=cfg.hidden_size,
            out_features=qkv_projection_size,
            dim=lora_cfg.adapter_dim,
            norm_position=None,
            norm_type=None,
            activation="identity",
            column_init_method=lora_cfg.get("column_init_method", "normal"),
            row_init_method=lora_cfg.get("row_init_method", "zero"),
            gather_output=False,
            dropout=lora_cfg.adapter_dropout,
            num_position_embeddings=cfg.num_layers,
            dim_position_embeddings=dim_position_embeddings,
            position_embedding_strategy=position_embedding_strategy,
        )

        self.name_key_to_cfg = {}
        self.name_key_to_mcore_mixins = {}
        for k in self.peft_name_keys:
            self.name_key_to_cfg[k] = adapter_cfg
            self.name_key_to_mcore_mixins[k] = [("self_attention", MCoreSelfAttentionMixin)]
        self.layer_selection = lora_cfg.get("layer_selection", None)
        if self.layer_selection is None:
            self.layer_selection = list(range(1, cfg.num_layers + 1))
        super().__init__(cfg, trainer)
        self.tie_weights()

    def tie_weights(self,):
        pos_idx = 0

        if self.mcore_gpt:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.decoder.layers
            else:
                layers = self.model.decoder.layers
        else:
            if self.cfg.megatron_amp_O2:
                layers = self.model.module.language_model.encoder.layers
            else:
                layers = self.model.language_model.encoder.layers

        layer0 = layers[0]
        for adapter_name in layer0.self_attention.adapter_layer:
            adapter = layer0.self_attention.get_adapter_module(adapter_name)
            print(adapter_name, pos_idx)
            adapter.set_position(pos_idx)
            pos_idx += 1

        for layer in layers[1:]:
            for adapter_name in layer.self_attention.adapter_layer:
                print(adapter_name, pos_idx)
                adapter_l = layer.self_attention.get_adapter_module(adapter_name)
                adapter_0 = layer0.self_attention.get_adapter_module(adapter_name)
                position_embeddings_0 = None
                if adapter_0.position_embedding_strategy:
                    position_embeddings_0 = adapter_0.position_embeddings
                adapter_l.tie_weights(pos_idx, adapter_0)
                pos_idx += 1
