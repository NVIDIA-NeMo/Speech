# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import os
import warnings
from contextlib import nullcontext
from dataclasses import fields
from functools import cache, partial
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm

from nemo.collections.multimodal.data.clip.clip_dataset import (
    build_imagenet_validation_dataloader,
    build_train_valid_datasets,
)
from nemo.collections.multimodal.losses.clip_loss import ClipLoss
from nemo.collections.multimodal.losses.siglip_loss import SigLipLoss
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import get_specs, mcore_supports_moe
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.utils_funcs import activation_to_func, get_last_rank
from nemo.collections.vision.modules.vit.vit_backbone import VitBackbone
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel as McoreDDP
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.models.vision.clip_vit_model import CLIPViTModel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import (
        drain_embedding_wgrad_compute,
        get_model_config,
        init_method_normal,
        scaled_init_method_normal,
    )

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

try:
    import transformer_engine
    from transformer_engine.pytorch import module as te_module

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@cache
def mcore_supports_moe() -> bool:
    global HAVE_MEGATRON_CORE
    if not HAVE_MEGATRON_CORE:
        return False
    try:
        from megatron.core.transformer.moe.router import TopKRouter

        return True
    except ImportError:
        return False


class CLIPVisionTransformer(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self, model_cfg, model_parallel_config, pre_process=True, post_process=True, skip_head=False):
        super(CLIPVisionTransformer, self).__init__()

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )

        self.config = model_parallel_config
        self.hidden_size = model_cfg.hidden_size
        self.global_average_pool = model_cfg.global_average_pool
        self.pre_process = pre_process
        self.post_process = post_process
        self.skip_head = skip_head

        if model_cfg.get("class_token_length") is None or model_cfg.get("class_token_length") <= 0:
            class_token = False
        else:
            class_token = True
        self.backbone = VitBackbone(
            model_cfg,
            model_parallel_config,
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            class_token=class_token,
            single_token_output=False,
        )

        if self.post_process and not skip_head:
            self.output_dim = model_cfg.output_dim
            self.head = torch.nn.Linear(
                self.hidden_size,
                self.output_dim,
                bias=False,
            )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.backbone.set_input_tensor(input_tensor)

    def forward(self, input):
        hidden_states = self.backbone(input)

        if self.post_process and not self.skip_head:
            if self.global_average_pool:
                hidden_states = hidden_states.mean(dim=1)
            else:
                hidden_states = hidden_states[:, 0]
            hidden_states = self.head(hidden_states)
        # print("vision_head", hidden_states.shape)
        return hidden_states


class CLIPTextTransformer(MegatronModule):
    """Text Transformer Model."""

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super(CLIPTextTransformer, self).__init__()

        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = model_cfg.fp16_lm_cross_entropy
        self.gradient_accumulation_fusion = model_cfg.gradient_accumulation_fusion

        scaled_init_method = (
            scaled_init_method_normal(model_cfg.init_method_std, model_cfg.num_layers)
            if model_cfg.use_scaled_init_method
            else init_method_normal(model_cfg.init_method_std)
        )
        self.language_model, self._language_model_key = get_language_model(
            config=model_parallel_config,
            vocab_size=padded_vocab_size,
            hidden_size=model_cfg.hidden_size,
            hidden_dropout=model_cfg.hidden_dropout,
            attention_dropout=model_cfg.attention_dropout,
            num_tokentypes=0,
            max_position_embeddings=model_cfg.max_position_embeddings,
            num_layers=model_cfg.num_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            apply_query_key_layer_scaling=model_cfg.apply_query_key_layer_scaling,
            kv_channels=model_cfg.kv_channels,
            ffn_hidden_size=model_cfg.ffn_hidden_size,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            position_embedding_type=model_cfg.get("position_embedding_type", "learned_absolute"),
            init_method=init_method_normal(model_cfg.init_method_std),
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            init_method_std=model_cfg.init_method_std,
            precision=model_cfg.precision,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            activations_checkpoint_granularity=model_cfg.activations_checkpoint_granularity,
            activations_checkpoint_method=model_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=model_cfg.activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline=model_cfg.activations_checkpoint_layers_per_pipeline,
            normalization=model_cfg.normalization,
            layernorm_epsilon=model_cfg.layernorm_epsilon,
            bias_activation_fusion=model_cfg.bias_activation_fusion,
            bias_dropout_add_fusion=model_cfg.bias_dropout_add_fusion,
            masked_softmax_fusion=model_cfg.masked_softmax_fusion,
            persist_layer_norm=model_cfg.persist_layer_norm,
            openai_gelu=model_cfg.openai_gelu,
            onnx_safe=model_cfg.onnx_safe,
            megatron_legacy=model_cfg.megatron_legacy,
            transformer_engine=False,
            fp8=model_cfg.fp8,
            fp8_e4m3=model_cfg.fp8_e4m3,
            fp8_hybrid=model_cfg.fp8_hybrid,
            fp8_margin=model_cfg.fp8_margin,
            fp8_interval=model_cfg.fp8_interval,
            fp8_amax_history_len=model_cfg.fp8_amax_history_len,
            fp8_amax_compute_algo=model_cfg.fp8_amax_compute_algo,
            reduce_amax=model_cfg.get('reduce_amax', True),
            use_emha=model_cfg.use_emha,
            activation=model_cfg.get('activation', 'gelu'),
            use_flash_attention=model_cfg.get('flash_attention', False),
        )

        self.initialize_word_embeddings(
            init_method=init_method_normal(model_cfg.init_method_std),
            vocab_size=padded_vocab_size,
            hidden_size=model_cfg.hidden_size,
        )

        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(model_cfg.max_position_embeddings).expand(1, -1).cuda()

        if self.post_process:
            self.output_dim = model_cfg.output_dim
            self.head = torch.nn.Linear(
                model_cfg.hidden_size,
                self.output_dim,
                bias=False,
            )

        self.attn_mask = self.build_attention_mask(model_cfg.max_position_embeddings)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def build_attention_mask(self, max_position_embeddings):
        # lazily create causal attention mask, with full attention between the tokens
        mask = torch.empty(max_position_embeddings, max_position_embeddings, dtype=bool, device='cuda')
        mask.fill_(True)
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.reshape(1, 1, max_position_embeddings, max_position_embeddings)
        return mask

    def forward(
        self,
        input_ids,
    ):
        # input_ids: [b, s]
        # position_ids: [b, s]
        # attention_mask: [1, 1, s, s]

        hidden_states = self.language_model(
            input_ids,
            self.position_ids,
            self.attn_mask,
            token_type_ids=None,
            layer_past=None,
            get_key_value=False,
            encoder_input=None,
            set_inference_key_value_memory=False,
            inference_max_sequence_len=None,
            checkpoint_activations_all_layers=None,
        )

        if self.post_process:
            # shape = [seq, bsz, hidden]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            hidden_states = hidden_states[input_ids.argmax(dim=-1), torch.arange(hidden_states.shape[1])]
            return self.head(hidden_states)

        return hidden_states


class SiglipMHAPoolingHead(TransformerLayer):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
    ):
        super().__init__(config, submodules)

        self.probe = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        # [s, b, h]
        probe = self.probe.repeat(1, batch_size, 1)
        hidden_state = hidden_state.transpose(0, 1)
        hidden_state, context = super().forward(
            probe,
            attention_mask=None,
            context=hidden_state,
        )

        return hidden_state[0]


class MCoreSiglipViTModel(CLIPViTModel):
    def __init__(self, *args, **kwargs):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = kwargs.pop('output_dim')
        kwargs['ln_pre_impl'] = IdentityOp
        super().__init__(*args, **kwargs)
        assert self.output_dim == self.config.hidden_size, "Siglip output_dim needs to be the same as hidden_size."
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=True,
        )
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.head = SiglipMHAPoolingHead(
            self.config,
            submodules=TransformerLayerSubmodules(
                cross_attention=ModuleSpec(
                    module=CrossAttention,
                    params={"attn_mask_type": MCoreAttnMaskType.no_mask},
                    submodules=CrossAttentionSubmodules(
                        linear_q=TEColumnParallelLinear,
                        linear_kv=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                cross_attn_bda=get_bias_dropout_add,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        )

    def forward(self, x):
        x = super().forward(
            x,
        )
        x = self.final_layernorm(x)
        x = self.head(x)
        return x


class MCoreSiglipTextModel(MCoreGPTModel):
    def __init__(self, *args, **kwargs):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = kwargs.pop('output_dim')
        kwargs['transformer_layer_spec'].submodules.self_attention.params['attn_mask_type'] = MCoreAttnMaskType.no_mask

        super().__init__(*args, **kwargs)
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=True,
        )

        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(kwargs['max_sequence_length']).expand(1, -1).cuda()

    def forward(self, input_ids):

        x = super().forward(input_ids, position_ids=self.position_ids, attention_mask=None)
        x = self.final_layernorm(x)
        x = x[-1]
        x = self.head(x)
        return x


class MCoreCLIPViTModel(CLIPViTModel):
    def __init__(self, *args, **kwargs):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = kwargs.pop('output_dim')
        super().__init__(*args, **kwargs)
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )

    def forward(self, x):
        x = super().forward(
            x,
        )
        x = self.final_layernorm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


class MCoreCLIPTextModel(MCoreGPTModel):
    def __init__(self, *args, **kwargs):
        # TODO (yuya): need to handle post_process correctly in order to enable PP
        self.output_dim = kwargs.pop('output_dim')

        super().__init__(*args, **kwargs)
        self.final_layernorm = TENorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.head = torch.nn.Linear(
            self.config.hidden_size,
            self.output_dim,
            bias=False,
        )
        self.position_ids = None
        if self.pre_process:
            self.position_ids = torch.arange(kwargs['max_sequence_length']).expand(1, -1).cuda()

    def forward(self, input_ids):
        x = super().forward(input_ids, position_ids=self.position_ids, attention_mask=None)
        x = self.final_layernorm(x)
        x = x[input_ids.argmax(dim=-1), torch.arange(x.shape[1])]
        x = self.head(x)
        return x


class CLIPModel(MegatronModule):
    """CLIP Model"""

    def __init__(
        self,
        model_cfg,
        model_parallel_config,
        vision_transformer_config,
        text_transformer_config,
        padded_vocab_size,
        pre_process=True,
        post_process=True,
    ):
        super(CLIPModel, self).__init__()

        self.config = model_parallel_config
        self.use_siglip = model_cfg.get("use_siglip", False)
        self.pre_process = pre_process
        self.post_process = post_process
        self.output_dim = model_cfg.output_dim
        self.get_attention_mask_from_fusion = model_cfg.get('get_attention_mask_from_fusion', True)

        if model_cfg.get("mcore_gpt", False):
            if model_cfg.vision.get("class_token_length") is None or model_cfg.vision.get("class_token_length") <= 0:
                add_class_token = False
            else:
                add_class_token = True
            vision_layer_spec = get_specs(
                model_cfg.text.get('name', ''),
                vision_transformer_config,
                model_cfg.get('transformer_engine', True),
            )
            vision_layer_spec.submodules.self_attention.params['attn_mask_type'] = MCoreAttnMaskType.no_mask

            if model_cfg.get("use_siglip", False):
                vision_module = MCoreSiglipViTModel
                text_module = MCoreSiglipTextModel
            else:
                vision_module = MCoreCLIPViTModel
                text_module = MCoreCLIPTextModel
            self.vision_encoder = vision_module(
                transformer_config=vision_transformer_config,
                transformer_layer_spec=vision_layer_spec,
                patch_dim=model_cfg.vision.get('patch_dim', 16),
                img_h=model_cfg.vision.get('img_h', 224),
                img_w=model_cfg.vision.get('img_w', 224),
                add_class_token=add_class_token,
                class_token_len=model_cfg.vision.get('class_token_length'),
                output_dim=model_cfg.output_dim,
            )
            self.text_encoder = text_module(
                config=text_transformer_config,
                transformer_layer_spec=get_specs(
                    model_cfg.text.get('name', ''),
                    text_transformer_config,
                    model_cfg.get('transformer_engine', True),
                ),
                vocab_size=model_cfg.text.get('override_vocab_size', padded_vocab_size),
                max_sequence_length=model_cfg.text.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=False,
                parallel_output=True,
                share_embeddings_and_output_weights=False,
                position_embedding_type=model_cfg.text.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=model_cfg.text.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=model_cfg.text.get('seq_len_interpolation_factor', None),
                rotary_base=model_cfg.text.get('rotary_base', 10000),
                output_dim=model_cfg.output_dim,
            )

        else:
            self.vision_encoder = CLIPVisionTransformer(
                model_cfg.vision,
                model_parallel_config,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
            self.text_encoder = CLIPTextTransformer(
                model_cfg.text,
                model_parallel_config,
                padded_vocab_size,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )

        if self.use_siglip:
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(10))
            self.logit_bias = torch.nn.Parameter(torch.ones([]) * (-10))
        else:
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def forward(self, images, captions):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(captions)

        if self.post_process:
            if self.use_siglip:
                return (
                    F.normalize(image_features, dim=-1),
                    F.normalize(text_features, dim=-1),
                    self.logit_scale.exp(),
                    self.logit_bias,
                )
            return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1), self.logit_scale.exp()

        return image_features, text_features

    def build_transformer_config(self) -> TransformerConfig:
        """Builds the megatron core gpt transformer config for the model.
        For attributes in the nemo model config that are the same
        as the megatron core TransformerConfig, we will use the value from the nemo model config.
        For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """

        normalization = self.cfg.get('normalization', 'layernorm').lower()
        layernorm_zero_centered_gamma = self.cfg.get('normalization', 'layernorm') == 'layernorm1p'
        if normalization == 'layernorm':
            normalization = 'LayerNorm'
        elif normalization == 'rmsnorm':
            normalization = 'RMSNorm'
        elif normalization == 'layernorm1p':
            normalization = 'LayerNorm'
            layernorm_zero_centered_gamma = True
        else:
            logging.warning(
                f"The normalization type: {normalization} might not be supported in megatron core."
                f"Supported types are LayerNorm and RMSNorm."
            )

        ub_tp_comm_overlap = self.cfg.get('ub_tp_comm_overlap', False)

        if not self.cfg.get('fp8', False):
            fp8 = None
        elif self.cfg.get('fp8_e4m3', False):
            fp8 = 'e4m3'
        elif self.cfg.get('fp8_hybrid', False):
            fp8 = 'hybrid'
        else:
            raise ValueError(f"fp8 enabled but fp8_format (fp8_e4m3 | fp8_hybrid) is not set.")

        # any configs that are not in the nemo model config will be added here
        model_specific_configs = {
            'layernorm_zero_centered_gamma': layernorm_zero_centered_gamma,
            'normalization': normalization,
            'fp8': fp8,
            'tp_comm_overlap': ub_tp_comm_overlap,
            # MoE related
            'num_moe_experts': self.cfg.get('num_moe_experts', None),
            'moe_router_load_balancing_type': self.cfg.get('moe_router_load_balancing_type', 'aux_loss'),
            'moe_router_topk': self.cfg.get('moe_router_topk', 2),
            'moe_grouped_gemm': self.cfg.get('moe_grouped_gemm', False),
            'moe_aux_loss_coeff': self.cfg.get(
                'moe_aux_loss_coeff', 0
            ),  # 1e-2 would be a good start value for load balance loss.
            'moe_z_loss_coeff': self.cfg.get('moe_z_loss_coeff', None),  # 1e-3 would be a good start value for z-loss
            'moe_input_jitter_eps': self.cfg.get('moe_input_jitter_eps', None),
            'moe_token_dropping': self.cfg.get('moe_token_dropping', False),  # TODO: Support token dropping.
        }
        if model_specific_configs['num_moe_experts'] is not None:
            assert mcore_supports_moe(), 'Megatron-core >= v0.5.0 is required for MoE'
        elif not mcore_supports_moe():
            if 'num_moe_experts' in model_specific_configs:
                del model_specific_configs['num_moe_experts']
            moe_keys = list(filter(lambda x: x.startswith('moe_'), model_specific_configs.keys()))
            for k in moe_keys:
                del model_specific_configs[k]

        transformer_config = super().build_transformer_config()

        for key, value in model_specific_configs.items():
            setattr(transformer_config, key, value)

        # pass mcore customization configs directly to mcore
        mcore_customization_config_dict = self.cfg.get('mcore_customization_config', {})
        for key, value in mcore_customization_config_dict.items():
            setattr(transformer_config, key, value)

        return transformer_config


class MegatronCLIPModel(MegatronBaseModel):
    """Megatron CLIP Model."""

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        self.imagenet_val = None
        super().__init__(cfg, trainer=trainer)

        self._validate_trainer()

        # placeholder for O2 wrapper
        self.transformer_config = self.build_transformer_config(self.cfg.text)

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        self.mcore_gpt = cfg.get('mcore_gpt', False)
        if cfg.get('fp8', False):
            self.prev_step_training = True
        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        self.transformer_engine = cfg.get('transformer_engine', False)
        if self.megatron_amp_O2 and not self.transformer_engine:
            logging.warning('megatron_amp_O2 is enabled but transformer-engine is not.')

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
            )
        else:
            build_model_context = nullcontext
            if HAVE_TE and self.cfg.get('fp8', False) and self.cfg.get('fp8_params', False):
                build_model_context = transformer_engine.pytorch.fp8_model_init
            with build_model_context():
                self.model = build_model(
                    model_provider_func=self.model_provider_func,
                    wrap_with_ddp=False,
                    virtual_pipeline_model_parallel_size=self.cfg.get('virtual_pipeline_model_parallel_size', None),
                    on_cpu=cfg.get('fsdp', False) and cfg.get('use_cpu_initialization', False),
                )

        # if we're not using interleaved, then self.model is a module.
        if self.cfg.get('virtual_pipeline_model_parallel_size', None) is None and (not self.use_mcore_dist_optim):
            self.model = self.model[0]

        if self.megatron_amp_O2:

            if not self.with_distributed_adam and not self.cfg.get("use_cpu_initialization", False):
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            self._wrap_model_for_O2()

        self.enable_autocast = (
            True if (not self.megatron_amp_O2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        # Convert the global-batch-based profile index to micro-batch index
        if hasattr(self, '_nsys_profile_enabled') or hasattr(self, '_memory_profile_enabled'):
            mp_size = cfg.get('tensor_model_parallel_size', 1) * cfg.get('pipeline_model_parallel_size', 1)
            cp_size = cfg.get('context_parallel_size', 1)
            data_parallel_world_size = trainer.world_size // (mp_size * cp_size)
            grad_accum_steps = cfg.get('global_batch_size') // (cfg.get('micro_batch_size') * data_parallel_world_size)
            if hasattr(self, '_nsys_profile_enabled'):
                self._nsys_profile_start_step *= grad_accum_steps
                self._nsys_profile_end_step *= grad_accum_steps
            if hasattr(self, '_memory_profile_enabled'):
                self._memory_profile_start_step *= grad_accum_steps
                self._memory_profile_end_step *= grad_accum_steps

        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)
        self.log_train_loss = bool(int(os.getenv("NEMO_LOG_TRAIN_LOSS", 1)))
        self.log_memory_usage = bool(int(os.getenv("NEMO_LOG_MEMORY_USAGE", 0)))
        self.loss_broadcast_src_rank = None
        data_cfg = cfg.get('data', {})
        self.return_output_tensors = data_cfg.get('return_output_tensors', False)
        self.validation_drop_last = data_cfg.get('validation_drop_last', True)
        self.sample_weight = data_cfg.get('sample_weight', 'token')
        self.validation_param_sync_overlap = self.cfg.get('validation_param_sync_overlap', False)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        vision_transformer_config = self.build_transformer_config(self.cfg.vision) if self.mcore_gpt else None
        text_transformer_config = self.build_transformer_config(self.cfg.text) if self.mcore_gpt else None

        if self.mcore_gpt and not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        model = CLIPModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            vision_transformer_config=vision_transformer_config,
            text_transformer_config=text_transformer_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def setup_mcore_distributed_parallel(self):
        """Set up mcore distributed data parallel"""
        if self.with_distributed_adam and self.use_mcore_dist_optim:
            config = get_model_config(self.model[0])
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=(self.cfg.optim.get('grad_sync_dtype', 'fp32') == 'fp32'),
                overlap_grad_reduce=self.cfg.optim.get('overlap_grad_sync', False),
                use_distributed_optimizer=True,
                check_for_nan_in_grad=self.cfg.optim.get('check_for_nan_in_grad', False),
                # mcore bucket_size is based on num of parameters, therefore not
                # using bucket_cap_mb to configure bucket_size here
                bucket_size=self.cfg.optim.get('ddp_bucket_size', None),
            )

            self.model = [
                McoreDDP(
                    config,
                    ddp_config,
                    model_chunk,
                    data_parallel_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    expert_data_parallel_group=parallel_state.get_data_modulo_expert_parallel_group(),
                    # Turn off bucketing for model_chunk 2 onwards, since communication for these
                    # model chunks is overlapped with compute anyway.
                    disable_bucketing=(model_chunk_idx > 0),
                )
                for (model_chunk_idx, model_chunk) in enumerate(self.model)
            ]

            # (TODO) Broadcast params from data parallel src rank to other data parallel ranks.
            # by calling model_module.broadcast_params() if the model is randomly initialized.

    def configure_optimizers(self):

        if self.with_distributed_adam and not self.use_mcore_dist_optim:

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, Float16Module):
                        module = module.module
                    stage_bucket = []
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, Float16Module):
                        module = module.module
                    for layer in itertools.chain(
                        module.vision_encoder.backbone.transformer.layers,
                        module.text_encoder.language_model.encoder.layers,
                    ):
                        buckets.append(
                            [p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)]
                        )
            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    def forward(self, image, text):
        output_tensor = self.model(image, text)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, forward_only):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam and not self.use_mcore_dist_optim:
            no_sync_func = partial(
                self._optimizer.no_sync,
                greedy_grad_copy=self.megatron_amp_O2,
            )
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # pipeline schedules will get these from self.model.config
        for module in self.get_model_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=None,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.stack(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                raise NotImplementedError("Losses of micro batches sizes must be uniform!")
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def initialize_ub_func(self):
        ub_cfgs = self.cfg.get('ub_tp_comm_overlap_cfg', None)
        if ub_cfgs is None:
            warnings.warn(
                "Couldn't find TP config. Please check the path correctness. Initializing TP comm overlap with the default config."
            )

        input_shape = [
            self.cfg.get('encoder_seq_length')
            * self.cfg.get('micro_batch_size')
            // self.cfg.get('context_parallel_size', 1),
            self.cfg.get('hidden_size'),
        ]

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=self.cfg.get('tensor_model_parallel_size'),
            use_fp8=self.cfg.get('fp8'),
            ub_cfgs=ub_cfgs,
        )
        self.initialize_ub = False

    def training_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        Batch should be a list of microbatches and those microbatches should on CPU.
        Microbatches are then moved to GPU during the pipeline.
        The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        if self.with_distributed_adam and not self.use_mcore_dist_optim:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                module = module.text_encoder
                if not self.mcore_gpt:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean = self.fwd_bwd_step(dataloader_iter, False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.cfg.get('fp8', False):
            self.prev_step_training = self.training

        # Optimization: Defer the embedding GEMM Wgrads of the last PP stage to pipeline flush waiting time
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            if (
                self.cfg.get('defer_embedding_wgrad_compute', False) and self.mcore_gpt
            ):  # Silently ignore the optimization if MCORE is not used
                module_list = self.get_model_module_list()
                if len(module_list) > 1:
                    embedding_module = module_list[-1]
                else:
                    embedding_module = module_list[0]

                embedding_activation_buffer = embedding_module.embedding_activation_buffer
                grad_output_buffer = embedding_module.grad_output_buffer
                weight = embedding_module.output_layer.weight

                drain_embedding_wgrad_compute(
                    embedding_module.config, embedding_activation_buffer, grad_output_buffer, weight
                )

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.megatron_timer_start('allreduce_sequence_parallel_gradients', log_level=1)
            self.allreduce_sequence_parallel_gradients()
            self.megatron_timer_stop('allreduce_sequence_parallel_gradients')

        self.megatron_timer_start('gradient_allreduce', log_level=1)
        if self.use_fsdp:
            # Reduce the gradients omitted from FSDP-sharding
            self.allreduce_fsdp_sharding_omitted_gradients()
        elif self.with_distributed_adam:
            if not self.use_mcore_dist_optim:
                # synchronize asynchronous grad reductions
                # note: not necessary, but reduces performance degradation
                # from multiple simultaneous NCCL calls
                self._optimizer._finish_bucket_grad_sync()
            # else: Mcore distributed optim calls finalize_model_grads to finish grad sync
        elif self.megatron_amp_O2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if (
                self.cfg.get('pipeline_model_parallel_size', 1) > 1
                or self.cfg.get('sequence_parallel', False)
                or not self.cfg.get('async_grad_allreduce', True)
            ):
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)
        self.megatron_timer_stop('gradient_allreduce')

        if (
            not self.use_mcore_dist_optim
            and self.cfg.get('pipeline_model_parallel_size', 1) > 1
            and self.cfg.get('share_embeddings_and_output_weights', True)
        ):
            self.megatron_timer_start('allreduce_first_last_embeddings', log_level=1)
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()
            self.megatron_timer_stop('allreduce_first_last_embeddings')

        if self.log_memory_usage:
            mem_reserved = torch.cuda.max_memory_reserved()
            self.log(
                'peak_memory_usage',
                mem_reserved,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )

        ## logging
        if self.log_train_loss:
            # When using pipeline parallelism, loss is calculated only in the last pipeline stage and
            # it should be casted to other pipeline stages for logging.
            # we can avoid this broadcast by updating the PTL log function to accept specific ranks
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if torch.distributed.get_rank() == get_last_rank():
                    torch.distributed.send(loss_mean, 0)
                elif torch.distributed.get_rank() == 0:
                    torch.distributed.recv(loss_mean, get_last_rank())
            self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)

            # (@adithyare) we need to check for the _scaler attribute to enable pp>1 for adapter training
            if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
                loss_scale = self.trainer.precision_plugin.scaler._scale
                if loss_scale is not None:
                    self.log('loss_scale', loss_scale, batch_size=1)

        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log(
            'global_step',
            self.trainer.global_step + 1,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        consumed_samples = self._compute_consumed_samples_after_training_step()
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples',
            consumed_samples,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )

        return loss_mean

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from apex.
        No need to call it here.
        """
        pass

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        pass

    def _append_sequence_parallel_module_grads(self, module, grads):
        """Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False)
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_O2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def get_forward_output_and_loss_func(self):
        if self.cfg.get("use_siglip", False):
            # TODO(yuya): fix rank
            loss_func = SigLipLoss(
                rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
                group=parallel_state.get_data_parallel_group(),
            )
        else:
            loss_func = ClipLoss(
                local_loss=self.cfg.local_loss,
                gather_with_grad=self.cfg.gather_with_grad,
            )

        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                images = batch["images"].cuda(non_blocking=True)
                captions = batch["captions"].cuda(non_blocking=True)
            else:
                # GPT3 uses only causal mask, which doesn't need attention mask
                if parallel_state.is_pipeline_first_stage():
                    # Fist pipeline stage needs only the tokens and position_ids
                    images = batch["images"].cuda(non_blocking=True)
                    captions = batch["captions"].cuda(non_blocking=True)
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    images, captions = None, None

            output_tensor = model(images, captions)
            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            raise NotImplementedError

        return fwd_output_only_func

    def zero_shot_classifier(self):
        if self.cfg.get("megatron_amp_O2", False):
            text_encoder = self.model.module.text_encoder
        else:
            text_encoder = self.model.text_encoder

        with torch.no_grad():
            zeroshot_weights = []
            for texts in self.imagenet_val["texts"]:
                texts = texts.cuda(non_blocking=True)
                # TODO (yuya): distributed not working
                with torch.cuda.amp.autocast(
                    enabled=self.autocast_dtype in (torch.half, torch.bfloat16),
                    dtype=self.autocast_dtype,
                ):
                    class_embeddings = text_encoder(texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    def zero_shot_eval(self):
        def accuracy(output, target, topk=(1,)):
            pred = output.topk(max(topk), 1, True, True)[1].t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

        logging.info('Starting zero-shot imagenet.')

        logging.info('Building zero-shot classifier')
        classifier = self.zero_shot_classifier()

        logging.info('Using classifier')

        if self.cfg.get("megatron_amp_O2", False):
            vision_encoder = self.model.module.vision_encoder
        else:
            vision_encoder = self.model.vision_encoder
        with torch.no_grad():
            top1, top5, n = 0.0, 0.0, 0.0
            for images, target in tqdm(self.imagenet_val["images"], desc="Imagenet Zero-shot Evaluation", leave=False):
                if images is None or target is None:
                    continue

                images = images.cuda(non_blocking=True).to(self.autocast_dtype)
                target = target.cuda(non_blocking=True)
                # predict
                with torch.cuda.amp.autocast(
                    enabled=self.autocast_dtype in (torch.half, torch.bfloat16),
                    dtype=self.autocast_dtype,
                ):
                    image_features = vision_encoder(images)
                    image_features = F.normalize(image_features, dim=-1)
                    logits = 100.0 * image_features @ classifier

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        logging.info('Finished zero-shot imagenet.')
        top1 = top1 / n
        top5 = top5 / n
        return top1, top5

    def validation_step(self, dataloader_iter):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions."""
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        loss = self.fwd_bwd_step(dataloader_iter, True)
        self.validation_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        # TODO (yuya): need fix later, check with Sean
        if not self.validation_step_outputs:
            return

        # Run zero shot imagenet evaluation
        if self.imagenet_val is not None:
            imagenet_metric = torch.zeros(2).cuda()
            imagenet_metric[0], imagenet_metric[1] = self.zero_shot_eval()
            imagenet_metric = average_losses_across_data_parallel_group(imagenet_metric)
            self.log('imagenet_top1', imagenet_metric[0], prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('imagenet_top5', imagenet_metric[1], prog_bar=True, rank_zero_only=True, batch_size=1)

        if parallel_state.is_pipeline_last_stage():
            averaged_metrics = torch.tensor(
                [torch.stack(self.validation_step_outputs).mean()], dtype=torch.float32, device='cuda'
            )
        else:
            averaged_metrics = torch.tensor([0.0], dtype=torch.float32, device='cuda')

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_metrics, get_last_rank())
        averaged_loss = averaged_metrics

        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for CLIP...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        self._train_ds, self._validation_ds = build_train_valid_datasets(
            model_cfg=self.cfg,
            consumed_samples=self.compute_consumed_samples(0),
            tokenizer=self.tokenizer,
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for CLIP.')

        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        # log number of parameters
        if isinstance(self.model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in self.model]
            )
        else:
            num_parameters_on_device = sum([p.nelement() for p in self.model.parameters()])

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()

        # Batch size need to be provided for webdatset
        self._num_micro_batches = get_num_microbatches()
        self._micro_batch_size = self.cfg.micro_batch_size

        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

        if self.cfg.data.get("imagenet_val") is not None:
            self.imagenet_val = build_imagenet_validation_dataloader(self.cfg, self.tokenizer)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        raise NotImplementedError

    def _validate_trainer(self):
        """Certain trainer configurations can break training.
        Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    def build_transformer_config(self, model_cfg=None) -> TransformerConfig:
        """Builds the megatron core gpt transformer config for the model.
        For attributes in the nemo model config that are the same
        as the megatron core TransformerConfig, we will use the value from the nemo model config.
        For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """
        if model_cfg is None:
            model_cfg = self.cfg
        normalization = model_cfg.get('normalization', 'layernorm').lower()
        layernorm_zero_centered_gamma = model_cfg.get('normalization', 'layernorm') == 'layernorm1p'
        if normalization == 'layernorm':
            normalization = 'LayerNorm'
        elif normalization == 'rmsnorm':
            normalization = 'RMSNorm'
        elif normalization == 'layernorm1p':
            normalization = 'LayerNorm'
            layernorm_zero_centered_gamma = True
        else:
            logging.warning(
                f"The normalization type: {normalization} might not be supported in megatron core."
                f"Supported types are LayerNorm and RMSNorm."
            )

        ub_tp_comm_overlap = model_cfg.get('ub_tp_comm_overlap', False)

        if not model_cfg.get('fp8', False):
            fp8 = None
        elif model_cfg.get('fp8_e4m3', False):
            fp8 = 'e4m3'
        elif model_cfg.get('fp8_hybrid', False):
            fp8 = 'hybrid'
        else:
            raise ValueError(f"fp8 enabled but fp8_format (fp8_e4m3 | fp8_hybrid) is not set.")

        # any configs that are not in the nemo model config will be added here
        model_specific_configs = {
            'layernorm_zero_centered_gamma': layernorm_zero_centered_gamma,
            'normalization': normalization,
            'fp8': fp8,
            'tp_comm_overlap': ub_tp_comm_overlap,
            # MoE related
            'num_moe_experts': model_cfg.get('num_moe_experts', None),
            'moe_router_load_balancing_type': model_cfg.get('moe_router_load_balancing_type', 'aux_loss'),
            'moe_router_topk': model_cfg.get('moe_router_topk', 2),
            'moe_grouped_gemm': model_cfg.get('moe_grouped_gemm', False),
            'moe_aux_loss_coeff': model_cfg.get(
                'moe_aux_loss_coeff', 0
            ),  # 1e-2 would be a good start value for load balance loss.
            'moe_z_loss_coeff': model_cfg.get('moe_z_loss_coeff', None),  # 1e-3 would be a good start value for z-loss
            'moe_input_jitter_eps': model_cfg.get('moe_input_jitter_eps', None),
            'moe_token_dropping': model_cfg.get('moe_token_dropping', False),  # TODO: Support token dropping.
        }
        if model_specific_configs['num_moe_experts'] is not None:
            assert mcore_supports_moe(), 'Megatron-core >= v0.5.0 is required for MoE'
        elif not mcore_supports_moe():
            if 'num_moe_experts' in model_specific_configs:
                del model_specific_configs['num_moe_experts']
            moe_keys = list(filter(lambda x: x.startswith('moe_'), model_specific_configs.keys()))
            for k in moe_keys:
                del model_specific_configs[k]

        # create a dictionary copy of the model config
        cfg = OmegaConf.to_container(model_cfg, resolve=True)

        # create a dict to store the transformer config arguments
        transformer_config_dict = {}

        # get model parallel configs from the base class
        model_parallel_config = self.build_model_parallel_config()

        add_bias_linear = model_cfg.get('bias', True)
        add_qkv_bias = model_cfg.get('qkv_bias', False)

        activation = model_cfg.get('activation', 'gelu')
        gated_linear_unit = activation.endswith('glu')
        # TODO: need to check which activation functions are supported in mcore
        activation_func = activation_to_func(activation, openai_gelu=model_cfg.get("openai_gelu", False))

        normalization = model_cfg.get('normalization', 'LayerNorm')

        init_method_std = model_cfg.get('init_method_std', 0.02)
        # default used in mcore
        init_method = init_method_normal(init_method_std)

        output_layer_init_method = init_method
        num_layers = model_cfg.get('num_layers', 1)
        use_scaled_init_method = model_cfg.get('use_scaled_init_method', True)
        if use_scaled_init_method:
            output_layer_init_method = scaled_init_method_normal(init_method_std, num_layers=num_layers)

        attention_softmax_in_fp32 = False  # not currently used in NeMo unless apply_query_key_layer_scaling is True
        apply_query_key_layer_scaling = model_cfg.get('apply_query_key_layer_scaling', False)

        rotary_interleaved = model_cfg.get('rotary_interleaved', False)

        fp16_enabled = self.trainer.precision in [16, '16', '16-mixed']
        if apply_query_key_layer_scaling:
            if fp16_enabled:
                os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
            else:
                logging.warning(
                    "apply_query_key_layer_scaling is only enabled when using FP16, setting it to False "
                    "and setting NVTE_APPLY_QK_LAYER_SCALING=0"
                )
                os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "0"
                apply_query_key_layer_scaling = False

        if apply_query_key_layer_scaling:
            attention_softmax_in_fp32 = True

        bias_activation_fusion = model_cfg.get('bias_activation_fusion', True)

        bias_dropout_fusion = model_cfg.get('bias_dropout_add_fusion', True)

        apply_rope_fusion = model_cfg.get('apply_rope_fusion', False)

        # TODO: need to check if recompute APIs are matching up properly
        recompute_granularity = model_cfg.get('activations_checkpoint_granularity', None)
        recompute_method = model_cfg.get('activations_checkpoint_method', None)
        recompute_num_layers = model_cfg.get('activations_checkpoint_num_layers', None)

        # any configs that are not in the nemo model config will be added here
        config_mapping = {
            'apply_query_key_layer_scaling': apply_query_key_layer_scaling,
            'apply_residual_connection_post_layernorm': False,  # we don't use this in NeMo
            'layernorm_zero_centered_gamma': False,
            'add_bias_linear': add_bias_linear,
            'add_qkv_bias': add_qkv_bias,
            'gated_linear_unit': gated_linear_unit,
            'activation_func': activation_func,
            'normalization': normalization,
            'init_method': init_method,
            'output_layer_init_method': output_layer_init_method,
            'attention_softmax_in_fp32': attention_softmax_in_fp32,
            'bias_activation_fusion': bias_activation_fusion,
            'bias_dropout_fusion': bias_dropout_fusion,
            'apply_rope_fusion': apply_rope_fusion,
            'recompute_granularity': recompute_granularity,
            'recompute_method': recompute_method,
            'recompute_num_layers': recompute_num_layers,
            'distribute_saved_activations': False,  # not currently used in NeMo
            'fp8': None,
            'rotary_interleaved': rotary_interleaved,
            'deallocate_pipeline_outputs': True,
        }

        # populate the transformer config dict
        for field in fields(TransformerConfig):
            # config mapping has second highest priority
            if field.name in config_mapping:
                transformer_config_dict[field.name] = config_mapping[field.name]
            # then config
            elif field.name in cfg:
                transformer_config_dict[field.name] = cfg[field.name]
            # then model parallel config
            elif field in fields(model_parallel_config):
                transformer_config_dict[field.name] = getattr(model_parallel_config, field.name)
            else:
                logging.warning(
                    f"The model: {self} does not have field.name: {field.name} in its cfg. "
                    f"Add this key to cfg or config_mapping to make to make it configurable."
                )

        transformer_config = TransformerConfig(**transformer_config_dict)

        for key, value in model_specific_configs.items():
            setattr(transformer_config, key, value)

        # pass mcore customization configs directly to mcore
        mcore_customization_config_dict = model_cfg.get('mcore_customization_config', {})
        for key, value in mcore_customization_config_dict.items():
            setattr(transformer_config, key, value)

        return transformer_config
