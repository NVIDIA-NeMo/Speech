# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import queue
import warnings
from contextlib import nullcontext
from dataclasses import fields
from functools import cache, partial
from importlib.metadata import version
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pkg_resources import packaging
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.utils import extend_instance
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronCorePretrainingSampler,
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.data.language_modeling.megatron.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from nemo.collections.nlp.models.language_modeling.megatron.falcon.falcon_spec import get_falcon_layer_spec
from nemo.collections.nlp.models.language_modeling.megatron.gpt_full_te_layer_autocast_spec import (
    get_gpt_full_te_layer_autocast_spec,
)
from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import get_gpt_layer_modelopt_spec
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_ltor_masks_and_position_ids,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_strategy import TextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.utils_funcs import activation_to_func, get_last_rank
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging
from nemo.utils.te_utils import is_float8tensor

try:
    from megatron.core import InferenceParams, parallel_state, tensor_parallel
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
    from megatron.core.datasets.utils import get_blend_from_list
    from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
    from megatron.core.dist_checkpointing.mapping import LocalNonpersitentObject, ShardedObject
    from megatron.core.distributed import DistributedDataParallel as McoreDDP
    from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads

    # NeMo's implementation of the get_gpt_layer_ammo_spec function is temporarily used
    # from megatron.core.inference.gpt.model_specs import get_gpt_layer_ammo_spec
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import (
        drain_embedding_wgrad_compute,
        get_model_config,
        init_method_normal,
        scaled_init_method_normal,
    )

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    TransformerConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import (
        get_current_global_batch_size,
        get_num_microbatches,
        update_num_microbatches,
    )

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import (
        get_current_global_batch_size,
        get_num_microbatches,
        update_num_microbatches,
    )

try:
    import transformer_engine
    from transformer_engine.pytorch import module as te_module

    from nemo.collections.nlp.modules.common.hyena.hyena_spec import get_gpt_layer_with_te_and_hyena_spec

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


## TODO: This function will not work if TE is not installed
def get_specs(spec_name, transformer_config=None, use_te=True, hyena_cfg: Dict = None):
    # else cases for backwards compatibility with neva
    num_experts = transformer_config.num_moe_experts if transformer_config else None
    moe_grouped_gemm = transformer_config.moe_grouped_gemm if transformer_config else False

    if num_experts is not None:
        assert mcore_supports_moe(), "Megatron-core >= v0.5.0 is required for MoE"

    if use_te and spec_name == '':
        spec_name = 'te_gpt'
    name_spec_dict = {
        "": get_gpt_layer_local_spec(num_experts, moe_grouped_gemm),
        "te_gpt": get_gpt_layer_with_transformer_engine_spec(num_experts, moe_grouped_gemm),
        "megatron_falcon_gpt": get_falcon_layer_spec(),
        "megatron_gpt_full_te_layer_autocast": get_gpt_full_te_layer_autocast_spec(transformer_config),
        "modelopt": get_gpt_layer_modelopt_spec(num_experts),
        "te_gpt_hyena": get_gpt_layer_with_te_and_hyena_spec(hyena_cfg),
    }
    if spec_name not in name_spec_dict:
        raise ValueError(f"Spec name '{spec_name}' is not recognized.")
    return name_spec_dict[spec_name]


class EmbeddingScalingMixin(torch.nn.Module):
    """
    A mixin class for scaling embeddings in Megatron GPT.
    The scaling is applied only if the configuration (accessible via `self.config`)
    includes `apply_embedding_scaling` set to True.
    """

    def forward(self, **kwargs):
        """
        Forward pass that scales the output embeddings from the `forward` method of
        the superclass by the square root of the hidden size specified in the configuration.
        """
        embeddings = super().forward(**kwargs)
        return embeddings * torch.tensor(self.config.hidden_size**0.5, dtype=embeddings.dtype)


class MegatronGPTExportableModel(torch.nn.Module, Exportable):
    """
    Megatron GPT Wrapper for ONNX export
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fp8_enabled = model.cfg.get('fp8', False)
        self.fp8_recipe = None
        if self.fp8_enabled and HAVE_TE:
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=0, interval=1, fp8_format=transformer_engine.common.recipe.Format.E4M3
            )

        self.dtype = utils_funcs.torch_dtype_from_precision(model.cfg.precision)

    def forward(self, tokens, position_ids, attention_mask):
        if self.fp8_enabled and HAVE_TE:
            with (
                transformer_engine.pytorch.onnx_export(self.fp8_enabled),
                transformer_engine.pytorch.fp8_autocast(enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe),
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast('cuda', dtype=self.dtype),
                warnings.catch_warnings(),
            ):
                warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning, module=r'.*')
                assert tokens.shape == position_ids.shape
                assert attention_mask.shape[2] == attention_mask.shape[3] == tokens.shape[1] == position_ids.shape[1]
                output_tensor = self.model.forward(
                    tokens=tokens.cuda(),
                    text_position_ids=position_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=None,
                )
        else:
            with (
                torch.no_grad(),
                torch.inference_mode(),
                torch.autocast('cuda', dtype=self.dtype),
                warnings.catch_warnings(),
            ):
                warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning, module=r'.*')
                assert tokens.shape == position_ids.shape
                assert attention_mask.shape[2] == attention_mask.shape[3] == tokens.shape[1] == position_ids.shape[1]
                output_tensor = self.model.forward(
                    tokens=tokens.cuda(),
                    text_position_ids=position_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=None,
                )

        return output_tensor

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def input_example(self, max_batch=1, max_dim=768, seq_len=6):
        ids = [self.model.tokenizer.text_to_ids(text) for text in ["how is the weather on           Sunday"]]
        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids]
        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, self.model.tokenizer.eos_id, False, False, False)
            for id_tensor in id_tensors
        ]
        for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
            attn_mask, _, pos_ids = attn_mask_and_pos_ids
            return tokens, pos_ids, attn_mask

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "position_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('D', 'D', 'T', 'T'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'position_ids', 'attention_mask']

    @property
    def output_names(self) -> List[str]:
        return ['logits']


class MegatronGPTModel(MegatronBaseModel, TextGeneration):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        if not HAVE_MEGATRON_CORE:
            logging.warning(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        super().__init__(cfg, trainer=trainer, no_lm_init=True)

        self._validate_trainer()

        # build the transformer config
        # TODO: add type hint once pip package is out
        self.transformer_config = self.build_transformer_config()

        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        self.mcore_gpt = cfg.get('mcore_gpt', False)
        self.spec_name = cfg.get('name', '')
        if cfg.get('fp8', False):
            self.prev_step_training = True
        self.continue_training = True if cfg.get("restore_from_ckpt") else False

        self.rampup_batch_size = self.cfg.get('rampup_batch_size', None)
        if self.rampup_batch_size:
            self.prev_consumed_samples = 0
            self.if_first_step = 0
            self.prev_global_batch_size = None

        if cfg.get('data', None) is not None:
            self.reset_position_ids = cfg.data.get('reset_position_ids', False)
            self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
            self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)

        if not self.megatron_amp_O2 and self.cfg.get('virtual_pipeline_model_parallel_size', None):
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        if not self.megatron_amp_O2 and self.cfg.get('expert_model_parallel_size', 1) > 1:
            raise ValueError('Expert parallelism is only supported when using megatron_amp_O2')

        if self.cfg.get('expert_model_parallel_size', 1) > 1 and self.with_distributed_adam:
            if not self.use_mcore_dist_optim:
                raise ValueError(
                    'Expert parallelism is currently not supporting Apex distributed optimizer, use Mcore distributed optimizer instead'
                )

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
                    on_cpu=cfg.get('use_cpu_initialization', False),
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

        # configuration used for inference
        self._inference_config = None

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

        self.get_attention_mask_from_fusion = self.cfg.get('get_attention_mask_from_fusion', True)
        self.initialize_ub = self.cfg.get('ub_tp_comm_overlap', False)
        self.log_train_loss = bool(int(os.getenv("NEMO_LOG_TRAIN_LOSS", 1)))
        self.log_memory_usage = bool(int(os.getenv("NEMO_LOG_MEMORY_USAGE", 0)))
        self.loss_broadcast_src_rank = None
        data_cfg = cfg.get('data', {})
        self.validation_drop_last = data_cfg.get('validation_drop_last', True)
        self.sample_weight = data_cfg.get('sample_weight', 'token')
        self.validation_param_sync_overlap = self.cfg.get('validation_param_sync_overlap', False)

        self.inference_params = None

        # Reset learning rate params
        self.if_init_step = True
        self.reset_lr = self.cfg.get('reset_lr', False)
        self.reset_lr_steps = self.cfg.get('reset_lr_steps', False)
        if self.reset_lr and (not self.with_distributed_adam or not self.megatron_amp_O2):
            raise ValueError(
                'Learning rate reset feature is only supported with the distributed optmizer and megatron_amp_O2 for now.'
            )

        # default to false since this doesn't work with sequence parallelism currently
        self.use_loss_mask = self.cfg.get('use_loss_mask', False)

        if self.use_loss_mask and self.transformer_config.sequence_parallel:
            raise ValueError('Loss mask is not supported with sequence parallelism.')

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
            model = MCoreGPTModel(
                config=self.transformer_config,
                transformer_layer_spec=get_specs(
                    self.spec_name,
                    self.transformer_config,
                    self.transformer_engine,
                    self.cfg.get('hyena', None),
                ),
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                max_sequence_length=self.cfg.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
            )
            if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
                extend_instance(model.embedding, EmbeddingScalingMixin)
        else:
            assert self.cfg.get('num_query_groups', None) is None or self.cfg.get(
                'num_query_groups', None
            ) == self.cfg.get(
                'num_attention_heads', None
            ), "Group Query Attention is only supported in Megatron Core. Set 'mcore_gpt' to use GQA."

            model = GPTModel(
                config=self.model_parallel_config,
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                hidden_size=self.cfg.hidden_size,
                max_position_embeddings=self.cfg.max_position_embeddings,
                num_layers=self.cfg.num_layers,
                num_attention_heads=self.cfg.num_attention_heads,
                apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=self.cfg.get('kv_channels', None),
                ffn_hidden_size=self.cfg.ffn_hidden_size,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=self.cfg.get('init_method_std', 0.02),
                use_scaled_init_method=self.cfg.get('use_scaled_init_method', True),
                fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
                hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
                attention_dropout=self.cfg.get('attention_dropout', 0.1),
                ffn_dropout=self.cfg.get('ffn_dropout', 0.0),
                precision=self.cfg.get('precision', 16),
                fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
                activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
                activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_layers_per_pipeline=self.cfg.get(
                    'activations_checkpoint_layers_per_pipeline', None
                ),
                normalization=self.cfg.get('normalization', 'layernorm'),
                layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
                onnx_safe=self.cfg.get('onnx_safe', False),
                bias=self.cfg.get('bias', True),
                bias_activation_fusion=self.cfg.get('bias_activation_fusion', True),
                bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
                activation=self.cfg.get('activation', 'gelu'),
                headscale=self.cfg.get('headscale', False),
                transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
                openai_gelu=self.cfg.get('openai_gelu', False),
                normalize_attention_scores=self.cfg.get('normalize_attention_scores', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percentage=self.cfg.get('rotary_percentage', 1.0),
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                attention_type=self.cfg.get('attention_type', 'multihead'),
                masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
                persist_layer_norm=self.cfg.get('persist_layer_norm', False),
                transformer_engine=self.cfg.get('transformer_engine', False),
                fp8=self.cfg.get('fp8', False),
                fp8_e4m3=self.cfg.get('fp8_e4m3', False),
                fp8_hybrid=self.cfg.get('fp8_hybrid', False),
                fp8_margin=self.cfg.get('fp8_margin', 0),
                fp8_interval=self.cfg.get('fp8_interval', 1),
                fp8_amax_history_len=self.cfg.get('fp8_amax_history_len', 1024),
                fp8_amax_compute_algo=self.cfg.get('fp8_amax_compute_algo', 'max'),
                reduce_amax=self.cfg.get('reduce_amax', True),
                use_emha=self.cfg.get('use_emha', False),
                ub_tp_comm_overlap=self.cfg.get('ub_tp_comm_overlap', False),
                use_flash_attention=self.cfg.get('use_flash_attention', False),
                megatron_legacy=self.cfg.get('megatron_legacy', False),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
            )
            if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
                extend_instance(model.language_model.embedding, EmbeddingScalingMixin)
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

            # Special handling for embedding grads
            with_fp32_embedding_grads = self.cfg.get('with_fp32_embedding_grads', True)
            modules = self.get_model_module_list()
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                module = modules[0]  # first virtual rank has the embeddings

                # Word embeddings: use FP32 grads and disable
                # overlapped grad sync with pipeline parallelism
                word_embeddings = (
                    module.shared_embedding_or_output_weight() if self.mcore_gpt else module.word_embeddings_weight()
                )
                word_embeddings._with_fp32_optimizer = with_fp32_embedding_grads
                if parallel_state.get_pipeline_model_parallel_world_size() > 1 and self.cfg.get(
                    'share_embeddings_and_output_weights', True
                ):
                    word_embeddings._disable_greedy_grad_copy = not self.megatron_amp_O2
                    word_embeddings._disable_overlap_grad_sync = True

                # Position embeddings: use FP32 grads
                position_embeddings = None
                if self.mcore_gpt:
                    if module.embedding.add_position_embedding:
                        position_embeddings = module.embedding.position_embeddings.weight
                else:
                    position_embeddings = module.position_embeddings_weight()
                if position_embeddings is not None:
                    position_embeddings._with_fp32_optimizer = with_fp32_embedding_grads

            # Handle case where embeddings are used in output layer
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and self.cfg.get(
                'share_embeddings_and_output_weights', True
            ):
                module = modules[-1]  # last virtual rank has the embeddings
                word_embeddings = (
                    module.shared_embedding_or_output_weight() if self.mcore_gpt else module.word_embeddings_weight()
                )
                word_embeddings._with_fp32_optimizer = with_fp32_embedding_grads
                if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                    word_embeddings._disable_greedy_grad_copy = not self.megatron_amp_O2
                    word_embeddings._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_O2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping and params in the
            # first layer are put together in a bucket. If FP8 tensors
            # are detected, those are also put in the first layer's
            # bucket.
            def make_parameter_bucket(module: torch.nn.Module) -> List[torch.nn.Parameter]:
                bucket = [
                    param for param in module.parameters() if not getattr(param, '_disable_overlap_grad_sync', False)
                ]
                if any(is_float8tensor(param) for param in bucket):
                    bucket = list(filter(is_float8tensor, bucket))
                return bucket

            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    stage_bucket = []
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    buckets.extend(make_parameter_bucket(layer) for layer in layers)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    buckets.extend(make_parameter_bucket(layer) for layer in layers)
            buckets.reverse()
            used_params = set(itertools.chain.from_iterable(buckets))
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    def forward(self, tokens, text_position_ids, attention_mask, labels):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels=labels)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, forward_only, first_val_step=None):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if self.with_distributed_adam:
            if forward_only:
                if self.validation_param_sync_overlap:
                    param_sync_func = self.sync_overlap_parameters
            elif not self.use_mcore_dist_optim:
                no_sync_func = partial(
                    self._optimizer.no_sync,
                    greedy_grad_copy=self.megatron_amp_O2,
                )
                grad_sync_func = self.reduce_overlap_gradients
                param_sync_func = self.sync_overlap_parameters
            else:
                if self.cfg.optim.get("overlap_grad_sync", False):
                    no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
                    no_sync_func = no_sync_func[0] if len(self.model) == 1 else no_sync_func

                    if self.cfg.optim.get("delay_grad_reduce", True):
                        grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.model]
                        grad_sync_func = grad_sync_func[0] if len(self.model) == 1 else grad_sync_func
                if self.cfg.optim.get("overlap_param_sync", False) and self.cfg.optim.get("delay_param_gather", False):
                    param_sync_func = [
                        lambda x, model_index=model_index: self._optimizer.finish_param_sync(model_index, x)
                        for model_index in range(len(self.model))
                    ]
                    param_sync_func = param_sync_func[0] if len(self.model) == 1 else param_sync_func

        # pipeline schedules will get these from self.model.config
        for module in self.get_model_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func
            if self.use_mcore_dist_optim:
                module.config.finalize_model_grads_func = finalize_model_grads

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.cfg.encoder_seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
            first_val_step=first_val_step,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.validation_drop_last:
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    loss_sum['loss_sum_and_ub_size']
                    for loss_sum in losses_reduced_per_micro_batch
                    if loss_sum['loss_sum_and_ub_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0]).cuda()
                )
                return loss_sum
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

    def training_step_fwd_bwd_step_call(self, dataloader_iter, forward_only):
        """
        This method is called from the training_step method.
        It is separated out to allow for overriding in the MegatronGPTEmbeddingModel
        """
        loss_mean = self.fwd_bwd_step(dataloader_iter, forward_only)
        return loss_mean

    def training_step(self, dataloader_iter):
        """
        We pass the dataloader iterator function to the micro-batch scheduler.
        The input batch to each micro-batch is fetched using the dataloader function
        in the micro-batch fwd function.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        # Reset learning rate
        if self.if_init_step and self.reset_lr:
            num_groups = len(self._optimizer.param_groups)
            for group in range(num_groups):
                self._optimizer.param_groups[group]['lr'] = (
                    0.0 if self.cfg.optim.sched.warmup_steps > 0 else self.cfg.optim.lr
                )
            self._optimizer.param_groups[0]['reset_lr'] = {
                'num_steps': self.trainer.global_step,
                'reset_lr_steps': True if self.reset_lr_steps else False,
                'if_init_step': self.if_init_step,
            }
            self.if_init_step = False

        if self.rampup_batch_size:
            current_global_batch_size = get_current_global_batch_size()
            # do validation and save the checkpoint when gbs is changed
            if self.prev_global_batch_size != current_global_batch_size and self.prev_global_batch_size:
                self.trainer.should_stop = True

        # zero out the mcore grad buf
        if self.use_mcore_dist_optim:
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()

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
                if not self.mcore_gpt:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            if (
                self.cfg.get('defer_embedding_wgrad_compute', False)
                and self.mcore_gpt
                and not self.use_mcore_dist_optim
            ):  # Silently ignore the optimization if MCORE is not used
                module_list = self.get_model_module_list()
                if len(module_list) > 1:
                    embedding_module = module_list[-1]
                else:
                    embedding_module = module_list[0]

                embedding_module.embedding_activation_buffer.clear()
                assert (
                    len(embedding_module.embedding_activation_buffer) == 0
                ), "When you defer wgrads, this buffer should not hold stray activations"

        loss_mean = self.training_step_fwd_bwd_step_call(dataloader_iter, forward_only=False)

        if self.cfg.get('fp8', False):
            self.prev_step_training = self.training

        # Optimization: Defer the embedding GEMM Wgrads of the last PP stage to pipeline flush waiting time
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1 and parallel_state.is_pipeline_last_stage(
            ignore_virtual=True
        ):
            if (
                self.cfg.get('defer_embedding_wgrad_compute', False)
                and self.mcore_gpt
                and not self.use_mcore_dist_optim
            ):  # Silently ignore the optimization if MCORE is not used
                module_list = self.get_model_module_list()
                if len(module_list) > 1:
                    embedding_module = module_list[-1]
                else:
                    embedding_module = module_list[0]

                embedding_activation_buffer = embedding_module.embedding_activation_buffer
                grad_output_buffer = embedding_module.grad_output_buffer
                if self.cfg.get('share_embeddings_and_output_weights', True):
                    weight = embedding_module.shared_embedding_or_output_weight()
                else:
                    weight = embedding_module.output_layer.weight

                drain_embedding_wgrad_compute(
                    embedding_module.config, embedding_activation_buffer, grad_output_buffer, weight
                )

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            # Mcore DistOpt handles this, so we don't have to
            if not self.use_mcore_dist_optim:
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
            max_memory_reserved = torch.cuda.max_memory_reserved()
            memory_allocated = torch.cuda.memory_allocated()
            self.log(
                'peak_memory_usage',
                max_memory_reserved,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )
            self.log(
                'memory_allocated',
                memory_allocated,
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
            self.trainer.global_step,
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

        if self.rampup_batch_size:
            self.prev_global_batch_size = current_global_batch_size
            self.prev_consumed_samples = consumed_samples
            num_microbatch_calculator.update(
                consumed_samples=consumed_samples,
                consistency_check=False,
            )
            current_global_batch_size = num_microbatch_calculator.current_global_batch_size
            self.log('global_batch_size', current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1)
            self.if_first_step = 1

        return loss_mean

    def backward(self, *args, **kwargs):
        """LightningModule hook to do backward.
        We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
        No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """LightningModule hook to zero grad.
        We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _append_sequence_parallel_module_grads(self, module, grads):
        """Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False) or getattr(
                param, 'sequence_parallel_enabled', False
            )
            # (@adithyare) adapter training now extends MegatronGPTModel
            # so we have to add this check here to ensure we do not
            # perform all_reduce when grad is None.
            # grad can be None when performing PEFT training.
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
        if not grads:
            # may be empty for PEFT training
            return
        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def allreduce_fsdp_sharding_omitted_gradients(self):
        """All-reduce gradients of FSDP-sharding-omitted parameters in sharding domain (data-parallel domain)."""
        assert isinstance(self.model, torch.nn.Module)
        grads = []
        for param in self.model._ignored_params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                grads.append(grad.data)
        if len(grads) > 0:
            coalesced = torch._utils._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def allreduce_first_last_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            module_list = self.get_model_module_list()
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                module = module_list[0]  # only the first virtual rank has the embeddings
            elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                module = module_list[-1]  # only the last virtual rank has the embeddings
            share_embeddings = (
                module.share_embeddings_and_output_weights if self.mcore_gpt else module.share_token_embeddings
            )
            if share_embeddings:
                word_embeddings_weight = (
                    module.shared_embedding_or_output_weight() if self.mcore_gpt else module.word_embeddings_weight()
                )
                # (@adithyare) adapter training now extends MegatronGPTModel so we have to add this check here to ensure we do not perform all_reduce when grad is None.
                # grad can be None when performing PeFT training.
                if word_embeddings_weight.requires_grad:
                    if self.megatron_amp_O2:
                        # O2 recipe stores a "main" copy of weights and grads
                        grad = word_embeddings_weight.main_grad
                    else:
                        grad = word_embeddings_weight.grad
                    torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def _make_data_iterator_list(self, data_iterator: Iterator) -> List[Iterator]:
        """Convert data iterator into form expected by Megatron

        With interleaved pipeline parallelism, Megatron expects a
        list of one data iterator per model chunk. Each model
        chunk independently gets data from its data iterator, so
        we need to interact with the data iterator multiple times
        for each microbatch step. Instead of incorporating this
        logic into the data loader, we cache the iterator's output
        to the first model chunk and reuse it in the other model
        chunks.
        """

        if not isinstance(self.model, list) or len(self.model) == 1:
            return data_iterator  # TODO @tmoon: Remove
            # TODO @tmoon: Use once available in Megatron-LM
            # return DataIteratorList([data_iterator])

        class CachingIterator:
            """Iterator wrapper that caches values"""

            class Proxy:
                """Returns values from caching iterator wrapper

                Assumed to never advance past the caching iterator.
                """

                def __init__(self):
                    self.cache = queue.Queue()

                def __iter__(self):
                    return self

                def __next__(self):
                    return self.cache.get_nowait()

            def __init__(self, iterator: Iterator):
                self.iterator = iterator
                self.proxies = []

            def make_proxy(self):
                self.proxies.append(CachingIterator.Proxy())
                return self.proxies[-1]

            def __iter__(self):
                return self

            def __next__(self):
                val = next(self.iterator)
                for proxy in self.proxies:
                    proxy.cache.put(val)
                return val

        # Make list of iterator wrappers
        iters = [CachingIterator(data_iterator)]
        while len(iters) < len(self.model):
            iters.append(iters[0].make_proxy())
        return iters  # TODO @tmoon: Remove
        # TODO @tmoon: Use once available in Megatron-LM
        # return DataIteratorList(iters)

    def get_batch(self, data_iterator, tuning):
        """Generate a batch."""

        # Broadcast data.
        if data_iterator is not None:
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            data = next(data_iterator)
            if isinstance(data, tuple):
                data = data[0]
        else:
            data = None

        # return batch for GPT SFT
        if tuning:
            return data

        batch = {
            'tokens': data["tokens"],
            'labels': data["labels"],
            'loss_mask': data["loss_mask"],
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"],
            'position_ids': data["position_ids"],
        }
        if "attention_mask" in data:
            batch['attention_mask'] = data["attention_mask"]

        return batch

    def get_batch_on_this_context_parallel_rank(self, batch):
        num_valid_tokens_in_ub = None
        if 'loss_mask' in batch and batch['loss_mask'] is not None:
            num_valid_tokens_in_ub = batch['loss_mask'].sum()

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            cp_rank = parallel_state.get_context_parallel_rank()
            for key, val in batch.items():
                if val is not None:
                    seq_dim = 1 if key != 'attention_mask' else 2
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                        non_blocking=True
                    )
                    val = val.index_select(seq_dim, index)
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    batch[key] = val

        batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub

        return batch

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = self.get_batch(dataloader_iter, tuning)

            # Transfer needed data to GPU
            required_keys = set()
            max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None
            cu_seqlens_argmin = batch['cu_seqlens_argmin'] if 'cu_seqlens_argmin' in batch else None
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if 'cu_seqlens' in batch:
                    required_keys.add('cu_seqlens')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')
            batch = {
                key: val.cuda(non_blocking=True) if key in required_keys and isinstance(val, torch.Tensor) else None
                for key, val in batch.items()
            }

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': None if self.get_attention_mask_from_fusion else batch['attention_mask'],
                'labels': batch['labels'] if 'labels' in batch else None,
                'loss_mask': batch['loss_mask'],
            }

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')

                if 'cu_seqlens' in batch:  # packed sequence from GPTSFTPackedDataset
                    # these args are passed eventually into TEDotProductAttention.forward()
                    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
                    # remove -1 "paddings" added in collate_fn
                    if cu_seqlens_argmin is not None:
                        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
                    else:
                        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

                    try:
                        from megatron.core.packed_seq_params import PackedSeqParams
                    except (ImportError, ModuleNotFoundError) as e:
                        mcore_version = packaging.version.Version(version('megatron-core'))
                        logging.error(
                            f"megatron-core v{mcore_version} does not support training with packed sequence. "
                            "Please use megatron-core >= 0.5.0, or set model.data.train_ds.packed_sequence=False"
                        )
                        raise e

                    forward_args['packed_seq_params'] = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                        qkv_format='thd',
                    )

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                cp_size = parallel_state.get_context_parallel_world_size()
                if isinstance(loss_for_ub, dict):
                    # TODO: need a better way to check if loss_func is returning more stuff than just loss... (@adithyare)

                    if set(loss_for_ub.keys()) == set(
                        ["loss", "query_hs", "pos_doc_hs", "pos_cs", "neg_cs", "diff_cs"]
                    ):  # (adithyare) this check will be True for GPT Embedding models
                        loss = loss_for_ub['loss']
                        reduced_loss = average_losses_across_data_parallel_group([loss])
                        pos_cs = average_losses_across_data_parallel_group([loss_for_ub['pos_cs']])
                        neg_cs = average_losses_across_data_parallel_group([loss_for_ub['neg_cs']])
                        diff_cs = average_losses_across_data_parallel_group([loss_for_ub['diff_cs']])
                        return (
                            loss * cp_size,
                            {
                                'avg': reduced_loss,
                                'query_hs': loss_for_ub['query_hs'],
                                'doc_hs': loss_for_ub['pos_doc_hs'],
                                'avg_pos_cs': pos_cs,
                                'avg_neg_cs': neg_cs,
                                'diff_cs': diff_cs,
                            },
                        )
                    elif set(loss_for_ub.keys()) == set(
                        ["loss", "query_pos_doc_logit", "query_neg_doc_logit", "logit_diff"]
                    ):  # (adithyare) this check will be True for GPT Reranker models

                        loss = loss_for_ub['loss']
                        reduced_loss = average_losses_across_data_parallel_group([loss])
                        logit_diff = average_losses_across_data_parallel_group([loss_for_ub['logit_diff']])
                        return (
                            loss * cp_size,
                            {
                                'avg': reduced_loss,
                                'query_pos_doc_logit': loss_for_ub['query_pos_doc_logit'],
                                'query_neg_doc_logit': loss_for_ub['query_neg_doc_logit'],
                                'logit_diff': logit_diff,
                            },
                        )
                    else:
                        raise RuntimeError(f"Dict loss_for_ub has unknown key set {loss_for_ub.keys()}")

                elif validation_step and not self.validation_drop_last:
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(loss_for_ub)
                        num_valid_tokens_in_ub = 0
                    else:
                        if self.sample_weight == 'constant':
                            num_valid_tokens_in_ub = 1
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub * cp_size, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            # If tuple, 1st element in it is the batch since dataloader_iter returns batch, batch_idx, dataloader_idx
            batch = next(dataloader_iter)
            if isinstance(batch, tuple):
                batch = batch[0]
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                if self.mcore_gpt:
                    # if first step, then clear KV cache, otherwise reuse inference_paarms
                    if set_inference_key_value_memory[0].item():
                        self.inference_params = InferenceParams(
                            max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                        )
                    extra_arg['inference_params'] = self.inference_params
                else:
                    extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                    extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            # Currently for all MCore transformer layer specs causal attention mask
            # is used so we can delegate creating it to MCore/TE and pass None below
            if (
                isinstance(model, MCoreGPTModel)
                or hasattr(model, "module")
                and isinstance(model.module, MCoreGPTModel)
            ):
                attention_mask = None
            output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def validation_step(self, dataloader_iter, dataloader_idx=0):
        """
        Our dataloaders produce a micro-batch and then we fetch
        a number of microbatches depending on the global batch size and model parallel size
        from the dataloader to produce a list of microbatches.
        The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        mode = 'test' if self.trainer.testing else 'val'
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()
        else:
            self.model.eval()

        if self.cfg.get('fp8', False):
            first_val_step = self.prev_step_training and not self.training
            self.prev_step_training = self.training
        else:
            first_val_step = None

        with torch.no_grad():
            loss = self.fwd_bwd_step(dataloader_iter, True, first_val_step)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        else:
            self.model.train()

        if mode == 'val':
            # Append with the correct dataloader_idx in case of multiple dataloaders
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        else:
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)

        return loss

    def on_validation_epoch_end(self):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.validation_drop_last:
                averaged_loss = torch.stack(self.validation_step_outputs).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                total_loss_and_total_samples = torch.vstack(self.validation_step_outputs).sum(axis=0)
                avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                averaged_loss = avg_loss.type(torch.float32).cuda()
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        # When using pipeline parallelism, loss is calculated only in the last pipeline stage and
        # it should be casted to other pipeline stages for logging.
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if self.loss_broadcast_src_rank is None:
                self.loss_broadcast_src_rank = parallel_state.get_pipeline_model_parallel_last_rank()
            torch.distributed.broadcast(
                averaged_loss,
                self.loss_broadcast_src_rank,
                group=parallel_state.get_pipeline_model_parallel_group(),
            )

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        return averaged_loss

    def test_step(self, dataloader_iter):
        return self.validation_step(dataloader_iter)

    def on_test_epoch_end(self):
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        self.test_step_outputs.clear()  # free memory

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / num_valid_tokens_in_ub  # sequence level nll
        if parallel_state.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss

    def build_train_valid_test_datasets(self):
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        logging.info('Building GPT datasets.')
        global_batch_size = self.cfg.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        # TODO: @athitten make num of eval and test samples 1 always, after it works with non DictConfig data_prefix.
        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        # The line below exploits a quirk in mcore dataset construction, to make number of epochs for validation and test equal to 1
        # The mcore dataset implementation uses the number N we provide via train_valid_test_num_samples to derive parameter E such that
        # E = argmin_e e * N_d >= N, or equivalently E = ceildiv(N, N_d)
        # Where N_d is the total number of samples in a dataset (files), and N is the requested number of samples (provided for every split in the list below).
        # Setting N = 1 we force E to be 1 as well
        legacy_dataset = self.cfg.data.get("legacy_dataset", False)
        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[1] = 1 if legacy_dataset else None
        # Add extra FIM tokens to tokenizer
        if self.cfg.data.get('add_fim', False) and self.cfg.tokenizer.library == 'megatron':
            fim_tokens = self.cfg.data.fim.extra_tokens
            fim_tokens = [fim_tokens.prefix, fim_tokens.middle, fim_tokens.suffix, fim_tokens.pad, fim_tokens.eod]
            self.tokenizer.add_special_tokens({'additional_special_tokens': fim_tokens})

        if legacy_dataset:
            self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
                cfg=self.cfg,
                trainer=self.trainer,
                data_prefix=self.cfg.data.data_prefix,
                data_impl=self.cfg.data.data_impl,
                splits_string=self.cfg.data.splits_string,
                train_valid_test_num_samples=train_valid_test_num_samples,
                seq_length=self.cfg.data.seq_length,
                seed=self.cfg.seed,
                skip_warmup=self.cfg.data.get('skip_warmup', True),
                tokenizer=self.tokenizer,
            )
        else:
            # Function needed for mcore GPTDataset
            is_dataset_built_on_rank = lambda: True

            mock_dataset = True if self.cfg.data.get("data_impl", "mmap") == "mock" else False
            add_extra_token = not self.cfg.data.get("no_seqlen_plus_one_input_tokens", False)
            kwargs = {
                "random_seed": self.cfg.seed,
                "sequence_length": self.cfg.data.seq_length,
                "path_to_cache": self.cfg.data.index_mapping_dir,
                "tokenizer": self.tokenizer,
                "reset_position_ids": self.reset_position_ids,
                "reset_attention_mask": self.reset_attention_mask,
                "eod_mask_loss": self.eod_mask_loss,
                "create_attention_mask": not self.get_attention_mask_from_fusion,
                "mmap_bin_files": self.cfg.data.get("mmap_bin_files", True),
                "drop_last_partial_validation_sequence": self.cfg.data.get("validation_drop_last", True),
                "num_dataset_builder_threads": self.cfg.data.get("num_dataset_builder_threads", 1),
                "add_extra_token_to_sequence": add_extra_token,
            }

            data_prefix = self.cfg.data.data_prefix

            # support for dict data input type
            if isinstance(data_prefix, DictConfig):
                kwargs['blend_per_split'] = [
                    get_blend_from_list(data_prefix.train),
                    get_blend_from_list(data_prefix.validation),
                    get_blend_from_list(data_prefix.test),
                ]
            else:
                kwargs['blend'] = None if mock_dataset else get_blend_from_list(data_prefix)
                kwargs["split"] = self.cfg.data.splits_string

            if self.cfg.data.get('add_fim', False):
                dataset_config = GPTFIMDatasetConfig(self.cfg.data.fim, **kwargs)
                dataset_type = GPTFIMDataset
            else:
                dataset_config = GPTDatasetConfig(**kwargs)
                dataset_type = MockGPTDataset if mock_dataset else GPTDataset

            self._train_ds, self._validation_ds, self._test_ds = BlendedMegatronDatasetBuilder(
                dataset_type,
                train_valid_test_num_samples,
                is_dataset_built_on_rank,
                dataset_config,
            ).build()

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            data_sampler = (
                MegatronPretrainingSampler
                if self.cfg.data.get('legacy_dataset', False)
                else MegatronCorePretrainingSampler
            )
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = data_sampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    rampup_batch_size=self.cfg.get('rampup_batch_size', None),
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def setup(self, stage=None):
        """
        PTL hook that is executed after DDP spawns.
        We setup datasets here as megatron datasets require DDP to instantiate.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.

        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert()

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Number of precise model parameters on device: {total_num_parameters}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path and not self.continue_training:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if self.rampup_batch_size:
            update_num_microbatches(self.init_consumed_samples, consistency_check=False)
            self.prev_consumed_samples = self.init_consumed_samples

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            self.build_train_valid_test_datasets()
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)
            # Override limit_train_batches in terms of num of microbatches
            self._reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_dl, 'train')
            # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting in between a step
            self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_dl, 'val')

        # Data cache generation only
        # Stops script execution after creating a data cache
        if self.cfg.data.get('data_cache_generation_only', False):
            self.trainer.num_sanity_val_steps = 0
            self.trainer.should_stop = True

        if stage == 'fit':
            self.initialize_last_rank_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_gpt', False):
            self.setup_transformer_engine_tp_groups()
            self.setup_transformer_engine_cp_groups()

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )

            drop_last = True
            if not self.validation_drop_last:
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            pad_samples_to_global_batch_size = False
            if self.cfg.data.get('pad_samples_to_global_batch_size', False):
                logging.info('pad_samples_to_global_batch_size set to True')
                pad_samples_to_global_batch_size = True

            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, "validation", drop_last, pad_samples_to_global_batch_size
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            if self._test_ds is not None:
                consumed_samples = 0
                logging.info(
                    f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
                )
                self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)
            else:
                self._test_dl = None

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        *,
        strategy: Optional[TextGenerationStrategy] = None,
    ) -> OutputType:

        # check whether the DDP is initialized
        if not parallel_state.is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            if self.cfg.get('transformer_engine', False):
                self.setup_transformer_engine_tp_groups()
                self.setup_transformer_engine_cp_groups()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        strategy_args = {} if strategy is None else {"strategy": strategy}

        return megatron_gpt_generate(
            self.cuda(), inputs, self.tokenizer, length_params, sampling_params, **strategy_args
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                inference_config['inputs'] = batch
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                inference_config['inputs'] = batch
                return generate(self, **inference_config)

    def list_available_models(self):
        return None

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        When using pipeline parallelism, we need the global batch to remain on the CPU,
        since the memory overhead will be too high when using a large number of microbatches.
        Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

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
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="megatron_gpt_345m",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_gpt_345m/versions/1/files/megatron_gpt_345m.nemo",
                description="345M parameter GPT generative Megatron model.",
            )
        )
        return result

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """

        # mcore uses distributed checkpointing
        # FSDP supports the lagecy checkpointing or torch-FSDP-native sharded checkpointing
        if self.mcore_gpt and not self.use_fsdp:
            checkpoint['sharded_state_dict'] = self.sharded_state_dict()

        # legacy checkpointing for interleaved
        else:
            if isinstance(self.model, list):
                for i in range(len(self.model)):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """

        # mcore uses distributed checkpointing
        # FSDP supports the lagecy checkpointing or torch-FSDP-native sharded checkpointing
        if self.mcore_gpt and not self.use_fsdp:
            if 'state_dict' in checkpoint and checkpoint['state_dict']:
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        checkpoint_state_dict = checkpoint['state_dict'][f'model_{index}']
                    else:
                        checkpoint_state_dict = checkpoint['state_dict']
                    # checkpoint_state_dict has "model." but module does not so we need to remove it when loading
                    checkpoint_state_dict = {
                        key.replace('model.', ''): checkpoint_state_dict.pop(key)
                        for key in list(checkpoint_state_dict.keys())
                    }
                    module.load_state_dict(checkpoint_state_dict, strict=True)
            else:
                # when restoring a distributed checkpoint from a ptl checkpoint we need to defer loading the state_dict
                # see NLPModel.on_load_checkpoint
                checkpoint['state_dict'] = {}

        # legacy checkpointing for interleaved
        else:
            if isinstance(self.model, list):
                for i in range(len(self.model)):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_validation_model_zero_grad(self) -> None:
        """
        Skip gradient zeroing at the beginning of validation routine.
        This is needed when overlapping the AllGather of the updated parameters with the following valdation step.
        """
        if not self.validation_param_sync_overlap:
            super().on_validation_model_zero_grad()

    def sharded_state_dict(self, prefix: str = '') -> Dict[str, Any]:
        """
        Creates the sharded state dict which is used by dist_checkpoint to save the sharded tensors to disk.
        When given the sharded_stated_dict, dist_checkpoint.load will load the tensors corresponding to
        self.state_dict().
        The sharded tensor mapping is defined in the GPTModel class from mcore.
        """

        if self.mcore_gpt:
            module_prefix = f'{prefix}model.'
            sharded_state_dict = {}
            for index, module in enumerate(self.get_model_module_list()):
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    # virtual pipline rank must be set so that GPTModel returns the correct sharded state dict
                    parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict[f'model_{index}'] = module_sharded_state_dict
                else:
                    module_sharded_state_dict = module.sharded_state_dict(prefix=module_prefix)
                    sharded_state_dict.update(module_sharded_state_dict)

            # reset vp rank
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)

            # WAR: This is a temporary fix to skip loading FP8 parameters for Dot Product Attention
            def skip_fp8_load(x):
                if isinstance(x, ShardedObject) and 'fused_attention' in x.key and '_extra_state' in x.key:
                    x = LocalNonpersitentObject(x.data)  # use the FP8 state from initialization, not from ckpt
                return x

            if self.cfg.get('skip_fp8_attention_checkpoint_load', True):
                dict_list_map_inplace(skip_fp8_load, sharded_state_dict)

            return sharded_state_dict

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    @property
    def mgpt_wrapper(self):
        return MegatronGPTExportableModel(self)

    def list_export_subnets(self):
        return ['mgpt_wrapper']

    def initialize_last_rank_embeddings(self):
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if self.cfg.get('share_embeddings_and_output_weights', True):
                for index, module in enumerate(self.get_model_module_list()):
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                    sync_embeddings = (
                        module.setup_embeddings_and_output_layer
                        if self.mcore_gpt
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                    parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def _reset_activation_checkpointing_args(self):
        """Disables activation checkpointing completely and saves the values so that
        _restore_activation_checkpointing_args can restore them later. This function must always be
        called before _restore_activation_checkpointing_args.
        """
        # Store values to restore them later.
        self.last_activations_checkpoint_granularity = self.cfg.activations_checkpoint_granularity
        self.last_activations_checkpoint_method = self.cfg.activations_checkpoint_method
        self.last_activations_checkpoint_num_layers = self.cfg.activations_checkpoint_num_layers
        self.last_activations_checkpoint_layers_per_pipeline = self.cfg.activations_checkpoint_layers_per_pipeline

        # Reset config values. Needed for calling generate.
        self.cfg.activations_checkpoint_granularity = None
        self.cfg.activations_checkpoint_method = None
        self.cfg.activations_checkpoint_num_layers = None
        self.cfg.activations_checkpoint_layers_per_pipeline = None

        # Reset model parameters.
        for module in self.get_model_module_list():
            if self.cfg.get('mcore_gpt', False):
                module.decoder.config.recompute_granularity = None
                module.decoder.config.recompute_method = None
                module.decoder.config.recompute_num_layers = None
            else:
                module.language_model.encoder.activations_checkpoint_granularity = None
                module.language_model.encoder.activations_checkpoint_method = None
                module.language_model.encoder.activations_checkpoint_num_layers = None
                module.language_model.encoder.activations_checkpoint_layers_per_pipeline = None

    def _restore_activation_checkpointing_args(self):
        """Restores the activation checkpointing parameters using the values saved by
        _reset_activation_checkpointing_args. This function must never be called before
        _reset_activation_checkpointing_args.
        """
        # Restore config values.
        self.cfg.activations_checkpoint_granularity = self.last_activations_checkpoint_granularity
        self.cfg.activations_checkpoint_method = self.last_activations_checkpoint_method
        self.cfg.activations_checkpoint_num_layers = self.last_activations_checkpoint_num_layers
        self.cfg.activations_checkpoint_layers_per_pipeline = self.last_activations_checkpoint_layers_per_pipeline

        # Restore model parameters.
        for module in self.get_model_module_list():
            if self.cfg.get('mcore_gpt', False):
                module.decoder.config.recompute_granularity = self.last_activations_checkpoint_granularity
                module.decoder.config.recompute_method = self.last_activations_checkpoint_method
                module.decoder.config.recompute_num_layers = self.last_activations_checkpoint_num_layers
            else:
                module.language_model.encoder.activations_checkpoint_granularity = (
                    self.last_activations_checkpoint_granularity
                )
                module.language_model.encoder.activations_checkpoint_method = self.last_activations_checkpoint_method
                module.language_model.encoder.activations_checkpoint_num_layers = (
                    self.last_activations_checkpoint_num_layers
                )
                module.language_model.encoder.activations_checkpoint_layers_per_pipeline = (
                    self.last_activations_checkpoint_layers_per_pipeline
                )

    def _reset_sequence_parallelism_args(self):
        """Disables sequence parallelism completely and saves the values so that
        _restore_sequence_parallelism_args can restore them later. This function must always be
        called before _restore_sequence_parallelism_args.
        """
        # Store values to restore them later.
        self.last_sequence_parallel = self.cfg.sequence_parallel

        # Reset config values. Needed for calling generate.
        self.cfg.sequence_parallel = False
        self.model_parallel_config.sequence_parallel = False
        self.transformer_config.sequence_parallel = False

        # Reset model parameters.
        for module in self.get_model_module_list():
            for mod in module.modules():
                if hasattr(mod, "sequence_parallel"):
                    mod.sequence_parallel = False

    def _restore_sequence_parallelism_args(self):
        """Restores the sequence parallelism parameters using the values saved by
        _reset_sequence_parallelism_args. This function must never be called before
        _reset_sequence_parallelism_args.
        """
        # Restore config values.
        self.cfg.sequence_parallel = self.last_sequence_parallel
        self.model_parallel_config.sequence_parallel = self.last_sequence_parallel
        self.transformer_config.sequence_parallel = self.last_sequence_parallel

        # Restore model parameters.
        for module in self.get_model_module_list():
            for mod in module.modules():
                if hasattr(mod, "sequence_parallel"):
                    mod.sequence_parallel = self.last_sequence_parallel

    def build_transformer_config(self) -> TransformerConfig:
        """Builds the megatron core gpt transformer config for the model.
        For attributes in the nemo model config that are the same
        as the megatron core TransformerConfig, we will use the value from the nemo model config.
        For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """

        if self.cfg.num_layers % self.cfg.get('pipeline_model_parallel_size', 1) != 0:
            raise ValueError(
                f"num_layers ({self.cfg.num_layers}) should be divisible by "
                f"pipeline_model_parallel_size ({self.cfg.get('pipeline_model_parallel_size', 1)})"
            )

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
