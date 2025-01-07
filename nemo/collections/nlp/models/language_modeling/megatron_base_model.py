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

import gc
import itertools
import os
import re
from dataclasses import fields
from datetime import datetime
from typing import Any, Dict, Optional, Union

import omegaconf
import torch
import torch.nn as nn
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import _FxValidator
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.attention import HAVE_FLASH_ATTENTION
from nemo.collections.nlp.modules.common.megatron.clip_grads import (
    clip_grad_norm_distributed_optimizer,
    clip_grad_norm_fp32,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.nlp_overrides import NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, GradScaler
from nemo.collections.nlp.parts.utils_funcs import activation_to_func
from nemo.core.optim import MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.utils import AppState, logging, str_to_dtype
from nemo.utils.get_rank import is_global_rank_zero

try:
    from megatron.core import ModelParallelConfig, parallel_state
    from megatron.core.distributed import DistributedDataParallel as McoreDDP
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal, scaled_init_method_normal

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = TransformerConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.num_microbatches_calculator import get_current_global_batch_size, get_num_microbatches

except (ImportError, ModuleNotFoundError):
    logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
    from apex.transformer.pipeline_parallel.utils import get_current_global_batch_size, get_num_microbatches

try:
    from megatron.core import Timers

    HAVE_MEGATRON_CORE_TIMERS = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE_TIMERS = False

__all__ = ["MegatronBaseModel"]


class MegatronBaseModel(NLPModel):
    """
    Megatron base class. All NeMo Megatron models inherit from this class.

    - Initialize the model parallel world for nemo.
    - Turn on all of the nvidia optimizations.
    - If `cfg.tokenizer` is available, it loads the tokenizer and pad the vocab to the
      correct size for tensor model parallelism.
    - If using distributed optimizer, configure to be compatible
      with O2 level optimizations and/or model parallelism.
    - Perform gradient clipping: `grad_clip_pl_default` triggers
      the PyTorch Lightning default implementation, `with_distributed_adam` triggers
      the distributed optimizer's implementation, `megatron_amp_O2` triggers gradient clipping on the main grads,
      and otherwise gradient clipping is performed on the model grads.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, no_lm_init=True):

        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        if trainer is None:
            raise ValueError(f"Trainer cannot be None for Megatron-based models. Please provide a PTL trainer object.")

        if cfg.get('use_flash_attention', False) and not HAVE_FLASH_ATTENTION:
            raise ImportError(
                "flash_attn was not found. Please see the installation instructions: https://github.com/HazyResearch/flash-attention."
                "If you use flash_attn with triton. Please install triton==2.0.0.dev20221202."
            )

        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None

        with open_dict(cfg):
            if cfg.get('precision', None) is None:
                cfg.precision = trainer.precision

        super().__init__(cfg, trainer=trainer, no_lm_init=no_lm_init)

        # TODO: @maanug-nv consolidate into one attribute (requires lots of changes in subclasses)
        self.torch_dtype = utils_funcs.torch_dtype_from_precision(self.cfg.precision)  # Mixed precision datatype
        self.autocast_dtype = self.torch_dtype  # Mixed precision datatype
        # instantiate weights in mixed precision datatype if using megatron amp O2
        self.params_dtype = (
            self.torch_dtype
            if self.torch_dtype in [torch.bfloat16, torch.float16] and self.cfg.get('megatron_amp_O2', False)
            else torch.float32
        )

        self.megatron_timers = None
        if self.cfg.get('enable_megatron_timers', False) and HAVE_MEGATRON_CORE_TIMERS:
            self.megatron_timers_cfg = dict(self.cfg.get('megatron_timer_kwargs', dict()))
            if 'log_every_n_steps' not in self.megatron_timers_cfg:
                self.megatron_timers_cfg['log_every_n_steps'] = self.trainer.log_every_n_steps
            if 'log_option' not in self.megatron_timers_cfg:
                self.megatron_timers_cfg['log_option'] = 'minmax'  # minmax, max, all
            if 'barrier' not in self.megatron_timers_cfg:
                self.megatron_timers_cfg['barrier'] = False
            self.megatron_timers = Timers(log_level=2, log_option=self.megatron_timers_cfg['log_option'])

        # set the megatron core model parallel config
        self.model_parallel_config: ModelParallelConfig = self.build_model_parallel_config()

        self.use_mcore_dist_optim = cfg.optim.get('name') == 'mcore_distributed_optim'
        self.with_distributed_adam = cfg.optim.get('name') == 'distributed_fused_adam' or self.use_mcore_dist_optim
        self.with_megatron_fused_adam = cfg.optim.get('name') == 'megatron_fused_adam'

        # used in NVIDIA NGC PyTorch containers
        self._enable_nvidia_optimizations()

        if self._cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        # Overrides used when converting checkpoints
        if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
            app_state = AppState()
            init_world_size = (
                app_state.tensor_model_parallel_size
                * app_state.pipeline_model_parallel_size
                * (app_state.expert_model_parallel_size or 1)
            )
            init_global_rank = app_state.global_rank
            init_local_rank = app_state.local_rank
        else:
            init_world_size = trainer.world_size
            init_global_rank = trainer.global_rank
            init_local_rank = trainer.local_rank

        # Set virtual pipeline size to None if it is 1 and
        # confirm that the number of model chunks is the same across all pipeline stages.
        vp_size = self.cfg.get('virtual_pipeline_model_parallel_size', None)

        if vp_size is not None:
            if vp_size == 1:
                vp_size = None
            else:
                assert (
                    self.cfg.num_layers // self.cfg.pipeline_model_parallel_size
                ) % vp_size == 0, 'Make sure the number of model chunks is the same across all pipeline stages.'

        initialize_model_parallel_for_nemo(
            world_size=init_world_size,
            global_rank=init_global_rank,
            local_rank=init_local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            expert_model_parallel_size=cfg.get('expert_model_parallel_size', 1),
            pipeline_model_parallel_size=cfg.get('pipeline_model_parallel_size', 1),
            virtual_pipeline_model_parallel_size=vp_size,
            pipeline_model_parallel_split_rank=cfg.get('pipeline_model_parallel_split_rank', 0),
            use_tp_pp_dp_mapping=cfg.get('use_tp_pp_dp_mapping', False),
            context_parallel_size=cfg.get('context_parallel_size', 1),
            micro_batch_size=cfg.get('micro_batch_size'),
            global_batch_size=cfg.get('global_batch_size'),
            rampup_batch_size=cfg.get('rampup_batch_size', None),
            use_fp8=cfg.get('fp8', False),
            init_mpi_proc_group=cfg.get('ub_tp_comm_overlap', False)
            and cfg.get('ub_tp_comm_bootstrap_backend', 'nccl') == 'mpi',
            seed=self.cfg.get('seed', 1234),
            apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
            use_te_rng_tracker=self.cfg.get('use_te_rng_tracker', False),
        )

        # This must be called after initialize model parallel since it needs to know the data parallel size
        self._validate_and_override_config()

        # set the megatron core model parallel config
        self.model_parallel_config: ModelParallelConfig = self.build_model_parallel_config()

        self.grad_clip_pl_default = False  # use pytorch default for gradient clipping. Default False

        if hasattr(self._cfg, "tokenizer") or (
            hasattr(self._cfg, "encoder_tokenizer") and hasattr(self._cfg, "decoder_tokenizer")
        ):
            # build tokenizer (defaults to nemo supported tokenizers)
            self._build_tokenizer()

            # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
            self._build_vocab()

        # TODO: remove this when PTL 1.7.3 is released
        _FxValidator.functions["configure_gradient_clipping"] = {
            "allowed_on_step": (False, True),
            "allowed_on_epoch": (False, True),
            "default_on_step": True,
            "default_on_epoch": False,
        }

        self.gc_interval = cfg.get('gc_interval', 0)
        # Do manual garbage collection during validation routine when gc_interval > 0
        self.gc_in_validation = bool(int(os.getenv("NEMO_MANUAL_GC_IN_VALIDATION", 1)))
        assert self.gc_interval >= 0, "gc_interval should be an integer value larger than or equal to 0."
        # If gc_interval > 0, memory garbage collection is manually controlled.
        # The automatic garbage collector sould be disabled before training starts.
        if self.gc_interval > 0:
            gc.disable()
            self.validation_global_step = 1

        self.use_fsdp = cfg.get('fsdp', False)

    def setup_transformer_engine_tp_groups(self):
        """This should be called after model parallel groups have been initialized
        and only needs to be called when using Transformer Engine.
        """
        for module in self.get_model_module_list():
            """Set TP group
            Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py#L398
            """
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(module.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_tensor_parallel_group"):
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    child.set_tensor_parallel_group(tp_group)

    def setup_transformer_engine_cp_groups(self):
        """This should be called after context parallel groups have been initialized
        and only needs to be called when using Transformer Engine.
        """
        cp_stream = torch.cuda.Stream()

        for module in self.get_model_module_list():
            """Set context parallel running
            Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            """
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(module.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_context_parallel_group"):
                    child.set_context_parallel_group(
                        parallel_state.get_context_parallel_group(),
                        parallel_state.get_context_parallel_global_ranks(),
                        cp_stream,
                    )

    def _wrap_model_for_O2(self):
        """Wraps self.model in a float16 wrapper if the model is using megatron amp O2.
        Args:
            model: The model to wrap. Can be a list of modules or a single module.
        Returns:
            The wrapped model. Returns a list of wrapped modules or a single wrapped module.
        """
        is_mcore_model = (
            self.__dict__.get('mcore_gpt', False)
            or self.__dict__.get('mcore_bert', False)
            or self.__dict__.get('mcore_t5', False)
        )

        Float16Wrapper = MCoreFloat16Module if is_mcore_model else Float16Module

        nemo_args = {'config': self.model_parallel_config, 'precision': self.cfg.precision}

        if type(self).__name__ == 'MegatronGPTModel':
            nemo_args['share_token_embeddings'] = self.cfg.get('share_embeddings_and_output_weights', True)

        if is_mcore_model:
            mcore_args = {
                'config': self.transformer_config,
            }
        else:
            mcore_args = None

        args = mcore_args if is_mcore_model else nemo_args
        # Model wrapper to convert both model and inputs to half precision
        if isinstance((self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model), list):
            converted_model = []
            for module in self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model:
                args['module'] = module
                converted_model.append(Float16Wrapper(**args))
            if hasattr(self, "enc_dec_model"):
                self.enc_dec_model = converted_model
            else:
                self.model = converted_model
        else:
            args['module'] = self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model
            if hasattr(self, "enc_dec_model"):
                self.enc_dec_model = Float16Wrapper(**args)
            else:
                self.model = Float16Wrapper(**args)
        args.pop('module')

    def get_model_module_list(self):
        def extract_module(model):
            if isinstance(model, (McoreDDP, Float16Module, MCoreFloat16Module)):
                return extract_module(model.module)
            else:
                return model

        if isinstance((self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model), list):
            return list(map(extract_module, (self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model)))
        else:
            return [extract_module(self.enc_dec_model if hasattr(self, "enc_dec_model") else self.model)]

    def _reconfigure_limit_batches(self, limit_batches, dataloader, mode):
        """
        Reconfigure trainer.limit_val_batches for pretraining
        """
        # Override limit_batches in terms of num microbatches and so there are limit_batches//num_micro_batches num of global batches
        if isinstance(limit_batches, int):
            limit_batches *= get_num_microbatches()
        else:
            assert isinstance(limit_batches, float)
            # Don't reconfigure if limit_batches is 0.0 or if there's no dataloader
            if limit_batches == 0.0 or dataloader is None:
                return
            # len(dataloader) returns len as num of microbatches
            dl_len_in_micro_batches = len(dataloader)
            if len(dataloader) != float("inf"):
                if limit_batches == 1.0:
                    limit_batches = dl_len_in_micro_batches
                else:
                    limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                    if limit_micro_batches == 0 and limit_batches > 0.0:
                        min_percentage = 1.0 / len(dataloader)
                        raise MisconfigurationException(
                            f"You requested to check {limit_batches} of the val_dataloader but"
                            f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                            f" `limit_val_batches` argument. Try at least"
                            f" `limit_val_batches={min_percentage}`"
                        )
                    # Make sure trainer.limit_val_batches is a multiple of num of microbatches
                    if limit_micro_batches < get_num_microbatches():
                        limit_batches = get_num_microbatches()
                    else:
                        limit_batches = limit_batches - limit_batches % get_num_microbatches()

        if mode == 'train':
            self.trainer.limit_train_batches = limit_batches
        else:
            self.trainer.limit_val_batches = limit_batches

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()

    def _enable_nvidia_optimizations(self):
        "These optimizations are present in NVIDIA NGC PyTorch Containers"

        # NVIDIA container version check
        nvidia_torch_version = os.getenv('NVIDIA_PYTORCH_VERSION', None)

        def is_official_release_version(nvidia_torch_version):
            return re.fullmatch("[0-9][0-9]\.[0-9][0-9].*", nvidia_torch_version)  # "YY.MM.*"

        # Support DLFW dev container
        if not is_official_release_version(nvidia_torch_version):
            nvidia_torch_version = datetime.now().strftime('%y.%m')

        if nvidia_torch_version is not None:
            try:
                NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split('.')[0])
            except Exception:
                NVIDIA_TORCH_MAJOR = 0
            try:
                NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split('.')[1][:2])
            except Exception:
                NVIDIA_TORCH_MINOR = 0

            # Apex Persistent layer norm is supported from Nvidia PyTorch container v21.11
            # This only depends on Apex version?
            if NVIDIA_TORCH_MAJOR < 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR < 11):
                self.cfg.persist_layer_norm = False

            # NVFUSER available starting with 21.11
            if (NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11)) and (
                NVIDIA_TORCH_MAJOR < 23 or (NVIDIA_TORCH_MAJOR == 23 and NVIDIA_TORCH_MINOR < 11)
            ):

                # NVFUSER
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(False)
                torch._C._jit_set_texpr_fuser_enabled(False)
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._debug_set_autodiff_subgraph_inlining(False)
        else:
            # Not a Nvidia container. NVFUSER Dependency check is on users
            pass

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        if hasattr(self._cfg.tokenizer, "sentencepiece_legacy"):
            legacy = self._cfg.tokenizer.sentencepiece_legacy
        else:
            legacy = True if self._cfg.tokenizer.library == 'sentencepiece' else False
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.get("type", None),
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.get('model', None)),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.get('vocab_file', None)),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.get('merge_file', None)),
            use_fast=self.cfg.tokenizer.get('use_fast', False),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            special_tokens=self.cfg.tokenizer.get('special_tokens', None),
            trust_remote_code=self.cfg.tokenizer.get('trust_remote_code', False),
            legacy=legacy,
            chat_template=getattr(self._cfg.tokenizer, "chat_template", None),
        )

        if self._cfg.tokenizer.get('additional_special_tokens', None) is not None:
            tokens_list = omegaconf.OmegaConf.to_object(self._cfg.tokenizer.additional_special_tokens)
            self.tokenizer.add_special_tokens(tokens_list)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.init_global_step = self.trainer.global_step

    def on_validation_start(self) -> None:
        super().on_validation_start()
        if self.gc_interval > 0 and self.gc_in_validation:
            gc.collect()

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if self.gc_interval > 0 and self.gc_in_validation:
            gc.collect()

    def build_transformer_config(self) -> TransformerConfig:
        """Builds the megatron core transformer config for the model.
        For attributes in the nemo model config that are the same
        as the megatron core TransformerConfig, we will use the value from the nemo model config.
        For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """

        # create a dictionary copy of the model config
        cfg = OmegaConf.to_container(self.cfg, resolve=True)

        # create a dict to store the transformer config arguments
        transformer_config_dict = {}

        # get model parallel configs from the base class
        model_parallel_config = self.build_model_parallel_config()

        add_bias_linear = self.cfg.get('bias', True)
        add_qkv_bias = self.cfg.get('qkv_bias', False)

        activation = self.cfg.get('activation', 'gelu')
        gated_linear_unit = activation.endswith('glu')
        # TODO: need to check which activation functions are supported in mcore
        activation_func = activation_to_func(activation, openai_gelu=self.cfg.get("openai_gelu", False))

        normalization = self.cfg.get('normalization', 'LayerNorm')

        init_method_std = self.cfg.get('init_method_std', 0.02)
        # default used in mcore
        init_method = init_method_normal(init_method_std)

        output_layer_init_method = init_method
        num_layers = self.cfg.get('num_layers', 1)
        use_scaled_init_method = self.cfg.get('use_scaled_init_method', True)
        if use_scaled_init_method:
            output_layer_init_method = scaled_init_method_normal(init_method_std, num_layers=num_layers)

        attention_softmax_in_fp32 = False  # not currently used in NeMo unless apply_query_key_layer_scaling is True
        apply_query_key_layer_scaling = self.cfg.get('apply_query_key_layer_scaling', False)

        rotary_interleaved = self.cfg.get('rotary_interleaved', False)

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

        bias_activation_fusion = self.cfg.get('bias_activation_fusion', True)

        bias_dropout_fusion = self.cfg.get('bias_dropout_add_fusion', True)

        apply_rope_fusion = self.cfg.get('apply_rope_fusion', False)

        # TODO: need to check if recompute APIs are matching up properly
        recompute_granularity = self.cfg.get('activations_checkpoint_granularity', None)
        recompute_method = self.cfg.get('activations_checkpoint_method', None)
        recompute_num_layers = self.cfg.get('activations_checkpoint_num_layers', None)

        tp_only_amax_red = self.cfg.get('tp_only_amax_red', False)

        attention_backend = self.cfg.get('attention_backend', "auto")
        attention_backend = AttnBackend[attention_backend]

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
            'tp_only_amax_red': tp_only_amax_red,
            'attention_backend': attention_backend,
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

        return transformer_config

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        after = ((after + multiple - 1) // multiple) * multiple
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def get_parameters_with_grad(self):
        """
        Get all parameters with grad from optimizer param groups
        """
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                if (
                    param.grad is not None
                ):  # (@adithyare) adapter training with pp>1 can result in params with no grads
                    params.append(param)
        return params

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
        We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.with_megatron_fused_adam or self.use_mcore_dist_optim:
            # Gradient clipping is done in optimizer step
            return

        if self.grad_clip_pl_default:
            # use the default behavior
            return super().configure_gradient_clipping(*args, **kwargs)

        if self.with_distributed_adam:
            grad_norm = clip_grad_norm_distributed_optimizer(self._optimizer, clip_val)
        else:
            if self.megatron_amp_O2:
                # grep fp32 master parameters for gradient clipping
                parameters = self._optimizer.get_parameters_with_grad()
            else:
                parameters = self.get_parameters_with_grad()
            grad_norm = clip_grad_norm_fp32(
                parameters=parameters,
                max_norm=clip_val,
                use_fsdp=self.use_fsdp,
            )

        self.log('grad_norm', grad_norm, rank_zero_only=True, batch_size=1)

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks.
        Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
        """
        # Bucketize and all-reduce
        buckets = {}
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            coalesced /= parallel_state.get_data_parallel_world_size(with_context_parallel=True)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def reduce_overlap_gradients(self, params=None):
        """Reduce grads if overlapped grad sync is enabled

        Used for pipeline parallelism with the distributed Adam
        optimizer. In the first pipeline stage, the grad sync is
        overlapped with the final backward pass. In other pipeline
        stages, the grad sync is deferred until the bubble overhead.

        """
        if self.with_distributed_adam and self._optimizer.overlap_grad_sync:
            if params is None:
                params = self._optimizer.parameters()
            self._optimizer.try_grad_sync(params)

    def sync_overlap_parameters(self, params=None):
        if self.with_distributed_adam:
            self._optimizer._try_start_bucket_param_sync(params)

    def on_train_batch_end(self, outputs, dataloader_iter: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        super().on_train_batch_end(outputs, dataloader_iter, batch_idx)

        # Megatron Timers
        if self.megatron_timers:
            if self.global_step % self.megatron_timers_cfg["log_every_n_steps"] == 0:
                logging.info(
                    "\n " + self.megatron_timers.get_all_timers_string(barrier=self.megatron_timers_cfg["barrier"])
                )

        # TODO: Replace with newer override for scheduler.step() instead of
        # search for plugins for fp16 GradScalar
        if self.trainer.precision_plugin is not None and isinstance(
            self.trainer.precision_plugin, MixedPrecisionPlugin
        ):
            precision_plugin = self.trainer.precision_plugin

            if (
                hasattr(precision_plugin, 'scaler')
                and precision_plugin.scaler is not None
                and isinstance(precision_plugin.scaler, GradScaler)
            ):
                grad_scaler = precision_plugin.scaler

                # If the grad scaler skipped its optimizer step due to infs/nans,
                # decrement the step of all schedulers.
                if grad_scaler.optimizer_update_skipped is not None and grad_scaler.optimizer_update_skipped is True:
                    scheduler_cfgs = self.trainer.lr_scheduler_configs

                    if not scheduler_cfgs or not self.trainer.lightning_module.automatic_optimization:
                        return

                    for scheduler_cfg in scheduler_cfgs:
                        # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                        # as well as update the optimizer lr in all param groups
                        scheduler_cfg.scheduler.last_epoch -= 2
                        scheduler_cfg.scheduler.step()

                    # Removing the line below because it messes up train_valid_test_num_samples calculation.
                    # self.trainer.fit_loop.max_steps = self.trainer.fit_loop.max_steps + 1

                    # Reset the optimizer update skipped to `None` - this is to prevent scheduler no-ops during
                    # accumulated gradient updates.
                    grad_scaler.optimizer_update_skipped = None

        if self.gc_interval > 0 and (self.trainer.global_step % self.gc_interval == 0):
            gc.collect()

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

        if self.gc_interval > 0 and self.gc_in_validation:
            if self.validation_global_step % self.gc_interval == 0:
                gc.collect()
            self.validation_global_step += 1

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Ensure `max_steps` is set correctly
        optim_config = self._optim_config_copy(optim_config)
        if optim_config is not None and 'sched' in optim_config and optim_config.sched.get('max_steps') is None:
            with open_dict(optim_config):
                optim_config.sched.max_steps = self._get_max_steps()

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()

        def get_config_arg(key: str, default_value: Optional[Any] = None) -> Any:
            """Get keyword argument from config"""
            val = None
            if val is None and optim_kwargs:
                val = optim_kwargs.get(key, None)
            if val is None and optim_config:
                val = optim_config.get(key, None)
            if val is None and self._cfg.optim:
                val = self._cfg.optim.get(key, None)
            if val is None:
                val = default_value
            return val

        if self.with_distributed_adam:
            # Allocate contiguous grad buffer to avoid extra copies
            optim_kwargs['contiguous_grad_buffer'] = get_config_arg('contiguous_grad_buffer', True)
            if self.megatron_amp_O2 and not optim_kwargs['contiguous_grad_buffer']:
                raise ValueError(
                    "Distributed Adam optimizer requires contiguous param buffer for O2. "
                    "Either enable contiguous_grad_buffer or disable megatron_amp_O2."
                )

            # Optimizer dtype
            optim_dtype = str_to_dtype(get_config_arg('dtype', torch.float32))
            optim_kwargs['dtype'] = optim_dtype

            # Match param allgather with model dtype
            model_dtype = torch.float32
            if self.megatron_amp_O2 and hasattr(self, 'autocast_dtype'):
                model_dtype = self.autocast_dtype
            # Don't override user desired value
            if 'param_sync_dtype' not in optim_config:
                optim_kwargs['param_sync_dtype'] = model_dtype

            # Determine whether to store master params in optimizer
            if 'store_params' not in optim_config:
                if self.cfg.get('fp8_params', False):
                    optim_kwargs['store_params'] = True
                elif optim_dtype == model_dtype:
                    optim_kwargs['store_params'] = False
                elif optim_dtype == torch.float32 and model_dtype == torch.bfloat16:
                    optim_kwargs['store_params'] = False
                    optim_kwargs['store_param_remainders'] = True
                else:
                    optim_kwargs['store_params'] = True

        return super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)

    def configure_optimizers(self):
        self.setup_optimization()

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_O2 and not self.with_distributed_adam and self._optimizer is not None:
            if self.torch_dtype == torch.bfloat16:
                fp32_grad_accum = True
                contiguous_grad_bucket = True
            elif self.torch_dtype == torch.float16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False
                raise ValueError(
                    "fp16 training is not yet supported with O2. Please set megatron_amp_O2 to False in the model config."
                )

            # if using tensor parallel only, we automatically use async grad all-reduce
            # if using pipeline parallel or sequence parallel or gradient accumulation fusion, then we disable it
            if (
                self.cfg.get('pipeline_model_parallel_size', 1) == 1
                and not (
                    self.cfg.get('sequence_parallel', False) or self.cfg.get('gradient_accumulation_fusion', False)
                )
                and self.cfg.get('async_grad_allreduce', True)
            ):
                async_grad_allreduce = True
            else:
                async_grad_allreduce = False

            if async_grad_allreduce:
                # we need this to be configurable until make_nccl_premul_sum is in public PyTorch.
                # currently cannot be imported in PyTorch 1.12.0
                grad_div_ar_fusion = self.cfg.get('grad_div_ar_fusion', False)
            else:
                grad_div_ar_fusion = False

            self._optimizer = MainParamsOptimizerWrapper(
                self._optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
                grad_div_ar_fusion=grad_div_ar_fusion,
                grad_allreduce_chunk_size_mb=self.cfg.get('grad_allreduce_chunk_size_mb', 125),
            )

            if hasattr(self._cfg.optim, 'sched'):
                sched_config = self._cfg.optim.sched
                self._scheduler = prepare_lr_scheduler(
                    optimizer=self._optimizer,
                    scheduler_config=sched_config,
                    train_dataloader=self._train_dl,
                )

        if getattr(self._cfg.optim, 'sched', None) is not None and self._scheduler is None:
            # The error below refers in particular to logs from `prepare_lr_scheduler()` (when it retunrs `None`).
            raise AssertionError(
                "A scheduler config exists but no scheduler was instantiated! Previous logs may help identify the "
                "root cause of this issue."
            )

        # Configure distributed optimizer
        if self.with_distributed_adam and not self.use_mcore_dist_optim:

            # Initialize param buckets if explicitly provided
            if getattr(self, 'distributed_adam_buckets', None) is not None:
                buckets = self.distributed_adam_buckets
                if self.cfg.get('distributed_adam_bucket_merge_size', 1) > 1:
                    # Merge buckets if needed
                    stride = self.cfg.get('distributed_adam_bucket_merge_size', 1)
                    buckets = [
                        list(itertools.chain.from_iterable(buckets[i : i + stride]))
                        for i in range(0, len(buckets), stride)
                    ]
                for bucket in buckets:
                    self._optimizer.init_params_bucket(bucket)
                try:
                    # We first attempt to only get the parameters that require grad.
                    # This is to support multimodal training in child classes
                    # where some modules might be pretrained and frozen.
                    params = self.parameters(requires_grad_only=True)
                except TypeError as e:
                    if "unexpected keyword argument 'requires_grad_only'" in str(e):
                        params = self.parameters()
                    else:
                        raise
                self._optimizer.init_params_bucket(params)
            if hasattr(self, 'distributed_adam_buckets'):
                del self.distributed_adam_buckets

            # Make sure all params are initialized so main grads are
            # available
            # Note: Consolidate grads without overlap
            overlap_params = []
            no_overlap_params = []
            for p in self.parameters():
                if p.requires_grad:
                    if getattr(p, '_disable_overlap_grad_sync', False):
                        no_overlap_params.append(p)
                    else:
                        overlap_params.append(p)
            self._optimizer.init_params(reversed(overlap_params))
            self._optimizer.init_params(reversed(no_overlap_params))

            # Initialize contiguous parameter buffer
            if self._optimizer.contiguous_param_buffer:
                self._optimizer.init_param_buffer()

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()

        if self.cfg.get('rampup_batch_size', None):
            current_global_batch_size = get_current_global_batch_size() if get_current_global_batch_size() else 1
            consumed_samples = self.prev_consumed_samples + self.if_first_step * current_global_batch_size
        else:
            consumed_samples = (
                self.init_consumed_samples
                + steps_since_resume
                * app_state.data_parallel_size
                * self.cfg.micro_batch_size
                * get_num_microbatches()
            )
        return int(consumed_samples)

    def _compute_consumed_samples_after_training_step(self):
        # Add +1 to account for the current batch, which is not counted yet in `trainer.global_step`.
        if not hasattr(self, 'init_global_step'):
            self.init_global_step = 0  # in case this method is called before training starts.
        return self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step)

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        try:
            init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
        except (ValueError, TypeError, IndexError):
            logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
            init_consumed_samples = 0

        return init_consumed_samples

    def _validate_and_override_config(self):
        """Certain configurations might be incompatible or discouraged.
        We can check for them here and override if necessary.
        """
        app_state = AppState()

        if self.cfg.get('sequence_parallel', False) and self.cfg.get('tensor_model_parallel_size', 1) == 1:
            logging.info(
                "Sequence parallel should only be used with tensor parallel size > 1. Setting sequence parallel to False"
            )
            with open_dict(self.cfg):
                self.cfg.sequence_parallel = False

        # Gradient accumulation fusion does not work with our baseline implementaiton of
        # async grad allreduce. This should be fixed!
        # For now we must disable it whenever using the baseline implementaion.
        # The distributed adam from apex does work with gradient accumulation fusion.
        distributed_fused_adam = (
            self.cfg.optim.get('name', 'fused_adam') == 'distributed_fused_adam'
            or self.cfg.optim.get('name', 'fused_adam') == 'mcore_distributed_optim'
        )
        pipeline_model_parallel_size = self.cfg.get('pipeline_model_parallel_size', 1)
        data_parallel_size = app_state.data_parallel_size

        if self.cfg.get('gradient_accumulation_fusion', False):
            if data_parallel_size > 1 and pipeline_model_parallel_size == 1 and not distributed_fused_adam:
                logging.info(
                    "When not using pipeline model parallel, gradient accumulation fusion can only be used with distributed_fused_adam."
                )
                with open_dict(self.cfg):
                    self.cfg.gradient_accumulation_fusion = False
            if self.cfg.get('fsdp', False):
                logging.info("When using FSDP, gradient accumulation cannot be fused to gradient computation.")
                with open_dict(self.cfg):
                    self.cfg.gradient_accumulation_fusion = False
            if not self.cfg.get('megatron_amp_O2', False):
                logging.info("Gradient accumulation fusion can only be used with megatron amp O2 mixed precision.")
                with open_dict(self.cfg):
                    self.cfg.gradient_accumulation_fusion = False

        if self.cfg.get('use_emha', False):
            raise ValueError('use_emha is not yet supported please set to False')

        vp_size = self.cfg.get('virtual_pipeline_model_parallel_size', None)

        if vp_size is not None:
            if vp_size == 1:
                self.cfg['virtual_pipeline_model_parallel_size'] = None
            else:
                assert (
                    self.cfg.num_layers // self.cfg.pipeline_model_parallel_size
                ) % vp_size == 0, 'Make sure the number of model chunks is the same across all pipeline stages.'

        if self.cfg.get('ub_tp_comm_overlap', False):
            if not self.cfg.get('sequence_parallel', False):
                logging.info(
                    "Pipelined tensor-parallel communication overlap is available with sequence-parallelism."
                    "Setting `ub_tp_comm_overlap` to False."
                )
                with open_dict(self.cfg):
                    self.cfg.ub_tp_comm_overlap = False

            if self.cfg.get('fsdp', False):
                logging.info(
                    "Userbuffer tensor-parallel communication overlap is not available with FSDP."
                    "Setting `ub_tp_comm_overlap` to False."
                )
                with open_dict(self.cfg):
                    self.cfg.ub_tp_comm_overlap = False

        if self.cfg.get('fsdp', False) and self.cfg.get('fp8', False):
            raise ValueError('Torch FSDP does not support FP8.')

    def is_data_parallel_rank_zero(self):
        if is_global_rank_zero():
            return True
        else:
            try:
                data_parallel_rank = parallel_state.get_data_parallel_rank()
            except:
                data_parallel_rank = None

            if data_parallel_rank is not None and data_parallel_rank == 0:
                return True
            else:
                return False

    def _get_total_params_across_model_parallel_groups_gpt_bert(self):
        """Returns the total number of parameters across all model parallel groups."""
        is_mcore_model = (
            self.__dict__.get('mcore_gpt', False)
            or self.__dict__.get('mcore_bert', False)
            or self.__dict__.get('mcore_t5', False)
        )
        # log number of parameters
        model = self.get_model_module_list()
        if isinstance(model, list):
            num_parameters_on_device = sum(
                [sum([p.nelement() for p in model_module.parameters()]) for model_module in model]
            )
            if (
                parallel_state.get_pipeline_model_parallel_world_size() > 1
                and parallel_state.is_pipeline_last_stage(ignore_virtual=True)
                and self.cfg.get('share_embeddings_and_output_weights', True)
            ):
                word_embeddings_weight = (
                    model[-1].shared_embedding_or_output_weight()
                    if is_mcore_model
                    else model[-1].word_embeddings_weight()
                )
                # substract the embedding weights on the last virtual stage
                num_word_embedding_parameters = sum([p.nelement() for p in word_embeddings_weight])
                num_parameters_on_device -= num_word_embedding_parameters
        else:
            num_parameters_on_device = sum([p.nelement() for p in model.parameters()])
            if (
                parallel_state.get_pipeline_model_parallel_world_size() > 1
                and parallel_state.is_pipeline_last_stage(ignore_virtual=True)
                and self.cfg.get('share_embeddings_and_output_weights', True)
            ):
                word_embeddings_weight = (
                    model.shared_embedding_or_output_weight() if is_mcore_model else model.word_embeddings_weight()
                )
                # substract the embedding weights on the last stage
                num_word_embedding_parameters = sum([p.nelement() for p in word_embeddings_weight])
                num_parameters_on_device -= num_word_embedding_parameters

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()

        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())

        return num_parameters_on_device, total_num_parameters

    def _get_total_params_across_model_parallel_groups_enc_dec(self, model):
        """Returns the total number of parameters across all model parallel groups."""
        # log number of parameters
        # TODO: If/when we add interleaved model parallelism, we will need to add another if/else here.
        num_parameters_on_device = sum([p.nelement() for p in model.parameters()])

        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.get_pipeline_model_parallel_rank() == self.cfg.get('pipeline_model_parallel_split_rank', 0)
            or parallel_state.is_pipeline_last_stage()
        ):
            # If the current rank is the in the decoder first stage (decoder emb) or last rank (output layer), subtract those weights since it is already accounted for in the encoder first stage.
            # TODO: If we support embedding untying with PP > 1, we will need to update this.
            num_word_embedding_parameters = sum([p.nelement() for p in model.word_embeddings_weight()])
            num_parameters_on_device -= num_word_embedding_parameters

            # Subtract decoder position embedding params that are shared with encoder.
            if (
                parallel_state.is_pipeline_stage_at_split()
                and self.cfg.encoder.get("position_embedding_type", "learned_absolute") == "learned_absolute"
            ):
                num_position_embedding_parameters = sum([p.nelement() for p in model.position_embeddings_weight()])
                num_parameters_on_device -= num_position_embedding_parameters

        # Check and remove RPE embeddings from the encoder that are replicated.
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.is_pipeline_stage_before_split()
            and not parallel_state.is_pipeline_first_stage()
            and self.cfg.encoder.get("position_embedding_type", "learned_absolute") == "relative"
        ):
            # substract the RPE params on intermediate pipeline stages.
            num_rpe_params = sum([p.nelement() for p in model.encoder_relative_position_embeddings_weight()])
            num_parameters_on_device -= num_rpe_params

        # Check and remove RPE embeddings from the decoder that are replicated.
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.is_pipeline_stage_after_split()
            and not parallel_state.is_pipeline_stage_at_split()
            and self.cfg.encoder.get("position_embedding_type", "learned_absolute") == "relative"
        ):
            # substract the RPE params on intermediate pipeline stages.
            num_rpe_params = sum([p.nelement() for p in model.decoder_relative_position_embeddings_weight()])
            num_parameters_on_device -= num_rpe_params

        # to be summed across data parallel group
        total_num_parameters = torch.tensor(num_parameters_on_device).cuda()
        torch.distributed.all_reduce(total_num_parameters, group=parallel_state.get_model_parallel_group())
        return num_parameters_on_device, total_num_parameters

    def build_model_parallel_config(self) -> ModelParallelConfig:
        """For attributes in the nemo model config that are the same as the
        megatron core ModelParallelConfig we will use the value from the nemo config.
        For attributes in ModelParallelConfig that are not in the nemo model config, we add custom logic.
        """
        cfg = OmegaConf.to_container(self.cfg, resolve=True)

        # map precision related configs
        megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        # dtype used in p2p communication
        pipeline_dtype = self.torch_dtype

        # maps NeMo model configs to ModelParallelConfig from megatron core
        config_mapping = {
            "perform_initialization": True,  # initailize weights when constructing the module
            "fp16": self.torch_dtype == torch.float16
            and megatron_amp_O2,  # NeMo does not currently support fp16 training with megatron amp O2, eval and inference is supported
            "bf16": self.torch_dtype == torch.bfloat16 and megatron_amp_O2,
            "params_dtype": self.params_dtype,
            "timers": self.megatron_timers,
            "async_tensor_model_parallel_allreduce": self.cfg.get('tensor_model_parallel_world_size', 1) > 1
            and not self.cfg.get('sequence_parallel', False),
            "pipeline_dtype": pipeline_dtype,
            "grad_scale_func": (
                self.trainer.precision_plugin.scaler.scale if self.trainer.precision in ["16", "16-mixed"] else None
            ),
            "enable_autocast": not megatron_amp_O2 and self.torch_dtype in [torch.bfloat16, torch.float16],
            "autocast_dtype": self.autocast_dtype,
            "variable_seq_lengths": False,  # set dynamically during training
            "num_microbatches_with_partial_activation_checkpoints": self.cfg.get(
                'num_micro_batches_with_partial_activation_checkpoints', None
            ),
            "batch_p2p_sync": True,  # call torch.cuda.synchronize() after batch isend/rcv
            "use_ring_exchange_p2p": False,  # not supported in NeMo
            "deallocate_pipeline_outputs": False,  # not supported in NeMo
            "no_sync_func": None,  # set dynamically during training
            "grad_sync_func": None,  # set dynamically during training
            "param_sync_func": None,  # set dynamically during training
            "tp_comm_overlap": self.cfg.get('ub_tp_comm_overlap', False),
            "tp_comm_bootstrap_backend": self.cfg.get('ub_tp_comm_bootstrap_backend', 'nccl'),
        }

        # instantitate ModelParallelConfig from this dict
        mp_config_dict = {}

        for field in fields(ModelParallelConfig):
            # model config has priority
            if field.name in cfg:
                mp_config_dict[field.name] = cfg[field.name]
            # then config_mapping
            elif field.name in config_mapping:
                mp_config_dict[field.name] = config_mapping[field.name]
            else:
                logging.warning(
                    f"The model: {self} does not have field.name: {field.name} in its cfg. "
                    f"Add this key to cfg or config_mapping to make to make it configurable."
                )

        model_parallel_config = ModelParallelConfig(**mp_config_dict)

        try:
            # hidden size is needed for pipeline schedules but is not currently in ModelParallelConfig
            setattr(model_parallel_config, 'hidden_size', self.cfg.hidden_size)
        except AttributeError:
            logging.warning(
                f'hidden_size not found in {self.cfg}. Set this in model_parallel_config if using pipeline parallelism.'
            )

        return model_parallel_config

    def _val_iterator_done(self, iterator):
        """
        Check if the iterator is exhausted, if so raise a StopIteration and exit validation_step
        """
        try:
            element = next(iterator)
        except StopIteration:
            return iterator, True
        # reinsert the element back to the iterator
        return itertools.chain([element], iterator), False

    def _get_max_steps(self):
        """
        Compute the maximum number of training steps (-1 if it cannot be computed).
        """
        if getattr(self, "_trainer", None) is None:
            logging.warning("Cannot compute `max_steps` as no trainer is set")
            return -1

        if self._trainer.max_steps >= 0:
            # Note that when `trainer.max_steps` is defined, we ignore `max_epochs` (even if training may end
            # before `max_steps` is reached due to `max_epochs`). This is for backward compatibility with older
            # versions of NeMo.
            if self._trainer.max_epochs is not None and self._trainer.max_epochs >= 0:
                logging.warning(
                    "Ignoring `trainer.max_epochs` when computing `max_steps` because `trainer.max_steps` is already "
                    f"set to {self._trainer.max_steps}."
                )
            return self._trainer.max_steps

        if self._trainer.max_epochs is None or self._trainer.max_epochs < 0:
            logging.warning(
                "Cannot compute `max_steps` if neither `trainer.max_steps` nor `trainer.max_epochs` is set"
            )
            return -1

        if getattr(self, "_train_dl", None) is None:
            logging.warning("Cannot compute `max_steps` from the number of epochs as the train dataloader is not set")
            return -1

        # The number of training step per epoch is typically the number of global batches in the training set...
        num_global_batches = len(self._train_dl)
        steps_per_epoch = num_global_batches

        # ... unless it is constrained by the `limit_train_batches` option.
        limit_batches = self._trainer.limit_train_batches
        if limit_batches is not None:
            if isinstance(limit_batches, float):
                limit_batches = int(limit_batches * num_global_batches)
            steps_per_epoch = min(num_global_batches, limit_batches)

        return steps_per_epoch * self._trainer.max_epochs

    def configure_sharded_model(self):
        def find_frozen_submodules(model):
            frozen_submodules = []
            frozen_submodule_names = []
            for name, module in model.named_modules():
                if (
                    isinstance(module, nn.Module)
                    and list(module.parameters())
                    and all(not param.requires_grad for param in module.parameters())
                ):
                    frozen_submodule_names.append(name)
                    frozen_submodules.append(module)
            return frozen_submodule_names, frozen_submodules

        if self.use_fsdp:
            """Top-evel FSDP model sharding"""
            # Shard the top-level model hierarchically. We shard the strategy-unwrapped model not
            # to lose the structure of non-FSDP wrapped parameters (e.g, embedding)
            # TODO: Currently the main parameter data type is kept in fp32 (when O2=False). This needs to be
            # extended to support lower precision main parameters.
            frozen_submodule_names, frozen_submodules = find_frozen_submodules(self.model)
            for submodule in frozen_submodule_names:
                logging.debug(f"Ignoring state {submodule} in FSDP.")
            self.trainer.strategy.kwargs['ignored_states'] = frozen_submodules
            # FSDP requires uniform status of require_grads
            # Diffusion models like SD has frozen parts and needs to be added to 'ignored_states' from sharding for FSDP to work
            self.model = self.trainer.strategy._setup_model(self.model)
            # Move the CPU-initialized model (with `use_cpu_initialization=True`) to GPU, which is to avoid
            # out-of-memory carash before sharding. In case of GPU-initialized model, this is no-op.
            self.model = self.model.cuda(torch.cuda.current_device())

    def megatron_timer_start(self, name, log_level):
        if self.megatron_timers:
            self.megatron_timers(name, log_level).start(barrier=False)

    def megatron_timer_stop(self, name):
        if self.megatron_timers:
            self.megatron_timers(name).stop()

    def optimizer_step(self, *args, **kwargs):
        self.megatron_timer_start('optimizer', log_level=1)
        super().optimizer_step(*args, **kwargs)
        self.megatron_timer_stop('optimizer')
