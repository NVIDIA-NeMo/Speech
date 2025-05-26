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

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoConfig
from vllm.config import ModelConfig, ModelImpl, PoolerConfig, _get_and_verify_dtype, _get_and_verify_max_len
from vllm.transformers_utils.config import get_hf_text_config

from nemo.export.tarutils import TarPath
from nemo.export.utils import is_nemo2_checkpoint
from nemo.export.vllm.model_converters import get_model_converter


class NemoModelConfig(ModelConfig):
    """
    This class pretents to be a vllm.config.ModelConfig (with extra fields) but skips
    some of its initialization code, and initializes the configuration from a Nemo checkpoint instead.
    """

    def __init__(
        self,
        nemo_checkpoint: str,
        model_dir: str,
        model_type: str,
        tokenizer_mode: str,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        override_neuron_config: Optional[Dict[str, Any]] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[dict] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: bool = False,
        max_seq_len_to_capture: Optional[int] = 8192,
        max_logprobs: int = 5,
        disable_sliding_window: bool = False,
        disable_cascade_attn: bool = False,
        use_async_output_proc: bool = False,
        disable_mm_preprocessor_cache: bool = False,
        logits_processor_pattern: Optional[str] = None,
        override_pooler_config: Optional[PoolerConfig] = None,
        override_generation_config: Optional[Dict[str, Any]] = None,
        enable_sleep_mode: bool = False,
        model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
    ) -> None:
        # Don't call ModelConfig.__init__ because we don't want it to call
        # transformers.AutoConfig.from_pretrained(...)

        # TODO: Do something about vLLM's call to _load_generation_config_dict in LLMEngine.__init__
        # because it calls transformers.GenerationConfig.from_pretrained(...), which tries to download things

        self.nemo_checkpoint = nemo_checkpoint
        self.model = model_dir
        self.model_type = model_type
        self.tokenizer = None
        self.tokenizer_mode = tokenizer_mode
        self.skip_tokenizer_init = False
        self.trust_remote_code = False
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.override_neuron_config = override_neuron_config
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.tokenizer_revision = tokenizer_revision
        self.model_impl = model_impl
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.disable_cascade_attn = disable_cascade_attn
        self.served_model_name = nemo_checkpoint
        self.multimodal_config = None
        self.mm_processor_kwargs = {}
        self.use_async_output_proc = use_async_output_proc
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.logits_processor_pattern = logits_processor_pattern
        self.generation_config = None
        self.task = "generate"  # Only the generate task is supported
        self.is_hybrid = False  # No hybrid models are supported
        self.attention_chunk_size = None  # Llama4-specific parameter
        self.override_generation_config = override_generation_config

        if self.task in ("draft", "generate"):
            self.truncation_side = "left"
        else:
            self.truncation_side = "right"

        self.encoder_config = self._get_encoder_config()
        self.pooler_config = self._init_pooler_config(override_pooler_config)
        self.enable_sleep_mode = enable_sleep_mode

        from vllm.platforms import current_platform  # vLLM uses local import for current_platform

        if self.enable_sleep_mode and not current_platform.is_cuda():
            raise ValueError("Sleep mode is only supported on CUDA devices.")

        self.model_converter = get_model_converter(model_type)
        if self.model_converter is None:
            raise RuntimeError(f'Unknown model type "{model_type}"')

        if is_nemo2_checkpoint(nemo_checkpoint):
            nemo_checkpoint: Path = Path(nemo_checkpoint)
            tokenizer_config = OmegaConf.load(nemo_checkpoint / "context/model.yaml").tokenizer
            if ('additional_special_tokens' in tokenizer_config) and len(
                tokenizer_config['additional_special_tokens']
            ) == 0:
                del tokenizer_config['additional_special_tokens']

            tokenizer_config = self._change_paths_to_absolute_paths(tokenizer_config, nemo_checkpoint)
            with (nemo_checkpoint / "context/model.yaml").open('r') as config_file:
                self.nemo_model_config: dict = yaml.load(config_file, Loader=yaml.SafeLoader)
            hf_args = self._load_hf_arguments(self.nemo_model_config['config'])

            tokenizer = instantiate(tokenizer_config)
            hf_args['vocab_size'] = tokenizer.original_vocab_size
            self.model_converter.convert_config(self.nemo_model_config['config'], hf_args)
            # In transformers ~= 4.52.0, the config for model_type="mixtral" loads with head_dim=None
            # which causes issues down the way in vLLM in MixtralAttention class. One possible fix is
            # to delete head_dim from the config if it is None.
            self.hf_config = AutoConfig.for_model(model_type, **hf_args)
            assert "huggingface" in tokenizer_config["_target_"]
            tokenizer_id = tokenizer_config["pretrained_model_name"]
        else:
            with TarPath(nemo_checkpoint) as archive:
                with (archive / "model_config.yaml").open("r") as model_config_file:
                    self.nemo_model_config = yaml.load(model_config_file, Loader=yaml.SafeLoader)
                    hf_args = self._load_hf_arguments(self.nemo_model_config)
                    self.model_converter.convert_config(self.nemo_model_config, hf_args)
                self.hf_config = AutoConfig.for_model(model_type, **hf_args)
            assert self.nemo_model_config["tokenizer"]["library"] == "huggingface"
            tokenizer_id = self.nemo_model_config["tokenizer"]["type"]
        self.tokenizer = tokenizer_id

        self.hf_config.architectures = [self.model_converter.get_architecture()]
        if self.rope_scaling is not None:
            self.hf_config['rope_scaling'] = rope_scaling

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
        )
        self.is_attention_free = self._init_attention_free()
        self.has_inner_state = self._init_has_inner_state()
        self.has_noops = self._init_has_noops()

        self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

    @staticmethod
    def _change_paths_to_absolute_paths(tokenizer_config: Dict[Any, Any], nemo_checkpoint: Path) -> Dict[Any, Any]:
        """
        Creates absolute path to the local tokenizers. Used for NeMo 2.0.

        Args:
            tokenizer_config (dict): Parameters for instantiating the tokenizer.
            nemo_checkpoint (path): Path to the NeMo2 checkpoint.
        Returns:
            dict: Updated tokenizer config.
        """
        context_path = nemo_checkpoint / 'context'

        # 'pretrained_model_name' -- huggingface tokenizer case
        # 'model_path' -- sentencepiece tokenizer
        path_keys = ['pretrained_model_name', 'model_path']

        for path_key in path_keys:
            if path := tokenizer_config.get(path_key, None):
                tokenizer_path = context_path / path
                if not tokenizer_path.exists():
                    continue

                tokenizer_config[path_key] = str(tokenizer_path.resolve())

        return tokenizer_config

    def _load_hf_arguments(self, nemo_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps argument names used in NeMo to their corresponding names in HF.
        """

        hf_to_nemo_dict = {
            'hidden_size': 'hidden_size',
            'intermediate_size': 'ffn_hidden_size',
            'num_hidden_layers': 'num_layers',
            'num_attention_heads': 'num_attention_heads',
            'num_key_value_heads': 'num_query_groups',
            # 'hidden_act': 'activation', ## <- vLLM has good defaults for the models, nemo values are wrong
            'num_local_experts': 'num_moe_experts',
            'max_position_embeddings': ['max_position_embeddings', 'encoder_seq_length'],
            'tie_word_embeddings': 'share_embeddings_and_output_weights',
            'rms_norm_eps': 'layernorm_epsilon',
            'attention_dropout': 'attention_dropout',
            'initializer_range': 'init_method_std',
            'norm_epsilon': 'layernorm_epsilon',
            'rope_theta': 'rotary_base',
            'use_bias': ['bias', 'add_bias_linear'],
        }

        hf_args = {}
        for hf_arg, nemo_arg in hf_to_nemo_dict.items():
            if not isinstance(nemo_arg, list):
                nemo_arg = [nemo_arg]

            for nemo_arg_option in nemo_arg:
                value = nemo_config.get(nemo_arg_option)
                if value is not None:
                    hf_args[hf_arg] = value
                    break

        return hf_args

    def try_get_generation_config(self, *args, **kwargs):
        """
        Prevent vLLM from trying to load a generation config
        """
        nemo_path = Path(self.nemo_checkpoint)
        generation_config_path = nemo_path / "context" / "artifacts" / "generation_config.json"
        if generation_config_path.exists():
            with generation_config_path.open("r") as f:
                return json.load(f)

        return {}
