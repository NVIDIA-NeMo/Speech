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

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from transformers import AutoConfig
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len
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
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 5,
        disable_sliding_window: bool = False,
        use_async_output_proc: bool = False,
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
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.served_model_name = nemo_checkpoint
        self.multimodal_config = None
        self.mm_processor_kwargs = {}
        self.use_async_output_proc = use_async_output_proc

        self.model_converter = get_model_converter(model_type)
        if self.model_converter is None:
            raise RuntimeError(f'Unknown model type "{model_type}"')

        if is_nemo2_checkpoint(nemo_checkpoint):
            from nemo.lightning.io import load_context

            nemo_checkpoint: Path = Path(nemo_checkpoint)

            with (nemo_checkpoint / "context/model.yaml").open('r') as config_file:
                self.nemo_model_config: dict = yaml.load(config_file, Loader=yaml.SafeLoader)

            hf_args = self._load_hf_arguments(self.nemo_model_config['config'])
            tokenizer = load_context((nemo_checkpoint / "context"), subpath="model.tokenizer")

            if hasattr(tokenizer, 'bos_id'):
                tokenizer.tokenizer.bos_token_id = tokenizer.bos_id
            if hasattr(tokenizer, 'eos_id'):
                tokenizer.tokenizer.eos_token_id = tokenizer.eos_id

            hf_args['vocab_size'] = tokenizer.original_vocab_size
            self.model_converter.convert_config(self.nemo_model_config['config'], hf_args)
            self.hf_config = AutoConfig.for_model(model_type, **hf_args)
            self.nemo_model_config['tokenizer'] = tokenizer
        else:
            with TarPath(nemo_checkpoint) as archive:
                with (archive / "model_config.yaml").open("r") as model_config_file:
                    self.nemo_model_config = yaml.load(model_config_file, Loader=yaml.SafeLoader)
                    hf_args = self._load_hf_arguments(self.nemo_model_config)
                    self.model_converter.convert_config(self.nemo_model_config, hf_args)
                self.hf_config = AutoConfig.for_model(model_type, **hf_args)

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

        self._verify_tokenizer_mode()
        self._verify_embedding_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

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
            'max_position_embeddings': ['max_position_embeddings', 'encoder_seq_length'],
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
