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
from typing import List

import numpy as np
import torch
from tensorrt_llm import default_net, str_dtype_to_trt
from tensorrt_llm._utils import trt_dtype_to_str
from tensorrt_llm.functional import expand_mask, gather_last_token_logits, recv, send, shape
from tensorrt_llm.layers import AttentionParams, ColumnLinear, KeyValueCacheParams, LoraParams
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.module import Module, ModuleList

from nemo.export.trt_llm.decoder import build_decoder_layer
from nemo.export.trt_llm.model_config import DECODER_GEMMA, ModelConfig
from nemo.export.trt_llm.quantization_utils import quantize_linear
from nemo.export.trt_llm.tensorrt_llm_build import build
from nemo.export.trt_llm.tensorrt_llm_utils import (
    build_embedding_from_config,
    build_layernorm_from_config,
    print_tensorrt_llm,
)


def get_transformer_layers(mapping, num_layers):
    layers_per_pipeline_stage = num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage, (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1)
    )
    return layers_range


class ModelBuilder(Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.quantization = model_config.quantization
        self.max_position_embeddings = model_config.max_position_embeddings
        self.hidden_act = model_config.hidden_act

        self._dtype = str_dtype_to_trt(model_config.dtype)
        self._kv_dtype = self._dtype
        self._tensor_parallel = model_config.mapping.tp_size
        self._vocab_size = model_config.vocab_size
        self._hidden_size = model_config.hidden_size
        self._num_layers = len(model_config.layers)
        self._num_heads = model_config.num_attention_heads
        self._num_kv_heads = model_config.num_kv_heads
        self._head_size = (
            model_config.hidden_size // model_config.num_attention_heads
            if model_config.head_size is None
            else model_config.head_size
        )
        self._use_prompt_tuning = model_config.use_prompt_tuning
        self._add_bos = model_config.layers[0].decoder_type == DECODER_GEMMA
        self._mapping = model_config.mapping
        self.rank = model_config.mapping.rank
        self.max_lora_rank = model_config.max_lora_rank

        if self._mapping.is_first_pp_rank():
            self.vocab_embedding = build_embedding_from_config(
                model_config.vocab_embedding,
                self._dtype,
                use_prompt_tuning=self._use_prompt_tuning,
                tensor_parallel=model_config.mapping.tp_size,
                tensor_parallel_rank=model_config.mapping.tp_rank,
            )

            if model_config.positional_embedding.weight is not None:
                self.positional_embedding = build_embedding_from_config(
                    model_config.positional_embedding,
                    self._dtype,
                    tensor_parallel=model_config.mapping.tp_size,
                    tensor_parallel_rank=model_config.mapping.tp_rank,
                )

        self.layers = []
        for layer_id in get_transformer_layers(self._mapping, self._num_layers):
            model_config.layers[layer_id].max_lora_rank = self.max_lora_rank
            self.layers.append(
                build_decoder_layer(
                    model_config.layers[layer_id],
                    layer_id,
                    self._num_layers,
                    dtype=self._dtype,
                    quantization=model_config.quantization,
                    rank=self.rank,
                    tensor_parallel=self._tensor_parallel,
                    tp_group=model_config.mapping.tp_group,
                )
            )

        self.layers = ModuleList(self.layers)

        if self._mapping.is_last_pp_rank():
            self.ln_f = build_layernorm_from_config(model_config.final_layernorm, self._dtype)

    def forward(
        self,
        input_ids,
        position_ids,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
        hidden_states=None,
        lora_params=None,
    ):
        ptuning_args = []
        if self._use_prompt_tuning:
            ptuning_args = [prompt_embedding_table, prompt_tasks, prompt_vocab_size]

        if self._mapping.is_first_pp_rank():
            x = self.vocab_embedding(input_ids, *ptuning_args)
            if hasattr(self, "positional_embedding") and self.positional_embedding:
                assert position_ids
                x = x + self.positional_embedding(position_ids)
            hidden_states = x
        else:
            hidden_states = recv(hidden_states, self._mapping.prev_pp_rank())

        kv_cache_params.fill_none_tensor_list(len(self.layers))

        if use_cache:
            presents = []

        if attention_mask is not None:
            attention_mask = expand_mask(attention_mask, shape(input_ids, -1))

        for layer_idx, (layer, past) in enumerate(zip(self.layers, kv_cache_params.past_key_value,)):

            decoder_params = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "kv_cache_params": KeyValueCacheParams(
                    past_key_value=[past],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    kv_cache_block_pointers=kv_cache_params.kv_cache_block_pointers,
                    host_max_attention_window_sizes=kv_cache_params.host_max_attention_window_sizes,
                    cache_indirection=kv_cache_params.cache_indirection,
                    host_sink_token_length=kv_cache_params.host_sink_token_length,
                    host_kv_cache_block_pointers=kv_cache_params.host_kv_cache_block_pointers,
                ),
                "attention_params": attention_params,
            }

            if lora_params.lora_ranks is not None:
                decoder_params["lora_layer_params"] = lora_params.get_layer_params(layer_idx)

            hidden_states = layer(**decoder_params)

            if use_cache:
                presents.append(hidden_states[1])
                hidden_states = hidden_states[0]

        if self._mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self._mapping.next_pp_rank())

        if use_cache:
            return hidden_states, tuple(presents)
        return hidden_states


class LMHeadModelBuilder(ModelBuilder, GenerationMixin):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        if self._mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(
                self._hidden_size,
                model_config.vocab_size_padded,
                bias=False,
                dtype=self._dtype,
                tp_group=self._mapping.tp_group,
                tp_size=self._tensor_parallel,
                gather_output=True,
                share_weight=None,
            )
            self.lm_head.weight.value = model_config.lm_head.weight
            if model_config.quantization:
                self.lm_head = quantize_linear(self.lm_head, model_config.quantization, model_config.lm_head)

    def forward(
        self,
        input_ids,
        position_ids,
        use_cache=False,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        inflight_batching_args=None,
        hidden_states=None,
        lora_params=None,
    ):

        hidden_states = super().forward(
            input_ids,
            position_ids,
            use_cache,
            attention_mask,
            kv_cache_params,
            attention_params,
            prompt_embedding_table,
            prompt_tasks,
            prompt_vocab_size,
            inflight_batching_args,
            hidden_states,
            lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        if self._mapping.is_last_pp_rank():
            assert last_token_ids is not None, "Expecting last token ids to be not None"
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
            )

            # [batch_size, hidden_size] -> [batch_size, vocab_size]
            lm_logits = self.lm_head(hidden_states)
            lm_logits.mark_output("logits", str_dtype_to_trt("float16"))
        else:
            hidden_states.mark_output('hidden_states_output', self._dtype)

        if use_cache:
            if not default_net().plugin_config.paged_kv_cache:
                for i, present in zip(self._mapping.pp_layers(self._num_layers), presents):
                    present.mark_output(f'present_key_value_{i}', self._kv_dtype)
            if self._mapping.is_last_pp_rank():
                return (lm_logits, presents)
            return (hidden_states, presents)
        else:
            if self._mapping.is_last_pp_rank():
                return lm_logits
            return hidden_states

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_new_tokens,
        use_cache=True,
        max_beam_width: int = 1,
        paged_kv_cache: bool = False,
        tokens_per_block: int = 64,
        prompt_embedding_table_size: int = 0,
        lora_target_modules: List[str] = None,
    ):

        # Prepare inputs
        head_size = self._head_size
        num_heads_kv = self._num_kv_heads
        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        use_custom_all_reduce = default_net().plugin_config.use_custom_all_reduce
        use_lora_plugin = default_net().plugin_config.lora_plugin

        model_inputs = self.prepare_basic_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_new_tokens,
            num_kv_heads=num_heads_kv,
            head_size=head_size,
            num_layers=self._num_layers,
            kv_dtype=self._kv_dtype,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            gather_context_logits=False,
            gather_generation_logits=False,
            dtype=self._dtype,
            num_heads=self._num_heads,
            mapping=self._mapping,
            max_num_tokens=None,
            prompt_embedding_table_size=prompt_embedding_table_size,
            position_encoding_2d=False,
            use_lora_plugin=use_lora_plugin,
            lora_target_modules=lora_target_modules,
            max_draft_len=0,
            use_custom_all_reduce=use_custom_all_reduce,
        )

        inflight_batching_args = None

        return (
            model_inputs["input_ids"],
            model_inputs["position_ids"],
            use_cache,
            model_inputs["last_token_ids"],
            model_inputs["attention_mask"],
            KeyValueCacheParams(
                past_key_value=model_inputs['past_key_value'],
                host_past_key_value_lengths=model_inputs['host_past_key_value_lengths'],
                host_max_attention_window_sizes=model_inputs['host_max_attention_window_sizes'],
                kv_cache_block_pointers=model_inputs['kv_cache_block_pointers'],
                host_kv_cache_block_pointers=model_inputs['host_kv_cache_block_pointers'],
                cache_indirection=model_inputs['cache_indirection'],
                host_sink_token_length=model_inputs['host_sink_token_length'],
            ),
            AttentionParams(
                sequence_length=model_inputs['sequence_length'],
                context_lengths=model_inputs['context_lengths'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'],
            ),
            model_inputs['prompt_embedding_table'],
            model_inputs['tasks'],
            model_inputs['prompt_vocab_size'],
            inflight_batching_args,
            model_inputs["hidden_states_input"],
            LoraParams(
                model_inputs['lora_ranks'],
                model_inputs['lora_weights_pointers'],
                host_context_lengths=model_inputs['host_context_lengths'],
                max_context_length=max_input_len,
                host_request_types=model_inputs['host_request_types'],
            ),
        )

    def build(
        self,
        output_dir: Path,
        timing_cache: str = "",
        log_level: str = "info",
        max_batch_size: int = 1,
        max_input_len: int = 200,
        max_output_len: int = 200,
        max_beam_width: int = 1,
        parallel_build: bool = False,
        max_prompt_embedding_table_size: int = 0,
        use_inflight_batching: bool = False,
        paged_kv_cache: bool = False,
        enable_context_fmha: bool = True,
        enable_multi_block_mode: bool = False,
        use_refit: bool = False,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
    ):

        if self.rank > torch.cuda.device_count():
            print(f"warning: Rank {self.rank} larger than GPUs available ({torch.cuda.device_count()})")

        build(
            tensorrt_llm_model=self,
            output_dir=output_dir,
            mapping=self._mapping,
            dtype=trt_dtype_to_str(self._dtype),
            timing_cache=timing_cache,
            log_level=log_level,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            parallel_build=parallel_build,
            gpus_per_node=torch.cuda.device_count(),
            quantization=self.quantization,
            use_inflight_batching=use_inflight_batching,
            paged_kv_cache=paged_kv_cache,
            enable_context_fmha=enable_context_fmha,
            enable_multi_block_mode=enable_multi_block_mode,
            use_refit=use_refit,
            use_lora_plugin=use_lora_plugin,
            lora_target_modules=lora_target_modules,
            max_lora_rank=max_lora_rank,
        )

    def print(self):
        np.set_printoptions(threshold=36)
        print_tensorrt_llm(f"rank.{self.rank}", self)
